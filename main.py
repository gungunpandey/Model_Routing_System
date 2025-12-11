import asyncio
import os
import time
from typing import Optional, Tuple

import psycopg2
import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()  # load environment variables from .env if present

# Labels used by the classifier
LABELS = ["simple", "medium", "complex"]
GPT_CLIENT: Tuple[OpenAI | None, str] | None = None  # (client, model_name)


def load_classifier():
    """Load the fine-tuned classifier from artifacts."""
    clf_path = os.getenv("CLASSIFIER_PATH", "artifacts/phi3-difficulty-classifier")
    tok = AutoTokenizer.from_pretrained(clf_path)
    model = AutoModelForSequenceClassification.from_pretrained(clf_path)
    model.eval()
    return tok, model


TOK, CLF = load_classifier()


def predict_difficulty(text: str):
    """Return (label, confidence, probabilities)."""
    with torch.no_grad():
        inputs = TOK(text, return_tensors="pt", truncation=True, max_length=512)
        logits = CLF(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        idx = int(torch.argmax(logits, dim=-1))
        return LABELS[idx], probs[idx], probs


def token_count(s: str) -> int:
    # Rough heuristic: ~4 chars/token
    return max(1, len(s) // 4)


def basic_safety_check(prompt: str) -> Tuple[bool, str]:
    """Very light safety/PII screen. Expand with a proper classifier if needed."""
    lowered = prompt.lower()
    banned = ["social security", "credit card", "ssn", "bank account", "password"]
    unsafe = ["self-harm", "terrorism", "weapon", "suicide"]
    if any(k in lowered for k in banned):
        return False, "Possible PII detected"
    if any(k in lowered for k in unsafe):
        return False, "Unsafe content detected"
    if len(prompt) > 8000:
        return False, "Prompt too long for small models; please shorten or chunk"
    return True, ""


def pick_model(label: str, conf: float, prompt: str) -> str:
    guard_escalate = any(k in prompt.lower() for k in [
        "production", "legal", "medical", "architecture", "optimize", "compliance"
    ])
    if guard_escalate:
        if label == "simple":
            return "llama-3"
        if label == "medium":
            return "gpt-4o"
        return "gpt-4o"  # complex

    if label == "simple" and conf >= 0.7:
        return "phi-3"
    if label == "medium" and conf >= 0.6:
        return "llama-3"
    return "gpt-4o"  # complex or low confidence -> gpt4


def fetch_rate(model: str) -> Tuple[int, int]:
    """
    Fetch cost rates (micros per 1k tokens) from DB.
    Falls back to hardcoded defaults if DB or row missing.
    """
    fallback = {
        "phi-3": (50, 100),
        "llama-3": (150, 300),
        "gpt-4o": (500, 1500),
    }
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute(
                """
                SELECT input_cost_per_1k_us, output_cost_per_1k_us
                FROM cost_rates
                WHERE model_name = %s
                ORDER BY effective_date DESC
                LIMIT 1
                """,
                (model,),
            )
            row = cur.fetchone()
            if row:
                return int(row[0]), int(row[1])
    except Exception as e:  # pragma: no cover
        print("rate lookup error:", e)
    return fallback[model]


def price_us(model: str, in_tokens: int, out_tokens: int) -> int:
    ri, ro = fetch_rate(model)
    return (in_tokens * ri // 1000) + (out_tokens * ro // 1000)


def get_gpt_client() -> Tuple[OpenAI | None, str] | None:
    """
    Return (client, model_name) for GPT calls.
    Uses Azure key + deployment; no OpenAI fallback.
    """
    global GPT_CLIENT
    if GPT_CLIENT:
        return GPT_CLIENT

    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GPT4")
    if not az_key or not az_deployment:
        return None

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    # Default to Azure global inference endpoint; no endpoint env needed.
    base_url = f"https://models.inference.ai.azure.com/openai/deployments/{az_deployment}"
    client = OpenAI(
        api_key=az_key,
        base_url=base_url,
        default_query={"api-version": api_version},
        default_headers={"api-key": az_key},
    )
    GPT_CLIENT = (client, az_deployment)
    return GPT_CLIENT


async def call_phi3(prompt):   return "phi3: " + prompt[:400]
async def call_llama3(prompt): return "llama3: " + prompt[:400]
async def call_gpt4o(prompt):
    gpt = get_gpt_client()
    if not gpt:
        return "gpt4o (stub): " + prompt[:400]

    client, model_name = gpt
    # Run sync OpenAI call in a thread to avoid blocking the event loop.
    resp = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
    )
    return resp.choices[0].message.content


def hf_generate(model_id: str, prompt: str) -> str:
    """
    Call HuggingFace Inference API for text generation.
    Requires HF_TOKEN in env. If missing, returns stub text.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        return f"{model_id} (stub): " + prompt[:400]

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.4}}

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # The inference API can return list or dict depending on model; handle common cases.
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    # Fallback: try first item text field
    if isinstance(data, list) and data and "text" in data[0]:
        return data[0]["text"]
    return str(data)


async def call_phi3(prompt: str):
    model_id = os.getenv("HF_PHI3_MODEL", "microsoft/phi-3-mini-4k-instruct")
    return await asyncio.to_thread(hf_generate, model_id, prompt)


async def call_llama3(prompt: str):
    model_id = os.getenv("HF_LLAMA3_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    return await asyncio.to_thread(hf_generate, model_id, prompt)


MODEL_CALL = {
    "phi-3": call_phi3,
    "llama-3": call_llama3,
    "gpt-4o": call_gpt4o,
}


def get_conn():
    return psycopg2.connect(os.getenv("PG_DSN", "postgresql://postgres:postgres@db:5432/router"))


app = FastAPI(title="Model Router", version="0.1.0")


class GenerateIn(BaseModel):
    user_id: Optional[str] = None
    prompt: str


class GenerateOut(BaseModel):
    model: str
    difficulty: str
    confidence: float
    response: str
    input_tokens: int
    output_tokens: int
    total_cost_us: int
    latency_ms: int


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateOut)
async def generate(inp: GenerateIn):
    ok, reason = basic_safety_check(inp.prompt)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    t0 = time.time()
    label, conf, _ = predict_difficulty(inp.prompt)
    model = pick_model(label, conf, inp.prompt)
    in_toks = token_count(inp.prompt)

    try:
        response = await MODEL_CALL[model](inp.prompt)
    except Exception as e:  # pragma: no cover
        if model != "gpt-4o":
            model = "gpt-4o"
            response = await MODEL_CALL[model](inp.prompt)
        else:
            raise HTTPException(status_code=502, detail=f"provider error: {e}")

    out_toks = token_count(response)
    cost = price_us(model, in_toks, out_toks)
    latency = int((time.time() - t0) * 1000)

    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO requests(user_id, prompt, difficulty, difficulty_conf,
                    routed_model, input_tokens, output_tokens, latency_ms,
                    cost_input_us, cost_output_us, total_cost_us, status, response_preview)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'success',%s)
                RETURNING id
                """,
                (
                    inp.user_id,
                    inp.prompt,
                    label,
                    conf,
                    model,
                    in_toks,
                    out_toks,
                    latency,
                    0,
                    0,
                    cost,
                    response[:500],
                ),
            )
    except Exception as e:  # pragma: no cover
        print("DB log error:", e)

    return GenerateOut(
        model=model,
        difficulty=label,
        confidence=conf,
        response=response,
        input_tokens=in_toks,
        output_tokens=out_toks,
        total_cost_us=cost,
        latency_ms=latency,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

