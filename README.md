# Cost-Control Smart Model Router

Minimal, stepwise scaffold for a cost-aware LLM router using FastAPI, a Phi-3 difficulty classifier, and Postgres logging.

## Structure
- `main.py` — FastAPI app with routing logic, pricing stub, and logging.
- `train_classifier.py` — single script to fine-tune the difficulty classifier (Colab-ready).
- `sql/001_schema.sql` — DB schema and sample cost rate seeds.
- `requirements.txt` — Python deps.
- `Dockerfile`, `docker-compose.yml` — container setup for API + Postgres.
- `data/`, `artifacts/` — placeholders for datasets and saved models.

Key behaviors:
- Safety/PII precheck rejects obvious sensitive or unsafe prompts.
- Routing picks Phi-3 / Llama-3 / GPT-4o based on predicted difficulty + confidence + guard words.
- Pricing reads from `cost_rates` in Postgres (falls back to defaults if missing).
- GPT-4o call uses OpenAI if `OPENAI_API_KEY` is set; otherwise returns a stub string. Phi-3/Llama-3 are still stubbed—swap with your real clients.
- HuggingFace Inference is wired for Phi-3 and Llama-3 if you set `HF_TOKEN`. Without it, those calls return stub text.

## Quickstart
1) Install deps (local dev):
```
pip install -r requirements.txt
```

2) Run API (requires classifier weights in `artifacts/phi3-difficulty-classifier`):
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

3) Or run via Docker Compose:
```
docker compose up --build
```
The classifier directory is mounted read-only into the container.

Environment vars you may set:
- `CLASSIFIER_PATH` if your classifier lives elsewhere.
- `PG_DSN` if not using the default Postgres DSN.
- Azure GPT-4 (only): `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME_GPT4`, optional `AZURE_OPENAI_API_VERSION` (default `2024-02-15-preview`). No OpenAI key needed.
- `HF_TOKEN` to call HuggingFace Inference.
- `HF_PHI3_MODEL` (default `microsoft/phi-3-mini-4k-instruct`) and `HF_LLAMA3_MODEL` (default `meta-llama/Meta-Llama-3-8B-Instruct`) to pick hosted models.

4) Call the endpoint:
```
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","prompt":"Write a 2-sentence summary of KNN vs K-means."}'
```

## Train the classifier (Colab)
Upload your `data/train.jsonl` and `data/val.jsonl` (fields: `text`, `label` in {simple, medium, complex}) then run:
```
!pip install -r requirements.txt
!python train_classifier.py --train data/train.jsonl --val data/val.jsonl --out artifacts/phi3-difficulty-classifier
```
Sync the resulting `artifacts/phi3-difficulty-classifier` back to this repo (and mount it in Compose).

## Notes / Next steps
- Replace stub model clients and pricing with real providers and DB `cost_rates` lookup.
- Add safety/PII prechecks and better token counting.
- Add tests for routing and pricing logic.

