# Cost-Control Smart Model Router

Minimal, stepwise scaffold for a cost-aware LLM router using FastAPI, a Phi-3 difficulty classifier, and Postgres logging.

## Structure
- `main.py` — FastAPI app with routing logic, pricing stub, and logging.
- `train_classifier.py` — single script to fine-tune the difficulty classifier (Colab-ready).
- `sql/001_schema.sql` — DB schema and sample cost rate seeds.
- `requirements.txt` — Python deps.
- `Dockerfile`, `docker-compose.yml` — container setup for API + Postgres.
- `data/`, `artifacts/` — placeholders for datasets and saved models.
- `data/label_questions.py` — heuristic labeling script to clean/label raw QA into train/val.

Key behaviors:
- Safety/PII precheck rejects obvious sensitive or unsafe prompts.
- Routing picks Phi-3 / Llama-3 / GPT-4o based on predicted difficulty + confidence + guard words.
- Pricing reads from `cost_rates` in Postgres (falls back to defaults if missing).
- GPT-4o call uses Azure OpenAI only (no OpenAI fallback). Phi-3/Llama-3 use HuggingFace Inference if `HF_TOKEN` is set; otherwise return stub text.
- Labeling script filters non-questions and labels difficulty; troubleshooting prompts are at least medium.

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

## Prepare & label data
Run heuristic labeling on your QA pairs (PowerShell example):
```
python data/label_questions.py --input data/qa_final_pairs.jsonl --output data/labeled_train.jsonl --val_output data/labeled_val.jsonl --val_ratio 0.1 --min_len 25
```
See `data/labeling_rules.txt` for how filters/labels work.

## Train the classifier (Colab or local GPU)
Use the labeled files (fields: `text`, `label` in {simple, medium, complex}):
```
!pip install -r requirements.txt
!python train_classifier.py --train data/train.jsonl --val data/val.jsonl --out artifacts/phi3-difficulty-classifier
```
Sync the resulting `artifacts/phi3-difficulty-classifier` back to this repo (and mount it in Compose).

