"""
Minimal Phi-3 difficulty classifier fine-tuning script (Colab-ready).

Usage (example):
!pip install -r requirements.txt
!python train_classifier.py --train data/train.jsonl --val data/val.jsonl --out artifacts/phi3-difficulty-classifier
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


LABELS = ["simple", "medium", "complex"]
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"


def encode_example(batch, tok, label2id):
    out = tok(batch["text"], truncation=True, padding=True, max_length=512)
    out["labels"] = [label2id[x] for x in batch["label"]]
    return out


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main(args):
    logger.info("=" * 60)
    logger.info("Starting Phi-3 Difficulty Classifier Training")
    logger.info("=" * 60)
    logger.info(f"Train file: {args.train}")
    logger.info(f"Val file: {args.val}")
    logger.info(f"Output dir: {args.out}")
    logger.info(f"Batch size: {args.batch_size}, Eval batch: {args.eval_batch_size}")
    logger.info(f"Learning rate: {args.lr}, Epochs: {args.epochs}")
    
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}
    logger.info(f"Labels: {LABELS}")

    logger.info("Loading datasets...")
    ds = load_dataset(
        "json",
        data_files={"train": args.train, "val": args.val},
    )
    logger.info(f"Loaded {len(ds['train'])} train examples, {len(ds['val'])} val examples")

    logger.info(f"Loading tokenizer from {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, padding_side="left", truncation_side="left")
    logger.info("Tokenizing datasets...")
    ds = ds.map(lambda batch: encode_example(batch, tok, label2id), batched=True).remove_columns(["text", "label"])
    logger.info("Tokenization complete")

    logger.info(f"Loading model from {MODEL_NAME}...")
    logger.info("(This may take a few minutes to download model weights...)")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    logger.info("Model loaded successfully")

    logger.info("Setting up training arguments...")
    train_args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=True,
    )

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    logger.info("DO NOT INTERRUPT - Training in progress...")
    trainer.train()
    logger.info("Training completed successfully!")

    logger.info("=" * 60)
    logger.info("Saving model and tokenizer...")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    logger.info(f"Model saved to: {out_dir.absolute()}")
    logger.info("=" * 60)
    logger.info("Training pipeline completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument("--val", required=True, help="Path to val.jsonl")
    parser.add_argument("--out", default="artifacts/phi3-difficulty-classifier", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)

