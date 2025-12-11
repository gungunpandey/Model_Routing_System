"""
Minimal Phi-3 difficulty classifier fine-tuning script (Colab-ready).

Usage (example):
!pip install -r requirements.txt
!python train_classifier.py --train data/train.jsonl --val data/val.jsonl --out artifacts/phi3-difficulty-classifier
"""
import argparse
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
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    ds = load_dataset(
        "json",
        data_files={"train": args.train, "val": args.val},
    )

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, padding_side="left", truncation_side="left")
    ds = ds.map(lambda batch: encode_example(batch, tok, label2id), batched=True).remove_columns(["text", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

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

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)


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

