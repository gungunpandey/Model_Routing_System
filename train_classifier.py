"""
Minimal Phi-3 difficulty classifier fine-tuning script (Colab-ready).

Usage (example):
!pip install -r requirements.txt
!python train_classifier.py --train data/train.jsonl --val data/val.jsonl --out artifacts/phi3-difficulty-classifier
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
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
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"GPU Memory Free: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        logger.warning("No GPU detected! Training will be very slow on CPU.")
    
    # Check system RAM (approximate)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        ram_available_gb = psutil.virtual_memory().available / 1e9
        logger.info(f"System RAM: {ram_gb:.2f} GB total, {ram_available_gb:.2f} GB available")
        if ram_available_gb < 10:
            logger.warning("⚠️  Low RAM detected! Model loading may fail.")
            logger.warning("   Consider: 1) Restart runtime, 2) Use Colab Pro, 3) Reduce batch_size to 2")
    except ImportError:
        logger.info("psutil not available - skipping RAM check")
        logger.info("Install with: pip install psutil")
    
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
    
    # Clear memory before loading model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory cleared before model loading")

    logger.info(f"Loading model from {MODEL_NAME}...")
    logger.info("(This may take a few minutes to download model weights...)")
    logger.info("Please be patient - do not interrupt during model loading!")
    logger.info("Using memory optimizations: float16, low_cpu_mem_usage, device_map")
    try:
        # Use device_map="auto" to automatically offload to GPU
        model_kwargs = {
            "num_labels": len(LABELS),
            "id2label": id2label,
            "label2id": label2id,
            "low_cpu_mem_usage": True,
        }
        
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"  # Automatically use GPU
        
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            **model_kwargs
        )
        logger.info("Model loaded successfully")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Model device: {next(model.parameters()).device}")
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda" in error_msg:
            logger.error("=" * 60)
            logger.error("⚠️  OUT OF MEMORY ERROR ⚠️")
            logger.error("=" * 60)
            logger.error("Colab free tier has limited RAM (~12GB). Phi-3-mini needs ~8GB+ to load.")
            logger.error("")
            logger.error("SOLUTIONS:")
            logger.error("  1. RESTART RUNTIME: Runtime → Restart runtime (clears memory)")
            logger.error("  2. USE COLAB PRO: More RAM available (recommended)")
            logger.error("  3. REDUCE BATCH SIZE: --batch_size 2 --eval_batch_size 4")
            logger.error("  4. USE SMALLER MODEL: Consider distilbert-base-uncased for testing")
            logger.error("  5. USE GRADIO/KAGGLE: Free GPU with more RAM")
            logger.error("")
            logger.error(f"Error details: {e}")
        else:
            logger.error(f"Failed to load model: {e}")
            logger.error("This might be due to:")
            logger.error("  1. Colab session timeout (restart runtime and try again)")
            logger.error("  2. Insufficient memory (try reducing batch_size)")
            logger.error("  3. Network issues (check internet connection)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Unexpected error - check logs above for details")
        sys.exit(1)

    logger.info("Setting up training arguments...")
    logger.info("Enabling memory optimizations: gradient_checkpointing, fp16, dataloader_pin_memory=False")
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
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
        fp16=True if torch.cuda.is_available() else False,
        dataloader_pin_memory=False,  # Save RAM
        gradient_accumulation_steps=2,  # Effective batch size = batch_size * 2
        max_grad_norm=1.0,
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
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.error("Training was interrupted! Model may be incomplete.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Check GPU memory and Colab session status.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Saving model and tokenizer...")
    try:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(out_dir)
        tok.save_pretrained(out_dir)
        logger.info(f"Model saved to: {out_dir.absolute()}")
        
        # Verify files were saved
        required_files = ["config.json", "pytorch_model.bin"]
        if not all((out_dir / f).exists() for f in required_files):
            # Check for safetensors format
            safetensors_files = list(out_dir.glob("*.safetensors"))
            if not safetensors_files:
                logger.warning("Warning: Some model files may be missing!")
        logger.info("=" * 60)
        logger.info("Training pipeline completed!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)


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

