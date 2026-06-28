"""
Spark Job 03 — Fine-tune IndoBERT Sentiment Model [FIXED v3 RTX5060]
====================================================================
Input  (HDFS):
  hdfs://namenode:9000/mbg/03_train_split/
  hdfs://namenode:9000/mbg/03_eval_split/

Output:
  /opt/models/indobert/              ← fine-tuned model & tokenizer
  /opt/models/indobert_eval_report.txt

FIX:
  - .master() dihapus dari SparkSession
  - Kolom yang dipakai dicek ketersediaannya sebelum digunakan
  - Tambah fallback: jika text_clean tidak ada, pakai text
  - Tambah os.makedirs untuk MODEL_OUT_DIR
  - RTX 5060 / CUDA support via PyTorch
  - overwrite_output_dir diganti overwrite manual pakai shutil.rmtree
  - Trainer tokenizer compatibility: tokenizer / processing_class
"""

import os
import sys
import inspect
import shutil
import numpy as np
from loguru import logger
from sklearn.metrics import classification_report, accuracy_score

import torch
from pyspark.sql import SparkSession
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
HDFS_BASE       = "hdfs://namenode:9000/mbg"
TRAIN_IN        = f"{HDFS_BASE}/03_train_split"
EVAL_IN         = f"{HDFS_BASE}/03_eval_split"
MODEL_OUT_DIR   = "/opt/models/indobert"
REPORT_OUT_PATH = "/opt/models/indobert_eval_report.txt"

MODEL_NAME      = "indobenchmark/indobert-base-p1"
LABELS          = ["negatif", "netral", "positif"]
LABEL_TO_ID     = {lbl: i for i, lbl in enumerate(LABELS)}
ID_TO_LABEL     = {i: lbl for lbl, i in LABEL_TO_ID.items()}

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ── Spark Session ─────────────────────────────────────────────────────────────
def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("MBG-03-Train-IndoBERT")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(labels, preds))}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(42)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Baca Data dari HDFS ────────────────────────────────────────────────
    logger.info("Membaca train & eval dari HDFS ...")
    train_spark = spark.read.parquet(TRAIN_IN)
    eval_spark  = spark.read.parquet(EVAL_IN)
    logger.info(f"Train: {train_spark.count()} | Eval: {eval_spark.count()}")

    train_pd = train_spark.toPandas()
    eval_pd  = eval_spark.toPandas()
    spark.stop()

    # FIX: Tentukan kolom teks yang dipakai
    text_col = "text_clean" if "text_clean" in train_pd.columns else "text"
    logger.info(f"Menggunakan kolom teks: '{text_col}'")

    # Pastikan kolom label_id ada
    if "label_id" not in train_pd.columns:
        if "sentiment" in train_pd.columns:
            train_pd["label_id"] = train_pd["sentiment"].map(LABEL_TO_ID)
            eval_pd["label_id"]  = eval_pd["sentiment"].map(LABEL_TO_ID)
        else:
            logger.error("✗ Kolom 'label_id' atau 'sentiment' tidak ditemukan!")
            sys.exit(1)

    # Pastikan label_id eval juga ada
    if "label_id" not in eval_pd.columns:
        if "sentiment" in eval_pd.columns:
            eval_pd["label_id"] = eval_pd["sentiment"].map(LABEL_TO_ID)
        else:
            logger.error("✗ Kolom 'label_id' atau 'sentiment' tidak ditemukan di eval!")
            sys.exit(1)

    # Buat HuggingFace Dataset
    train_dataset = Dataset.from_pandas(
        train_pd[[text_col, "label_id"]].rename(columns={text_col: "text"})
    )
    eval_dataset = Dataset.from_pandas(
        eval_pd[[text_col, "label_id"]].rename(columns={text_col: "text"})
    )

    train_dataset = train_dataset.rename_column("label_id", "label")
    eval_dataset  = eval_dataset.rename_column("label_id", "label")

    # ── 2. Load Tokenizer & Model ─────────────────────────────────────────────
    logger.info(f"Loading model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    # ── 3. Tokenisasi ─────────────────────────────────────────────────────────
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=160)

    train_tok = train_dataset.map(tokenize_fn, batched=True)
    eval_tok  = eval_dataset.map(tokenize_fn, batched=True)

    # ── 4. TrainingArguments ──────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device.upper()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.warning("CUDA tidak tersedia, training akan jalan di CPU.")

    # Overwrite manual: hapus model lama supaya hasil tidak numpuk
    if os.path.exists(MODEL_OUT_DIR):
        logger.warning(f"Menghapus model lama di {MODEL_OUT_DIR}")
        shutil.rmtree(MODEL_OUT_DIR)

    if os.path.exists(REPORT_OUT_PATH):
        logger.warning(f"Menghapus report lama di {REPORT_OUT_PATH}")
        os.remove(REPORT_OUT_PATH)

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    args_dict = {
        "output_dir": MODEL_OUT_DIR,

        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "num_train_epochs": 5.0,

        # Aman untuk RTX 5060
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "fp16": torch.cuda.is_available(),
        "gradient_checkpointing": True,

        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": 42,
    }

    # FIX: Kompatibilitas eval_strategy vs evaluation_strategy lintas versi HF
    sig = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in sig:
        args_dict["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig:
        args_dict["evaluation_strategy"] = "epoch"

    # Buang argumen yang tidak didukung oleh versi transformers di container
    args_dict = {k: v for k, v in args_dict.items() if k in sig}

    training_args = TrainingArguments(**args_dict)

    # ── 5. Training ───────────────────────────────────────────────────────────
    logger.info("Memulai fine-tuning IndoBERT ...")

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tok,
        "eval_dataset": eval_tok,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }

    trainer_sig = inspect.signature(Trainer.__init__).parameters

    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    # ── 6. Evaluasi ───────────────────────────────────────────────────────────
    eval_metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {eval_metrics}")

    pred_out   = trainer.predict(eval_tok)
    y_true     = eval_pd["sentiment"].tolist() if "sentiment" in eval_pd.columns else [ID_TO_LABEL[i] for i in eval_pd["label_id"].tolist()]
    y_pred_ids = np.argmax(pred_out.predictions, axis=-1)
    y_pred     = [ID_TO_LABEL[i] for i in y_pred_ids]

    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")

    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.write("=== INDOBERT SENTIMENT MODEL EVALUATION ===\n")
        f.write(f"Accuracy: {eval_metrics.get('eval_accuracy', 0.0):.4f}\n\n")
        f.write(report)

    # ── 7. Simpan Model ───────────────────────────────────────────────────────
    logger.info(f"Menyimpan model ke {MODEL_OUT_DIR} ...")
    trainer.save_model(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)
    logger.success("✓ Job 03 selesai!")


if __name__ == "__main__":
    main()