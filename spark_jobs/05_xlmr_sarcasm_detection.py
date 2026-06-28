"""
Spark Job 05 — Sarcasm Detection using RoBERTa [DRIVER GPU SAFE]
================================================================
Input  (HDFS):
  hdfs://namenode:9000/mbg/04_active_learning_dataset/

Output (HDFS):
  hdfs://namenode:9000/mbg/05_sarcasm_predictions/

FIX:
  - Model hanya di-load SEKALI di driver
  - Inferensi batch di driver
  - Menghindari OOM akibat model di-load per partisi worker
  - Output tetap: pipeline_row, sarcasm_label, sarcasm_confidence
  - Fallback text_clean → text
"""

import sys
import math
import numpy as np
import pandas as pd
from loguru import logger

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType


# ── Konfigurasi ───────────────────────────────────────────────────────────────
HDFS_BASE      = "hdfs://namenode:9000/mbg"
AL_DATASET_IN  = f"{HDFS_BASE}/04_active_learning_dataset"
SARCASM_OUT    = f"{HDFS_BASE}/05_sarcasm_predictions"

MODEL_NAME     = "cardiffnlp/twitter-roberta-base-irony"

BATCH_SIZE     = 32
MAX_LENGTH     = 160


# ── Spark Session ─────────────────────────────────────────────────────────────
def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("MBG-05-Sarcasm-Detection")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )


# ── Inference di Driver ───────────────────────────────────────────────────────
def run_sarcasm_inference_on_driver(texts: list, model_name: str) -> list:
    import gc
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info(f"Loading sarcasm model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger.info(f"Model loaded on {device.upper()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    results = []
    total_batches = math.ceil(len(texts) / BATCH_SIZE)

    logger.info(f"Mulai inferensi sarkasme untuk {len(texts):,} teks")
    logger.info(f"Batch size: {BATCH_SIZE} | Max length: {MAX_LENGTH}")

    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(texts), BATCH_SIZE), start=1):
            batch = texts[start:start + BATCH_SIZE]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            logits = model(**inputs).logits.detach().cpu().numpy()
            probs = softmax(logits)

            for p in probs:
                pred_label = int(np.argmax(p))
                confidence = float(p[pred_label])

                results.append({
                    "sarcasm_label": pred_label,
                    "sarcasm_confidence": confidence,
                })

            if batch_idx % 100 == 0 or batch_idx == total_batches:
                logger.info(f"Progress: {batch_idx}/{total_batches} batch selesai")

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Inferensi sarkasme selesai, model dibebaskan dari memory.")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(f"Membaca AL dataset dari: {AL_DATASET_IN}")
    al_df = spark.read.parquet(AL_DATASET_IN)

    total_rows = al_df.count()
    logger.info(f"Total rows: {total_rows:,}")

    if total_rows == 0:
        logger.error("Dataset AL kosong. Job dihentikan.")
        spark.stop()
        sys.exit(1)

    # Fallback text_clean -> text
    columns = al_df.columns

    if "id" not in columns:
        logger.error("Kolom id tidak ditemukan. Tidak bisa membuat output join key.")
        logger.error(f"Kolom tersedia: {columns}")
        spark.stop()
        sys.exit(1)

    if "text_clean" in columns:
        text_col = "text_clean"
    elif "text" in columns:
        text_col = "text"
    else:
        logger.error("Kolom text_clean atau text tidak ditemukan.")
        logger.error(f"Kolom tersedia: {columns}")
        spark.stop()
        sys.exit(1)

    logger.info(f"Menggunakan kolom teks: {text_col}")

    # Ambil hanya kolom yang dibutuhkan supaya tidak boros memory driver
    logger.info("Collect pipeline_row dan teks ke driver ...")
    al_small_pd = (
        al_df
        .select(
            F.col("id").cast("string").alias("id"),
            F.col(text_col).alias("text_for_sarcasm"),
        )
        .toPandas()
    )

    logger.info(f"Collected rows ke driver: {len(al_small_pd):,}")

    al_small_pd["text_for_sarcasm"] = (
        al_small_pd["text_for_sarcasm"]
        .fillna("")
        .astype(str)
    )

    texts = al_small_pd["text_for_sarcasm"].tolist()

    logger.info(f"Menjalankan deteksi sarkasme: {MODEL_NAME}")
    inference_results = run_sarcasm_inference_on_driver(texts, MODEL_NAME)

    results_pd = pd.DataFrame(inference_results)

    sarcasm_pd = pd.concat(
        [
            al_small_pd[["id"]].reset_index(drop=True),
            results_pd.reset_index(drop=True),
        ],
        axis=1,
    )

    # Pastikan tipe jelas sebelum Pandas -> Spark
    sarcasm_pd["id"] = sarcasm_pd["id"].astype(str)
    sarcasm_pd["sarcasm_label"] = sarcasm_pd["sarcasm_label"].astype("int32")
    sarcasm_pd["sarcasm_confidence"] = sarcasm_pd["sarcasm_confidence"].astype("float64")

    logger.info("Membuat Spark DataFrame hasil prediksi sarkasme ...")
    sarcasm_df = spark.createDataFrame(sarcasm_pd)

    logger.info("Distribusi prediksi sarkasme:")
    sarcasm_df.groupBy("sarcasm_label").count().show()

    logger.info(f"Menyimpan output ke: {SARCASM_OUT}")
    sarcasm_df.write.mode("overwrite").parquet(SARCASM_OUT)

    logger.success(f"✓ Sarcasm predictions → {SARCASM_OUT}")

    spark.stop()
    logger.success("✓ Job 05 selesai!")


if __name__ == "__main__":
    main()