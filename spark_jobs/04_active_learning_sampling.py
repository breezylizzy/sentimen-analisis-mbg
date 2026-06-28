"""
Spark Job 04 — Active Learning Uncertainty Sampling [FIXED v3]
==============================================================
FIX v3:
  - Model hanya di-load SEKALI di driver (bukan per partisi worker)
  - Inferensi dilakukan di driver dengan batch processing
  - Menghindari OOM akibat multi-instance model di worker
  - Tetap support pool besar (30k+)
"""

import os
import sys
import math
import numpy as np
from loguru import logger

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


HDFS_BASE      = "hdfs://namenode:9000/mbg"
BASELINE_IN    = f"{HDFS_BASE}/03_baseline"
POOL_IN        = f"{HDFS_BASE}/03_unlabeled_pool"
AL_DATASET_OUT = f"{HDFS_BASE}/04_active_learning_dataset"

MODEL_DIR      = "/opt/models/indobert"
CANDIDATES_CSV = "/opt/data/labeled/active_learning_candidates.csv"
TARGET_SIZE    = 40000
BATCH_SIZE     = 32

LABELS         = ["negatif", "netral", "positif"]
ID_TO_LABEL    = {i: lbl for i, lbl in enumerate(LABELS)}


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("MBG-04-Active-Learning-Sampling")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "10")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )


def run_inference_on_driver(texts: list, model_dir: str) -> list:
    """Load model SEKALI di driver, proses semua teks dalam batch."""
    import torch
    import gc
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info(f"Loading model dari {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model loaded ({device.upper()}), proses {len(texts):,} teks ...")

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    all_probs = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits.cpu().numpy()
            all_probs.extend(softmax(logits))

            if (i // BATCH_SIZE + 1) % 100 == 0:
                logger.info(f"  Progress: {i // BATCH_SIZE + 1}/{total_batches} batch selesai")

    results = []
    for p in all_probs:
        sorted_p = sorted(p, reverse=True)
        results.append({
            "predicted_sentiment": ID_TO_LABEL[int(np.argmax(p))],
            "model_confidence":    float(sorted_p[0]),
            "uncertainty_margin":  float(sorted_p[0] - sorted_p[1]),
            "uncertainty_entropy": float(-sum(x * math.log(x + 1e-12) for x in p)),
            "prob_negatif":        float(p[0]),
            "prob_netral":         float(p[1]),
            "prob_positif":        float(p[2]),
        })

    del model
    gc.collect()
    logger.info("Inferensi selesai, model dibebaskan dari memory.")
    return results


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    baseline_df = spark.read.parquet(BASELINE_IN)
    pool_df     = spark.read.parquet(POOL_IN)

    baseline_count = baseline_df.count()
    pool_count     = pool_df.count()
    logger.info(f"Baseline: {baseline_count:,} | Pool: {pool_count:,}")

    needed_count = max(0, TARGET_SIZE - baseline_count)
    logger.info(f"Target AL candidates: {needed_count:,}")

    if needed_count <= 0:
        logger.warning("Baseline sudah >= TARGET_SIZE. Simpan baseline sebagai output.")
        baseline_df.write.mode("overwrite").parquet(AL_DATASET_OUT)
        logger.success(f"✓ Output → {AL_DATASET_OUT}")
        spark.stop()
        sys.exit(0)

    if pool_count == 0:
        logger.error("✗ Pool kosong!")
        spark.stop()
        sys.exit(1)

    # Ambil sample dari pool (2x needed agar ada buffer untuk sorting)
    sample_size = min(pool_count, needed_count * 2)
    logger.info(f"Collecting {sample_size:,} rows dari pool ke driver ...")
    pool_pd = pool_df.limit(sample_size).toPandas()
    logger.info(f"Collected {len(pool_pd):,} rows")

    # Inferensi di driver
    text_col = "text_clean" if "text_clean" in pool_pd.columns else "text"
    texts = pool_pd[text_col].fillna("").tolist()
    inference_results = run_inference_on_driver(texts, MODEL_DIR)

    import pandas as pd
    results_pd = pd.DataFrame(inference_results)
    pool_scored_pd = pd.concat([pool_pd.reset_index(drop=True), results_pd], axis=1)

    # Drop kolom kosong/tidak terpakai dari raw data
    DROP_UNUSED_COLS = ["source", "isRetweet", "isQuote"]

    existing_drop_cols = [col for col in DROP_UNUSED_COLS if col in pool_scored_pd.columns]
    if existing_drop_cols:
        logger.warning(f"Drop kolom tidak terpakai: {existing_drop_cols}")
        pool_scored_pd = pool_scored_pd.drop(columns=existing_drop_cols)

    # Uncertainty sampling: ambil yang paling tidak pasti
    pool_scored_pd = pool_scored_pd.sort_values(
        ["uncertainty_margin", "uncertainty_entropy"],
        ascending=[True, False]
    )
    candidates_pd = pool_scored_pd.head(needed_count).copy()
    logger.info(f"AL candidates terpilih: {len(candidates_pd):,}")

    candidates_pd["sentiment"]       = candidates_pd["predicted_sentiment"]
    candidates_pd["label_source"]    = "indobert_al_prediction"
    candidates_pd["review_required"] = True

    al_candidates_df = spark.createDataFrame(candidates_pd)

    baseline_prepared = (
        baseline_df
        .withColumn("predicted_sentiment", F.col("sentiment"))
        .withColumn("model_confidence",    F.lit(1.0))
        .withColumn("uncertainty_margin",  F.lit(None).cast(DoubleType()))
        .withColumn("uncertainty_entropy", F.lit(None).cast(DoubleType()))
        .withColumn("prob_negatif",        F.when(F.col("sentiment") == "negatif", 1.0).otherwise(0.0))
        .withColumn("prob_netral",         F.when(F.col("sentiment") == "netral",  1.0).otherwise(0.0))
        .withColumn("prob_positif",        F.when(F.col("sentiment") == "positif", 1.0).otherwise(0.0))
    )

    for col in DROP_UNUSED_COLS:
        if col in baseline_prepared.columns:
            baseline_prepared = baseline_prepared.drop(col)
        if col in al_candidates_df.columns:
            al_candidates_df = al_candidates_df.drop(col)

    common_cols = list(set(baseline_prepared.columns) & set(al_candidates_df.columns))

    final_al_dataset = (
        baseline_prepared.select(common_cols)
        .unionByName(al_candidates_df.select(common_cols))
    )

    total_final = final_al_dataset.count()
    logger.info(f"Total AL dataset final: {total_final:,}")

    final_al_dataset.write.mode("overwrite").parquet(AL_DATASET_OUT)
    logger.success(f"✓ AL Dataset → {AL_DATASET_OUT}")

    os.makedirs(os.path.dirname(CANDIDATES_CSV), exist_ok=True)
    candidates_pd.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8")
    logger.success(f"✓ Kandidat CSV → {CANDIDATES_CSV}")

    logger.info("Distribusi label final:")
    final_al_dataset.groupBy("sentiment").count().show()

    spark.stop()
    logger.success("✓ Job 04 selesai!")


if __name__ == "__main__":
    main()