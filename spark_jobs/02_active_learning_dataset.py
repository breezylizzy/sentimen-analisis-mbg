"""
Spark Job 02 — Active Learning Dataset Builder [FIXED v2]
==========================================================
Input  (HDFS):
  hdfs://namenode:9000/mbg/02_cleaned/
  /opt/data/labeled/mbg2_sample_100_labeled.csv

Output (HDFS):
  hdfs://namenode:9000/mbg/03_baseline/
  hdfs://namenode:9000/mbg/03_unlabeled_pool/
  hdfs://namenode:9000/mbg/03_train_split/
  hdfs://namenode:9000/mbg/03_eval_split/

FIX:
  - .master() dihapus dari SparkSession (konflik dengan spark-submit)
  - drop_duplicates dipanggil dengan list (bukan *args)
  - Broadcast filter diganti dengan LEFT ANTI JOIN (lebih efisien & tidak pakai UDF)
  - Schema validation lebih robust
"""

import sys
from loguru import logger

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ── Konfigurasi ───────────────────────────────────────────────────────────────
HDFS_BASE       = "hdfs://namenode:9000/mbg"
CLEANED_IN      = f"{HDFS_BASE}/02_cleaned"
BASELINE_LOCAL  = "/opt/data/labeled/mbg2_1000_labeled.csv"

BASELINE_OUT    = f"{HDFS_BASE}/03_baseline"
POOL_OUT        = f"{HDFS_BASE}/03_unlabeled_pool"
TRAIN_OUT       = f"{HDFS_BASE}/03_train_split"
EVAL_OUT        = f"{HDFS_BASE}/03_eval_split"

VALID_LABELS    = ["negatif", "netral", "positif"]


# ── Spark Session ─────────────────────────────────────────────────────────────
def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("MBG-02-ActiveLearning-Dataset")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Baca cleaned data dari HDFS ────────────────────────────────────────
    logger.info(f"Membaca cleaned data dari: {CLEANED_IN}")
    cleaned_df = spark.read.parquet(CLEANED_IN)
    logger.info(f"  Total cleaned rows: {cleaned_df.count():,}")

    # ── 2. Baca baseline labels ───────────────────────────────────────────────
    logger.info(f"Membaca baseline CSV dari: {BASELINE_LOCAL}")

    local_baseline_path = f"file://{BASELINE_LOCAL}"
    
    baseline_raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(local_baseline_path) # Menggunakan path yang sudah diperbaiki
    )

    baseline_raw = (
        baseline_raw
        .withColumn("sentiment", F.trim(F.lower(F.col("sentiment"))))
        .filter(F.col("sentiment").isin(VALID_LABELS))
        .filter(F.col("text").isNotNull())
        .filter(F.trim(F.col("text")) != "")
    )
    logger.info(f"  Baseline valid rows: {baseline_raw.count():,}")

    logger.info("── Distribusi Label Baseline ──")
    baseline_raw.groupBy("sentiment").count().show()

    # ── 3. Tambahkan label_id ─────────────────────────────────────────────────
    label_map_expr = (
        F.when(F.col("sentiment") == "negatif", 0)
         .when(F.col("sentiment") == "netral",  1)
         .when(F.col("sentiment") == "positif", 2)
         .otherwise(-1)
    )

    baseline_df = (
        baseline_raw
        .withColumn("label_id",       label_map_expr)
        .withColumn("label_source",   F.lit("baseline_manual"))
        .withColumn("review_required", F.lit(False))
        .dropDuplicates(["text"])   # FIX: list, bukan *args
    )

    # ── 4. Buat unlabeled pool via LEFT ANTI JOIN ─────────────────────────────
    # FIX: Ganti broadcast UDF dengan join Spark native (lebih scalable)
    logger.info("Membuat unlabeled pool via LEFT ANTI JOIN ...")

    # Pool: semua cleaned data yang URL-nya TIDAK ada di baseline
    pool_by_url = (
        cleaned_df
        .join(
            baseline_df.select(F.col("url").alias("baseline_url")).dropDuplicates(["baseline_url"]),
            on=(cleaned_df.url == F.col("baseline_url")),
            how="left_anti"
        )
    )

    # Tambahan: exclude juga berdasarkan teks persis sama
    pool_df = (
        pool_by_url
        .join(
            baseline_df.select(F.col("text").alias("baseline_text")).dropDuplicates(["baseline_text"]),
            on=(pool_by_url.text == F.col("baseline_text")),
            how="left_anti"
        )
        .dropDuplicates(["text_clean"])
        .withColumn("label_source",    F.lit("unlabeled"))
        .withColumn("review_required", F.lit(True))
    )

    pool_count = pool_df.count()
    logger.info(f"Unlabeled pool: {pool_count:,} rows")

    # ── 5. Stratified split baseline → train 80% / eval 20% ──────────────────
    train_parts, eval_parts = [], []
    for label in VALID_LABELS:
        subset = baseline_df.filter(F.col("sentiment") == label)
        tr, ev = subset.randomSplit([0.8, 0.2], seed=42)
        train_parts.append(tr)
        eval_parts.append(ev)

    train_df = train_parts[0]
    for t in train_parts[1:]:
        train_df = train_df.unionByName(t, allowMissingColumns=True)

    eval_df = eval_parts[0]
    for e in eval_parts[1:]:
        eval_df = eval_df.unionByName(e, allowMissingColumns=True)

    logger.info(f"Train: {train_df.count():,} | Eval: {eval_df.count():,}")
    logger.info("── Distribusi Train ──")
    train_df.groupBy("sentiment").count().show()
    logger.info("── Distribusi Eval ──")
    eval_df.groupBy("sentiment").count().show()

    # ── 6. Simpan ke HDFS ─────────────────────────────────────────────────────
    baseline_df.write.mode("overwrite").parquet(BASELINE_OUT)
    pool_df.write.mode("overwrite").parquet(POOL_OUT)
    train_df.write.mode("overwrite").parquet(TRAIN_OUT)
    eval_df.write.mode("overwrite").parquet(EVAL_OUT)

    logger.success(f"✓ Baseline   → {BASELINE_OUT}")
    logger.success(f"✓ Pool       → {POOL_OUT}")
    logger.success(f"✓ Train      → {TRAIN_OUT}")
    logger.success(f"✓ Eval       → {EVAL_OUT}")

    spark.stop()
    logger.success("✓ Job 02 selesai!")


if __name__ == "__main__":
    main()