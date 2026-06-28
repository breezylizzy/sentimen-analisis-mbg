"""
Spark Job 06 — Final Sentiment Merging [FIXED v3 ID JOIN]
=========================================================
Input  (HDFS):
  hdfs://namenode:9000/mbg/04_active_learning_dataset/   ← AL dataset (IndoBERT)
  hdfs://namenode:9000/mbg/05_sarcasm_predictions/       ← sarcasm predictions

Output (HDFS):
  hdfs://namenode:9000/mbg/06_final_sentiment/

FIX v3:
  - JOIN pakai id, bukan pipeline_row
  - Sesuai output Task 5:
      id, sarcasm_label, sarcasm_confidence
  - Left join agar data AL tidak hilang jika ada hasil sarcasm yang missing
  - Missing sarcasm_label dianggap 0 / non-sarcasm
  - Tambah sarcasm_text agar output lebih mudah dibaca
"""

import sys
from loguru import logger

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ── Konfigurasi ───────────────────────────────────────────────────────────────
HDFS_BASE    = "hdfs://namenode:9000/mbg"
AL_IN        = f"{HDFS_BASE}/04_active_learning_dataset"
SARCASM_IN   = f"{HDFS_BASE}/05_sarcasm_predictions"
FINAL_OUT    = f"{HDFS_BASE}/06_final_sentiment"


# ── Spark Session ─────────────────────────────────────────────────────────────
def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("MBG-06-Final-Sentiment-Merging")
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

    # ── 1. Baca Data ──────────────────────────────────────────────────────────
    logger.info("Membaca data dari HDFS ...")
    al_df = spark.read.parquet(AL_IN)
    sarcasm_df = spark.read.parquet(SARCASM_IN)

    al_count = al_df.count()
    sarcasm_count = sarcasm_df.count()
    logger.info(f"AL Dataset: {al_count:,} | Sarcasm: {sarcasm_count:,}")

    # ── 2. Validasi kolom wajib ───────────────────────────────────────────────
    if "id" not in al_df.columns:
        logger.error("Kolom id tidak ditemukan di AL dataset.")
        logger.error(f"Kolom AL tersedia: {al_df.columns}")
        spark.stop()
        sys.exit(1)

    if "id" not in sarcasm_df.columns:
        logger.error("Kolom id tidak ditemukan di sarcasm predictions.")
        logger.error(f"Kolom sarcasm tersedia: {sarcasm_df.columns}")
        spark.stop()
        sys.exit(1)

    if "sentiment" not in al_df.columns:
        logger.error("Kolom sentiment tidak ditemukan di AL dataset.")
        logger.error(f"Kolom AL tersedia: {al_df.columns}")
        spark.stop()
        sys.exit(1)

    if "sarcasm_label" not in sarcasm_df.columns:
        logger.error("Kolom sarcasm_label tidak ditemukan di sarcasm predictions.")
        logger.error(f"Kolom sarcasm tersedia: {sarcasm_df.columns}")
        spark.stop()
        sys.exit(1)

    if "sarcasm_confidence" not in sarcasm_df.columns:
        logger.error("Kolom sarcasm_confidence tidak ditemukan di sarcasm predictions.")
        logger.error(f"Kolom sarcasm tersedia: {sarcasm_df.columns}")
        spark.stop()
        sys.exit(1)

    # ── 3. Normalisasi tipe id ────────────────────────────────────────────────
    al_df = al_df.withColumn("id", F.col("id").cast("string"))
    sarcasm_df = sarcasm_df.withColumn("id", F.col("id").cast("string"))

    # Pastikan sarcasm_df hanya punya kolom yang diperlukan agar tidak tabrakan
    sarcasm_df = sarcasm_df.select(
        "id",
        F.col("sarcasm_label").cast("int").alias("sarcasm_label"),
        F.col("sarcasm_confidence").cast("double").alias("sarcasm_confidence"),
    )

    # Rename kolom sentiment sebelum join agar jelas
    al_df = al_df.withColumnRenamed("sentiment", "sentiment_label")

    # ── 4. JOIN ───────────────────────────────────────────────────────────────
    logger.info("Menjalankan LEFT JOIN berdasarkan id ...")
    merged_df = al_df.join(sarcasm_df, on="id", how="left")

    merged_count = merged_df.count()
    logger.info(f"Setelah JOIN: {merged_count:,} rows")

    missing_sarcasm = merged_df.filter(F.col("sarcasm_label").isNull()).count()
    logger.info(f"Rows tanpa hasil sarcasm: {missing_sarcasm:,}")

    # Jika ada data yang tidak punya prediksi sarcasm, anggap non-sarcasm
    merged_df = (
        merged_df
        .withColumn("sarcasm_label", F.coalesce(F.col("sarcasm_label"), F.lit(0)))
        .withColumn("sarcasm_confidence", F.coalesce(F.col("sarcasm_confidence"), F.lit(0.0)))
        .withColumn(
            "sarcasm_text",
            F.when(F.col("sarcasm_label") == 1, F.lit("sarcasm"))
             .otherwise(F.lit("non_sarcasm"))
        )
    )

    # ── 5. Resolusi Sentimen berdasarkan Sarkasme ─────────────────────────────
    # Aturan:
    # - Sarkas + Positif → Negatif
    # - Sarkas + Netral  → Negatif
    # - Lainnya          → Tetap sentiment IndoBERT
    resolve_expr = (
        F.when(
            (F.col("sarcasm_label") == 1) & (F.col("sentiment_label") == "positif"),
            F.lit("negatif")
        )
        .when(
            (F.col("sarcasm_label") == 1) & (F.col("sentiment_label") == "netral"),
            F.lit("negatif")
        )
        .otherwise(F.col("sentiment_label"))
    )

    final_df = merged_df.withColumn("final_sentiment", resolve_expr)

    # ── 6. Statistik ──────────────────────────────────────────────────────────
    logger.info("── Sentimen Awal (IndoBERT / AL dataset) ──")
    final_df.groupBy("sentiment_label").count().orderBy("sentiment_label").show()

    logger.info("── Distribusi Sarkasme ──")
    final_df.groupBy("sarcasm_label", "sarcasm_text").count().orderBy("sarcasm_label").show()

    logger.info("── Sentimen Akhir (setelah resolusi sarcasm) ──")
    final_df.groupBy("final_sentiment").count().orderBy("final_sentiment").show()

    changed = final_df.filter(
        F.col("sentiment_label") != F.col("final_sentiment")
    ).count()

    logger.info(f"Sentimen yang berubah karena sarkasme: {changed:,}")

    # Optional: tampilkan contoh yang berubah
    logger.info("Contoh data yang berubah karena sarkasme:")
    final_df.filter(
        F.col("sentiment_label") != F.col("final_sentiment")
    ).select(
        "id",
        "text",
        "sentiment_label",
        "sarcasm_label",
        "sarcasm_confidence",
        "final_sentiment",
    ).show(10, truncate=False)

    # ── 7. Simpan ke HDFS ─────────────────────────────────────────────────────
    logger.info(f"Menyimpan final sentiment ke: {FINAL_OUT}")
    final_df.write.mode("overwrite").parquet(FINAL_OUT)

    logger.success(f"✓ Final sentiment → {FINAL_OUT}")

    spark.stop()
    logger.success("✓ Job 06 selesai!")


if __name__ == "__main__":
    main()