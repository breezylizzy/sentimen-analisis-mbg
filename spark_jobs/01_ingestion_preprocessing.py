"""
Spark Job 01 — Ingestion & Preprocessing [FIXED v2]
====================================================
Input  (local mount):
  /opt/data/raw/mbg1.csv
  /opt/data/raw/mbg2.csv          (opsional)
  /opt/data/raw/mbg3.csv          (opsional)

Output (HDFS):
  hdfs://namenode:9000/mbg/01_raw_merged/
  hdfs://namenode:9000/mbg/02_cleaned/

FIX:
  - .master() dihapus dari SparkSession (konflik dengan spark-submit --master)
  - UDF is_relevant diganti dengan filter SQL native (jauh lebih cepat)
  - Kolom isReply/isRetweet/isQuote: handle nilai string "true"/"false"/""/null
  - Slang path dibuat opsional tanpa crash
  - drop_duplicates dipanggil dengan list (bukan *args)
"""

import json
import re
import sys
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from loguru import logger


# ── Konfigurasi ───────────────────────────────────────────────────────────────
HDFS_BASE       = "hdfs://namenode:9000/mbg"
RAW_LOCAL_PATH  = "/opt/data/raw"
SLANG_PATH      = "/opt/configs/slangwords.json"

RAW_OUT         = f"{HDFS_BASE}/01_raw_merged"
CLEAN_OUT       = f"{HDFS_BASE}/02_cleaned"

MBG_KEYWORDS = [
    "mbg",
    "makan bergizi gratis",
    "makan gratis",
    "makan siang gratis",
    "program makan bergizi",
    "bgn",
    "badan gizi nasional",
    "sppg",
    "satuan pelayanan pemenuhan gizi",
    "dapur mbg",
    "menu mbg",
    "penerima mbg",
]


# ── Spark Session ─────────────────────────────────────────────────────────────
def build_spark() -> SparkSession:
    """
    PENTING: Jangan set .master() di sini.
    Master URL sudah ditentukan lewat spark-submit --master spark://spark-master:7077
    Jika ingin run lokal (tanpa cluster), set env SPARK_MASTER=local[*]
    """
    try:
        spark = (
            SparkSession.builder
            .appName("MBG-01-Ingestion-Preprocessing")
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "1g")
            .config("spark.sql.shuffle.partitions", "20")
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
            .getOrCreate()
        )
        logger.success("✓ Spark Session berhasil dibuat")
        return spark
    except Exception as e:
        logger.error(f"✗ Gagal membuat Spark Session: {e}")
        raise


# ── Slang Dictionary ──────────────────────────────────────────────────────────
def load_slang_dict(path: str) -> dict:
    if not os.path.exists(path):
        logger.warning(f"⚠ File slang dictionary tidak ditemukan: {path}. Lanjut tanpa slang normalization.")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"✗ Error membaca slang dictionary: {e}")
        return {}


# ── Cleaner UDF ───────────────────────────────────────────────────────────────
def make_cleaner(slang_dict: dict):
    """Full cleaning pipeline. Returned sebagai closure agar slang_dict di-capture."""

    def clean(text: str) -> str:
        if not text:
            return ""
        text = str(text).strip()

        # Hapus URL
        text = re.sub(r"https?://\S+", " URL ", text)
        # Hapus mention
        text = re.sub(r"@\w+", " USER ", text)
        # Pertahankan kata di balik hashtag
        text = re.sub(r"#(\w+)", r"\1", text)
        # Hapus karakter non-alfanumerik (kecuali tanda baca umum)
        text = re.sub(r"[^\w\s.,?!]", " ", text)
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Lowercase
        text = text.lower()

        # Slang normalization (jika kamus tersedia)
        if slang_dict:
            tokens = text.split()
            text = " ".join(slang_dict.get(tok, tok) for tok in tokens)

        return text.strip()

    return clean


# ── Boolean helper ────────────────────────────────────────────────────────────
def str_to_bool_col(col_name: str):
    """
    Konversi kolom string "true"/"false"/"" / null ke boolean.
    FIX: CSV dari Twitter Scraper kadang isi kolom ini string, bukan boolean native.
    """
    return (
        F.when(F.lower(F.trim(F.col(col_name))).isin("true", "1", "yes"), True)
         .otherwise(False)
         .cast("boolean")
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # Load slang dictionary (opsional)
    slang_dict = load_slang_dict(SLANG_PATH)
    logger.info(f"Slang entries loaded: {len(slang_dict)}")

    # ── 1. Scan & baca CSV files ──────────────────────────────────────────────
    logger.info(f"Scanning folder: {RAW_LOCAL_PATH}")
    if not os.path.exists(RAW_LOCAL_PATH):
        logger.error(f"✗ Folder tidak ada: {RAW_LOCAL_PATH}. Pastikan Docker volume sudah ter-mount.")
        sys.exit(1)

    sources = ["mbg1.csv", "mbg2.csv", "mbg3.csv"]
    available_files = [
        (src, os.path.join(RAW_LOCAL_PATH, src))
        for src in sources
        if os.path.exists(os.path.join(RAW_LOCAL_PATH, src))
    ]

    if not available_files:
        logger.error("✗ Tidak ada CSV ditemukan. Letakkan file di folder data/raw/")
        sys.exit(1)

    logger.info(f"Files ditemukan: {[f for f, _ in available_files]}")

    # ── 2. Baca & union semua CSV ─────────────────────────────────────────────
    dfs = []
    for src, path in available_files:
        try:
            # FIX: Tambahkan prefix 'file://' agar Spark tahu ini file lokal container, bukan HDFS
            local_spark_path = f"file://{path}"
            
            df = (
                spark.read
                .option("header", "true")
                .option("inferSchema", "false")   # FIX: paksa semua kolom string dulu
                .option("multiLine", "true")
                .option("escape", '"')
                .option("encoding", "UTF-8")
                .csv(local_spark_path)            # Menggunakan path dengan prefix file://
                .withColumn("source_file", F.lit(src))
            )
            cnt = df.count()
            dfs.append(df)
            logger.info(f"  ✓ {src}: {cnt:,} rows")
        except Exception as exc:
            logger.error(f"  ✗ Error membaca {src}: {exc}")
    if not dfs:
        logger.error("✗ Tidak ada CSV yang berhasil dibaca!")
        sys.exit(1)

    raw_df = dfs[0]
    for df in dfs[1:]:
        raw_df = raw_df.unionByName(df, allowMissingColumns=True)

    raw_df = raw_df.withColumn("pipeline_row", F.monotonically_increasing_id())
    total_before = raw_df.count()
    logger.info(f"Total sebelum dedup: {total_before:,}")

    # Simpan raw merged ke HDFS
    raw_df.write.mode("overwrite").parquet(RAW_OUT)
    logger.success(f"✓ Raw merged → {RAW_OUT}")

    # ── 3. Deduplikasi ────────────────────────────────────────────────────────
    # Pilih kolom dedup secara prioritas
    if "id" in raw_df.columns:
        dedup_col = "id"
    elif "url" in raw_df.columns:
        dedup_col = "url"
    else:
        dedup_col = "text"

    logger.info(f"Dedup berdasarkan kolom: '{dedup_col}'")
    dedup_df = raw_df.dropDuplicates([dedup_col])
    removed = total_before - dedup_df.count()
    logger.info(f"  Removed {removed:,} duplikasi | Remaining: {dedup_df.count():,}")

    # ── 4. Filter relevansi MBG ───────────────────────────────────────────────
    # FIX: Pakai filter SQL native (contains) — jauh lebih cepat dari UDF Python
    if "text" not in dedup_df.columns:
        logger.error("✗ Kolom 'text' tidak ditemukan di CSV!")
        sys.exit(1)

    # Bangun kondisi OR dari keyword list
    text_lower = F.lower(F.col("text"))
    keyword_filter = F.lit(False)
    for kw in MBG_KEYWORDS:
        keyword_filter = keyword_filter | text_lower.contains(kw)

    filtered_df = dedup_df.filter(keyword_filter)
    logger.info(f"Setelah filter MBG: {filtered_df.count():,} rows")

    # ── 5. Cleaning teks ──────────────────────────────────────────────────────
    cleaner = make_cleaner(slang_dict)
    clean_udf = F.udf(cleaner, StringType())

    cleaned_df = filtered_df.withColumn("text_clean", clean_udf(F.col("text")))

    # ── 6. Filter tweet terlalu pendek (< 5 token) ───────────────────────────
    cleaned_df = (
        cleaned_df
        .withColumn("token_count", F.size(F.split(F.col("text_clean"), r"\s+")))
        .filter(F.col("token_count") >= 5)
    )

    # ── 7. Normalisasi kolom boolean ─────────────────────────────────────────
    # FIX: Handle nilai string "true"/"false"/null dari CSV Twitter
    bool_cols = {
        "isReply":   "is_reply",
        "isRetweet": "is_retweet",
        "isQuote":   "is_quote",
    }
    for src_col, dst_col in bool_cols.items():
        if src_col in cleaned_df.columns:
            cleaned_df = cleaned_df.withColumn(dst_col, str_to_bool_col(src_col))
        else:
            cleaned_df = cleaned_df.withColumn(dst_col, F.lit(False).cast("boolean"))

    # ── 8. Tambahkan metadata ─────────────────────────────────────────────────
    cleaned_df = cleaned_df.withColumn("ingestion_ts", F.current_timestamp())

    final_count = cleaned_df.count()
    logger.info(f"Final cleaned rows: {final_count:,}")

    # ── 9. Simpan ke HDFS ─────────────────────────────────────────────────────
    (
        cleaned_df
        .repartition(10)
        .write
        .mode("overwrite")
        .parquet(CLEAN_OUT)
    )
    logger.success(f"✓ Cleaned data → {CLEAN_OUT}")

    # ── 10. Statistik ─────────────────────────────────────────────────────────
    logger.info("=== Distribusi per source_file ===")
    cleaned_df.groupBy("source_file").count().show()

    if "lang" in cleaned_df.columns:
        logger.info("=== Distribusi per bahasa ===")
        cleaned_df.groupBy("lang").count().orderBy("count", ascending=False).show(10)

    spark.stop()
    logger.success("✓ Job 01 selesai!")


if __name__ == "__main__":
    main()