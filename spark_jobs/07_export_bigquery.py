"""
Spark Job 07 — Export Final Sentiment Results to BigQuery
=========================================================

Input  (HDFS):
  hdfs://namenode:9000/mbg/06_final_sentiment

Output:
  BigQuery table:
    {GCP_PROJECT_ID}.{BQ_DATASET}.sentiment_results

Optional output:
  CSV lokal:
    /opt/data/output/sentiment_results.csv
"""

import os
import sys
from loguru import logger

import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

from pyspark.sql import SparkSession


# ── Konfigurasi HDFS ──────────────────────────────────────────────────────────
HDFS_BASE = "hdfs://namenode:9000/mbg"
FINAL_IN  = f"{HDFS_BASE}/06_final_sentiment"


# ── Konfigurasi BigQuery ──────────────────────────────────────────────────────
GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "final-project-500709")
BQ_DATASET  = os.getenv("BQ_DATASET",     "mbg_sentiment")
BQ_TABLE    = "sentiment_results"
BQ_TABLE_ID = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

# GOOGLE_APPLICATION_CREDENTIALS dibaca otomatis oleh library Google
# selama environment variable-nya sudah di-set di docker-compose.yml


# ── Konfigurasi CSV ───────────────────────────────────────────────────────────
EXPORT_CSV = True
CSV_OUT    = "/opt/data/output/sentiment_results.csv"


# ── Spark Session ─────────────────────────────────────────────────────────────
def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("MBG-07-Export-BigQuery")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory",   "2g")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )


# ── BigQuery Client ───────────────────────────────────────────────────────────
def build_bq_client() -> bigquery.Client:
    """
    Membuat BigQuery client.
    Jika dataset belum ada, dataset akan dibuat otomatis.
    """
    logger.info(f"Mencoba koneksi BigQuery: project={GCP_PROJECT}, dataset={BQ_DATASET}")

    try:
        client = bigquery.Client(project=GCP_PROJECT)

        dataset_id = f"{GCP_PROJECT}.{BQ_DATASET}"

        try:
            dataset_ref = client.get_dataset(dataset_id)
            logger.success(f"✓ Dataset ditemukan: {dataset_ref.full_dataset_id}")
        except Exception:
            logger.warning(f"Dataset belum ada, membuat dataset: {dataset_id}")

            dataset = bigquery.Dataset(dataset_id)

            # Pilih salah satu location.
            # Kalau di Console sudah biasa pakai US, ubah jadi "US".
            dataset.location = os.getenv("BQ_LOCATION", "asia-southeast2")

            dataset_ref = client.create_dataset(dataset, exists_ok=True)
            logger.success(f"✓ Dataset berhasil dibuat: {dataset_ref.full_dataset_id}")

        return client

    except Exception as e:
        logger.error("✗ Gagal koneksi ke BigQuery.")
        logger.error(f"Detail error: {e}")
        logger.error("")
        logger.error("Cek hal berikut:")
        logger.error("1. File gcp-key.json ter-mount di container dan path-nya benar")
        logger.error("2. GOOGLE_APPLICATION_CREDENTIALS sudah di-set di docker-compose.yml")
        logger.error(f"3. Dataset '{BQ_DATASET}' bisa dibuat/diakses di project '{GCP_PROJECT}'")
        logger.error("4. Service account punya role BigQuery Data Editor + BigQuery Job User")
        sys.exit(1)

# ── Data Preparation ──────────────────────────────────────────────────────────
def prepare_dataframe(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Pilih dan rename kolom yang akan masuk BigQuery.
    Kolom yang tidak ada akan otomatis di-skip.
    """

    column_mapping = {
        # Identitas
        "id":          "tweet_id",
        "url":         "url",
        "createdAt":   "created_at",

        # Teks
        "text":        "text",
        "text_clean":  "text_clean",
        "lang":        "lang",

        # Engagement
        "retweetCount": "retweet_count",
        "replyCount":   "reply_count",
        "likeCount":    "like_count",
        "quoteCount":   "quote_count",
        "viewCount":    "view_count",

        # Flags
        "is_reply":    "is_reply",
        "is_retweet":  "is_retweet",
        "is_quote":    "is_quote",

        # Sentiment
        "sentiment_label":      "sentiment_label",
        "predicted_sentiment":  "predicted_sentiment",
        "model_confidence":     "model_confidence",
        "prob_negatif":         "prob_negatif",
        "prob_netral":          "prob_netral",
        "prob_positif":         "prob_positif",

        # Active learning
        "label_source":         "label_source",
        "review_required":      "review_required",
        "uncertainty_entropy":  "uncertainty_entropy",
        "uncertainty_margin":   "uncertainty_margin",

        # Sarcasm
        "sarcasm_label":        "sarcasm_label",
        "sarcasm_text":         "sarcasm_text",
        "sarcasm_confidence":   "sarcasm_confidence",

        # Final
        "final_sentiment":  "final_sentiment",
        "ingestion_ts":     "ingestion_ts",
        "source_file":      "source_file",
        "token_count":      "token_count",
    }

    available_mapping = {
        src: dst
        for src, dst in column_mapping.items()
        if src in pdf.columns
    }

    if not available_mapping:
        logger.error("✗ Tidak ada kolom yang cocok untuk diekspor.")
        logger.error(f"Kolom tersedia: {list(pdf.columns)}")
        sys.exit(1)

    missing = [k for k in column_mapping if k not in pdf.columns]
    logger.info(f"Kolom yang ditemukan: {list(available_mapping.keys())}")
    logger.warning(f"Kolom yang di-skip (tidak ada): {missing}")

    pdf = pdf[list(available_mapping.keys())].rename(columns=available_mapping)

    # tweet_id → string (ID Twitter besar, bisa overflow kalau int)
    if "tweet_id" in pdf.columns:
        pdf["tweet_id"] = pdf["tweet_id"].astype(str)

    # URL — potong kalau terlalu panjang
    if "url" in pdf.columns:
        pdf["url"] = pdf["url"].astype(str).str.slice(0, 512)

    # Boolean columns
    # BigQuery tidak kenal pandas BooleanDtype, konversi ke Python bool biasa
    bool_cols = ["is_reply", "is_retweet", "is_quote", "review_required"]
    for col in bool_cols:
        if col in pdf.columns:
            pdf[col] = (
                pdf[col]
                .astype(str).str.strip().str.lower()
                .map({"true": True, "1": True, "false": False, "0": False})
            )

    # Integer columns
    int_cols = ["retweet_count", "reply_count", "like_count", "quote_count",
                "view_count", "sarcasm_label", "token_count"]
    for col in int_cols:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
            # BigQuery butuh tipe native Python int, bukan Int64 (nullable)
            pdf[col] = pdf[col].where(pdf[col].notna(), other=None)

    # Float columns
    float_cols = ["model_confidence", "prob_negatif", "prob_netral", "prob_positif",
                  "uncertainty_entropy", "uncertainty_margin", "sarcasm_confidence"]
    for col in float_cols:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    # Datetime
    if "created_at" in pdf.columns:
        pdf["created_at"] = pd.to_datetime(pdf["created_at"], errors="coerce")

    logger.info(f"Final kolom untuk BigQuery: {list(pdf.columns)}")
    return pdf


# ── Upload ke BigQuery ────────────────────────────────────────────────────────
def upload_to_bigquery(client: bigquery.Client, pdf: pd.DataFrame) -> None:
    """
    Upload Pandas DataFrame ke BigQuery menggunakan load_table_from_dataframe.
    if_exists = WRITE_TRUNCATE: hapus data lama, isi dengan yang baru
                (sama seperti if_exists='replace' di pandas to_sql)
    """

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        # BigQuery akan auto-detect schema dari DataFrame
        autodetect=True,
    )

    logger.info(f"Mengupload {len(pdf):,} baris ke BigQuery: {BQ_TABLE_ID}")

    try:
        job = client.load_table_from_dataframe(
            pdf,
            BQ_TABLE_ID,
            job_config=job_config,
        )

        # Tunggu sampai job selesai
        job.result()

        # Verifikasi
        table = client.get_table(BQ_TABLE_ID)
        logger.success(f"✓ Upload selesai! Total baris di BigQuery: {table.num_rows:,}")

    except GoogleAPIError as e:
        logger.error("✗ Gagal upload ke BigQuery.")
        logger.error(f"Detail error: {e}")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Baca final sentiment dari HDFS ─────────────────────────────────────
    logger.info(f"Membaca final sentiment dari HDFS: {FINAL_IN}")

    try:
        final_df = spark.read.parquet(FINAL_IN)
    except Exception as e:
        logger.error("✗ Gagal membaca output Task 6.")
        logger.error(f"Path: {FINAL_IN}")
        logger.error(f"Detail error: {e}")
        spark.stop()
        sys.exit(1)

    total_rows = final_df.count()
    logger.info(f"Total rows dari Task 6: {total_rows:,}")

    if total_rows == 0:
        logger.error("✗ Data Task 6 kosong. Tidak ada yang diekspor.")
        spark.stop()
        sys.exit(1)

    logger.info("Schema Task 6:")
    final_df.printSchema()

    logger.info("Preview Task 6:")
    final_df.show(5, truncate=False)

    # ── 2. Convert ke Pandas ──────────────────────────────────────────────────
    logger.info("Mengubah Spark DataFrame ke Pandas ...")
    pdf = final_df.toPandas()
    spark.stop()

    logger.info(f"Pandas shape awal: {pdf.shape}")

    # ── 3. Siapkan kolom ──────────────────────────────────────────────────────
    pdf = prepare_dataframe(pdf)
    logger.info(f"Pandas shape setelah prepare: {pdf.shape}")

    # ── 4. Koneksi + Upload ke BigQuery ───────────────────────────────────────
    client = build_bq_client()
    upload_to_bigquery(client, pdf)

    # ── 5. Export CSV lokal ───────────────────────────────────────────────────
    if EXPORT_CSV:
        try:
            os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
            logger.info(f"Export CSV: {CSV_OUT}")

            # Baca balik dari BigQuery untuk memastikan data konsisten
            query = f"SELECT * FROM `{BQ_TABLE_ID}`"
            export_pdf = client.query(query).to_dataframe()

            export_pdf.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
            logger.success(f"✓ CSV berhasil dibuat: {CSV_OUT}")

        except Exception as e:
            logger.warning("CSV export gagal, tapi upload BigQuery sudah selesai.")
            logger.warning(f"Detail error: {e}")

    logger.success("✓ Job 07 selesai!")


if __name__ == "__main__":
    main()