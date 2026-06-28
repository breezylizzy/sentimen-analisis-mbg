#!/bin/bash
# ============================================================
#  MBG Pipeline - Job Runner
#  Cara pakai:
#    ./run_job.sh 01   → jalankan job 01
#    ./run_job.sh all  → jalankan semua job berurutan
# ============================================================

SPARK_MASTER="spark://spark-master:7077"
JOBS_DIR="/opt/spark_jobs"

SPARK_SUBMIT="/opt/spark/bin/spark-submit"

run_job() {
    JOB=$1
    case $JOB in
        01) SCRIPT="01_ingestion_preprocessing.py" ;;
        02) SCRIPT="02_active_learning_dataset.py" ;;
        03) SCRIPT="03_train_indobert.py" ;;
        04) SCRIPT="04_active_learning_sampling.py" ;;
        05) SCRIPT="05_xlmr_sarcasm_detection.py" ;;
        06) SCRIPT="06_final_sentiment_merging.py" ;;
        07) SCRIPT="07_export_postgresql.py" ;;
        *)  echo "Job tidak dikenal: $JOB"; exit 1 ;;
    esac

    echo "======================================================"
    echo " Menjalankan Job $JOB: $SCRIPT"
    echo "======================================================"

    $SPARK_SUBMIT \
        --master "$SPARK_MASTER" \
        --deploy-mode client \
        --executor-memory 2g \
        --driver-memory 1g \
        --conf spark.pyspark.python=/usr/bin/python3 \
        "$JOBS_DIR/$SCRIPT"

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "✓ Job $JOB selesai!"
    else
        echo "✗ Job $JOB GAGAL (exit code: $STATUS)"
        exit $STATUS
    fi
}

TARGET=${1:-"help"}

if [ "$TARGET" = "all" ]; then
    for JOB in 01 02 03 04 05 06 07; do
        run_job $JOB
    done
    echo "======================================================"
    echo "✓ Semua job selesai!"
elif [ "$TARGET" = "help" ]; then
    echo "Usage: ./run_job.sh [01|02|03|04|05|06|07|all]"
else
    run_job "$TARGET"
fi