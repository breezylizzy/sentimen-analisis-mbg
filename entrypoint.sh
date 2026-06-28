#!/bin/bash
set -e

# ============================================================
#  MBG Spark - Entrypoint Script [FIXED v2]
#  Compatible dengan apache/spark:3.4.1 docker image
#
#  FIX: Dihapus pengecekan 'hdfs dfs' karena binary hdfs
#       tidak tersedia di image apache/spark:3.4.1.
#       HDFS sudah dihandle oleh bde2020/hadoop-* images.
#  FIX: Worker webui-port bisa di-override via argumen ke-3.
# ============================================================

SPARK_MODE=${1:-master}
SPARK_MASTER=${2:-}
WORKER_WEBUI_PORT=${3:-8081}   # default 8081; worker-2 kirim 8082

echo "======================================================"
echo "MBG Spark Entrypoint"
echo "  Mode       : $SPARK_MODE"
echo "  Master URL : ${SPARK_MASTER:-N/A}"
echo "  WebUI Port : $WORKER_WEBUI_PORT"
echo "======================================================"

# ── Setup environment ──────────────────────────────────────
export SPARK_HOME=${SPARK_HOME:-/opt/spark}
export PYSPARK_PYTHON=/usr/bin/python3
export PYSPARK_DRIVER_PYTHON=/usr/bin/python3

# Buat direktori yang diperlukan
mkdir -p /opt/spark_jobs /opt/configs /opt/data/raw \
         /opt/data/labeled /opt/models "$SPARK_HOME/logs"

echo "SPARK_HOME  : $SPARK_HOME"
echo "PYSPARK     : $PYSPARK_PYTHON"
echo ""

# ── Mulai Spark service ────────────────────────────────────
if [ "$SPARK_MODE" = "master" ]; then
    echo ">>> Starting Spark MASTER pada port 7077 (WebUI: 8080) ..."
    exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.master.Master \
        --host 0.0.0.0 \
        --port 7077 \
        --webui-port 8080

elif [ "$SPARK_MODE" = "worker" ]; then
    if [ -z "$SPARK_MASTER" ]; then
        echo "ERROR: SPARK_MASTER URL wajib diisi untuk mode worker!"
        exit 1
    fi
    echo ">>> Starting Spark WORKER -> $SPARK_MASTER (WebUI: $WORKER_WEBUI_PORT) ..."
    exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker \
        --webui-port "$WORKER_WEBUI_PORT" \
        "$SPARK_MASTER"

else
    echo "ERROR: Mode tidak dikenal: '$SPARK_MODE'"
    echo "Gunakan: entrypoint.sh [master|worker] [spark_master_url] [webui_port]"
    exit 1
fi