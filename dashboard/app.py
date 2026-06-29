import os
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from google.cloud import bigquery
from google.oauth2 import service_account


# =========================
# CONFIG
# =========================
PROJECT_ID = "final-project-500709"
TABLE_ID = "final-project-500709.mbg_sentiment.sentiment_results"
BQ_TIMEOUT_SECONDS = 60

# Ubah ke True kalau masih testing agar query ringan
DEBUG_LIMIT = None


COLOR_MAP = {
    "positif": "#1baf7a",
    "negatif": "#e34948",
    "netral": "#eda100",
}


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Dashboard Sentimen MBG",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

      html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
      }

      .block-container {
        padding: 2rem 2.5rem;
      }

      .kpi-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
      }

      .kpi-label {
        font-size: 12px;
        color: #888;
        margin-bottom: 6px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: .05em;
      }

      .kpi-value {
        font-size: 28px;
        font-weight: 600;
        line-height: 1;
      }

      .kpi-sub {
        font-size: 12px;
        color: #aaa;
        margin-top: 5px;
      }

      .insight-card {
        background: #f8f9ff;
        border-left: 4px solid #4F6EF7;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 13px;
        color: #333;
        line-height: 1.6;
      }

      .insight-card b {
        color: #4F6EF7;
      }

      .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: 1.8rem 0 .8rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def safe_pct(part, total):
    return part / total * 100 if total else 0


def plotly_layout(fig, title=None, legend=True, height=420):
    fig.update_layout(
        title=dict(
            text=title,
            y=0.96,
            x=0.0,
            xanchor="left",
            yanchor="top",
            font=dict(size=15),
        ),
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=70, b=70, l=20, r=20),
        font=dict(family="Inter"),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title_text="",
        ),
    )
    return fig


def kpi_card(col, label, value, sub, color="#333"):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(icon, title, body):
    st.markdown(
        f'<div class="insight-card"><b>{icon} {title}</b><br>{body}</div>',
        unsafe_allow_html=True,
    )


# =========================
# BIGQUERY
# =========================
@st.cache_resource(show_spinner=False)
def get_bq_client():
    # Coba dari st.secrets dulu (Streamlit Cloud / secrets.toml lokal)
    if "gcp_service_account" in st.secrets:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=PROJECT_ID)

    # Fallback: file gcp-key.json untuk keperluan lokal tanpa secrets.toml
    app_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(app_dir, "gcp-key.json")

    if os.path.exists(key_path):
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=PROJECT_ID)

    raise FileNotFoundError(
        "Kredensial GCP tidak ditemukan. "
        "Di Streamlit Cloud: tambahkan [gcp_service_account] di Settings → Secrets. "
        "Lokal: buat .streamlit/secrets.toml atau letakkan gcp-key.json satu folder dengan app.py."
    )


def build_query():
    limit_clause = f"LIMIT {DEBUG_LIMIT}" if DEBUG_LIMIT else ""

    return f"""
        SELECT
            tweet_id,
            created_at,
            final_sentiment,
            model_confidence,
            sarcasm_label,
            sarcasm_confidence,
            label_source,
            review_required,
            uncertainty_entropy,
            uncertainty_margin,
            retweet_count,
            like_count,
            view_count,
            reply_count,
            quote_count,
            prob_positif,
            prob_negatif,
            prob_netral,
            lang
        FROM `{TABLE_ID}`
        WHERE created_at IS NOT NULL
        {limit_clause}
    """


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    client = get_bq_client()

    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        job_timeout_ms=BQ_TIMEOUT_SECONDS * 1000,
    )

    query_job = None

    try:
        query_job = client.query(build_query(), job_config=job_config)
        rows = query_job.result(timeout=BQ_TIMEOUT_SECONDS)

        # Sengaja tidak pakai BigQuery Storage API agar tidak butuh dependency/permission tambahan.
        df = rows.to_dataframe(create_bqstorage_client=False)

    except FuturesTimeoutError:
        if query_job:
            query_job.cancel()

        raise TimeoutError(
            f"Query BigQuery lebih dari {BQ_TIMEOUT_SECONDS} detik, jadi dicancel agar Streamlit tidak nyangkut."
        )

    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])

    df["final_sentiment"] = df["final_sentiment"].astype(str).str.lower()
    df["bulan"] = df["created_at"].dt.to_period("M").astype(str)
    df["tahun"] = df["created_at"].dt.year

    numeric_cols = [
        "model_confidence",
        "sarcasm_label",
        "sarcasm_confidence",
        "uncertainty_entropy",
        "uncertainty_margin",
        "retweet_count",
        "like_count",
        "view_count",
        "reply_count",
        "quote_count",
        "prob_positif",
        "prob_negatif",
        "prob_netral",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df.sort_values("created_at")


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🍱 MBG Sentiment")
    st.markdown("Dashboard analisis sentimen publik terhadap program Makan Bergizi Gratis.")
    st.divider()

    if st.button("🔄 Clear cache & reload"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.caption(f"Project: `{PROJECT_ID}`")
    st.caption(f"Timeout BigQuery: `{BQ_TIMEOUT_SECONDS}` detik")
    st.caption(f"Debug limit: `{DEBUG_LIMIT if DEBUG_LIMIT else 'OFF'}`")


# =========================
# LOAD DATA
# =========================
try:
    with st.spinner("Mengambil data dari BigQuery..."):
        df = load_data()

except Exception as e:
    st.error("❌ Gagal mengambil data dari BigQuery")
    st.warning(str(e))
    st.info(
        "Cek: di Streamlit Cloud sudah ada `[gcp_service_account]` di Settings → Secrets, "
        "BigQuery API aktif, billing aktif, dan service account punya akses BigQuery."
    )
    st.stop()

if df.empty:
    st.warning("Query berhasil, tapi data kosong.")
    st.stop()


# =========================
# FILTERS
# =========================
with st.sidebar:
    st.markdown("### Filter data")

    min_date = df["created_at"].min().date()
    max_date = df["created_at"].max().date()
    default_start = max(pd.Timestamp("2025-01-01").date(), min_date)

    date_range = st.date_input(
        "Rentang tanggal",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    sentimen_filter = st.multiselect(
        "Sentimen",
        options=["positif", "negatif", "netral"],
        default=["positif", "negatif", "netral"],
    )

    show_sarcasm = st.checkbox("Tampilkan hanya tweet sarkasme", value=False)

    st.divider()
    st.caption(f"Data terakhir: {df['created_at'].max().strftime('%d %b %Y')}")


mask = df["final_sentiment"].isin(sentimen_filter)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask &= df["created_at"].dt.date.between(start_date, end_date)

if show_sarcasm:
    mask &= df["sarcasm_label"].eq(1)

dff = df[mask].copy()

if dff.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()


# =========================
# HEADER
# =========================
st.markdown("## Dashboard Analisis Sentimen MBG")
st.markdown(
    f"""
    <span style='color:#888;font-size:13px'>
        {len(dff):,} tweet ·
        {dff['created_at'].min().strftime('%b %Y')} –
        {dff['created_at'].max().strftime('%b %Y')} ·
        Pipeline IndoBERT + XLM-R
    </span>
    """,
    unsafe_allow_html=True,
)


# =========================
# KPI
# =========================
section("Ringkasan keseluruhan")

total = len(dff)
n_pos = dff["final_sentiment"].eq("positif").sum()
n_neg = dff["final_sentiment"].eq("negatif").sum()
n_neu = dff["final_sentiment"].eq("netral").sum()
n_sarc = int(dff["sarcasm_label"].sum())
avg_conf = dff["model_confidence"].mean() * 100

c1, c2, c3, c4, c5 = st.columns(5)

kpi_card(c1, "Total Tweet", f"{total:,}", "terklasifikasi")
kpi_card(c2, "Sentimen Positif", f"{safe_pct(n_pos, total):.1f}%", f"{n_pos:,} tweet", COLOR_MAP["positif"])
kpi_card(c3, "Sentimen Negatif", f"{safe_pct(n_neg, total):.1f}%", f"{n_neg:,} tweet", COLOR_MAP["negatif"])
kpi_card(c4, "Avg. Confidence", f"{avg_conf:.1f}%", "IndoBERT model", "#4F6EF7")
kpi_card(c5, "Tweet Sarkasme", f"{n_sarc:,}", f"{safe_pct(n_sarc, total):.1f}% dari total", COLOR_MAP["netral"])


# =========================
# MONTHLY TREND
# =========================
section("① Tren sentimen bulanan")

monthly = (
    dff.groupby(["bulan", "final_sentiment"])
    .size()
    .reset_index(name="count")
    .sort_values("bulan")
)

col_a, col_b = st.columns([3, 2])

with col_a:
    fig = px.bar(
        monthly,
        x="bulan",
        y="count",
        color="final_sentiment",
        color_discrete_map=COLOR_MAP,
        barmode="stack",
        labels={"bulan": "", "count": "Jumlah tweet", "final_sentiment": "Sentimen"},
    )
    fig.update_xaxes(tickangle=45, showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(plotly_layout(fig, "Volume tweet per bulan"), use_container_width=True)

with col_b:
    pivot = (
        monthly.pivot_table(
            index="bulan",
            columns="final_sentiment",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .sort_values("bulan")
    )

    for col in ["positif", "negatif", "netral"]:
        if col not in pivot:
            pivot[col] = 0

    pivot["total"] = pivot[["positif", "negatif", "netral"]].sum(axis=1)
    pivot["pct_neg"] = pivot["negatif"] / pivot["total"] * 100

    fig = go.Figure()
    fig.add_hrect(y0=30, y1=100, fillcolor="#ffe5e5", opacity=0.3, line_width=0)
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=COLOR_MAP["negatif"],
        annotation_text="Threshold 30%",
        annotation_position="top right",
    )
    fig.add_trace(
        go.Scatter(
            x=pivot["bulan"],
            y=pivot["pct_neg"],
            mode="lines+markers",
            name="% Negatif",
            line=dict(color=COLOR_MAP["negatif"], width=2.5),
            marker=dict(size=6, color=COLOR_MAP["negatif"]),
            fill="tozeroy",
            fillcolor="rgba(227,73,72,0.08)",
        )
    )
    fig.update_xaxes(tickangle=45, showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0", ticksuffix="%", range=[0, 100])
    st.plotly_chart(plotly_layout(fig, "% sentimen negatif — indikator krisis", legend=False), use_container_width=True)


# =========================
# SENTIMENT + CONFIDENCE
# =========================
section("② Distribusi sentimen & kualitas model")

col1, col2, col3 = st.columns(3)

with col1:
    sent_counts = dff["final_sentiment"].value_counts().reset_index()
    sent_counts.columns = ["sentimen", "count"]

    fig = px.pie(
        sent_counts,
        values="count",
        names="sentimen",
        color="sentimen",
        color_discrete_map=COLOR_MAP,
        hole=0.55,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    st.plotly_chart(plotly_layout(fig, "Distribusi final sentiment", legend=False), use_container_width=True)

with col2:
    bins = [0, 0.5, 0.7, 0.85, 0.95, 1.0]
    labels = ["<50%", "50–70%", "70–85%", "85–95%", ">95%"]

    dff["conf_bucket"] = pd.cut(
        dff["model_confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    conf_dist = dff["conf_bucket"].value_counts().sort_index().reset_index()
    conf_dist.columns = ["bucket", "count"]

    fig = px.bar(
        conf_dist,
        x="bucket",
        y="count",
        color="bucket",
        color_discrete_sequence=["#f7c1c1", "#f09595", "#eda100", "#1baf7a", "#185fa5"],
        labels={"bucket": "Confidence", "count": "Jumlah tweet"},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(plotly_layout(fig, "Distribusi confidence model", legend=False), use_container_width=True)

with col3:
    conf_sent = (
        dff.groupby(["final_sentiment", "conf_bucket"], observed=False)
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        conf_sent,
        x="count",
        y="final_sentiment",
        color="conf_bucket",
        orientation="h",
        color_discrete_sequence=["#f7c1c1", "#f09595", "#eda100", "#1baf7a", "#185fa5"],
        labels={"final_sentiment": "", "count": "Tweet", "conf_bucket": "Confidence"},
    )
    fig.update_xaxes(gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(plotly_layout(fig, "Confidence per sentimen"), use_container_width=True)


# =========================
# ENGAGEMENT
# =========================
section("③ Engagement — siapa yang lebih viral?")

col1, col2 = st.columns(2)

with col1:
    eng_mean = (
        dff.groupby("final_sentiment")[["retweet_count", "like_count", "reply_count"]]
        .mean()
        .reset_index()
    )

    eng_mean = eng_mean.melt(
        id_vars="final_sentiment",
        var_name="metric",
        value_name="mean",
    )

    eng_mean["metric"] = eng_mean["metric"].map(
        {
            "retweet_count": "Retweet",
            "like_count": "Like",
            "reply_count": "Reply",
        }
    )

    fig = px.bar(
        eng_mean,
        x="metric",
        y="mean",
        color="final_sentiment",
        barmode="group",
        color_discrete_map=COLOR_MAP,
        labels={"metric": "", "mean": "Rata-rata", "final_sentiment": "Sentimen"},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(plotly_layout(fig, "Rata-rata engagement per sentimen"), use_container_width=True)

with col2:
    eng_median = (
        dff.groupby("final_sentiment")[["retweet_count", "like_count", "view_count"]]
        .median()
        .reset_index()
    )

    eng_median = eng_median.melt(
        id_vars="final_sentiment",
        var_name="metric",
        value_name="median",
    )

    eng_median["metric"] = eng_median["metric"].map(
        {
            "retweet_count": "Retweet",
            "like_count": "Like",
            "view_count": "View",
        }
    )

    fig = px.bar(
        eng_median,
        x="metric",
        y="median",
        color="final_sentiment",
        barmode="group",
        color_discrete_map=COLOR_MAP,
        labels={"metric": "", "median": "Median", "final_sentiment": "Sentimen"},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(plotly_layout(fig, "Median engagement per sentimen"), use_container_width=True)


# =========================
# SARCASM
# =========================
section("④ Analisis sarkasme")

col1, col2 = st.columns(2)

with col1:
    sarc = (
        dff.groupby(["sarcasm_label", "final_sentiment"])
        .size()
        .reset_index(name="count")
    )

    sarc["sarcasm_label"] = sarc["sarcasm_label"].map(
        {
            0: "Non-sarkasme",
            1: "Sarkasme",
        }
    ).fillna("Unknown")

    fig = px.bar(
        sarc,
        x="sarcasm_label",
        y="count",
        color="final_sentiment",
        color_discrete_map=COLOR_MAP,
        barmode="stack",
        labels={"sarcasm_label": "", "count": "Jumlah tweet", "final_sentiment": "Sentimen"},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(plotly_layout(fig, "Distribusi sarkasme × sentimen akhir"), use_container_width=True)

with col2:
    st.markdown("**Insight sarkasme**")
    pct_sarc = safe_pct(n_sarc, total)

    st.markdown(
        f"""
        <div class="insight-card">
            📊 <b>{n_sarc:,} tweet sarkasme</b> ({pct_sarc:.1f}% dari total).
            Lapisan sarkasme membantu membaca ekspresi positif palsu yang sebenarnya mengandung kritik.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="insight-card">
            🤖 <b>Pipeline berlapis</b> membantu mengurangi risiko tweet bernada sarkastik
            salah masuk sebagai sentimen positif.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="insight-card">
            📉 <b>Engagement negatif</b> bisa dibandingkan lewat median retweet, like,
            dan view untuk membaca efek negativity bias.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# UNCERTAINTY
# =========================
section("⑤ Kualitas model — uncertainty & label source")

col1, col2, col3 = st.columns(3)

with col1:
    high_unc = dff[dff["uncertainty_entropy"] > 0.8]
    unc_counts = high_unc["final_sentiment"].value_counts().reset_index()
    unc_counts.columns = ["sentimen", "count"]

    if high_unc.empty:
        st.info("Tidak ada tweet high-uncertainty dengan entropi > 0.8.")
    else:
        fig = px.pie(
            unc_counts,
            values="count",
            names="sentimen",
            color="sentimen",
            color_discrete_map=COLOR_MAP,
            hole=0.5,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        title = f"High uncertainty<br><sup>{len(high_unc):,} tweet ({safe_pct(len(high_unc), total):.1f}%)</sup>"
        st.plotly_chart(plotly_layout(fig, title, legend=False), use_container_width=True)

with col2:
    src_counts = dff["label_source"].value_counts().reset_index()
    src_counts.columns = ["source", "count"]

    fig = px.pie(
        src_counts,
        values="count",
        names="source",
        color_discrete_sequence=["#4F6EF7", "#eda100"],
        hole=0.5,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    st.plotly_chart(plotly_layout(fig, "Sumber label klasifikasi", legend=False), use_container_width=True)

with col3:
    sample_df = dff.sample(min(3000, len(dff)), random_state=42)

    fig = px.scatter(
        sample_df,
        x="model_confidence",
        y="uncertainty_entropy",
        color="final_sentiment",
        color_discrete_map=COLOR_MAP,
        opacity=0.4,
        labels={
            "model_confidence": "Confidence",
            "uncertainty_entropy": "Entropy",
            "final_sentiment": "Sentimen",
        },
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_xaxes(gridcolor="#f0f0f0")
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(plotly_layout(fig, "Confidence vs uncertainty"), use_container_width=True)


# =========================
# FINDINGS
# =========================
section("⑥ Key findings untuk skripsi")

findings = [
    (
        "📊",
        "Dominasi sentimen",
        f"Positif {safe_pct(n_pos, total):.1f}%, negatif {safe_pct(n_neg, total):.1f}%, dan netral {safe_pct(n_neu, total):.1f}%.",
    ),
    (
        "🔥",
        "Negativity bias",
        "Bandingkan median view, retweet, dan like untuk melihat apakah konten kritik menyebar lebih kuat.",
    ),
    (
        "🤖",
        "Kualitas model",
        f"Confidence rata-rata {avg_conf:.1f}%. Tweet high-uncertainty dapat dijadikan kandidat review manual.",
    ),
    (
        "🎭",
        "Sarkasme",
        f"Terdapat {n_sarc:,} tweet terdeteksi sarkasme pada data terfilter.",
    ),
]

cols = st.columns(2)

for i, item in enumerate(findings):
    with cols[i % 2]:
        insight(*item)


# =========================
# FOOTER
# =========================
st.divider()
st.markdown(
    """
    <p style='text-align:center;color:#aaa;font-size:12px'>
        Dashboard Analisis Sentimen MBG · Final Project Semester 6 ·
        Pipeline: Docker → Spark → HDFS → IndoBERT → BigQuery
    </p>
    """,
    unsafe_allow_html=True,
)
