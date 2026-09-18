##################################################
# EŞİK - FİNANSAL RİSK ERKEN UYARI DASHBOARDU
##################################################

import json
import hashlib
import os
import uuid
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from n8n_client import N8nClientError, send_chat_message
from esik_voice import voice_chat_input
from esik_email import render_report_email
from esik_company_query import CompanyQuery, CompanyQueryError
from esik_chat_context import CHAT_CONTEXT_VERSION
from esik_calibration import load_probability_layer, company_probability
from esik_assessment import (render_company_assessment, render_calibration_validation,
                             probability_answer, percent, observed_scenario_candidates)
from esik_artifacts import validate_artifacts, priority_fields
from esik_ui import (apply_visual_theme, sidebar_brand, page_header,
                     context_strip, go_to_page, outcome)
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)


##################################################
# SAYFA AYARLARI
##################################################

st.set_page_config(
    page_title="Eşik | Finansal Risk",
    page_icon="◒",
    layout="wide"
)
apply_visual_theme()


##################################################
# DOSYA YOLLARI
##################################################

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "esik_final_model_pipeline.pkl"
)

METADATA_PATH = (
    BASE_DIR
    / "outputs"
    / "esik_model_metadata.json"
)

RISK_LIST_PATH = (
    BASE_DIR
    / "outputs"
    / "esik_dashboard_risk_list.csv"
)

SCORED_COMPANIES_PATH = (
    BASE_DIR
    / "outputs"
    / "esik_scored_test_companies.csv"
)

SHAP_GLOBAL_PATH = (
    BASE_DIR
    / "outputs"
    / "esik_shap_global_importance.csv"
)

SHAP_LOCAL_PATH = (
    BASE_DIR
    / "outputs"
    / "esik_shap_local_values.csv"
)


##################################################
# DOSYA KONTROLÜ
##################################################

required_files = [
    MODEL_PATH,
    METADATA_PATH,
    RISK_LIST_PATH,
    SCORED_COMPANIES_PATH,
    SHAP_GLOBAL_PATH,
    SHAP_LOCAL_PATH
]

missing_files = [
    str(file_path)
    for file_path in required_files
    if not file_path.exists()
]

if missing_files:
    st.error(
        "Dashboard için gerekli dosyalardan bazıları bulunamadı."
    )

    for missing_file in missing_files:
        st.write(missing_file)

    st.stop()


##################################################
# VERİ VE MODEL YÜKLEME FONKSİYONLARI
##################################################

artifact_version = "reference-v1"
if (BASE_DIR / "outputs" / "artifact_manifest.json").exists():
    try:
        artifact_version = validate_artifacts(BASE_DIR)["model_sha256"]
    except ValueError as error:
        st.error(str(error))
        st.stop()
elif json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("model_version"):
    st.error("Bu model sürümünün artifact_manifest.json dosyası eksik. Aynı sürümün tüm çıktılarını birlikte yükleyin.")
    st.stop()

@st.cache_data
def load_csv(file_path, version):
    return pd.read_csv(file_path)


@st.cache_data
def load_metadata(file_path, version):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


@st.cache_resource
def load_model_package(file_path, version):
    return joblib.load(file_path)


@st.cache_resource
def load_company_query(version, chat_url, chat_token):
    return CompanyQuery(BASE_DIR, chat_url=chat_url, chat_token=chat_token)


risk_df = load_csv(RISK_LIST_PATH, artifact_version)
scored_companies_df = load_csv(SCORED_COMPANIES_PATH, artifact_version)
shap_global_df = load_csv(SHAP_GLOBAL_PATH, artifact_version)
shap_local_df = load_csv(SHAP_LOCAL_PATH, artifact_version)
metadata = load_metadata(METADATA_PATH, artifact_version)
model_package = load_model_package(MODEL_PATH, artifact_version)

try:
    probability_layer = load_probability_layer(BASE_DIR, artifact_version)
except ValueError as error:
    probability_layer = None
    st.warning(str(error))
probability_manifest = BASE_DIR / 'outputs/calibration_v1/manifest.json'
analysis_version = CHAT_CONTEXT_VERSION + '|' + artifact_version + '|' + str(probability_manifest.stat().st_mtime_ns if probability_manifest.exists() else 0)

model_ready = "pipeline" in model_package
feature_columns = metadata.get(
    "feature_columns",
    [column for column in scored_companies_df.columns if column.startswith("Attr")]
)


##################################################
# YAN MENÜ
##################################################

with st.sidebar:
    sidebar_brand()

selected_page = st.sidebar.radio(
    "Navigasyon",
    [
        "Yönetici Özeti",
        "Analist Öncelik Listesi",
        "Şirket Detayı",
        "Yeni Veri Analizi",
        "Eşik AI",
        "Model Açıklaması",
        "Model Doğrulaması",
        "Model Kartı"
    ],
    key="esik_page",
    label_visibility="collapsed",
    width="stretch",
)

st.sidebar.divider()

if model_ready:
    st.sidebar.success("Model kullanıma hazır")
else:
    st.sidebar.error("Model yüklenemedi")

st.sidebar.caption(
    f"Model: {metadata['model_name']}\n\n"
    "Tahmin ufku: 1 yıl\n\n"
    "Operasyonel kapasite: İlk %10"
)
context_strip()


##################################################
# ORTAK HESAPLAMALAR
##################################################

analyst_review_df = risk_df[
    risk_df["ANALYST_REVIEW"] == 1
]

total_company_count = len(risk_df)
reviewed_company_count = len(analyst_review_df)

total_bankrupt_count = int(
    risk_df["ACTUAL_BANKRUPT"].sum()
)

captured_bankrupt_count = int(
    analyst_review_df["ACTUAL_BANKRUPT"].sum()
)

recall_at_top_10 = (
    captured_bankrupt_count
    / total_bankrupt_count
)

precision_at_top_10 = (
    captured_bankrupt_count
    / reviewed_company_count
)

lift_at_top_10 = metadata[
    "test_lift_at_top_10"
]


##################################################
# RİSK GRUBU GRAFİĞİ
##################################################

def create_risk_group_chart(dataframe):
    risk_group_order = [
        "Kritik - İlk %5",
        "Yüksek - %5 ile %10",
        "Orta - %10 ile %20",
        "Düşük - %20 sonrası"
    ]

    risk_group_counts = (
        dataframe["RISK_GROUP"]
        .value_counts()
        .reindex(
            risk_group_order,
            fill_value=0
        )
        .rename("Şirket Sayısı")
        .reset_index()
    )

    risk_group_counts.columns = [
        "Risk Grubu",
        "Şirket Sayısı"
    ]

    bars = (
        alt.Chart(risk_group_counts)
        .mark_bar(
            cornerRadiusEnd=7,
            size=27
        )
        .encode(
            y=alt.Y(
                "Risk Grubu:N",
                sort=risk_group_order,
                title=None
            ),
            x=alt.X(
                "Şirket Sayısı:Q",
                title="Şirket Sayısı"
            ),
            color=alt.Color(
                "Risk Grubu:N",
                scale=alt.Scale(
                    domain=risk_group_order,
                    range=[
                        "#FF725C",
                        "#B6A0FF",
                        "#67CEDA",
                        "#718096"
                    ]
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip(
                    "Risk Grubu:N",
                    title="Risk Grubu"
                ),
                alt.Tooltip(
                    "Şirket Sayısı:Q",
                    title="Şirket Sayısı"
                )
            ]
        )
    )

    labels = (
        alt.Chart(risk_group_counts)
        .mark_text(
            align="left",
            baseline="middle",
            dx=5,
            color="#F4F5F0"
        )
        .encode(
            y=alt.Y(
                "Risk Grubu:N",
                sort=risk_group_order
            ),
            x=alt.X("Şirket Sayısı:Q"),
            text=alt.Text("Şirket Sayısı:Q")
        )
    )

    return (
        bars
        + labels
    ).properties(height=260)


##################################################
# FİNANSAL GÖSTERGE TANIMLARI
##################################################

shap_global_df = shap_global_df.sort_values(
    "SHAP_RANK"
).reset_index(drop=True)

TOP_FEATURES = (
    shap_global_df
    .head(15)["VARIABLE"]
    .tolist()
)

FEATURE_DESCRIPTIONS = dict(
    zip(
        shap_global_df["VARIABLE"],
        shap_global_df["DESCRIPTION"]
    )
)

FEATURE_IMPORTANCE = dict(
    zip(
        shap_global_df["VARIABLE"],
        shap_global_df["SHAP_IMPORTANCE_RATIO"]
    )
)


def get_financial_driver_group(description):
    """Map technical ratio descriptions to a readable financial theme."""

    description_lower = str(description).lower()

    if any(
        keyword in description_lower
        for keyword in [
            "likit",
            "asit-test",
            "cari oran",
            "cari varlık",
            "çalışma sermayesi",
            "kısa vadeli yükümlülük"
        ]
    ):
        return "Likidite"

    if any(
        keyword in description_lower
        for keyword in [
            "borç",
            "yükümlülük",
            "finansman gider",
            "özkaynak",
            "sermaye"
        ]
    ):
        return "Borçluluk ve finansman"

    if any(
        keyword in description_lower
        for keyword in [
            "kâr",
            "kar",
            "satış",
            "faaliyet gider"
        ]
    ):
        return "Kârlılık ve operasyon"

    return "Diğer finansal gösterge"


def create_plain_language_summary(shap_data):
    """Create a concise, non-causal explanation from local SHAP values."""

    increasing = (
        shap_data
        .query("SHAP_VALUE > 0")
        .sort_values("SHAP_VALUE", ascending=False)
        .head(3)
    )

    reducing = (
        shap_data
        .query("SHAP_VALUE < 0")
        .sort_values("SHAP_VALUE")
        .head(2)
    )

    increasing_groups = [
        get_financial_driver_group(description)
        for description in increasing["DESCRIPTION"]
    ]

    increasing_groups = list(dict.fromkeys(increasing_groups))

    if increasing_groups:
        group_text = ", ".join(increasing_groups).lower()
        summary = (
            f"Model skorunun yükselmesinde özellikle {group_text} "
            "başlıklarındaki göstergeler etkili oldu."
        )
    else:
        summary = (
            "Bu şirket için model skorunu yükselten belirgin bir "
            "pozitif SHAP katkısı görülmedi."
        )

    if not reducing.empty:
        reducing_groups = list(dict.fromkeys(
            get_financial_driver_group(description)
            for description in reducing["DESCRIPTION"]
        ))
        reducing_text = ", ".join(reducing_groups).lower()
        summary += (
            f" {reducing_text.capitalize()} başlıklarındaki bazı "
            "göstergeler ise riski azaltıcı yönde katkı sağladı."
        )

    return summary


def create_action_recommendations(shap_data):
    """Translate top positive SHAP drivers into analyst checks."""

    action_map = {
        "Likidite": (
            "Likidite oranlarını, kısa vadeli yükümlülükleri ve nakit "
            "karşılama kapasitesini kontrol et."
        ),
        "Borçluluk ve finansman": (
            "Borç servis kapasitesini, finansman giderlerini ve özkaynak "
            "yeterliliğini incele."
        ),
        "Kârlılık ve operasyon": (
            "Faaliyet kârlılığını, satış maliyetlerini ve operasyonel gider "
            "trendini incele."
        ),
        "Diğer finansal gösterge": (
            "Göstergenin hesaplanma biçimini ve şirketin finansal tablolarındaki "
            "karşılığını analist olarak doğrula."
        )
    }

    top_drivers = (
        shap_data
        .query("SHAP_VALUE > 0")
        .sort_values("SHAP_VALUE", ascending=False)
        .head(3)
    )

    recommendations = []
    seen_groups = set()

    for _, row in top_drivers.iterrows():
        group = get_financial_driver_group(row["DESCRIPTION"])
        if group in seen_groups:
            continue

        recommendations.append({
            "group": group,
            "feature": row["DESCRIPTION"],
            "action": action_map[group]
        })
        seen_groups.add(group)

    return recommendations


def create_data_quality_report(company_row, feature_columns, reference_df):
    """Return transparent checks for one company's model inputs."""

    numeric_values = pd.to_numeric(
        company_row[feature_columns],
        errors="coerce"
    )

    missing_features = numeric_values.index[numeric_values.isna()].tolist()

    extreme_feature_count = 0
    for feature in feature_columns:
        feature_reference = pd.to_numeric(
            reference_df[feature],
            errors="coerce"
        ).dropna()

        if feature_reference.empty or pd.isna(numeric_values[feature]):
            continue

        lower_bound = feature_reference.quantile(0.01)
        upper_bound = feature_reference.quantile(0.99)

        if (
            numeric_values[feature] < lower_bound
            or numeric_values[feature] > upper_bound
        ):
            extreme_feature_count += 1

    return {
        "missing_count": len(missing_features),
        "missing_features": missing_features,
        "extreme_count": extreme_feature_count,
        "total_count": len(feature_columns)
    }


def predict_model_score(model_package, feature_values, feature_columns):
    """Predict one scenario without presenting the score as calibrated probability."""

    input_frame = pd.DataFrame(
        [feature_values],
        columns=feature_columns
    )

    return float(
        model_package["pipeline"]
        .predict_proba(input_frame)[:, 1][0]
    )


##################################################
# ŞİRKET KARŞILAŞTIRMA FONKSİYONLARI
##################################################

def create_company_comparison(selected_row, reference_df):
    comparison_rows = []

    for feature in TOP_FEATURES:
        reference_values = pd.to_numeric(
            reference_df[feature],
            errors="coerce"
        )

        company_value = pd.to_numeric(
            pd.Series([selected_row[feature]]),
            errors="coerce"
        ).iloc[0]

        portfolio_median = reference_values.median()

        if pd.isna(company_value):
            portfolio_comparison = "Veri eksik"

        elif company_value < portfolio_median:
            portfolio_comparison = "Medyanın altında"

        elif company_value > portfolio_median:
            portfolio_comparison = "Medyanın üzerinde"

        else:
            portfolio_comparison = "Medyanla aynı"

        comparison_rows.append({
            "VARIABLE": feature,
            "DESCRIPTION": FEATURE_DESCRIPTIONS[feature],
            "COMPANY_VALUE": company_value,
            "PORTFOLIO_MEDIAN": portfolio_median,
            "SHAP_IMPORTANCE": FEATURE_IMPORTANCE[feature] * 100,
            "PORTFOLIO_COMPARISON": portfolio_comparison
        })

    return pd.DataFrame(comparison_rows)


def create_company_shap_chart(company_shap_df):
    chart_df = (
        company_shap_df
        .sort_values(
            "ABS_SHAP_VALUE",
            ascending=False
        )
        .head(10)
        .copy()
    )

    chart_df["FEATURE_LABEL"] = (
        chart_df["VARIABLE"]
        + " · "
        + chart_df["DESCRIPTION"]
    )

    bars = (
        alt.Chart(chart_df)
        .mark_bar(
            cornerRadius=5,
            size=25
        )
        .encode(
            y=alt.Y(
                "FEATURE_LABEL:N",
                sort=chart_df["FEATURE_LABEL"].tolist(),
                title=None
            ),
            x=alt.X(
                "SHAP_VALUE:Q",
                title="SHAP Katkısı"
            ),
            color=alt.Color(
                "SHAP_DIRECTION:N",
                scale=alt.Scale(
                    domain=[
                        "Riski Artırıyor",
                        "Riski Azaltıyor",
                        "Nötr"
                    ],
                    range=[
                        "#FF725C",
                        "#D7FF42",
                        "#ACB3C2"
                    ]
                ),
                title=None
            ),
            tooltip=[
                alt.Tooltip(
                    "DESCRIPTION:N",
                    title="Gösterge"
                ),
                alt.Tooltip(
                    "MODEL_INPUT_VALUE:Q",
                    title="Model Girdisi",
                    format=".3f"
                ),
                alt.Tooltip(
                    "SHAP_VALUE:Q",
                    title="SHAP Katkısı",
                    format=".3f"
                ),
                alt.Tooltip(
                    "SHAP_DIRECTION:N",
                    title="Etki Yönü"
                )
            ]
        )
    )

    zero_line = (
        alt.Chart(
            pd.DataFrame({"ZERO": [0]})
        )
        .mark_rule(
            color="#ACB3C2",
            strokeDash=[4, 4]
        )
        .encode(x="ZERO:Q")
    )

    return (bars + zero_line).properties(
        height=380
    )


def create_global_shap_chart(global_shap_data):
    chart_df = (
        global_shap_data
        .sort_values("SHAP_RANK")
        .head(15)
        .copy()
    )

    chart_df["FEATURE_LABEL"] = (
        chart_df["VARIABLE"]
        + " · "
        + chart_df["DESCRIPTION"]
    )

    return (
        alt.Chart(chart_df)
        .mark_bar(
            cornerRadiusEnd=7,
            size=23,
            color="#B6A0FF"
        )
        .encode(
            y=alt.Y(
                "FEATURE_LABEL:N",
                sort=chart_df["FEATURE_LABEL"].tolist(),
                title=None
            ),
            x=alt.X(
                "MEAN_ABS_SHAP:Q",
                title="Ortalama Mutlak SHAP Değeri"
            ),
            tooltip=[
                alt.Tooltip(
                    "DESCRIPTION:N",
                    title="Gösterge"
                ),
                alt.Tooltip(
                    "MEAN_ABS_SHAP:Q",
                    title="Ortalama |SHAP|",
                    format=".3f"
                ),
                alt.Tooltip(
                    "SHAP_IMPORTANCE_RATIO:Q",
                    title="Önem Payı",
                    format=".1%"
                )
            ]
        )
        .properties(height=480)
    )


##################################################
# YÖNETİCİ ÖZETİ
##################################################

if selected_page == "Yönetici Özeti":

    page_header(
        "Önce hangi şirket?",
        f"{total_company_count:,}".replace(",", ".")
        + " şirket arasından inceleme önceliğini belirleyin. "
        + "Risk sırasını, olasılık tahminini ve finansal dayanakları birlikte inceleyin.",
        section="YÖNETİCİ ÖZETİ", illustration="inspect", hero=True,
    )
    with st.container(horizontal=True, gap="small"):
        st.button("Öncelik listesini aç", type="primary", on_click=go_to_page,
                  args=("Analist Öncelik Listesi",), key="home_priority")
        st.button("Şirket incele", on_click=go_to_page,
                  args=("Şirket Detayı",), key="home_company")
        st.button("Eşik AI ile sor", on_click=go_to_page,
                  args=("Eşik AI",), key="home_ai")

    (
        metric_col_1,
        metric_col_2,
        metric_col_3,
        metric_col_4,
        metric_col_5
    ) = st.columns(5)

    with metric_col_1:
        st.metric(
            "Değerlendirilen Şirket",
            f"{total_company_count:,}".replace(",", ".")
        )

    with metric_col_2:
        st.metric(
            "Analist İncelemesi",
            f"{reviewed_company_count}",
            help="Risk sıralamasındaki ilk yüzde 10"
        )

    with metric_col_3:
        st.metric(
            "Yakalanan İflas",
            (
                f"{captured_bankrupt_count}"
                f" / {total_bankrupt_count}"
            )
        )

    with metric_col_4:
        st.metric(
            "Yakalama@%10",
            f"%{recall_at_top_10 * 100:.2f}".replace(".", ","),
            help="İlk %10 inceleme listesinde yakalanan gerçek iflasların oranı (Recall@10%).",
        )

    with metric_col_5:
        st.metric(
            "Lift@Top10",
            f"{lift_at_top_10:.2f}×".replace(".", ",")
        )

    outcome(captured_bankrupt_count, total_bankrupt_count,
            reviewed_company_count, precision_at_top_10)

    left_column, right_column = st.columns(
        [1, 1.6]
    )

    with left_column:
        st.subheader("Risk Gruplarının Dağılımı")

        risk_group_chart = create_risk_group_chart(
            risk_df
        )

        st.altair_chart(
            risk_group_chart,
            width="stretch"
        )

    with right_column:
        st.subheader("Öncelikli Analist Listesi")

        priority_preview_df = (
            risk_df
            .head(10)
            .copy()
        )

        priority_preview_df["RISK_SCORE"] = (
            priority_preview_df["RISK_SCORE"]
            * 100
        ).round(1)

        priority_preview_df = (
            priority_preview_df.rename(
                columns={
                    "COMPANY_ID": "Şirket",
                    "RISK_RANK": "Sıra",
                    "RISK_SCORE": "Risk Skoru",
                    "RISK_GROUP": "Risk Grubu",
                    "RECOMMENDED_ACTION":
                        "Önerilen Aksiyon"
                }
            )
        )

        st.dataframe(
            priority_preview_df[
                [
                    "Sıra",
                    "Şirket",
                    "Risk Skoru",
                    "Risk Grubu",
                    "Önerilen Aksiyon"
                ]
            ],
            width="stretch",
            hide_index=True
        )

    st.caption(
        "Risk skoru, modelin ham (kalibre edilmemiş) iflas olasılığı "
        "tahminidir. Şirketleri sıralamak için bu tahmini kullanıyoruz. "
        "Kalibre olasılık, Şirket Detayı bölümünde ayrıca gösterilir."
    )


##################################################
# ANALİST ÖNCELİK LİSTESİ
##################################################

elif selected_page == "Analist Öncelik Listesi":

    page_header(
        "Analist Öncelik Listesi",
        "İnceleme kapasitenize göre filtreleyin. Model sırası ve olasılık tahmini ayrı ölçümlerdir.",
        section="01 / ÖNCELİKLENDİRME",
    )

    filter_col_1, filter_col_2 = st.columns(2)

    risk_group_options = [
        "Kritik - İlk %5",
        "Yüksek - %5 ile %10",
        "Orta - %10 ile %20",
        "Düşük - %20 sonrası"
    ]

    with filter_col_1:
        selected_groups = st.multiselect(
            "Risk grubu",
            options=risk_group_options,
            default=[
                "Kritik - İlk %5",
                "Yüksek - %5 ile %10"
            ]
        )

    with filter_col_2:
        minimum_risk_score = st.slider(
            "Minimum risk skoru",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )

    filtered_risk_df = risk_df[
        risk_df["RISK_GROUP"].isin(
            selected_groups
        )
    ].copy()

    filtered_risk_df = filtered_risk_df[
        filtered_risk_df["RISK_SCORE"] * 100
        >= minimum_risk_score
    ]

    show_actual_status = st.checkbox(
        "Gerçek test sonucunu göster",
        value=False,
        help=(
            "Gerçek uygulamada bu bilgi bilinmez. "
            "Yalnızca model değerlendirmesi için bulunur."
        )
    )

    filtered_risk_df["RISK_SCORE"] = (
        filtered_risk_df["RISK_SCORE"]
        * 100
    ).round(1)

    display_columns = [
        "RISK_RANK",
        "COMPANY_ID",
        "RISK_SCORE",
        "RISK_PERCENTILE",
        "RISK_GROUP",
        "RECOMMENDED_ACTION"
    ]

    if probability_layer is not None:
        filtered_risk_df['CALIBRATED_PROBABILITY_PCT'] = filtered_risk_df.COMPANY_ID.map(
            probability_layer['rows'].CALIBRATED_PROBABILITY) * 100
        display_columns.insert(3, 'CALIBRATED_PROBABILITY_PCT')

    if show_actual_status:
        display_columns.append(
            "ACTUAL_STATUS"
        )

    analyst_table_df = (
        filtered_risk_df[display_columns]
        .rename(
            columns={
                "RISK_RANK": "Risk Sırası",
                "COMPANY_ID": "Şirket",
                "RISK_SCORE": "Risk Skoru",
                "CALIBRATED_PROBABILITY_PCT": "1 yıllık olasılık tahmini (%)",
                "RISK_PERCENTILE": "Risk Yüzdeliği",
                "RISK_GROUP": "Risk Grubu",
                "RECOMMENDED_ACTION":
                    "Önerilen Aksiyon",
                "ACTUAL_STATUS": "Gerçek Sonuç"
            }
        )
    )

    st.metric(
        "Listelenen Şirket",
        len(analyst_table_df)
    )

    if analyst_table_df.empty:
        st.info(
            "Seçilen filtrelere uygun şirket bulunamadı."
        )
    else:
        selection = st.dataframe(
            analyst_table_df,
            width="stretch",
            hide_index=True,
            key="analyst_company_table",
            on_select="rerun",
            selection_mode="single-row",
            column_config={"1 yıllık olasılık tahmini (%)": st.column_config.NumberColumn(format="%.1f")}
        )
        from esik_ui import open_company
        if selection.selection.rows:
            chosen_id = analyst_table_df.iloc[selection.selection.rows[0]]["Şirket"]
            st.button(f"{chosen_id} · Şirketi incele", type="primary",
                      on_click=open_company, args=(chosen_id,))
        else:
            st.caption("Bir satır seçin; şirketin finansal kanıtlarına ve senaryolarına doğrudan geçin.")
        if probability_layer is not None:
            probabilities = filtered_risk_df['CALIBRATED_PROBABILITY_PCT'] / 100
            st.metric('Bu listedeki beklenen iflas sayısı', f'{probabilities.sum():.1f}')
            st.caption('Şirketlerin kalibrasyon tahminlerinin toplamı. Gerçekleşmiş vaka sayımı değildir; tarihsel veri kapsamındaki keşifsel beklentidir. Ham risk skoru, risk yüzdeliği ve olasılık farklı ölçümlerdir.')
        st.download_button('Bu listenin analizini CSV indir',analyst_table_df.to_csv(index=False).encode('utf-8-sig'),
                           file_name='ESIK_inceleme_ve_olasilik_listesi.csv',mime='text/csv')


##################################################
# ŞİRKET DETAYI
##################################################

elif selected_page == "Şirket Detayı":

    page_header(
        "Şirket Detayı",
        "Olasılık tahminini, modelin dayanaklarını ve eksik finansal girdileri birlikte inceleyin.",
        section="02 / FİNANSAL KANIT",
        illustration="inspect",
    )

    company_selection_df = (
        scored_companies_df
        .sort_values("RISK_RANK")
        .reset_index(drop=True)
    )

    company_labels = {
        row["COMPANY_ID"]: (
            f"{row['COMPANY_ID']} · "
            f"Risk sırası {int(row['RISK_RANK'])} · "
            f"{row['RISK_GROUP']}"
        )
        for _, row in company_selection_df.iterrows()
    }

    if "esik_company_pending" in st.session_state:
        st.session_state["esik_company_selector"] = st.session_state.pop("esik_company_pending")
    selected_company_id = st.selectbox(
        "İncelenecek şirket",
        options=company_selection_df["COMPANY_ID"].tolist(),
        key="esik_company_selector",
        format_func=lambda company_id: company_labels[company_id]
    )

    selected_company = company_selection_df.loc[
        company_selection_df["COMPANY_ID"]
        == selected_company_id
    ].iloc[0]

    company_comparison_df = create_company_comparison(
        selected_company,
        scored_companies_df
    )

    selected_company_shap_df = shap_local_df.loc[
        shap_local_df["COMPANY_ID"]
        == selected_company_id
    ].copy()

    from esik_ui import company_navigation
    company_navigation()

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = (
        st.columns(4)
    )

    metric_col_1.metric(
        "Risk Skoru",
        f"{selected_company['RISK_SCORE'] * 100:.1f}/100"
    )

    metric_col_2.metric(
        "Risk Sırası",
        (
            f"{int(selected_company['RISK_RANK'])} "
            f"/ {len(company_selection_df)}"
        )
    )

    metric_col_3.metric(
        "Risk Yüzdeliği",
        f"%{selected_company['RISK_PERCENTILE']:.1f}"
    )

    metric_col_4.metric(
        "Risk Grubu",
        selected_company["RISK_GROUP"]
    )

    st.html('<div id="finansal-kanit" class="esik-anchor"></div>')
    render_company_assessment(probability_layer, selected_company, selected_company_shap_df)
    st.html('<div id="rapor-paylasimi" class="esik-anchor"></div>')
    render_report_email(selected_company, selected_company_shap_df, probability_layer, metadata['model_name'])

    recommended_action = (
        selected_company["RECOMMENDED_ACTION"]
    )

    if selected_company["RISK_GROUP"] == "Kritik - İlk %5":
        st.error(
            f"Önerilen aksiyon: {recommended_action}"
        )

    elif selected_company["RISK_GROUP"] == "Yüksek - %5 ile %10":
        st.warning(
            f"Önerilen aksiyon: {recommended_action}"
        )

    elif selected_company["RISK_GROUP"] == "Orta - %10 ile %20":
        st.info(
            f"Önerilen aksiyon: {recommended_action}"
        )

    else:
        st.success(
            f"Önerilen aksiyon: {recommended_action}"
        )

    chart_column, summary_column = st.columns(
        [3, 2]
    )

    with chart_column:
        st.subheader("Şirket Bazlı SHAP Açıklaması")

        st.altair_chart(
            create_company_shap_chart(
                selected_company_shap_df
            ),
            width="stretch"
        )

        st.caption(
            "Pozitif SHAP değerleri model skorunu "
            "iflas riski yönünde, negatif değerler "
            "ise riskin azalması yönünde etkiler."
        )

    with summary_column:
        st.subheader("Otomatik Finansal Özet")

        risk_increasing_factors = (
            selected_company_shap_df
            .query("SHAP_VALUE > 0")
            .sort_values(
                "SHAP_VALUE",
                ascending=False
            )
            .head(3)
        )

        risk_reducing_factors = (
            selected_company_shap_df
            .query("SHAP_VALUE < 0")
            .sort_values("SHAP_VALUE")
            .head(2)
        )

        if risk_increasing_factors.empty:
            st.success(
                "Bu şirket için belirgin bir pozitif "
                "SHAP risk katkısı bulunmuyor."
            )

        else:
            st.write("**Riski en fazla artıran faktörler**")

            for _, shap_row in risk_increasing_factors.iterrows():
                st.markdown(
                    f"- **{shap_row['DESCRIPTION']}** — "
                    f"SHAP katkısı: +{shap_row['SHAP_VALUE']:.3f}"
                )

        if not risk_reducing_factors.empty:
            st.write("**Riski en fazla azaltan faktörler**")

            for _, shap_row in risk_reducing_factors.iterrows():
                st.markdown(
                    f"- **{shap_row['DESCRIPTION']}** — "
                    f"SHAP katkısı: {shap_row['SHAP_VALUE']:.3f}"
                )

        st.info(
            create_plain_language_summary(
                selected_company_shap_df
            )
        )

        st.subheader("Analist İçin İlk Kontrol Önerileri")

        action_recommendations = create_action_recommendations(
            selected_company_shap_df
        )

        if action_recommendations:
            for recommendation in action_recommendations:
                st.markdown(
                    f"- **{recommendation['group']}** "
                    f"({recommendation['feature']}): "
                    f"{recommendation['action']}"
                )
        else:
            st.caption(
                "Pozitif SHAP katkısı bulunmadığı için öncelikli finansal "
                "aksiyon önerisi üretilemedi."
            )

        quality_report = create_data_quality_report(
            selected_company,
            feature_columns,
            scored_companies_df
        )

        with st.expander("Veri kalitesi kontrolü", expanded=False):
            quality_col_1, quality_col_2 = st.columns(2)

            quality_col_1.metric(
                "Eksik model girdisi",
                quality_report["missing_count"]
            )

            quality_col_2.metric(
                "Referans aralığı dışında",
                quality_report["extreme_count"]
            )

            if quality_report["missing_count"]:
                st.warning(
                    "Eksik değer bulunan göstergeler: "
                    + ", ".join(quality_report["missing_features"][:8])
                )
            else:
                st.success(
                    "Model girdilerinde eksik değer bulunmadı."
                )

            st.caption(
                "Referans aralığı dışındaki değerler otomatik olarak "
                "hatalı kabul edilmez; test portföyünün yaklaşık 1.–99. "
                "yüzdelik aralığına göre inceleme sinyali olarak gösterilir."
            )

        missing_important_features = (
            company_comparison_df.loc[
                company_comparison_df["COMPANY_VALUE"].isna(),
                "DESCRIPTION"
            ]
            .tolist()
        )

        if missing_important_features:
            st.warning(
                "Eksik önemli gösterge sayısı: "
                f"{len(missing_important_features)}. "
                "Eksiklik, analist incelemesinde ayrıca "
                "kontrol edilmelidir."
            )

        st.info(
            "Risk skoru, modelin ham (kalibre edilmemiş) iflas olasılığı "
            "tahminidir; inceleme sıralamasında kullanılır. Kalibre "
            "olasılık ayrı gösterilir. "
            "SHAP değerleri nedensellik göstermez."
        )

    st.subheader("Önemli Finansal Oranların Karşılaştırılması")

    comparison_table_df = (
        company_comparison_df[
            [
                "VARIABLE",
                "DESCRIPTION",
                "COMPANY_VALUE",
                "PORTFOLIO_MEDIAN",
                "SHAP_IMPORTANCE",
                "PORTFOLIO_COMPARISON"
            ]
        ]
        .rename(
            columns={
                "VARIABLE": "Değişken",
                "DESCRIPTION": "Finansal Açıklama",
                "COMPANY_VALUE": "Şirket Değeri",
                "PORTFOLIO_MEDIAN": "Portföy Medyanı",
                "SHAP_IMPORTANCE": "Global SHAP Önem Payı",
                "PORTFOLIO_COMPARISON": "Portföy Karşılaştırması"
            }
        )
    )

    st.dataframe(
        comparison_table_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Şirket Değeri": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Portföy Medyanı": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Global SHAP Önem Payı": st.column_config.NumberColumn(
                format="%.1f%%"
            )
        }
    )

    st.html('<div id="senaryo-analizi" class="esik-anchor"></div>')
    st.subheader("Sınırlı What-if Senaryo Analizi")

    st.write(
        "Model skoruna en fazla katkı sağlayan ve verisi mevcut olan "
        "en fazla üç finansal göstergeyi değiştirerek, tahmin edilen "
        "öncelik skorunun nasıl değiştiğini inceleyin."
    )

    scenario_candidates = observed_scenario_candidates(
        selected_company_shap_df, selected_company, limit=3
    )
    st.caption("Medyanla doldurulan oranlar senaryo seçimine alınmaz. Bu analiz gözlenen girdilere modelin duyarlılığını gösterir; muhasebe tutarlılığını veya nedensel etkiyi garanti etmez.")

    if scenario_candidates.empty:
        st.warning(
            "Senaryo analizi için yeterli sayısal gösterge bulunamadı."
        )
    else:
        scenario_mode = st.radio(
            "Senaryo modu",
            [
                "Tek değişken analizi",
                "Çoklu senaryo"
            ],
            horizontal=True,
            help=(
                "Tek değişken analizinde yalnızca bir gösterge değişir. "
                "Çoklu senaryoda en etkili üç gösterge birlikte değiştirilebilir."
            )
        )

        if scenario_mode == "Tek değişken analizi":
            scenario_candidates = scenario_candidates.head(1)

        st.caption(
            "Değer aralığı, test portföyünün yaklaşık 1.–99. yüzdelik "
            "aralığıyla sınırlandırılır. Mevcut şirket değeri bu aralığın "
            "dışındaysa korunur; böylece analiz tamamen kopuk senaryolara "
            "izin vermez."
        )

        with st.form("what_if_form"):
            scenario_values = {}
            scenario_columns = st.columns(len(scenario_candidates))

            for column, (_, shap_row) in zip(
                scenario_columns,
                scenario_candidates.iterrows()
            ):
                feature = shap_row["VARIABLE"]
                current_value = float(selected_company[feature])
                reference_values = pd.to_numeric(
                    scored_companies_df[feature],
                    errors="coerce"
                ).dropna()

                if len(reference_values) >= 5:
                    lower_bound = float(reference_values.quantile(0.01))
                    upper_bound = float(reference_values.quantile(0.99))
                else:
                    lower_bound = current_value - abs(current_value or 1)
                    upper_bound = current_value + abs(current_value or 1)

                step = max(
                    abs(upper_bound - lower_bound) / 1000,
                    0.0001
                )

                control_min = min(
                    lower_bound,
                    current_value
                )
                control_max = max(
                    upper_bound,
                    current_value
                )

                with column:
                    scenario_values[feature] = st.number_input(
                        shap_row["DESCRIPTION"],
                        value=current_value,
                        min_value=control_min,
                        max_value=control_max,
                        step=step,
                        format="%.4f",
                        key=f"scenario_{selected_company_id}_{feature}"
                    )

                    st.caption(
                        f"Mevcut: {current_value:.4f} · "
                        f"SHAP: {shap_row['SHAP_VALUE']:+.3f}\n\n"
                        f"Referans aralığı: {lower_bound:.4f} – "
                        f"{upper_bound:.4f}"
                    )

            scenario_submitted = st.form_submit_button(
                "Senaryoyu hesapla",
                type="primary"
            )

        if scenario_submitted:
            scenario_input = selected_company[feature_columns].to_dict()
            scenario_input.update(scenario_values)

            try:
                scenario_score = predict_model_score(
                    model_package,
                    scenario_input,
                    feature_columns
                )
                current_score = float(selected_company["RISK_SCORE"])
                score_change = (scenario_score - current_score) * 100

                result_col_1, result_col_2, result_col_3 = st.columns(3)

                result_col_1.metric(
                    "Mevcut model skoru",
                    f"{current_score * 100:.1f}"
                )

                result_col_2.metric(
                    "Senaryo model skoru",
                    f"{scenario_score * 100:.1f}"
                )

                result_col_3.metric(
                    "Skor değişimi",
                    f"{score_change:+.1f} puan"
                )

                comparison_chart_df = pd.DataFrame({
                    "Durum": ["Mevcut durum", "Senaryo durumu"],
                    "Model Skoru": [
                        current_score * 100,
                        scenario_score * 100
                    ]
                })

                st.altair_chart(
                    alt.Chart(comparison_chart_df)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        x=alt.X("Durum:N", title=None),
                        y=alt.Y(
                            "Model Skoru:Q",
                            title="Model skoru"
                        ),
                        color=alt.Color(
                            "Durum:N",
                            scale=alt.Scale(
                                range=["#B6A0FF", "#FF725C"]
                            ),
                            legend=None
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Durum:N",
                                title="Durum"
                            ),
                            alt.Tooltip(
                                "Model Skoru:Q",
                                title="Skor",
                                format=".1f"
                            )
                        ]
                    )
                    .properties(
                        title="Mevcut ve senaryo skoru",
                        height=260
                    ),
                    width="stretch"
                )

                sensitivity_feature = scenario_candidates.iloc[0]["VARIABLE"]
                sensitivity_description = scenario_candidates.iloc[0]["DESCRIPTION"]
                sensitivity_reference = pd.to_numeric(
                    scored_companies_df[sensitivity_feature],
                    errors="coerce"
                ).dropna()

                sensitivity_values = [
                    float(sensitivity_reference.quantile(quantile))
                    for quantile in [0.10, 0.25, 0.50, 0.75, 0.90]
                ]

                sensitivity_rows = []
                for reference_quantile, sensitivity_value in zip([10, 25, 50, 75, 90], sensitivity_values):
                    sensitivity_input = selected_company[feature_columns].to_dict()
                    sensitivity_input[sensitivity_feature] = sensitivity_value
                    sensitivity_score = predict_model_score(
                        model_package,
                        sensitivity_input,
                        feature_columns
                    )
                    sensitivity_rows.append({
                        "Referans": f"Gösterge dağılımının %{reference_quantile} yüzdeliği",
                        "Gösterge değeri": sensitivity_value,
                        "Model skoru": sensitivity_score * 100
                    })

                sensitivity_df = pd.DataFrame(sensitivity_rows)
                st.altair_chart(
                    alt.Chart(sensitivity_df)
                    .mark_line(point=True, color="#B6A0FF")
                    .encode(
                        x=alt.X(
                            "Gösterge değeri:Q",
                            title=sensitivity_description
                        ),
                        y=alt.Y(
                            "Model skoru:Q",
                            title="Model skoru"
                        ),
                        tooltip=[
                            alt.Tooltip("Referans:N"),
                            alt.Tooltip(
                                "Gösterge değeri:Q",
                                format=".4f"
                            ),
                            alt.Tooltip(
                                "Model skoru:Q",
                                format=".1f"
                            )
                        ]
                    )
                    .properties(
                        title=(
                            "Duyarlılık eğrisi: "
                            f"{sensitivity_description}"
                        ),
                        height=280
                    ),
                    width="stretch"
                )

                st.caption(
                    "Duyarlılık eğrisi, diğer göstergeler sabitken seçilen "
                    "tek göstergenin test portföyündeki beş referans değerinde "
                    "model skorunun nasıl değiştiğini gösterir. Bu beş değer, ilgili finansal oranın "
                    "%10, %25, %50 (medyan), %75 ve %90 yüzdelikleridir; iflas olasılıkları değildir. "
                    "Bu bir model duyarlılığı incelemesidir; gerçek hayatta nedensel sonuç garantisi vermez."
                )

                changed_features = []
                for feature, new_value in scenario_values.items():
                    feature_row = scenario_candidates.loc[
                        scenario_candidates["VARIABLE"] == feature
                    ].iloc[0]
                    changed_features.append({
                        "feature": feature,
                        "description": feature_row["DESCRIPTION"],
                        "current_value": round(
                            float(selected_company[feature]),
                            6
                        ),
                        "scenario_value": round(float(new_value), 6)
                    })

                st.session_state["latest_what_if"] = {
                    "company_id": str(selected_company_id),
                    "current_score": round(current_score, 6),
                    "scenario_score": round(scenario_score, 6),
                    "score_change_points": round(score_change, 3),
                    "changed_features": changed_features
                }

                if score_change < 0:
                    st.success(
                        "Bu senaryoda model skoru azaldı. Bu sonuç, "
                        "girdi değişikliklerinin model tahmini üzerindeki "
                        "etkisini gösterir; nedensellik veya garanti anlamına gelmez."
                    )
                elif score_change > 0:
                    st.warning(
                        "Bu senaryoda model skoru yükseldi. Bu sonuç, "
                        "girdi değişikliklerinin model tahmini üzerindeki "
                        "etkisini gösterir; nedensellik veya garanti anlamına gelmez."
                    )
                else:
                    st.info(
                        "Bu senaryoda model skorunda anlamlı bir değişiklik olmadı."
                    )

            except Exception as error:
                st.error(
                    "Senaryo hesaplanırken bir hata oluştu. "
                    "Girdi değerlerinin modelin beklediği finansal oranlarla "
                    "uyumlu olduğundan emin olun."
                )
                st.caption(f"Teknik ayrıntı: {error}")

    st.caption(
        "Karşılaştırma referansı test portföyünün "
        "medyanıdır; gerçek iflas etiketi hesaplamaya "
        "dahil edilmez. Attr55 mutlak çalışma sermayesi "
        "olduğu için şirket büyüklüğünden etkilenebilir."
    )

    show_company_actual_status = st.checkbox(
        "Bu şirketin gerçek test sonucunu göster",
        value=False,
        help=(
            "Gerçek kullanım anında sonuç bilinmez. "
            "Bu alan yalnızca model değerlendirmesi "
            "ve proje sunumu için kullanılmalıdır."
        )
    )

    if show_company_actual_status:
        if selected_company["ACTUAL_BANKRUPT"] == 1:
            st.error(
                "Gerçek test sonucu: İflas Etti"
            )
        else:
            st.success(
                "Gerçek test sonucu: İflas Etmedi"
            )


##################################################
# YENİ VERİ ANALİZİ
##################################################

elif selected_page == "Yeni Veri Analizi":

    page_header(
        "Yeni Veri Analizi",
        "CSV dosyanızı yükleyin. Kayıtlı model, finansal oranlardan inceleme sırası üretsin.",
        section="03 / PORTFÖY YÜKLEME",
    )

    uploaded_file = st.file_uploader(
        "Şirket verisi CSV dosyası",
        type=["csv"],
        help=(
            "CSV dosyasında modelin beklediği Attr1–Attr64 sütunları "
            "bulunmalıdır. COMPANY_ID sütunu isteğe bağlıdır."
        )
    )

    if uploaded_file is None:
        st.info(
            "Başlamak için Attr1–Attr64 sütunlarını içeren bir CSV dosyası yükleyin."
        )
    else:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except (ValueError, UnicodeError) as error:
            st.error(f"CSV okunamadı: {error}")
            st.stop()
        if uploaded_df.empty:
            st.error("CSV dosyasında en az bir şirket kaydı bulunmalıdır.")
            st.stop()
        missing_columns = [
            feature
            for feature in feature_columns
            if feature not in uploaded_df.columns
        ]

        if missing_columns:
            st.error(
                f"Dosyada {len(missing_columns)} gerekli model girdisi eksik."
            )
            st.write(
                "Eksik sütunlar: "
                + ", ".join(missing_columns[:20])
            )
            st.stop()

        uploaded_model_df = uploaded_df[feature_columns].copy()
        original_missing_count = int(uploaded_model_df.isna().sum().sum())

        for feature in feature_columns:
            uploaded_model_df[feature] = pd.to_numeric(
                uploaded_model_df[feature],
                errors="coerce"
            )

        converted_missing_count = int(
            uploaded_model_df.isna().sum().sum()
        )
        newly_invalid_count = max(
            converted_missing_count - original_missing_count,
            0
        )

        if uploaded_model_df.isin([float("inf"), float("-inf")]).any().any():
            st.error("CSV sonsuz değer içeriyor. İlgili finansal oranları düzeltip yeniden yükleyin.")
            st.stop()
        if uploaded_model_df.isna().all(axis=1).any():
            st.error("Bazı şirketlerde 64 göstergenin tamamı eksik veya geçersiz. Bu kayıtları düzeltin.")
            st.stop()

        if newly_invalid_count:
            st.warning(
                f"Sayısal olmayan {newly_invalid_count} hücre eksik değer olarak "
                "işaretlendi. Pipeline eksik değerleri kendi imputasyon adımıyla "
                "işleyecek; ancak veri kaynağını kontrol etmeniz önerilir."
            )

        uploaded_scores = model_package["pipeline"].predict_proba(
            uploaded_model_df
        )[:, 1]

        scored_upload_df = uploaded_df.copy()
        scored_upload_df["RISK_SCORE"] = uploaded_scores
        row_count = len(scored_upload_df)
        upload_priority = priority_fields(uploaded_scores)
        for column in upload_priority:
            scored_upload_df[column] = upload_priority[column].to_numpy()
        scored_upload_df["RISK_PERCENTILE"] = scored_upload_df["RISK_PERCENTILE"].round(1)

        scored_upload_df["RISK_SCORE"] = (
            scored_upload_df["RISK_SCORE"] * 100
        ).round(2)

        if "COMPANY_ID" not in scored_upload_df.columns:
            scored_upload_df["COMPANY_ID"] = [
                f"UPLOADED_{index + 1}"
                for index in range(len(scored_upload_df))
            ]

        uploaded_top_10 = (
            scored_upload_df
            .sort_values("RISK_RANK")
            .head(10)[
                [
                    "COMPANY_ID",
                    "RISK_RANK",
                    "RISK_SCORE",
                    "RISK_GROUP"
                ]
            ]
            .copy()
        )
        uploaded_top_10["RISK_GROUP"] = (
            uploaded_top_10["RISK_GROUP"].astype(str)
        )
        st.session_state["uploaded_portfolio_context"] = {
            "risk_score_scale": "0–100 model skoru; kalibre edilmiş olasılık değil",
            "row_count": int(row_count),
            "missing_value_count": int(converted_missing_count),
            "top_10": uploaded_top_10.to_dict(orient="records")
        }

        display_columns = []
        if "COMPANY_ID" in scored_upload_df.columns:
            display_columns.append("COMPANY_ID")

        display_columns.extend([
            "RISK_RANK",
            "RISK_SCORE",
            "RISK_PERCENTILE",
            "RISK_GROUP"
        ])

        quality_col_1, quality_col_2, quality_col_3 = st.columns(3)
        quality_col_1.metric("Analiz edilen satır", row_count)
        quality_col_2.metric("Eksik model girdisi", converted_missing_count)
        quality_col_3.metric("Model girdisi", len(feature_columns))

        st.subheader("Risk Öncelik Listesi")
        st.dataframe(
            scored_upload_df[
                display_columns
            ].sort_values("RISK_RANK"),
            width="stretch",
            hide_index=True
        )

        download_df = scored_upload_df.sort_values("RISK_RANK")
        st.download_button(
            "Sonuçları CSV olarak indir",
            data=download_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="esik_yeni_veri_risk_sonuclari.csv",
            mime="text/csv"
        )

        st.info(
            "Yüklenen kayıtlar üzerinde etiketli doğrulama yapılmadığı için bu ekran "
            "doğruluk veya recall hesaplamaz; yalnızca modelin ürettiği "
            "öncelik skorlarını ve sıralamayı gösterir."
        )


##################################################
# EŞİK AI - N8N FİNANSAL RİSK ASİSTANI
##################################################

elif selected_page == "Eşik AI":

    page_header(
        "Eşik AI",
        "Şirketin rakamlarını sorun, dayanaklarını görün. Sorunuzu yazın veya mikrofonla kaydedin.",
        section="04 / FİNANSAL RİSK ASİSTANI", illustration="voice",
    )

    def _runtime_setting(name):
        value = os.getenv(name, "").strip()
        try:
            if not value and name in st.secrets:
                value = str(st.secrets[name]).strip()
        except Exception:
            pass
        return value

    n8n_webhook_url = _runtime_setting("N8N_WEBHOOK_URL")
    n8n_webhook_token = _runtime_setting("N8N_WEBHOOK_TOKEN")
    transcription_url = _runtime_setting("N8N_TRANSCRIBE_WEBHOOK_URL")
    chat_lookup = load_company_query(analysis_version, "", "")
    pending_company = st.session_state.pop("esik_ai_pending_company", None)
    if pending_company is not None:
        st.session_state["ai_company_select"] = pending_company
    with st.expander("Kodla doğrudan şirket sorgula", expanded=False):
        st.caption(
            "Şirket kodunu sorunun içine yazın. Kayıt özeti AI bağlantısı olmadan da çalışır. "
            "Kodlar tarihsel demo portföyüne aittir."
        )
        with st.form("single_company_query"):
            single_question = st.text_input(
                "Şirket kodu ve sorunuz",
                value="ESIK-05817 için en önemli 3 risk nedir?",
                max_chars=2000,
                key="single_company_question",
            )
            add_ai = st.checkbox("AI yorumu da ekle", value=False,
                                 disabled=not bool(n8n_webhook_url))
            lookup_submitted = st.form_submit_button("Şirketi sorgula", type="primary")
        if lookup_submitted:
            with st.spinner("Şirket kaydı okunuyor..."):
                lookup = load_company_query(
                    analysis_version,
                    n8n_webhook_url if add_ai else "",
                    n8n_webhook_token if add_ai else "",
                )
                st.session_state["single_company_result"] = lookup.query(single_question)
        if "single_company_result" in st.session_state:
            single_result = st.session_state["single_company_result"]
            st.text(single_result["answer"])
            if single_result.get("company_id"):
                st.caption("Yanıtlanan kayıt: " + single_result["company_id"])
    uploaded_context = st.session_state.get("uploaded_portfolio_context")

    context_options = ["Şirket analizi"]
    if uploaded_context:
        context_options.append("Yüklenen CSV özeti")

    analysis_context = (
        st.radio("Asistan bağlamı", context_options, horizontal=True)
        if uploaded_context else "Şirket analizi"
    )

    if analysis_context == "Şirket analizi":
        ai_company_df = (
            scored_companies_df
            .sort_values("RISK_RANK")
            .reset_index(drop=True)
        )
        ai_labels = {
            row["COMPANY_ID"]: (
                f"{row['COMPANY_ID']} · "
                f"Risk sırası {int(row['RISK_RANK'])} · "
                f"{row['RISK_GROUP']}"
            )
            for _, row in ai_company_df.iterrows()
        }
        ai_company_id = st.selectbox(
            "Analiz edilecek şirket",
            options=ai_company_df["COMPANY_ID"].tolist(),
            format_func=lambda company_id: ai_labels[company_id],
            key="ai_company_select"
        )
        ai_company = ai_company_df.loc[
            ai_company_df["COMPANY_ID"] == ai_company_id
        ].iloc[0]
        ai_shap_df = shap_local_df.loc[
            shap_local_df["COMPANY_ID"] == ai_company_id
        ].copy()
        increasing_df = (
            ai_shap_df.loc[ai_shap_df["SHAP_VALUE"] > 0]
            .sort_values("ABS_SHAP_VALUE", ascending=False)
            .head(5)
        )
        reducing_df = (
            ai_shap_df.loc[ai_shap_df["SHAP_VALUE"] < 0]
            .sort_values("ABS_SHAP_VALUE", ascending=False)
            .head(5)
        )

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric(
            "Risk Skoru",
            f"{ai_company['RISK_SCORE'] * 100:.1f}/100"
        )
        metric_2.metric("Risk Grubu", ai_company["RISK_GROUP"])
        metric_3.metric(
            "İlk %10 İnceleme",
            "Evet" if int(ai_company["ANALYST_REVIEW"]) == 1 else "Hayır"
        )

        latest_what_if = st.session_state.get("latest_what_if")
        if (
            latest_what_if
            and latest_what_if.get("company_id") != str(ai_company_id)
        ):
            latest_what_if = None

        chatbot_context = {
            "context_type": "company",
            "company": chat_lookup.company_context(str(ai_company_id), latest_what_if),
            "portfolio": None,
        }
        if chatbot_context['company']['bankruptcy_probability']:
            st.info('Bir yıllık iflas olasılığı tahmini: '+percent(chatbot_context['company']['bankruptcy_probability']['estimate'])+' · Keşifsel kalibrasyon, tarihsel veri.')
        context_key = f"{CHAT_CONTEXT_VERSION}:{artifact_version}:company:{ai_company_id}"
        quick_questions = [
            "Bu şirket neden riskli görünüyor?",
            "En önemli 3 risk faktörünü açıkla.",
            "Bir yıllık iflas olasılığı tahmini kaç?",
            "Son What-if sonucunu yorumla."
        ]
    else:
        top_10_df = pd.DataFrame(uploaded_context["top_10"])
        st.metric("Yüklenen şirket", uploaded_context["row_count"])
        st.dataframe(top_10_df, width="stretch", hide_index=True)
        chatbot_context = {
            "context_type": "uploaded_portfolio",
            "company": None,
            "portfolio": uploaded_context
        }
        portfolio_fingerprint = hashlib.sha256(json.dumps(uploaded_context, sort_keys=True).encode()).hexdigest()
        context_key = f"{CHAT_CONTEXT_VERSION}:{artifact_version}:uploaded_portfolio:{portfolio_fingerprint}"
        quick_questions = [
            "En riskli 10 şirketi kısaca özetle.",
            "Kritik gruptaki şirketleri belirt.",
            "Analist inceleme sırasını açıkla.",
            "Bu listenin sınırlılıkları nelerdir?"
        ]

    if "esik_ai_session_id" not in st.session_state:
        st.session_state["esik_ai_session_id"] = str(uuid.uuid4())

    if st.session_state.get("esik_ai_context_key") != context_key:
        st.session_state["esik_ai_context_key"] = context_key
        st.session_state["esik_ai_messages"] = []

    if "esik_ai_messages" not in st.session_state:
        st.session_state["esik_ai_messages"] = []

    status_col, clear_col = st.columns([4, 1])
    with status_col:
        if n8n_webhook_url:
            st.caption("AI bağlantısı yapılandırıldı")
        else:
            st.warning(
                "N8N_WEBHOOK_URL tanımlanmadı. n8n workflow'unu içe aktarıp "
                "Webhook URL'sini secrets.toml dosyasına ekleyin."
            )
    with clear_col:
        if st.button("Sohbeti temizle"):
            st.session_state["esik_ai_messages"] = []
            st.rerun()

    st.caption("Hazır sorular")
    quick_columns = st.columns(2)
    quick_question = None
    for index, question in enumerate(quick_questions):
        with quick_columns[index % 2]:
            if st.button(
                question,
                key=f"quick_{context_key}_{index}",
                disabled=not bool(n8n_webhook_url),
                width="stretch"
            ):
                quick_question = question

    for message in st.session_state["esik_ai_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = voice_chat_input(
        context_key=context_key,
        transcription_url=transcription_url,
        webhook_token=n8n_webhook_token,
        disabled=not bool(n8n_webhook_url),
    )
    submitted_question = quick_question or typed_question or st.session_state.pop("esik_ai_pending_question", None)

    # Resolve entity identity before showing an answer or sending anything to n8n.
    if submitted_question and analysis_context == "Şirket analizi":
        try:
            target_company = chat_lookup.resolve_company_id(submitted_question, str(ai_company_id))
            if target_company != str(ai_company_id):
                st.session_state["esik_ai_pending_company"] = target_company
                st.session_state["esik_ai_pending_question"] = submitted_question
                st.rerun()
        except CompanyQueryError as error:
            st.error(str(error))
            submitted_question = None

    if submitted_question:
        st.session_state["esik_ai_messages"].append({
            "role": "user",
            "content": submitted_question
        })
        with st.chat_message("user"):
            st.markdown(submitted_question)

        request_payload = {
            "schema_version": "1.0",
            "session_id": st.session_state["esik_ai_session_id"],
            "question": submitted_question,
            "context": chatbot_context,
            "conversation_history": st.session_state[
                "esik_ai_messages"
            ][-7:-1],
            "model_validation": chat_lookup.model_validation()
        }

        try:
            with st.chat_message("assistant"):
                assistant_answer = probability_answer(submitted_question, chatbot_context.get('company'))
                if assistant_answer is None:
                    with st.spinner("n8n workflow'u yanıt hazırlıyor..."):
                        assistant_answer = send_chat_message(
                            webhook_url=n8n_webhook_url,
                            payload=request_payload,
                            webhook_token=n8n_webhook_token
                        )
                st.markdown(assistant_answer)

            st.session_state["esik_ai_messages"].append({
                "role": "assistant",
                "content": assistant_answer
            })
        except N8nClientError as error:
            st.error(str(error))
            st.caption(
                "n8n editöründe workflow'un aktif/test modunda olduğunu ve "
                "doğru test veya production webhook URL'sini kullandığınızı "
                "kontrol edin."
            )

    st.info(
        "Eşik AI yalnızca uygulamanın gönderdiği model ve SHAP çıktılarıyla "
        "konuşur. Model skorunu yeniden hesaplamaz; kredi kararı, yatırım "
        "tavsiyesi veya nedensellik iddiası üretmemelidir. Nihai değerlendirme "
        "insan analiste aittir."
    )


##################################################
# MODEL AÇIKLAMASI
##################################################

elif selected_page == "Model Açıklaması":

    page_header(
        "Model Açıklaması",
        "Modelin sıralamada hangi finansal göstergelerden yararlandığını SHAP ile inceleyin.",
        section="05 / AÇIKLANABİLİRLİK",
    )

    top_global_feature = shap_global_df.iloc[0]

    summary_col_1, summary_col_2, summary_col_3 = (
        st.columns(3)
    )

    summary_col_1.metric(
        "Açıklanan Şirket",
        f"{scored_companies_df['COMPANY_ID'].nunique():,}"
    )

    summary_col_2.metric(
        "Açıklanan Finansal Oran",
        len(shap_global_df)
    )

    summary_col_3.metric(
        "En Etkili Gösterge",
        top_global_feature["VARIABLE"]
    )

    st.subheader("Global SHAP Önem Sıralaması")

    st.altair_chart(
        create_global_shap_chart(
            shap_global_df
        ),
        width="stretch"
    )

    global_display_df = (
        shap_global_df
        .head(20)
        .copy()
    )

    global_display_df["SHAP_IMPORTANCE_RATIO"] = (
        global_display_df["SHAP_IMPORTANCE_RATIO"]
        * 100
    )

    global_display_df = global_display_df[
        [
            "SHAP_RANK",
            "VARIABLE",
            "DESCRIPTION",
            "MEAN_ABS_SHAP",
            "SHAP_IMPORTANCE_RATIO"
        ]
    ].rename(
        columns={
            "SHAP_RANK": "Sıra",
            "VARIABLE": "Değişken",
            "DESCRIPTION": "Finansal Açıklama",
            "MEAN_ABS_SHAP": "Ortalama Mutlak SHAP",
            "SHAP_IMPORTANCE_RATIO": "Önem Payı"
        }
    )

    st.dataframe(
        global_display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Ortalama Mutlak SHAP": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Önem Payı": st.column_config.NumberColumn(
                format="%.1f%%"
            )
        }
    )

    st.info(
        "Global SHAP sıralaması, bir değişkenin tüm "
        "şirketlerdeki ortalama mutlak katkısını gösterir. "
        "Yüksek önem, nedensellik veya tek başına risk "
        "anlamına gelmez. Etkinin yönü şirket bazında "
        "Şirket Detayı sayfasında incelenmelidir."
    )


##################################################
# MODEL DOĞRULAMASI
##################################################

elif selected_page == "Model Doğrulaması":

    page_header("Model Doğrulaması", "Kalibrasyon, hata dağılımı ve eşik duyarlılığı.",
                section="06 / PERFORMANS VE SINIRLAR")
    render_calibration_validation(probability_layer)

    st.write(
        "Bu sayfa, model skorunun farklı operasyonel eşiklerde "
        "şirketleri nasıl sınıflandırdığını gösterir. Eşik, kalibre "
        "edilmiş olasılık değil; inceleme önceliği için kullanılan bir "
        "model skoru olarak yorumlanmalıdır."
    )

    validation_df = scored_companies_df[
        [
            "ACTUAL_BANKRUPT",
            "RISK_SCORE"
        ]
    ].dropna().copy()

    y_true = validation_df["ACTUAL_BANKRUPT"].astype(int)
    y_score = validation_df["RISK_SCORE"].astype(float)

    threshold = st.slider(
        "Operasyonel sınıflandırma eşiği",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help=(
            "Skoru bu eşikten yüksek şirketler yüksek riskli kabul edilir. "
            "Bu eşik, model skorunu yönetimsel karar kuralına dönüştürür."
        )
    )

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    threshold_recall = tp / (tp + fn) if (tp + fn) else 0.0
    threshold_precision = tp / (tp + fp) if (tp + fp) else 0.0
    threshold_specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

    metric_col_1.metric(
        "Recall",
        f"%{threshold_recall * 100:.1f}"
    )
    metric_col_2.metric(
        "Precision",
        f"%{threshold_precision * 100:.1f}"
    )
    metric_col_3.metric(
        "Specificity",
        f"%{threshold_specificity * 100:.1f}"
    )
    metric_col_4.metric(
        "İnceleme adedi",
        int(y_pred.sum())
    )

    left_validation, right_validation = st.columns(2)

    with left_validation:
        st.subheader("Confusion Matrix")

        confusion_df = pd.DataFrame(
            [
                {
                    "Gerçek": "İflas Etmedi",
                    "Tahmin": "İflas Etmedi",
                    "Şirket Sayısı": int(tn)
                },
                {
                    "Gerçek": "İflas Etmedi",
                    "Tahmin": "İflas Etti",
                    "Şirket Sayısı": int(fp)
                },
                {
                    "Gerçek": "İflas Etti",
                    "Tahmin": "İflas Etmedi",
                    "Şirket Sayısı": int(fn)
                },
                {
                    "Gerçek": "İflas Etti",
                    "Tahmin": "İflas Etti",
                    "Şirket Sayısı": int(tp)
                }
            ]
        )

        st.dataframe(
            confusion_df,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Yanlış negatif sayısı, gerçekte iflas eden ancak bu eşikte "
            "yüksek riskli yakalanamayan şirketleri gösterir."
        )

    with right_validation:
        st.subheader("Eşik Yorumu")

        if threshold_recall >= 0.80:
            st.success(
                "Bu eşikte iflas eden şirketlerin büyük bölümü yakalanıyor."
            )
        elif threshold_recall >= 0.60:
            st.warning(
                "Bu eşikte yakalama oranı orta düzeyde; yanlış negatifler "
                "ayrıca incelenmeli."
            )
        else:
            st.error(
                "Bu eşikte birçok iflas vakası kaçırılıyor."
            )

        st.write(
            f"Skoru **{threshold:.2f}** üzerinde olan "
            f"**{int(y_pred.sum())}** şirket yüksek riskli kabul ediliyor."
        )

        st.info(
            "Eşik düşürüldüğünde daha fazla şirket incelemeye alınır ve "
            "iflas vakalarını yakalama ihtimali artabilir; ancak analist "
            "yükü ve yanlış alarm sayısı da artabilir."
        )

    st.subheader("ROC ve Precision-Recall Eğrileri")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision_values, recall_values, _ = precision_recall_curve(
        y_true,
        y_score
    )

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall_values, precision_values)

    roc_df = pd.DataFrame({
        "Yanlış Pozitif Oranı": fpr,
        "Doğru Pozitif Oranı": tpr
    })

    pr_df = pd.DataFrame({
        "Recall": recall_values,
        "Precision": precision_values
    })

    curve_col_1, curve_col_2 = st.columns(2)

    with curve_col_1:
        st.altair_chart(
            alt.Chart(roc_df)
            .mark_line(color="#FF725C")
            .encode(
                x=alt.X("Yanlış Pozitif Oranı:Q"),
                y=alt.Y("Doğru Pozitif Oranı:Q"),
                tooltip=[
                    "Yanlış Pozitif Oranı",
                    "Doğru Pozitif Oranı"
                ]
            )
            .properties(
                title=f"ROC-AUC: {roc_auc:.3f}",
                height=300
            ),
            width="stretch"
        )

    with curve_col_2:
        st.altair_chart(
            alt.Chart(pr_df)
            .mark_line(color="#B6A0FF")
            .encode(
                x=alt.X("Recall:Q"),
                y=alt.Y("Precision:Q"),
                tooltip=["Recall", "Precision"]
            )
            .properties(
                title=f"PR-AUC (trapez): {pr_auc:.3f}",
                height=300
            ),
            width="stretch"
        )

    st.metric(
        "Brier Score",
        f"{brier_score_loss(y_true, y_score):.3f}",
        help=(
            "Tahmin ile gerçekleşen etiket arasındaki karesel hatayı ölçer. "
            "Kalibrasyon ve ayrım gücünü birlikte değerlendirir. Bu alttaki değer "
            "ham model skoruna, üstteki karşılaştırma kalibrasyon deneyine aittir."
        )
    )


##################################################
# MODEL KARTI
##################################################

elif selected_page == "Model Kartı":

    page_header("Model Kartı", "Veri kapsamı, seçim protokolü ve kullanım sınırları.",
                section="07 / MODEL KAYDI")

    st.caption("XGBoost karşılaştırma sürümü. Aile tercihi, daha önce incelenmiş beş dış kattaki ortalama Recall@Top10'a dayanır. Bu katlar yeni bağımsız test değildir.")
    nested = metadata.get("nested_validation", {}).get(metadata.get("validation_key", "selected_procedure"))
    if nested:
        st.write(
            f"**Bu model ailesinin dış CV Recall@Top10 ortalaması:** "
            f"%{nested['mean_recall_at_10'] * 100:.1f} · "
            f"**Katlar arası standart sapma:** {nested['std_recall_at_10'] * 100:.1f} yüzde puan"
        )
        st.caption("Her dış katta XGBoost ayarları yalnız o katın eğitim bölümüyle seçildi. Tarihsel testte XGBoost 67/82, LightGBM 70/82 iflas yakaladı. XGBoost her ölçütte üstün değildir.")

    model_col_1, model_col_2 = st.columns(2)

    with model_col_1:
        st.subheader("Model Bilgileri")

        st.write(
            f"**Model:** "
            f"{metadata['model_name']}"
        )

        st.write(
            f"**Tahmin hedefi:** "
            f"{metadata['prediction_target']}"
        )

        st.write(
            f"**Veri seti:** "
            f"{metadata['dataset']}"
        )

        st.write(
            f"**Bağımsız değişken sayısı:** "
            f"{metadata['feature_count']}"
        )

        st.write(
            "**Varsayılan inceleme kapasitesi:** %10"
        )

    with model_col_2:
        st.subheader("Final Performans")

        performance_df = pd.DataFrame({
            "Metrik": [
                "Test ROC-AUC",
                "Test Average Precision (AP)",
                "Standart Recall",
                "Recall@Top10",
                "Precision@Top10",
                "Lift@Top10"
            ],
            "Değer": [
                f"%{metadata['test_roc_auc'] * 100:.1f}",
                f"%{metadata.get('test_average_precision', metadata.get('test_pr_auc')) * 100:.1f}",
                f"%{metadata['test_recall'] * 100:.1f}",
                (
                    f"%{metadata['test_recall_at_top_10'] * 100:.1f}"
                ),
                (
                    f"%{metadata['test_precision_at_top_10'] * 100:.1f}"
                ),
                (
                    f"{metadata['test_lift_at_top_10']:.2f}x"
                )
            ]
        })

        st.dataframe(
            performance_df,
            width="stretch",
            hide_index=True
        )

    st.subheader("Seçilen Model Parametreleri")

    fitted_model = model_package["pipeline"].named_steps["model"]
    if fitted_model.__class__.__name__.startswith("LGBM") and hasattr(fitted_model, "n_estimators_"):
        st.caption(
            f"İstenen boosting turu üst sınırı: {fitted_model.n_estimators}; "
            f"gerçekleşen tur/ağaç sayısı: {fitted_model.n_estimators_}. "
            "Yeni geçerli bölünme üretilemediğinde LightGBM daha az ağaç oluşturabilir."
        )

    if metadata.get("feature_dictionary_note"):
        st.caption(metadata["feature_dictionary_note"])

    parameter_df = pd.DataFrame(
        metadata["best_parameters"].items(),
        columns=[
            "Parametre",
            "Seçilen Değer"
        ]
    )

    st.dataframe(
        parameter_df,
        width="stretch",
        hide_index=True
    )

    st.info(
        metadata["score_definition"]
    )

    st.warning(
        "Model, tarihsel Polonya şirket verileriyle "
        "geliştirilmiş bir karar destek çalışmasıdır. "
        "Gerçek kredi veya yatırım kararlarında doğrudan "
        "ve tek başına kullanılmamalıdır."
    )
