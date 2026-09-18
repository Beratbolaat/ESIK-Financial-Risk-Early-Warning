################################################################
# Eşik - Kurumsal Finansal Risk Erken Uyarı Sistemi
################################################################


# 1. GEREKLİLİKLER

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import arff
from sklearn.exceptions import ConvergenceWarning


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)


######################################
# VERİNİN YÜKLENMESİ
######################################

data, meta = arff.loadarff("datasets/5year.arff")

df = pd.DataFrame(data)

print("Veri setinin ilk 5 gözlemi:")
print(df.head())

print("\nVeri setinin boyutu:")
print(df.shape)



######################################
# HEDEF DEĞİŞKENİN DÜZENLENMESİ
######################################

df["class"] = df["class"].apply(lambda x: x.decode("utf-8"))
df["class"] = df["class"].astype(int)

df.rename(columns={"class": "BANKRUPT"}, inplace=True)

print("\nHedef değişkenin sınıf dağılımı:")
print(df["BANKRUPT"].value_counts())

print("\nHedef değişkenin yüzdelik dağılımı:")
print(df["BANKRUPT"].value_counts(normalize=True) * 100)

print("\nHedef değişkenin veri tipi:")
print(df["BANKRUPT"].dtype)


######################################
# VERİ SETİNİN GENEL RESMİ
######################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)



######################################
# DEĞİŞKEN TÜRLERİNİN YAKALANMASI
######################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, sayısal ve kategorik fakat kardinal
    değişkenlerin isimlerini verir.
    """

    cat_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtype == "O"
    ]

    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th
        and dataframe[col].dtype != "O"
    ]

    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th
        and dataframe[col].dtype == "O"
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [
        col for col in cat_cols
        if col not in cat_but_car
    ]

    num_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtype != "O"
    ]

    num_cols = [
        col for col in num_cols
        if col not in cat_cols
    ]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


features_df = df.drop("BANKRUPT", axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(features_df)

print("\nKategorik değişkenler:")
print(cat_cols)

print("\nSayısal değişkenler:")
print(num_cols)

print("\nKategorik fakat kardinal değişkenler:")
print(cat_but_car)



######################################
# EKSİK DEĞER ANALİZİ
######################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [
        col for col in dataframe.columns
        if dataframe[col].isnull().sum() > 0
    ]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(
        ascending=False
    )

    ratio = (
        dataframe[na_columns].isnull().sum()
        / dataframe.shape[0] * 100
    ).sort_values(ascending=False)

    missing_df = pd.concat(
        [n_miss, np.round(ratio, 2)],
        axis=1,
        keys=["n_miss", "ratio"]
    )

    print(missing_df)

    if na_name:
        return na_columns


na_cols = missing_values_table(df, na_name=True)

print("\nEksik değer bulunan değişken sayısı:")
print(len(na_cols))

print("\nToplam eksik hücre sayısı:")
print(df.isnull().sum().sum())

print("\nEn az bir eksik değeri bulunan gözlem sayısı:")
print(df.isnull().any(axis=1).sum())

print("\nTekrarlanan gözlem sayısı:")
print(df.duplicated().sum())



######################################
# EKSİK DEĞERLERİN HEDEF DEĞİŞKENE GÖRE ANALİZİ
######################################

missing_by_target = pd.DataFrame({
    "NON_BANKRUPT_MISSING_RATIO":
        df.loc[df["BANKRUPT"] == 0, na_cols].isnull().mean() * 100,

    "BANKRUPT_MISSING_RATIO":
        df.loc[df["BANKRUPT"] == 1, na_cols].isnull().mean() * 100
})

missing_by_target["DIFFERENCE"] = (
    missing_by_target["BANKRUPT_MISSING_RATIO"]
    - missing_by_target["NON_BANKRUPT_MISSING_RATIO"]
)

missing_by_target["ABS_DIFFERENCE"] = (
    missing_by_target["DIFFERENCE"].abs()
)

missing_by_target = missing_by_target.sort_values(
    by="ABS_DIFFERENCE",
    ascending=False
)

print("\nEksik değer oranlarının hedef değişkene göre karşılaştırılması:")
print(missing_by_target.head(10))



######################################
# TEKRARLAYAN GÖZLEMLERİN İNCELENMESİ
######################################

duplicate_rows = df[df.duplicated(keep=False)]

print("Tekrarlı gruplarda yer alan toplam satır sayısı:")
print(duplicate_rows.shape[0])

print("\nTekrarlı gruplardaki hedef dağılımı:")
print(duplicate_rows["BANKRUPT"].value_counts())

print("\nSilinmesi muhtemel tekrarların hedef dağılımı:")
print(df.loc[df.duplicated(), "BANKRUPT"].value_counts())

duplicate_feature_rows = df[
    df.duplicated(subset=num_cols, keep=False)
]

duplicate_target_control = (
    duplicate_feature_rows
    .groupby(num_cols, dropna=False)["BANKRUPT"]
    .nunique()
)

print("\nAynı finansal oranlara fakat farklı hedefe sahip grup sayısı:")
print((duplicate_target_control > 1).sum())



######################################
# TEKRARLAYAN GÖZLEMLERİN KALDIRILMASI
######################################

print("Tekrarlar kaldırılmadan önce veri boyutu:")
print(df.shape)

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nTekrarlar kaldırıldıktan sonra veri boyutu:")
print(df.shape)

print("\nKalan tekrarlı gözlem sayısı:")
print(df.duplicated().sum())

print("\nGüncel hedef değişken dağılımı:")
print(df["BANKRUPT"].value_counts())

print("\nGüncel hedef değişken yüzdeleri:")
print(df["BANKRUPT"].value_counts(normalize=True) * 100)






######################################
# EKSİK DEĞER ANALİZİ
######################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [
        col for col in dataframe.columns
        if dataframe[col].isnull().sum() > 0
    ]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(
        ascending=False
    )

    ratio = (
        dataframe[na_columns].isnull().sum()
        / dataframe.shape[0] * 100
    ).sort_values(ascending=False)

    missing_df = pd.concat(
        [n_miss, np.round(ratio, 2)],
        axis=1,
        keys=["n_miss", "ratio"]
    )

    print(missing_df)

    if na_name:
        return na_columns


na_cols = missing_values_table(df, na_name=True)

print("\nEksik değer bulunan değişken sayısı:")
print(len(na_cols))

print("\nToplam eksik hücre sayısı:")
print(df.isnull().sum().sum())

print("\nEn az bir eksik değeri bulunan gözlem sayısı:")
print(df.isnull().any(axis=1).sum())

print("\nTekrarlanan gözlem sayısı:")
print(df.duplicated().sum())



######################################
# EKSİK DEĞERLERİN HEDEF DEĞİŞKENE GÖRE ANALİZİ
######################################

missing_by_target = pd.DataFrame({
    "NON_BANKRUPT_MISSING_RATIO":
        df.loc[df["BANKRUPT"] == 0, na_cols].isnull().mean() * 100,

    "BANKRUPT_MISSING_RATIO":
        df.loc[df["BANKRUPT"] == 1, na_cols].isnull().mean() * 100
})

missing_by_target["DIFFERENCE"] = (
    missing_by_target["BANKRUPT_MISSING_RATIO"]
    - missing_by_target["NON_BANKRUPT_MISSING_RATIO"]
)

missing_by_target["ABS_DIFFERENCE"] = (
    missing_by_target["DIFFERENCE"].abs()
)

missing_by_target = missing_by_target.sort_values(
    by="ABS_DIFFERENCE",
    ascending=False
)

print("\nEksik değer oranlarının hedef değişkene göre karşılaştırılması:")
print(missing_by_target.head(10))



######################################
# HEDEF DEĞİŞKEN ANALİZİ
######################################

def target_summary(dataframe, target, plot=False):
    target_counts = dataframe[target].value_counts().sort_index()

    target_ratios = (
        dataframe[target].value_counts(normalize=True)
        .sort_index() * 100
    )

    target_df = pd.DataFrame({
        "COUNT": target_counts,
        "RATIO": target_ratios
    })

    print(target_df)

    if plot:
        labels = ["İflas Etmedi", "İflas Etti"]

        plt.figure(figsize=(8, 5))

        ax = sns.barplot(
            x=labels,
            y=target_counts.values,
            color="#355C7D"
        )

        for index, value in enumerate(target_counts.values):
            ratio = target_ratios.iloc[index]

            ax.text(
                index,
                value,
                f"{value}\n%{ratio:.2f}",
                ha="center",
                va="bottom"
            )

        plt.title(
            "Şirketlerin İflas Durumuna Göre Dağılımı",
            pad=15
        )
        plt.ylim(0, target_counts.max() * 1.15)
        plt.xlabel("İflas Durumu")
        plt.ylabel("Şirket Sayısı")
        plt.tight_layout()
        plt.show()


target_summary(df, "BANKRUPT", plot=True)



######################################
# SAYISAL DEĞİŞKENLERİN HEDEFE GÖRE ANALİZİ
######################################

feature_descriptions = {
    "Attr1": "Net kâr / toplam varlıklar",
    "Attr2": "Toplam yükümlülükler / toplam varlıklar",
    "Attr3": "Çalışma sermayesi / toplam varlıklar",
    "Attr4": "Dönen varlıklar / kısa vadeli yükümlülükler",
    "Attr7": "EBIT / toplam varlıklar",
    "Attr21": "Cari yıl satışları / önceki yıl satışları",
    "Attr27": "Faaliyet kârı / finansman giderleri",
    "Attr37": "(Dönen varlıklar - stoklar) / uzun vadeli yükümlülükler",
    "Attr59": "Uzun vadeli yükümlülükler / özkaynaklar"
}


def target_summary_with_num(dataframe, target, numerical_col):
    summary_df = dataframe.groupby(target)[numerical_col].agg(
        ["count", "mean", "median"]
    )

    summary_df["missing"] = (
        dataframe.groupby(target).size()
        - summary_df["count"]
    )

    summary_df.rename(
        index={
            0: "İflas Etmedi",
            1: "İflas Etti"
        },
        inplace=True
    )

    print(
        f"\n{numerical_col} - "
        f"{feature_descriptions[numerical_col]}"
    )

    print(summary_df)


for col in feature_descriptions:
    target_summary_with_num(
        df,
        "BANKRUPT",
        col
    )

    ######################################
    # SAYISAL DEĞİŞKENLERİN GÖRSEL ANALİZİ
    ######################################

    selected_cols = list(feature_descriptions.keys())

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(18, 14)
    )

    axes = axes.flatten()

    for index, col in enumerate(selected_cols):
        plot_df = df[["BANKRUPT", col]].dropna().copy()

        lower_limit = plot_df[col].quantile(0.01)
        upper_limit = plot_df[col].quantile(0.99)

        plot_df = plot_df[
            plot_df[col].between(lower_limit, upper_limit)
        ]

        sns.boxplot(
            data=plot_df,
            x="BANKRUPT",
            y=col,
            palette=["#496A81", "#C8553D"],
            ax=axes[index]
        )

        axes[index].set_title(
            f"{col} - {feature_descriptions[col]}",
            fontsize=10
        )

        axes[index].set_xlabel("")
        axes[index].set_ylabel("Oran")
        axes[index].set_xticks([0, 1])
        axes[index].set_xticklabels(
            ["İflas Etmedi", "İflas Etti"]
        )

    plt.suptitle(
        "Finansal Oranların İflas Durumuna Göre Dağılımı",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


######################################
# AYKIRI DEĞER ANALİZİ
######################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(
        dataframe,
        col_name
    )

    outlier_exists = dataframe[
        (dataframe[col_name] > up_limit)
        | (dataframe[col_name] < low_limit)
        ].any(axis=None)

    return outlier_exists


outlier_cols = [
    col for col in num_cols
    if check_outlier(df, col)
]

print("Aykırı değer bulunan değişken sayısı:")
print(len(outlier_cols))

print("\nAykırı değer bulunan değişkenler:")
print(outlier_cols)

outlier_summary = []

for col in num_cols:
    low_limit, up_limit = outlier_thresholds(df, col)

    outlier_count = df[
        (df[col] < low_limit)
        | (df[col] > up_limit)
        ][col].count()

    valid_count = df[col].notnull().sum()

    outlier_ratio = (
        outlier_count / valid_count * 100
        if valid_count > 0
        else 0
    )

    if outlier_count > 0:
        outlier_summary.append({
            "VARIABLE": col,
            "LOW_LIMIT": low_limit,
            "UP_LIMIT": up_limit,
            "OUTLIER_COUNT": outlier_count,
            "OUTLIER_RATIO": outlier_ratio
        })

outlier_summary_df = pd.DataFrame(
    outlier_summary
).sort_values(
    by="OUTLIER_RATIO",
    ascending=False
)

print("\nAykırı değer özeti:")
print(outlier_summary_df.head(15))


######################################
# KORELASYON ANALİZİ
######################################

def high_correlated_pairs(dataframe, corr_th=0.90):
    corr_matrix = dataframe.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )

    high_corr_df = (
        upper_triangle
        .stack()
        .reset_index()
    )

    high_corr_df.columns = [
        "VARIABLE_1",
        "VARIABLE_2",
        "CORRELATION"
    ]

    high_corr_df = high_corr_df[
        high_corr_df["CORRELATION"] > corr_th
        ]

    high_corr_df = high_corr_df.sort_values(
        by="CORRELATION",
        ascending=False
    )

    return high_corr_df


high_corr_df = high_correlated_pairs(
    df[num_cols],
    corr_th=0.90
)

print("Yüksek korelasyonlu değişken çifti sayısı:")
print(high_corr_df.shape[0])

print("\nEn yüksek korelasyonlu ilk 20 değişken çifti:")
print(high_corr_df.head(20))



######################################
# SEÇİLMİŞ DEĞİŞKENLER İÇİN KORELASYON HARİTASI
######################################

correlation_labels = {
    "Attr1": "Net Kârlılık",
    "Attr2": "Borç / Varlık",
    "Attr3": "Çalışma Sermayesi",
    "Attr4": "Cari Oran",
    "Attr7": "EBIT / Varlık",
    "Attr21": "Satış Büyümesi",
    "Attr27": "Faiz Karşılama",
    "Attr37": "Likit Varlık / UV Borç",
    "Attr59": "UV Borç / Özkaynak"
}

selected_corr_df = (
    df[list(correlation_labels.keys())]
    .rename(columns=correlation_labels)
    .corr()
)

plt.figure(figsize=(11, 8))

sns.heatmap(
    selected_corr_df,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.5
)

plt.title(
    "Temel Finansal Oranlar Arasındaki Korelasyonlar",
    pad=15
)

plt.tight_layout()
plt.show()


######################################
# EDA SONUÇLARI
######################################

# 1. Tekrarlanan 60 kayıt kaldırılmış ve 5850 benzersiz
#    gözlemle analize devam edilmiştir.
#
# 2. İflas eden şirketlerin oranı yaklaşık %6.97'dir.
#    Hedef değişken belirgin biçimde dengesizdir.
#
# 3. Şirketlerin yaklaşık %48.80'inde en az bir eksik
#    finansal oran bulunmaktadır.
#
# 4. Satış büyümesi (Attr21) ve faaliyet kârı /
#    finansman giderleri (Attr27) değişkenlerindeki
#    eksiklik, iflas eden şirketlerde daha yüksektir.
#
# 5. İflas eden şirketlerde daha düşük kârlılık,
#    daha yüksek borçluluk, daha zayıf likidite ve
#    satış daralması gözlenmiştir.
#
# 6. Finansal oranların tamamında aykırı değer vardır.
#    Bu değerler finansal sıkıntı sinyali taşıyabileceği
#    için EDA aşamasında silinmemiştir.
#
# 7. Benzer finansal bilgileri ölçen yüksek korelasyonlu
#    değişken grupları bulunmaktadır. Değişken çıkarma
#    kararı model performansı ile birlikte verilecektir.
#
# 8. Model değerlendirmesinde accuracy tek başına
#    kullanılmamalıdır. Recall, precision, F1, ROC-AUC
#    ve analist kapasitesi metriği incelenmelidir.


