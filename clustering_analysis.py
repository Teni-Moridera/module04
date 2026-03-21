# -*- coding: utf-8 -*-
"""
Кластеризация данных клиентов: подготовка, k-means, иерархическая кластеризация,
DBSCAN, метод локтя, силуэт, визуализации, интерпретация.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "PP13_ISP23V_clustering.csv"
OUT_DIR = BASE_DIR / "clustering_output"
FEATURES = [
    "customer_age",
    "annual_income",
    "spending_score",
    "purchase_frequency_per_month",
]


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates().dropna(how="any")
    for c in FEATURES:
        df = df[df[c] >= 0]
    print(f"Строк после загрузки: {before}, после очистки: {len(df)}")
    return df.reset_index(drop=True)


def correlation_analysis(df: pd.DataFrame, out_dir: Path) -> None:
    corr = df[FEATURES].corr()
    print("\n--- Матрица корреляций ---")
    print(corr.round(3).to_string())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def _elbow_k_from_inertia(k_vals: list[int], inertias: list[float]) -> int:
    """Точка «локтя»: макс. расстояние от кривой инерции до прямой (первый–последний узел)."""
    x = np.asarray(k_vals, dtype=float)
    y = np.asarray(inertias, dtype=float)
    a = np.array([x[0], y[0]])
    b = np.array([x[-1], y[-1]])
    ab = b - a
    ab_len = np.linalg.norm(ab) + 1e-12
    dists = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        # |(b-a)×(a-p)| / |b-a| в 2D
        cross = (b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1])
        dists.append(abs(cross) / ab_len)
    return int(k_vals[int(np.argmax(dists))])


def elbow_and_silhouette(X: np.ndarray, k_min: int, k_max: int, out_dir: Path) -> tuple[int, list[float], list[float]]:
    inertias: list[float] = []
    silhouettes: list[float] = []
    k_range = range(k_min, k_max + 1)
    k_list = list(k_range)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X, km.labels_)))

    k_elbow = _elbow_k_from_inertia(k_list, inertias)
    k_sil = int(k_list[int(np.argmax(silhouettes))])
    # Если силуэт при локте близок к максимуму — предпочитаем более простую модель (локоть)
    sil_at_elbow = silhouettes[k_list.index(k_elbow)]
    sil_max = max(silhouettes)
    if sil_max - sil_at_elbow <= 0.03:
        best_k = k_elbow
    else:
        best_k = k_sil

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_list, inertias, "o-", color="steelblue")
    axes[0].axvline(k_elbow, color="crimson", linestyle="--", alpha=0.8, label=f"elbow k={k_elbow}")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("Elbow method (k-means)")
    axes[0].legend()
    axes[1].plot(k_list, silhouettes, "o-", color="darkgreen")
    axes[1].axvline(k_sil, color="orange", linestyle="--", alpha=0.8, label=f"max silhouette k={k_sil}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    axes[1].set_title("Silhouette vs k (k-means)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "elbow_silhouette.png", dpi=150)
    plt.close(fig)

    print("\n--- Выбор k ---")
    for k, s in zip(k_range, silhouettes):
        mark = ""
        if k == k_elbow:
            mark += " [локоть]"
        if k == k_sil:
            mark += " [max силуэт]"
        print(f"  k={k}: силуэт = {s:.4f}{mark}")
    print(f"k по методу локтя (хорда): {k_elbow}")
    print(f"k по максимуму силуэта: {k_sil}")
    print(f"Итоговое k для моделей (согласование): {best_k}")
    return best_k, inertias, silhouettes


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray, name: str) -> None:
    df_l = df[FEATURES].copy()
    df_l["cluster"] = labels
    print(f"\n--- Профили кластеров ({name}) ---")
    grp = df_l.groupby("cluster", sort=True)[FEATURES].agg(["mean", "std", "count"])
    print(grp.round(2).to_string())


def plot_pca_2d(X: np.ndarray, labels: np.ndarray, title: str, fname: str, out_dir: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", alpha=0.75, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def plot_dendrogram_sample(X: np.ndarray, out_dir: Path, max_leaf: int = 40) -> None:
    n = X.shape[0]
    idx = np.random.RandomState(42).choice(n, size=min(max_leaf, n), replace=False)
    Z_link = linkage(X[idx], method="ward")
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z_link, ax=ax, leaf_rotation=90)
    ax.set_title(f"Dendrogram (Ward, sample n={len(idx)})")
    fig.tight_layout()
    fig.savefig(out_dir / "dendrogram_sample.png", dpi=150)
    plt.close(fig)


def business_interpretation(df: pd.DataFrame, labels: np.ndarray) -> None:
    df_l = df[FEATURES].copy()
    df_l["cluster"] = labels
    means = df_l.groupby("cluster")[FEATURES].mean()
    print("\n--- Интерпретация (бизнес-смысл) ---")
    global_means = df[FEATURES].mean()
    for c in sorted(means.index):
        row = means.loc[c]
        parts = []
        for f in FEATURES:
            rel = (row[f] - global_means[f]) / (global_means[f] + 1e-9) * 100
            if rel > 8:
                parts.append(f"{f} выше среднего (~{rel:.0f}%)")
            elif rel < -8:
                parts.append(f"{f} ниже среднего (~{rel:.0f}%)")
        desc = "; ".join(parts) if parts else "признаки близки к средним по выборке"
        print(f"Кластер {c}: {desc}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_clean(DATA_PATH)
    X_raw = df[FEATURES].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    correlation_analysis(df, OUT_DIR)

    k_min, k_max = 2, 10
    best_k, _, _ = elbow_and_silhouette(X, k_min, k_max, OUT_DIR)

    km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    labels_km = km.fit_predict(X)
    print(f"\nk-means: число кластеров k = {best_k}, inertia = {km.inertia_:.2f}")
    print(f"Силуэт (k-means): {silhouette_score(X, labels_km):.4f}")
    cluster_profiles(df, labels_km, "k-means")
    plot_pca_2d(X, labels_km, f"k-means, k={best_k}", "pca_kmeans.png", OUT_DIR)

    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    labels_hier = hier.fit_predict(X)
    print(f"\nИерархическая (Ward): силуэт = {silhouette_score(X, labels_hier):.4f}")
    cluster_profiles(df, labels_hier, "иерархическая")
    plot_pca_2d(X, labels_hier, f"Hierarchical Ward, k={best_k}", "pca_hierarchical.png", OUT_DIR)
    plot_dendrogram_sample(X, OUT_DIR)

    db = DBSCAN(eps=0.85, min_samples=5)
    labels_db = db.fit_predict(X)
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise = int(np.sum(labels_db == -1))
    print(f"\nDBSCAN: eps=0.85, min_samples=5")
    print(f"  Выделено кластеров: {n_clusters_db}, шумовых точек: {n_noise}")
    if n_clusters_db >= 2:
        mask = labels_db >= 0
        if mask.sum() > 1:
            sil_db = silhouette_score(X[mask], labels_db[mask])
            print(f"  Силуэт (без шума): {sil_db:.4f}")
        plot_pca_2d(
            X,
            labels_db,
            "DBSCAN (noise = -1)",
            "pca_dbscan.png",
            OUT_DIR,
        )
    else:
        print("  DBSCAN дал мало кластеров — при необходимости подберите eps/min_samples.")

    business_interpretation(df, labels_km)

    print("\n--- Рекомендации ---")
    print(
        "1) Использовать кластеры k-means для сегментации клиентов в маркетинге и персонализации предложений.\n"
        "2) Сравнивать с иерархической кластеризацией: при согласованности меток усиливается доверие к сегментам.\n"
        "3) Точки DBSCAN как «шум» — отдельно анализировать как аномальные или переходные клиенты.\n"
        "4) Регулярно переобучать модель при обновлении данных и проверять устойчивость сегментов."
    )

    print(f"\nГрафики сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    main()
