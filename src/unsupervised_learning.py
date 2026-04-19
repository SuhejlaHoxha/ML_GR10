import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.decomposition      import PCA
from sklearn.cluster            import (KMeans, DBSCAN,
                                        AgglomerativeClustering)
from sklearn.mixture            import GaussianMixture
from sklearn.metrics            import (silhouette_score,
                                        davies_bouldin_score,
                                        calinski_harabasz_score,
                                        adjusted_rand_score,
                                        normalized_mutual_info_score)
from scipy.cluster.hierarchy    import dendrogram, linkage
from scipy.spatial.distance     import cdist

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
TARGET_COL  = "actor_is_bot"




def load_and_prepare(csv_path: str):
    """Load processed CSV, encode object columns, return scaled X, y, feature list."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(df.median(numeric_only=True))

    # Drop leakage
    leakage = ["actor_bot_ratio"]
    df = df.drop(columns=[c for c in leakage if c in df.columns])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names, df


def reduce_for_plot(X_scaled, n_components=2):
    """PCA projection for 2D visualisation."""
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    return pca.fit_transform(X_scaled)




def cluster_metrics(X, labels, y_true=None):
    """Return a dict of internal + external cluster quality metrics."""
    unique = np.unique(labels[labels != -1])   
    n_clusters = len(unique)

    if n_clusters < 2:
        return {"n_clusters": n_clusters,
                "silhouette": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan,
                "ARI": np.nan,
                "NMI": np.nan}


    mask = labels != -1
    X_m, lbl_m = X[mask], labels[mask]

    sil  = silhouette_score(X_m, lbl_m) if len(np.unique(lbl_m)) >= 2 else np.nan
    db   = davies_bouldin_score(X_m, lbl_m)
    ch   = calinski_harabasz_score(X_m, lbl_m)

    ari = nmi = np.nan
    if y_true is not None:
        yt_m = y_true[mask]
        ari  = adjusted_rand_score(yt_m, lbl_m)
        nmi  = normalized_mutual_info_score(yt_m, lbl_m)

    return {
        "n_clusters"         : n_clusters,
        "noise_points"       : int((labels == -1).sum()),
        "silhouette"         : round(float(sil), 4),
        "davies_bouldin"     : round(float(db),  4),
        "calinski_harabasz"  : round(float(ch),  4),
        "ARI"                : round(float(ari), 4),
        "NMI"                : round(float(nmi), 4),
    }


def cluster_purity(labels, y_true):
    """Unsupervised cluster purity against ground truth."""
    mask = labels != -1
    labels, y_true = labels[mask], y_true[mask]
    total = 0
    for c in np.unique(labels):
        idx    = labels == c
        counts = np.bincount(y_true[idx])
        total += counts.max()
    return total / len(labels)




def scatter_2d(X_2d, labels, title, path, y_true=None, alpha=0.4):
    """2D PCA scatter coloured by cluster."""
    fig, axes = plt.subplots(1, 2 if y_true is not None else 1,
                             figsize=(12 if y_true is not None else 6, 5))
    if y_true is None:
        axes = [axes]

    palette = sns.color_palette("tab10", len(np.unique(labels)))
    for ax, (lbl_arr, lbl_title) in zip(
        axes,
        [(labels, f"Clusters – {title}")]
        + ([(y_true, "True Labels")] if y_true is not None else [])
    ):
        unique_lbls = np.unique(lbl_arr)
        colors = sns.color_palette("tab10", len(unique_lbls))
        for i, lv in enumerate(unique_lbls):
            mask = lbl_arr == lv
            label = f"Noise" if lv == -1 else f"Cluster {lv}"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       s=10, alpha=alpha, color=colors[i],
                       label=label)
        ax.set_title(lbl_title, fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(markerscale=2, fontsize=7,
                  bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def cluster_profile_heatmap(X_df, labels, feature_names, title, path, top_n=12):
    """Heatmap of mean standardised feature values per cluster."""
    df = pd.DataFrame(X_df, columns=feature_names)
    df["cluster"] = labels
    profile = df.groupby("cluster").mean()


    variances = profile.var()
    top_feats = variances.nlargest(top_n).index.tolist()
    profile   = profile[top_feats]

    fig, ax = plt.subplots(figsize=(min(14, top_n + 2), 4))
    sns.heatmap(profile, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, annot_kws={"size": 7})
    ax.set_title(f"Cluster Profile – {title}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def cluster_size_plot(labels, title, path):
    """Bar chart of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    cluster_labels = [f"Noise" if c == -1 else f"Cluster {c}"
                      for c in unique]
    colors = sns.color_palette("pastel", len(unique))

    fig, ax = plt.subplots(figsize=(max(6, len(unique) * 0.8), 4))
    bars = ax.bar(cluster_labels, counts, color=colors, edgecolor="grey")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(f"Cluster Sizes – {title}")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()




def run_kmeans(X, y, feature_names, X_2d, plots_dir):
    print("\n  ── K-Means Clustering ──────────────────────────────────")

   
    k_range = range(2, 11)
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(k_range, inertias, "o-", color="steelblue")
    axes[0].set_title("K-Means Elbow Curve")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[1].plot(k_range, sil_scores, "o-", color="seagreen")
    axes[1].set_title("K-Means Silhouette Scores")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "kmeans_elbow.png"), dpi=130)
    plt.close()

    best_k = int(k_range[np.argmax(sil_scores)])
    print(f"    Best k (max silhouette): {best_k}")

    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    labels  = km_best.fit_predict(X)

    metrics = cluster_metrics(X, labels, y)
    metrics["purity"] = round(cluster_purity(labels, y), 4)
    print(f"    Metrics: {metrics}")

    scatter_2d(X_2d, labels, "K-Means",
               os.path.join(plots_dir, "kmeans_scatter.png"), y_true=y)
    cluster_size_plot(labels, "K-Means",
                      os.path.join(plots_dir, "kmeans_sizes.png"))
    cluster_profile_heatmap(X, labels, feature_names,
                             "K-Means",
                             os.path.join(plots_dir, "kmeans_profile.png"))

    return {"Algorithm": "K-Means", **metrics}, labels




def run_dbscan(X, y, feature_names, X_2d, plots_dir):
    print("\n  ── DBSCAN Clustering ────────────────────────────────────")

    
    best_sil, best_eps, best_ms, best_labels = -1, 0.5, 5, None

    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for ms in [3, 5, 10]:
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            lbl = db.fit_predict(X)
            n_clust = len(set(lbl)) - (1 if -1 in lbl else 0)
            if n_clust < 2:
                continue
            mask = lbl != -1
            if mask.sum() < 2:
                continue
            try:
                s = silhouette_score(X[mask], lbl[mask])
            except Exception:
                continue
            if s > best_sil:
                best_sil, best_eps, best_ms = s, eps, ms
                best_labels = lbl

    if best_labels is None:
        print("    ⚠ DBSCAN: no valid clustering found – using eps=0.5, ms=5")
        db = DBSCAN(eps=0.5, min_samples=5)
        best_labels = db.fit_predict(X)

    labels = best_labels
    print(f"    Best eps={best_eps}, min_samples={best_ms}")
    print(f"    Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}  "
          f"| Noise points: {(labels == -1).sum()}")

    metrics = cluster_metrics(X, labels, y)
    noise_frac = (labels == -1).sum() / len(labels)
    metrics["noise_fraction"] = round(float(noise_frac), 4)
    if metrics["n_clusters"] >= 2:
        metrics["purity"] = round(cluster_purity(labels, y), 4)
    print(f"    Metrics: {metrics}")

    scatter_2d(X_2d, labels, "DBSCAN",
               os.path.join(plots_dir, "dbscan_scatter.png"), y_true=y)
    cluster_size_plot(labels, "DBSCAN",
                      os.path.join(plots_dir, "dbscan_sizes.png"))
    if metrics["n_clusters"] >= 2:
        cluster_profile_heatmap(X, labels, feature_names,
                                 "DBSCAN",
                                 os.path.join(plots_dir, "dbscan_profile.png"))

    return {"Algorithm": "DBSCAN", **metrics}, labels




def run_agglomerative(X, y, feature_names, X_2d, plots_dir, n_clusters=3):
    print("\n  ── Agglomerative Clustering ─────────────────────────────")

   
    sample_size = min(300, len(X))
    idx         = np.random.RandomState(RANDOM_SEED).choice(
        len(X), sample_size, replace=False
    )
    Z = linkage(X[idx], method="ward")
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
               leaf_rotation=90, leaf_font_size=7)
    ax.set_title(f"Dendrogram (Ward linkage, n={sample_size} sample)")
    ax.set_xlabel("Sample index"); ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "agglo_dendrogram.png"), dpi=130)
    plt.close()

   
    best_sil, best_n, best_labels = -1, n_clusters, None
    for nc in range(2, 7):
        agg = AgglomerativeClustering(n_clusters=nc, linkage="ward")
        lbl = agg.fit_predict(X)
        s   = silhouette_score(X, lbl)
        if s > best_sil:
            best_sil, best_n, best_labels = s, nc, lbl

    labels = best_labels
    print(f"    Best n_clusters: {best_n}  (sil={best_sil:.4f})")

    metrics = cluster_metrics(X, labels, y)
    metrics["purity"] = round(cluster_purity(labels, y), 4)
    print(f"    Metrics: {metrics}")

    scatter_2d(X_2d, labels, "Agglomerative",
               os.path.join(plots_dir, "agglo_scatter.png"), y_true=y)
    cluster_size_plot(labels, "Agglomerative",
                      os.path.join(plots_dir, "agglo_sizes.png"))
    cluster_profile_heatmap(X, labels, feature_names,
                             "Agglomerative",
                             os.path.join(plots_dir, "agglo_profile.png"))

    return {"Algorithm": "Agglomerative", **metrics}, labels




def run_gmm(X, y, feature_names, X_2d, plots_dir):
    print("\n  ── Gaussian Mixture Model (GMM) ─────────────────────────")

  
    bics  = []
    n_range = range(2, 9)
    for n in n_range:
        gmm = GaussianMixture(n_components=n, covariance_type="full",
                              random_state=RANDOM_SEED, n_init=3)
        gmm.fit(X)
        bics.append(gmm.bic(X))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_range, bics, "o-", color="purple")
    ax.set_title("GMM – BIC Score vs. Number of Components")
    ax.set_xlabel("n_components"); ax.set_ylabel("BIC (lower=better)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gmm_bic.png"), dpi=120)
    plt.close()

    best_n = int(n_range[np.argmin(bics)])
    print(f"    Best n_components (min BIC): {best_n}")

    gmm_best = GaussianMixture(n_components=best_n, covariance_type="full",
                                random_state=RANDOM_SEED, n_init=5)
    gmm_best.fit(X)
    labels = gmm_best.predict(X)

    metrics = cluster_metrics(X, labels, y)
    metrics["purity"] = round(cluster_purity(labels, y), 4)
    print(f"    Metrics: {metrics}")

    scatter_2d(X_2d, labels, "GMM",
               os.path.join(plots_dir, "gmm_scatter.png"), y_true=y)
    cluster_size_plot(labels, "GMM",
                      os.path.join(plots_dir, "gmm_sizes.png"))
    cluster_profile_heatmap(X, labels, feature_names,
                             "GMM",
                             os.path.join(plots_dir, "gmm_profile.png"))

    return {"Algorithm": "GMM", **metrics}, labels




def plot_cluster_comparison(all_metrics: list, plots_dir: str):
    """Bar chart comparing silhouette, Davies-Bouldin, ARI, NMI across algos."""
    df = pd.DataFrame(all_metrics)
    df = df.set_index("Algorithm")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics_cols = [
        ("silhouette",        "Silhouette ↑"),
        ("davies_bouldin",    "Davies-Bouldin ↓"),
        ("ARI",               "ARI ↑"),
        ("NMI",               "NMI ↑"),
    ]
    colors = sns.color_palette("Set2", len(df))
    for ax, (col, label) in zip(axes, metrics_cols):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
        bars = ax.bar(df.index, vals, color=colors, edgecolor="grey")
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, max(vals.max() * 1.2, 0.1))

    plt.suptitle("Clustering Algorithm Comparison", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cluster_comparison.png"),
                dpi=130, bbox_inches="tight")
    plt.close()




def run_unsupervised(csv_path: str, plots_dir: str = "phase2_plots") -> pd.DataFrame:
    """
    Full unsupervised learning pipeline.
    Returns a DataFrame of cluster quality metrics.
    """
    os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "=" * 68)
    print("  PHASE II · Unsupervised Learning & Clustering Analysis")
    print("=" * 68)

    X, y, feature_names, df = load_and_prepare(csv_path)
    print(f"\n  Dataset: {X.shape[0]:,} rows × {X.shape[1]} features")

   
    X_2d = reduce_for_plot(X)

    
    scatter_2d(X_2d, y, "Ground Truth (Bot=1, Human=0)",
               os.path.join(plots_dir, "ground_truth_scatter.png"))

    all_metrics = []

   
    m_km, lbl_km     = run_kmeans(X, y, feature_names, X_2d, plots_dir)
    m_db, lbl_db     = run_dbscan(X, y, feature_names, X_2d, plots_dir)
    m_ag, lbl_ag     = run_agglomerative(X, y, feature_names, X_2d, plots_dir)
    m_gm, lbl_gm     = run_gmm(X, y, feature_names, X_2d, plots_dir)

    all_metrics = [m_km, m_db, m_ag, m_gm]

    
    plot_cluster_comparison(all_metrics, plots_dir)

    results_df = pd.DataFrame(all_metrics)
    results_csv = os.path.join(plots_dir, "clustering_metrics.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n  ✓ Metrics saved → {results_csv}")

    print("\n  " + "=" * 64)
    print("  CLUSTERING SUMMARY")
    print("  " + "=" * 64)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV  = os.path.join(ROOT, "processed dataset", "processed_github.csv")
    OUT  = os.path.join(ROOT, "phase2_plots")
    run_unsupervised(CSV, OUT)
