
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class EDAAnalyzer:


    def __init__(self, save_plots: bool = True, output_dir: str = "eda_plots"):
        self.summary:    dict = {}
        self.save_plots: bool = save_plots
        self.output_dir: str  = output_dir
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────
    def _save(self, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    [plot saved → {path}]")

    # ─────────────────────────────────────────────────────────────
    # 1. SUMMARY STATISTICS
    # ─────────────────────────────────────────────────────────────
    def numerical_summary(self, df: pd.DataFrame,
                           columns: list | None = None) -> pd.DataFrame:
        """Compute describe() + skewness + kurtosis + missing count."""
        if columns is None:
            columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        summary = df[columns].describe().T
        summary["missing"]  = df[columns].isnull().sum()
        summary["skewness"] = df[columns].skew()
        summary["kurtosis"] = df[columns].kurtosis()
        self.summary["numerical_summary"] = summary
        return summary

    def categorical_summary(self, df: pd.DataFrame,
                             columns: list | None = None) -> dict:
        """Value-count tables for categorical columns."""
        if columns is None:
            columns = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        result = {col: df[col].value_counts(dropna=False) for col in columns}
        self.summary["categorical_summary"] = result
        return result

    # ─────────────────────────────────────────────────────────────
    # 2. UNIVARIATE PLOTS
    # ─────────────────────────────────────────────────────────────
    def distribution_plots(self, df: pd.DataFrame, columns: list) -> None:
        """Histogram + KDE for each numeric column."""
        for col in columns:
            if col not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#5b9bd5")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            fig.tight_layout()
            self._save(f"dist_{col}.png")

    def boxplot(self, df: pd.DataFrame, columns: list) -> None:
        """Box-and-whisker plot for each numeric column."""
        for col in columns:
            if col not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df[col].dropna(), ax=ax, color="#70ad47")
            ax.set_title(f"Boxplot of {col}")
            fig.tight_layout()
            self._save(f"box_{col}.png")

    # ─────────────────────────────────────────────────────────────
    # 3. MULTIVARIATE ANALYSIS
    # ─────────────────────────────────────────────────────────────
    def correlation_matrix(self, df: pd.DataFrame,
                            columns: list | None = None,
                            figsize: tuple = (12, 10)) -> pd.DataFrame:
        """Compute and plot a correlation heatmap."""
        if columns is None:
            columns = df.select_dtypes(
                include=["float64", "int64"]
            ).columns.tolist()
        corr = df[columns].corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax,
                    linewidths=0.3, square=True)
        ax.set_title("Feature Correlation Heatmap")
        fig.tight_layout()
        self._save("correlation_heatmap.png")
        self.summary["correlation_matrix"] = corr
        return corr

    def pairplot(self, df: pd.DataFrame,
                  columns: list,
                  hue: str | None = None) -> None:
        """Seaborn pairplot (use a sample to keep it fast)."""
        data = df[columns].dropna()
        if len(data) > 1000:
            data = data.sample(n=1000, random_state=42)
        g = sns.pairplot(data, hue=hue, diag_kind="kde",
                         plot_kws={"alpha": 0.4, "s": 10})
        path = os.path.join(self.output_dir, "pairplot.png")
        g.savefig(path, dpi=150)
        plt.close()
        print(f"    [plot saved → {path}]")

    def pca_analysis(self, df: pd.DataFrame,
                      columns: list,
                      n_components: int = 2) -> tuple:
        """Run PCA and return (component_df, explained_variance_ratio)."""
        scaler = StandardScaler()
        X = scaler.fit_transform(df[columns].fillna(df[columns].mean()))
        pca = PCA(n_components=n_components)
        comps = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_
        self.summary["pca_explained_variance"] = ev
        pca_df = pd.DataFrame(
            comps,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        return pca_df, ev

    def plot_pca(self, pca_df: pd.DataFrame,
                  labels=None) -> None:
        """Scatter plot of first two PCA components."""
        fig, ax = plt.subplots(figsize=(9, 6))
        sc = ax.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1],
                        c=labels, cmap="coolwarm", alpha=0.5, s=8)
        if labels is not None:
            plt.colorbar(sc, ax=ax, label="Label")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("PCA – PC1 vs PC2")
        fig.tight_layout()
        self._save("pca_plot.png")

    # ─────────────────────────────────────────────────────────────
    # 4. GROUPED / AGGREGATED SUMMARIES
    # ─────────────────────────────────────────────────────────────
    def grouped_summary(self, df: pd.DataFrame,
                         group_col: str,
                         target_col: str) -> pd.DataFrame:
        """Group by group_col and compute descriptive stats for target_col."""
        grouped = df.groupby(group_col, observed=True)[target_col].agg(
            ["mean", "median", "std", "count"]
        )
        self.summary[f"grouped_{target_col}_by_{group_col}"] = grouped
        return grouped

    # ─────────────────────────────────────────────────────────────
    # 5. GITHUB-SPECIFIC PLOTS
    # ─────────────────────────────────────────────────────────────
    def plot_class_distribution(self, df: pd.DataFrame,
                                 target_col: str = "actor_is_bot") -> None:
        """Bar chart of class counts for the binary target."""
        if target_col not in df.columns:
            print(f"  ⚠  '{target_col}' not in dataframe.")
            return
        counts = df[target_col].value_counts().sort_index()
        labels = {0.0: "Human (0)", 1.0: "Bot (1)"}
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            [labels.get(k, str(k)) for k in counts.index],
            counts.values,
            color=["#5b9bd5", "#e07070"],
        )
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + counts.max() * 0.01,
                    f"{int(b.get_height()):,}",
                    ha="center", va="bottom", fontsize=11)
        ax.set_title(f"Class Distribution: {target_col}")
        ax.set_ylabel("Count")
        fig.tight_layout()
        self._save("class_distribution.png")

    def plot_events_by_hour(self, df: pd.DataFrame,
                             hour_col: str = "event_hour") -> None:
        """Bar chart of events by hour of day."""
        if hour_col not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        df[hour_col].value_counts().sort_index().plot(
            kind="bar", ax=ax, color="#5b9bd5"
        )
        ax.set_title("Events by Hour of Day")
        ax.set_xlabel("Hour"); ax.set_ylabel("Count")
        fig.tight_layout()
        self._save("events_by_hour.png")

    def plot_events_by_dayofweek(self, df: pd.DataFrame,
                                  dow_col: str = "event_dayofweek") -> None:
        """Bar chart of events by day of week (0=Mon … 6=Sun)."""
        if dow_col not in df.columns:
            return
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        counts = df[dow_col].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([day_names[i] for i in counts.index], counts.values,
               color="#70ad47")
        ax.set_title("Events by Day of Week")
        ax.set_xlabel("Day"); ax.set_ylabel("Count")
        fig.tight_layout()
        self._save("events_by_dayofweek.png")

    def plot_top_actions(self, df: pd.DataFrame,
                          action_col: str = "action",
                          top_n: int = 15) -> None:
        """Horizontal bar chart of top N action types."""
        if action_col not in df.columns:
            return
        counts = df[action_col].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(10, 5))
        counts[::-1].plot(kind="barh", ax=ax, color="#9e6db0")
        ax.set_title(f"Top {top_n} Action Types")
        ax.set_xlabel("Count")
        fig.tight_layout()
        self._save("top_actions.png")

    def plot_resampling_comparison(self,
                                    before: dict,
                                    after_smote: dict,
                                    after_adasyn: dict) -> None:
        """
        Three-panel bar chart comparing class counts:
        Original | After SMOTE | After ADASYN
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        labels = ["Human (0)", "Bot (1)"]
        colours = ["#5b9bd5", "#e07070"]

        for ax, title, data in [
            (axes[0], "Original",     before),
            (axes[1], "After SMOTE",  after_smote),
            (axes[2], "After ADASYN", after_adasyn),
        ]:
            vals = [data.get(0, data.get(0.0, 0)),
                    data.get(1, data.get(1.0, 0))]
            bars = ax.bar(labels, vals, color=colours)
            ax.set_title(title); ax.set_ylabel("Count")
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + max(vals) * 0.01,
                        f"{int(b.get_height()):,}",
                        ha="center", va="bottom")

        fig.suptitle("Class Distribution: Original vs SMOTE vs ADASYN",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save("resampling_comparison.png")

    # ─────────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────────
    def list_saved_plots(self) -> list:
        if not os.path.exists(self.output_dir):
            return []
        return sorted(f for f in os.listdir(self.output_dir)
                      if f.endswith(".png"))

    def get_summary(self) -> dict:
        return self.summary

    def __repr__(self) -> str:
        return (f"EDAAnalyzer(save_plots={self.save_plots}, "
                f"output_dir='{self.output_dir}')")
