import numpy as np
import pandas as pd
from scipy.stats import zscore
from numpy.linalg import inv, LinAlgError
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetector:

    def __init__(self):
        self.summary: dict = {}

    
    def detect_iqr(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Flag values outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR]."""
        for col in columns:
            if col not in df.columns:
                continue
            q1  = df[col].quantile(0.25)
            q3  = df[col].quantile(0.75)
            iqr = q3 - q1
            flag = f"outlier_iqr_{col}"
            df[flag] = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            self.summary[f"iqr_{col}"] = int(df[flag].sum())
        return df

  
    def detect_zscore(self, df: pd.DataFrame,
                       columns: list,
                       threshold: float = 3.0) -> pd.DataFrame:
        """Flag values whose |Z-score| exceeds *threshold*."""
        for col in columns:
            if col not in df.columns:
                continue
            filled = df[col].fillna(df[col].mean())
            flag   = f"outlier_zscore_{col}"
            df[flag] = (np.abs(zscore(filled)) > threshold)
            self.summary[f"zscore_{col}"] = int(df[flag].sum())
        return df


    def detect_isolation_forest(self, df: pd.DataFrame,
                                  features: list,
                                  contamination: float = 0.035) -> pd.DataFrame:
    
        X = df[features].fillna(0).values
        iso = IsolationForest(
            n_estimators=250,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        preds = iso.fit_predict(X)
        df["outlier_iforest"] = (preds == -1)
        self.summary["isolation_forest"] = int(df["outlier_iforest"].sum())
        print(f"    Isolation Forest outliers: {self.summary['isolation_forest']:,}")
        return df


    def detect_lof(self, df: pd.DataFrame,
                    features: list,
                    contamination: float = 0.035) -> pd.DataFrame:
        """Fit LOF and flag outliers."""
        X = df[features].fillna(0).values
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
        )
        preds = lof.fit_predict(X)
        df["outlier_lof"] = (preds == -1)
        self.summary["lof"] = int(df["outlier_lof"].sum())
        print(f"    LOF outliers: {self.summary['lof']:,}")
        return df


    def detect_mahalanobis(self, df: pd.DataFrame,
                            pca_components: list,
                            threshold: float = 3.5) -> pd.DataFrame:
                              
        if len(pca_components) < 2:
            print("    ⚠  Mahalanobis skipped: need ≥ 2 PCA components.")
            return df
        X    = df[pca_components].fillna(0).values
        mean = X.mean(axis=0)
        try:
            cov     = np.cov(X, rowvar=False)
            inv_cov = inv(cov)
        except LinAlgError:
            print("    ⚠  Mahalanobis skipped: singular covariance matrix.")
            return df
        diff = X - mean
        md   = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
        df["outlier_mahalanobis"] = (md > threshold)
        self.summary["mahalanobis"] = int(df["outlier_mahalanobis"].sum())
        print(f"    Mahalanobis outliers: {self.summary['mahalanobis']:,}")
        return df

    def detect_rare_categories(self, df: pd.DataFrame,
                                 column: str,
                                 min_freq: float = 0.01) -> pd.DataFrame:
        """Flag rows whose category value appears in < min_freq of all rows."""
        if column not in df.columns:
            return df
        freq  = df[column].value_counts(normalize=True)
        rare  = freq[freq < min_freq].index
        flag  = f"outlier_rare_{column}"
        df[flag] = df[column].isin(rare)
        self.summary[flag] = int(df[flag].sum())
        print(f"    Rare-category outliers in '{column}': {self.summary[flag]:,}")
        return df


    def compute_outlier_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sum all boolean outlier flags into a single composite score.
        Higher = flagged by more methods.
        """
        outlier_cols = [c for c in df.columns
                        if c.startswith("outlier_")
                        and not c.endswith("_score")
                        and c not in ("outlier_type",
                                      "outlier_validated",
                                      "outlier_confirmed",
                                      "outlier_false_positive",
                                      "outlier_agreement_count")]
        df["outlier_score"] = df[outlier_cols].sum(axis=1)
        return df

    @staticmethod
    def map_outlier_type(score: int) -> str:
        if score == 0:   return "normal"
        elif score == 1: return "mild"
        elif score == 2: return "strong"
        else:            return "extreme"


    def validate_outliers(self, df: pd.DataFrame,
                           min_agreement: int = 2,
                           use_multivariate: bool = True) -> pd.DataFrame:

        uni_flags  = [c for c in df.columns
                      if c.startswith("outlier_")
                      and ("iqr_" in c or "zscore_" in c)]
        multi_flags = [c for c in df.columns
                       if c in ("outlier_iforest",
                                "outlier_lof",
                                "outlier_mahalanobis")]
        all_flags  = uni_flags + (multi_flags if use_multivariate else [])

        if not all_flags:
            print("  ⚠  No outlier flags found for validation.")
            return df

        df["outlier_agreement_count"] = df[all_flags].sum(axis=1)
        df["outlier_validated"]       = df["outlier_agreement_count"] >= min_agreement

        if "outlier_score" in df.columns:
            detected = (df["outlier_score"] > 0)
            fp = detected & ~df["outlier_validated"]
            df["outlier_false_positive"]    = fp
            self.summary["validated_outliers"] = int(df["outlier_validated"].sum())
            self.summary["false_positives"]    = int(fp.sum())
            self.summary["false_positive_rate"] = round(
                fp.sum() / max(detected.sum(), 1) * 100, 2
            )
        return df

    def filter_false_detections(self, df: pd.DataFrame,
                                  method: str = "agreement",
                                  min_agreement: int = 2,
                                  confidence_threshold: float = 0.5
                                  ) -> pd.DataFrame:

        if method == "agreement":
            df = self.validate_outliers(df, min_agreement=min_agreement)
            df["outlier_confirmed"] = df.get("outlier_validated",
                                              pd.Series(False, index=df.index))

        elif method == "confidence":
            flag_cols = [c for c in df.columns
                         if c.startswith("outlier_")
                         and not c.endswith(("_score", "_type",
                                             "_validated", "_confirmed",
                                             "_false_positive",
                                             "_agreement_count"))]
            max_score = len(flag_cols)
            if max_score > 0 and "outlier_score" in df.columns:
                df["outlier_confidence"] = df["outlier_score"] / max_score
                df["outlier_confirmed"]  = (
                    (df["outlier_confidence"] >= confidence_threshold) &
                    (df["outlier_score"] > 0)
                )
            else:
                df["outlier_confirmed"] = False
        else:
            print(f"  Unknown filter method '{method}', using 'agreement'.")
            df = self.filter_false_detections(df, method="agreement",
                                               min_agreement=min_agreement)

        if "outlier_confirmed" in df.columns:
            confirmed = int(df["outlier_confirmed"].sum())
            self.summary["confirmed_outliers"] = confirmed
            orig_flagged = int((df.get("outlier_score",
                                       pd.Series(0, index=df.index)) > 0).sum())
            filtered = orig_flagged - confirmed
            self.summary["filtered_out"]  = filtered
            self.summary["filter_rate"]   = round(
                filtered / max(orig_flagged, 1) * 100, 2
            )
        return df

    def remove_outliers(self, df: pd.DataFrame,
                         method: str = "agreement",
                         min_agreement: int = 2,
                         remove_extreme_only: bool = False,
                         keep_flags: bool = True) -> pd.DataFrame:
        original = len(df)
        df = self.filter_false_detections(df, method=method,
                                           min_agreement=min_agreement)

        if remove_extreme_only:
            to_remove = df.get("outlier_score",
                                pd.Series(0, index=df.index)) >= 3
        else:
            to_remove = df.get("outlier_confirmed",
                                pd.Series(False, index=df.index))

        self.summary["outliers_removed"] = int(to_remove.sum())
        self.summary["removal_rate"]     = round(
            to_remove.sum() / original * 100, 2
        )
        cleaned = df[~to_remove].copy()
        if not keep_flags:
            flag_cols = [c for c in cleaned.columns if c.startswith("outlier_")]
            cleaned   = cleaned.drop(columns=flag_cols)
        print(f"  Rows removed as confirmed outliers: "
              f"{self.summary['outliers_removed']:,}  "
              f"({self.summary['removal_rate']:.2f}%)")
        return cleaned

    def get_false_detection_report(self, df: pd.DataFrame) -> dict:
        report: dict = {}
        score_col  = "outlier_score"
        fp_col     = "outlier_false_positive"
        conf_col   = "outlier_confirmed"

        if score_col in df.columns:
            detected = int((df[score_col] > 0).sum())
            report["total_detected"] = detected

        if fp_col in df.columns:
            fp = int(df[fp_col].sum())
            tp = report.get("total_detected", 0) - fp
            report["false_positives"]   = fp
            report["true_positives"]    = tp
            report["false_positive_rate"] = round(
                fp / max(report.get("total_detected", 1), 1) * 100, 2
            )
            if fp > 0:
                flag_cols = [c for c in df.columns
                             if c.startswith("outlier_")
                             and c not in (score_col, "outlier_type",
                                           "outlier_validated", conf_col,
                                           fp_col, "outlier_agreement_count")]
                fp_rows = df[df[fp_col]]
                report["method_contributions"] = {
                    c: int(fp_rows[c].sum()) for c in flag_cols if c in fp_rows
                }

        if conf_col in df.columns:
            report["confirmed_outliers"] = int(df[conf_col].sum())
            report["filtered_out"] = (
                report.get("total_detected", 0) - report["confirmed_outliers"]
            )
        return report

    def get_summary(self) -> dict:
        return self.summary
