import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import (StandardScaler, RobustScaler,
                                   MinMaxScaler, LabelEncoder)

warnings.filterwarnings("ignore")
MAX_LABEL_ENCODE_CARDINALITY = 200


class AdvancedPreprocessor:

    def __init__(self):
        self.scalers: dict             = {}
        self.encoders: dict            = {}
        self.pca_models: dict          = {}
        self.feature_selectors: dict   = {}
        self.discretization_bins: dict = {}

    # ─────────────────────────────────────────────────────────────
    # 1. DERIVED FEATURE CREATION
    # ─────────────────────────────────────────────────────────────
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("  Creating derived features…")
        result = df.copy()
        created: list[str] = []

        # ── Temporal features ─────────────────────────────────────
        ts_col = "@timestamp" if "@timestamp" in result.columns else None
        if ts_col and pd.api.types.is_datetime64_any_dtype(result[ts_col]):
            result["event_hour"]      = result[ts_col].dt.hour
            result["event_dayofweek"] = result[ts_col].dt.dayofweek
            result["event_month"]     = result[ts_col].dt.month
            result["event_year"]      = result[ts_col].dt.year
            result["is_weekend"]      = (result["event_dayofweek"] >= 5).astype(int)
            created += ["event_hour", "event_dayofweek", "event_month",
                        "event_year", "is_weekend"]

        # ── Actor-level aggregates ────────────────────────────────
        if "actor" in result.columns:
            actor_counts = result["actor"].value_counts().rename("actor_event_count")
            result = result.join(actor_counts, on="actor")
            created.append("actor_event_count")

            if "actor_is_bot" in result.columns:
                bot_ratio = (
                    result.groupby("actor", observed=True)["actor_is_bot"]
                          .mean()
                          .rename("actor_bot_ratio")
                )
                result = result.join(bot_ratio, on="actor")
                created.append("actor_bot_ratio")

            if ts_col and pd.api.types.is_datetime64_any_dtype(result[ts_col]):
                actor_days = (
                    result.groupby("actor", observed=True)[ts_col]
                          .nunique()
                          .rename("actor_active_days")
                )
                result = result.join(actor_days, on="actor")
                result["actor_event_velocity"] = (
                    result["actor_event_count"] /
                    result["actor_active_days"].replace(0, 1)
                )
                created += ["actor_active_days", "actor_event_velocity"]

        # ── Bot / automation indicators ───────────────────────────
        if "programmatic_access_type" in result.columns:
            result["is_programmatic"] = (
                result["programmatic_access_type"]
                .astype(str)
                .isin(["Unknown", "nan", ""])
                .astype(int)
                .rsub(1)              
            )
            created.append("is_programmatic")

        if "integration_id" in result.columns:
            result["is_integration_event"] = (
                result["integration_id"].notna().astype(int)
            )
            created.append("is_integration_event")

        print(f"  Created {len(created)} derived features: {created}")
        return result

    # ─────────────────────────────────────────────────────────────
    # 2. LABEL ENCODING
    # ─────────────────────────────────────────────────────────────
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        print("  Encoding categorical columns…")
        result = df.copy()
        encoded, skipped = [], []

        obj_cols = result.select_dtypes(include=["object", "category"]).columns
        for col in obj_cols:
            n_unique = result[col].nunique()
            if n_unique <= MAX_LABEL_ENCODE_CARDINALITY:
                le = LabelEncoder()
                result[col] = le.fit_transform(result[col].astype(str))
                self.encoders[col] = le
                encoded.append(col)
            else:
                skipped.append(col)

        print(f"    Label-encoded : {len(encoded)} columns "
              f"(cardinality ≤ {MAX_LABEL_ENCODE_CARDINALITY})")
        if skipped:
            print(f"    Skipped       : {len(skipped)} high-cardinality columns")
            print(f"      → {skipped[:8]}{'…' if len(skipped) > 8 else ''}")
        return result

    # ─────────────────────────────────────────────────────────────
    # 3. DISCRETISATION & BINARISATION
    # ─────────────────────────────────────────────────────────────
    def discretize_and_binarize(self, df: pd.DataFrame,
                                 config: dict | None = None) -> pd.DataFrame:
        result = df.copy()

        # ── Discretisation config ─────────────────────────────────
        if config is None:
            config = {}
            if "actor_event_count" in result.columns:
                config["actor_event_count"] = {
                    "method": "quantile",
                    "bins"  : 4,
                    "labels": ["Low", "Medium", "High", "Very High"],
                }
            if "event_hour" in result.columns:
                config["event_hour"] = {
                    "method": "custom",
                    "bins"  : [-1, 6, 12, 18, 23],
                    "labels": ["Night", "Morning", "Afternoon", "Evening"],
                }
            if "actor_event_velocity" in result.columns:
                config["actor_event_velocity"] = {
                    "method": "quantile",
                    "bins"  : 3,
                    "labels": ["Slow", "Moderate", "Fast"],
                }

        for col, cfg in config.items():
            if col not in result.columns:
                print(f"'{col}' not found – skipping discretisation.")
                continue
            try:
                if cfg["method"] == "quantile":
                    result[f"{col}_bin"] = pd.qcut(
                        result[col], q=cfg["bins"],
                        labels=cfg.get("labels"), duplicates="drop"
                    )
                elif cfg["method"] in ("equal_width", "custom"):
                    result[f"{col}_bin"] = pd.cut(
                        result[col], bins=cfg["bins"],
                        labels=cfg.get("labels")
                    )
                print(f"    Discretised '{col}' → '{col}_bin'"
                      f"  ({cfg['method']}, {cfg['bins']} bins)")
                self.discretization_bins[col] = cfg
            except Exception as exc:
                print(f"Discretisation failed for '{col}': {exc}")

        # ── Binarisation ──────────────────────────────────────────
        binarize_cfg = {}
        if "actor_is_bot" in result.columns:
            binarize_cfg["actor_is_bot"] = {
                "threshold": 0.5,
                "name"     : "is_bot_binary",
            }
        if "actor_event_count" in result.columns:
            med = result["actor_event_count"].median()
            binarize_cfg["actor_event_count"] = {
                "threshold": med,
                "name"     : "is_high_activity_actor",
            }
        if "is_weekend" in result.columns:
            binarize_cfg["is_weekend"] = {
                "threshold": 0.5,
                "name"     : "is_weekend_event",
            }

        for col, cfg in binarize_cfg.items():
            if col not in result.columns:
                continue
            try:
                result[cfg["name"]] = (
                    result[col] >= cfg["threshold"]
                ).astype(int)
                print(f"    Binarised '{col}' "
                      f"(threshold={cfg['threshold']}) → '{cfg['name']}'")
            except Exception as exc:
                print(f"Binarisation failed for '{col}': {exc}")

        print(f"  Shape after discretisation/binarisation: {result.shape}")
        return result

    # ─────────────────────────────────────────────────────────────
    # 4. DATA TRANSFORMATIONS  (scaling + log/sqrt)
    # ─────────────────────────────────────────────────────────────
    def apply_transformations(self, df: pd.DataFrame,
                               transform_cfg: dict | None = None
                               ) -> pd.DataFrame:
        print("  Applying data transformations…")
        result = df.copy()

        if transform_cfg is None:
            transform_cfg = {
                "scaling"       : {"method": "standard", "columns": "numeric"},
                "robust_scaling": {"method": "robust",
                                   "columns": ["actor_event_count",
                                               "actor_event_velocity"]},
                "log_transform" : ["actor_event_count", "actor_event_velocity"],
                "sqrt_transform": ["event_hour"],
            }

        num_cols = result.select_dtypes(include="number").columns.tolist()

        # ── Standard scaling ──────────────────────────────────────
        sc_cfg = transform_cfg.get("scaling", {})
        if sc_cfg:
            cols = (num_cols if sc_cfg.get("columns") == "numeric"
                    else [c for c in sc_cfg.get("columns", [])
                          if c in result.columns])
            cols = [c for c in cols if result[c].var() > 0]
            if cols:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(result[cols])
                for i, col in enumerate(cols):
                    result[f"{col}_scaled"] = scaled[:, i]
                self.scalers["standard"] = scaler
                print(f"    StandardScaler applied to {len(cols)} columns.")

        # ── Robust scaling ────────────────────────────────────────
        rb_cfg = transform_cfg.get("robust_scaling", {})
        if rb_cfg:
            cols = [c for c in rb_cfg.get("columns", []) if c in result.columns]
            cols = [c for c in cols if result[c].var() > 0]
            if cols:
                scaler = RobustScaler()
                scaled = scaler.fit_transform(result[cols])
                for i, col in enumerate(cols):
                    result[f"{col}_robust"] = scaled[:, i]
                self.scalers["robust"] = scaler
                print(f"    RobustScaler applied to: {cols}")

        # ── Log transform ─────────────────────────────────────────
        for col in transform_cfg.get("log_transform", []):
            if col in result.columns:
                result[f"{col}_log"] = np.log1p(result[col].clip(lower=0))
                print(f"    log1p transform → '{col}_log'")

        # ── Square-root transform ─────────────────────────────────
        for col in transform_cfg.get("sqrt_transform", []):
            if col in result.columns:
                result[f"{col}_sqrt"] = np.sqrt(result[col].clip(lower=0))
                print(f"    sqrt transform  → '{col}_sqrt'")

        return result

    # ─────────────────────────────────────────────────────────────
    # 5. SUBSET SELECTION  (GitHub-specific subsets)
    # ─────────────────────────────────────────────────────────────
    def select_event_subsets(self, df: pd.DataFrame,
                              subset_type: str = "bot_events") -> pd.DataFrame:

        print(f"  Selecting subset: '{subset_type}'…")
        original = len(df)

        if subset_type == "bot_events":
            if "actor_is_bot" not in df.columns:
                print("'actor_is_bot' not found.")
                return df
            subset = df[df["actor_is_bot"] == 1.0].copy()

        elif subset_type == "human_events":
            if "actor_is_bot" not in df.columns:
                print("'actor_is_bot' not found.")
                return df
            subset = df[df["actor_is_bot"] == 0.0].copy()

        elif subset_type == "high_activity":
            if "actor_event_count" not in df.columns:
                print("'actor_event_count' not found.")
                return df
            threshold = df["actor_event_count"].quantile(0.75)
            subset = df[df["actor_event_count"] >= threshold].copy()
            print(f"    Threshold: actor_event_count ≥ {threshold:.0f}")

        elif subset_type == "weekend_events":
            if "is_weekend" not in df.columns:
                print("'is_weekend' not found.")
                return df
            subset = df[df["is_weekend"] == 1].copy()

        elif subset_type == "programmatic":
            if "is_programmatic" not in df.columns:
                print("'is_programmatic' not found.")
                return df
            subset = df[df["is_programmatic"] == 1].copy()

        else:
            print(f"Unknown subset_type '{subset_type}'.")
            return df

        print(f"    {original:,} → {len(subset):,} rows "
              f"({len(subset)/original*100:.1f}%)")
        return subset.reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────
    # 6. DIMENSION REDUCTION
    # ─────────────────────────────────────────────────────────────
    def dimension_reduction(self, df: pd.DataFrame,
                             target_col: str | None = "actor_is_bot",
                             method: str = "pca",
                             n_components: int | float = 0.95,
                             feature_types: str = "numeric") -> pd.DataFrame:

        print(f"\n  === DIMENSION REDUCTION: {method.upper()} ===")

        if feature_types == "numeric":
            feat_cols = df.select_dtypes(include="number").columns.tolist()
        else:
            feat_cols = df.columns.tolist()

        if target_col and target_col in feat_cols:
            feat_cols.remove(target_col)

        # Remove columns already tagged as outlier flags
        feat_cols = [c for c in feat_cols if not c.startswith("outlier_")]

        print(f"  Working with {len(feat_cols)} features.")

        if len(feat_cols) == 0:
            print("  No suitable features found.")
            return df

        # Fill remaining NaN before reduction
        df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))

        if method == "pca":
            return self._apply_pca(df, feat_cols, n_components)
        elif method == "variance_threshold":
            return self._apply_variance_threshold(df, feat_cols, n_components)
        elif method == "univariate" and target_col:
            return self._apply_univariate(df, feat_cols, target_col, n_components)
        else:
            print(f"  Method '{method}' not supported or target missing.")
            return df

    def _apply_pca(self, df, feat_cols, n_components):
        numeric = df[feat_cols].select_dtypes(include="number").columns.tolist()
        if len(numeric) < 2:
            print("  Need ≥ 2 numeric features for PCA.")
            return df
        X = df[numeric].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        pca = PCA(n_components=n_components if isinstance(n_components, float)
                  and n_components < 1
                  else min(int(n_components), len(numeric)))
        X_pca = pca.fit_transform(X_s)
        pca_cols = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        non_num = [c for c in df.columns if c not in numeric]
        result = pd.concat([df[non_num], pca_df], axis=1)
        self.scalers["pca_scaler"] = scaler
        self.pca_models["pca"] = pca
        print(f"  PCA: {len(numeric)} features → {X_pca.shape[1]} components")
        print(f"  Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
        return result

    def _apply_variance_threshold(self, df, feat_cols, threshold):
        numeric = df[feat_cols].select_dtypes(include="number").columns.tolist()
        sel = VarianceThreshold(threshold=float(threshold) if isinstance(threshold, float)
                                else 0.0)
        sel.fit_transform(df[numeric])
        kept = [c for c, s in zip(numeric, sel.get_support()) if s]
        non_num = [c for c in df.columns if c not in numeric]
        self.feature_selectors["variance_threshold"] = sel
        print(f"  VarianceThreshold: {len(numeric)} → {len(kept)} features kept.")
        return df[non_num + kept].copy()

    def _apply_univariate(self, df, feat_cols, target_col, k):
        numeric = df[feat_cols].select_dtypes(include="number").columns.tolist()
        if target_col not in df.columns:
            print(f"  Target '{target_col}' not in dataframe.")
            return df
        mask = df[target_col].notna()
        X = df.loc[mask, numeric]
        y = df.loc[mask, target_col].astype(int)
        if len(X) == 0:
            return df
        k_best = min(int(k), len(numeric))
        sel = SelectKBest(score_func=f_classif, k=k_best)
        sel.fit(X, y)
        kept = [c for c, s in zip(numeric, sel.get_support()) if s]
        non_num = [c for c in df.columns if c not in numeric]
        self.feature_selectors["univariate"] = sel
        print(f"  Univariate (f_classif): {len(numeric)} → {len(kept)} features kept.")
        print(f"  Selected: {kept}")
        return df[non_num + kept].copy()

    # ─────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────
    def get_preprocessing_summary(self, original_df: pd.DataFrame,
                                   processed_df: pd.DataFrame) -> dict:
        """Print and return a before/after summary of all preprocessing."""
        print("\n  === PREPROCESSING SUMMARY ===")
        print(f"  Original  : {original_df.shape[0]:,} rows × "
              f"{original_df.shape[1]} columns")
        print(f"  Processed : {processed_df.shape[0]:,} rows × "
              f"{processed_df.shape[1]} columns")

        new_cols = set(processed_df.columns) - set(original_df.columns)
        dropped  = set(original_df.columns) - set(processed_df.columns)
        print(f"  Features added  : {len(new_cols)}")
        print(f"  Features dropped: {len(dropped)}")
        if new_cols:
            for c in sorted(new_cols)[:20]:
                print(f"    + {c}")
            if len(new_cols) > 20:
                print(f"    … and {len(new_cols)-20} more")

        orig_mb = original_df.memory_usage(deep=True).sum() / 1024**2
        proc_mb = processed_df.memory_usage(deep=True).sum() / 1024**2
        print(f"  Memory: {orig_mb:.1f} MB → {proc_mb:.1f} MB")

        return {
            "original_shape" : original_df.shape,
            "processed_shape": processed_df.shape,
            "features_added" : len(new_cols),
            "features_dropped": len(dropped),
            "new_columns"    : sorted(new_cols),
            "memory_change_mb": round(proc_mb - orig_mb, 2),
        }
