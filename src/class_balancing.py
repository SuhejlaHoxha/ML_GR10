import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# SMOTE  (Synthetic Minority Oversampling TEchnique)
# ─────────────────────────────────────────────────────────────────────
def smote_oversample(X: np.ndarray,
                     y: np.ndarray,
                     k: int = 5,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    majority_cls = classes[np.argmax(counts)]
    minority_cls = classes[np.argmin(counts)]

    X_min = X[y == minority_cls]
    n_gen = int(counts.max()) - int(counts.min())

    synthetic = []
    for _ in range(n_gen):
        idx    = rng.integers(len(X_min))
        anchor = X_min[idx]
        dists  = np.linalg.norm(X_min - anchor, axis=1)
        dists[idx] = np.inf
        nn_idx = np.argsort(dists)[:k]
        nb     = X_min[rng.choice(nn_idx)]
        lam    = rng.random()
        synthetic.append(anchor + lam * (nb - anchor))

    X_syn = np.vstack(synthetic) if synthetic else np.empty((0, X.shape[1]))
    y_syn = np.full(len(synthetic), minority_cls)

    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


# ─────────────────────────────────────────────────────────────────────
# ADASYN  (Adaptive Synthetic Sampling)
# ─────────────────────────────────────────────────────────────────────
def adasyn_oversample(X: np.ndarray,
                      y: np.ndarray,
                      k: int = 5,
                      seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    minority_cls = classes[np.argmin(counts)]
    majority_cls = classes[np.argmax(counts)]

    X_min = X[y == minority_cls]
    X_maj = X[y == majority_cls]
    G     = int(counts.max()) - int(counts.min())  # total to generate

    # Difficulty ratio per minority sample
    all_X   = np.vstack([X_min, X_maj])
    all_lbl = np.concatenate([np.zeros(len(X_min)), np.ones(len(X_maj))])
    ratios  = []
    for pt in X_min:
        dists  = np.linalg.norm(all_X - pt, axis=1)
        nn_idx = np.argsort(dists)[1: k + 1]
        ratio  = np.sum(all_lbl[nn_idx]) / k
        ratios.append(ratio)

    ratios   = np.array(ratios)
    total_r  = ratios.sum()
    if total_r == 0:                   # no neighbours are majority → uniform
        ratios   = np.ones(len(X_min))
        total_r  = ratios.sum()

    n_per_pt = np.round((ratios / total_r) * G).astype(int)

    synthetic = []
    for i, pt in enumerate(X_min):
        dists  = np.linalg.norm(X_min - pt, axis=1)
        dists[i] = np.inf
        nn_idx = np.argsort(dists)[:k]
        for _ in range(n_per_pt[i]):
            nb  = X_min[rng.choice(nn_idx)]
            lam = rng.random()
            synthetic.append(pt + lam * (nb - pt))

    if not synthetic:
        return X, y

    X_syn = np.vstack(synthetic)
    y_syn = np.full(len(synthetic), minority_cls)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


# ─────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────
def apply_resampling(df: pd.DataFrame,
                     target_col: str = "actor_is_bot",
                     feature_cols: list | None = None,
                     k: int = 5,
                     seed: int = 42
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
 
    # Resolve target column name (may have been renamed during cleaning)
    target_clean = target_col.replace(".", "_")
    if target_clean not in df.columns:
        print(f"  ⚠  Target '{target_clean}' not found – skipping resampling.")
        return df, df

    # Drop rows with missing target
    valid = df[df[target_clean].notna()].copy()

    # Feature columns
    if feature_cols is None:
        feature_cols = [
            c for c in valid.select_dtypes(include="number").columns
            if c != target_clean and not c.startswith("outlier_")
        ][:30]                         # cap for speed

    X = valid[feature_cols].fillna(0).values
    y = valid[target_clean].values.astype(int)

    classes, counts = np.unique(y, return_counts=True)
    print(f"  Before resampling: "
          f"{dict(zip(classes.tolist(), counts.tolist()))}")

    # ── SMOTE ─────────────────────────────────────────────────────
    X_smote, y_smote = smote_oversample(X, y, k=k, seed=seed)
    cls_s, cnt_s = np.unique(y_smote, return_counts=True)
    print(f"  After  SMOTE    : "
          f"{dict(zip(cls_s.tolist(), cnt_s.tolist()))}")
    df_smote = pd.DataFrame(X_smote, columns=feature_cols)
    df_smote[target_clean] = y_smote

    # ── ADASYN ────────────────────────────────────────────────────
    X_ada, y_ada = adasyn_oversample(X, y, k=k, seed=seed)
    cls_a, cnt_a = np.unique(y_ada, return_counts=True)
    print(f"  After  ADASYN   : "
          f"{dict(zip(cls_a.tolist(), cnt_a.tolist()))}")
    df_adasyn = pd.DataFrame(X_ada, columns=feature_cols)
    df_adasyn[target_clean] = y_ada

    return df_smote, df_adasyn
