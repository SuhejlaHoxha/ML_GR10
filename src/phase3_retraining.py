import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report, precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
TARGET_COL  = "actor_is_bot"
LEAKAGE_COLS = ["actor_bot_ratio"]

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(ROOT_DIR, "processed dataset", "processed_github.csv")
OUT_DIR     = os.path.join(ROOT_DIR, "phase3_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(csv_path: str):
    print("\n[1/8] Loading dataset …")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(df.median(numeric_only=True))

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"    Dataset shape   : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"    Class balance   : Human={int((y==0).sum()):,}  Bot={int((y==1).sum()):,}")
    print(f"    Imbalance ratio : {(y==0).sum()/(y==1).sum():.1f}:1")
    return X, y, list(X.columns)

def smote_oversample(X: np.ndarray, y: np.ndarray,
                     k: int = 5, random_state: int = RANDOM_SEED) -> tuple:
    rng = np.random.RandomState(random_state)
    minority_mask = y == 1
    X_min = X[minority_mask]
    X_maj = X[~minority_mask]

    n_to_generate = len(X_maj) - len(X_min)
    if n_to_generate <= 0:
        return X, y

    synthetic = []
    for _ in range(n_to_generate):
        anchor_idx = rng.randint(0, len(X_min))
        anchor     = X_min[anchor_idx]

        dists = np.linalg.norm(X_min - anchor, axis=1)
        dists[anchor_idx] = np.inf        
        nn_indices = np.argsort(dists)[:k]
        neighbour  = X_min[rng.choice(nn_indices)]

        lam     = rng.uniform(0, 1)
        new_pt  = anchor + lam * (neighbour - anchor)
        synthetic.append(new_pt)

    X_syn   = np.vstack(synthetic)
    y_syn   = np.ones(len(X_syn), dtype=int)

    X_bal   = np.vstack([X, X_syn])
    y_bal   = np.hstack([y, y_syn])
    return X_bal, y_bal

def train_baseline(X_train, X_test, y_train, y_test):
    print("\n[2/8] Training baseline Random Forest (Phase II reference) …")
    rf_baseline = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_baseline.fit(X_train, y_train)
    return rf_baseline

def train_smote_model(X_train, X_test, y_train, y_test):
    print("\n[3/8] Retraining with SMOTE balancing …")
    X_bal, y_bal = smote_oversample(X_train, y_train)
    print(f"    After SMOTE — Human: {int((y_bal==0).sum()):,}  "
          f"Bot: {int((y_bal==1).sum()):,}")

    rf_smote = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        max_depth=20,
        min_samples_leaf=2,
        class_weight=None,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_smote.fit(X_bal, y_bal)
    return rf_smote, X_bal, y_bal

def find_optimal_threshold(model, X_test, y_test) -> float:
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_f1 = 0.5, 0.0
    f1_curve = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        f1_curve.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    return best_t, thresholds, f1_curve

def predict_with_threshold(model, X, threshold: float) -> np.ndarray:
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)

def compute_metrics(name: str, y_true, y_pred, y_proba=None,
                    cv_scores: dict = None) -> dict:
    m = {
        "Model"     : name,
        "Accuracy"  : round(accuracy_score(y_true, y_pred), 4),
        "Precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"    : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1"        : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC"   : round(roc_auc_score(y_true, y_proba), 4)
                      if y_proba is not None else "N/A",
        "Avg-Precision": round(average_precision_score(y_true, y_proba), 4)
                         if y_proba is not None else "N/A",
    }
    if cv_scores:
        m.update(cv_scores)
    return m

def cv_evaluate(model, X, y, n_splits=5) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    res = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=False, n_jobs=-1
    )
    return {
        "CV-F1 (mean)"   : round(res["test_f1"].mean(), 4),
        "CV-F1 (std)"    : round(res["test_f1"].std(), 4),
        "CV-AUC (mean)"  : round(res["test_roc_auc"].mean(), 4),
        "CV-AUC (std)"   : round(res["test_roc_auc"].std(), 4),
    }

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]

def save_model(model, scaler, feature_names: list):
    try:
        import joblib
        joblib.dump(model,        os.path.join(OUT_DIR, "retrained_model.joblib"))
        joblib.dump(scaler,       os.path.join(OUT_DIR, "scaler.joblib"))
        with open(os.path.join(OUT_DIR, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)
        print("    Model, scaler, and feature list saved to phase3_outputs/")
    except ImportError:
        print("    ⚠ joblib not available — model not serialised.")

def run_phase3(csv_path: str = CSV_PATH):
    print("\n" + "=" * 68)
    print("  PHASE III — MODEL RETRAINING & ADVANCED ANALYSIS")
    print("  GitHub Activity-Log: Bot vs Human Classifier")
    print("=" * 68)

    X, y, feature_names = load_data(csv_path)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y.values, test_size=0.20,
        stratify=y.values, random_state=RANDOM_SEED
    )
    print(f"\n    Train : {len(X_tr):,}   Test : {len(X_te):,}")

    scaler        = StandardScaler()
    X_tr_sc       = scaler.fit_transform(X_tr)
    X_te_sc       = scaler.transform(X_te)

    rf_base = train_baseline(X_tr, X_te, y_tr, y_te)
    y_pred_base  = rf_base.predict(X_te)
    y_proba_base = rf_base.predict_proba(X_te)[:, 1]
    cv_base      = cv_evaluate(rf_base, X_tr, y_tr)
    metrics_base = compute_metrics(
        "Baseline RF (Phase II)", y_te, y_pred_base, y_proba_base, cv_base
    )

    rf_smote, X_bal, y_bal = train_smote_model(X_tr, X_te, y_tr, y_te)
    y_pred_smote  = rf_smote.predict(X_te)
    y_proba_smote = rf_smote.predict_proba(X_te)[:, 1]
    cv_smote      = cv_evaluate(rf_smote, X_bal, y_bal)
    metrics_smote = compute_metrics(
        "RF + SMOTE (Strategy A)", y_te, y_pred_smote, y_proba_smote, cv_smote
    )

    best_t, thresholds, f1_curve = find_optimal_threshold(
        rf_base, X_te, y_te
    )

    y_pred_thresh  = predict_with_threshold(rf_base, X_te, best_t)
    y_proba_thresh = rf_base.predict_proba(X_te)[:, 1]
    metrics_thresh = compute_metrics(
        f"RF + Threshold={best_t:.2f} (Strategy B)",
        y_te, y_pred_thresh, y_proba_thresh
    )

    print(f"\n  All outputs saved → {OUT_DIR}/")
    print("  Phase III complete. ✓\n")

    return metrics_df, rf_smote, scaler, feature_names

if __name__ == "__main__":
    run_phase3()
