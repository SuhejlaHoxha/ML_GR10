
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    RandomizedSearchCV, cross_validate, learning_curve
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, label_binarize
)
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier, export_text
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TARGET_COL  = "actor_is_bot"
# ─────────────────────────────────────────────────────────────────────


# ─── Helpers ─────────────────────────────────────────────────────────

def load_and_prepare(csv_path: str):
    """Load processed CSV, encode remaining object columns, return X, y."""
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(df.median(numeric_only=True))

    leakage = ["actor_bot_ratio"]   
    df = df.drop(columns=[c for c in leakage if c in df.columns])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y, df


def build_splits(X, y):
    """Stratified 80/20 train-test split."""
    return train_test_split(X, y, test_size=0.20,
                            stratify=y, random_state=RANDOM_SEED)


def evaluate_model(name, model, X_train, X_test, y_train, y_test,
                   cv: StratifiedKFold = None, plots_dir: str = "phase2_plots"):
    """
    Train, evaluate and return a metrics dictionary.
    Saves confusion matrix + ROC curve plots.
    """
    os.makedirs(plots_dir, exist_ok=True)

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = (model.predict_proba(X_test)[:, 1]
               if hasattr(model, "predict_proba") else None)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

    # Cross-validation
    cv_scores = {}
    if cv is not None:
        cv_res = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["accuracy", "f1", "roc_auc"],
            return_train_score=True
        )
        cv_scores = {
            "cv_acc_mean"   : cv_res["test_accuracy"].mean(),
            "cv_acc_std"    : cv_res["test_accuracy"].std(),
            "cv_f1_mean"    : cv_res["test_f1"].mean(),
            "cv_f1_std"     : cv_res["test_f1"].std(),
            "cv_roc_mean"   : cv_res["test_roc_auc"].mean(),
            "cv_roc_std"    : cv_res["test_roc_auc"].std(),
        }

    # ── Confusion matrix plot ──────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Human", "Bot"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, f"cm_{name.replace(' ', '_')}.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()

    # ── ROC curve ─────────────────────────────────────────────────
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve – {name}")
        ax.legend()
        plt.tight_layout()
        roc_path = os.path.join(plots_dir,
                                f"roc_{name.replace(' ', '_')}.png")
        plt.savefig(roc_path, dpi=120)
        plt.close()

    metrics = {
        "Model"     : name,
        "Accuracy"  : round(acc,  4),
        "Precision" : round(prec, 4),
        "Recall"    : round(rec,  4),
        "F1"        : round(f1,   4),
        "ROC-AUC"   : round(roc,  4) if not np.isnan(roc) else "N/A",
        "CM"        : cm,
        **{k: round(v, 4) for k, v in cv_scores.items()},
    }
    return metrics, model


def plot_feature_importance(name, model, feature_names, plots_dir, top_n=15):
    """Bar chart of feature importances or absolute coefficients."""
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        return

    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(idx)),
           importances[idx],
           color=sns.color_palette("viridis", len(idx)))
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([feature_names[i] for i in idx],
                       rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Feature Importance – {name}")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    path = os.path.join(plots_dir,
                        f"fi_{name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()


def plot_learning_curve(name, model, X, y, cv, plots_dir):
    """Training vs validation learning curve."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_mean, "o-", color="royalblue",
            label="Training F1")
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color="royalblue")
    ax.plot(train_sizes, val_mean, "o-", color="tomato",
            label="Validation F1")
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color="tomato")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Learning Curve – {name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(plots_dir,
                        f"lc_{name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()


def plot_all_roc(results_list, X_test, y_test, plots_dir):
    """Overlay ROC curves for all models on one chart."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("tab10", len(results_list))

    for (name, model), color in zip(results_list, colors):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            proba = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{name} (AUC={roc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_all_models.png"), dpi=130)
    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame, plots_dir: str):
    """Grouped bar chart comparing all models across key metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    df_plot = metrics_df[["Model"] + metrics].copy()
    for m in metrics:
        df_plot[m] = pd.to_numeric(df_plot[m], errors="coerce")

    df_melted = df_plot.melt(id_vars="Model", var_name="Metric",
                             value_name="Score")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_melted, x="Metric", y="Score",
                hue="Model", ax=ax, palette="tab10")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison – Key Metrics")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_comparison.png"), dpi=130)
    plt.close()


# ─── Model definitions & hyperparameter grids ─────────────────────────

def get_models_and_grids(X_train):
    scaler = StandardScaler()
    X_scaled_check = scaler.fit_transform(X_train)

    models = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED,
                                             class_weight="balanced"),
            "param_grid": {
                "C"      : [0.01, 0.1, 1, 10],
                "solver" : ["lbfgs", "liblinear"],
                "penalty": ["l2"],
            },
            "needs_scaling": True,
            "search_type": "grid",
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_SEED,
                                                class_weight="balanced"),
            "param_grid": {
                "max_depth"       : [3, 5, 10, None],
                "min_samples_leaf": [1, 5, 10],
                "criterion"       : ["gini", "entropy"],
            },
            "needs_scaling": False,
            "search_type": "grid",
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_SEED,
                                                 class_weight="balanced",
                                                 n_jobs=-1),
            "param_grid": {
                "n_estimators" : [100, 200],
                "max_depth"    : [5, 10, None],
                "max_features" : ["sqrt", "log2"],
            },
            "needs_scaling": False,
            "search_type": "random",
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_SEED),
            "param_grid": {
                "n_estimators"  : [100, 200],
                "learning_rate" : [0.05, 0.1, 0.2],
                "max_depth"     : [3, 5],
                "subsample"     : [0.8, 1.0],
            },
            "needs_scaling": False,
            "search_type": "random",
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=RANDOM_SEED,
                             class_weight="balanced", max_iter=2000),
            "param_grid": {
                "C"      : [1, 10],
                "kernel" : ["rbf"],
                "gamma"  : ["scale"],
            },
            "needs_scaling": True,
            "search_type": "grid",
            "subsample": 2000,   # use a subsample for speed
        },
        "K-Nearest Neighbours": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [5, 11],
                "weights"    : ["uniform", "distance"],
                "metric"     : ["euclidean"],
            },
            "needs_scaling": True,
            "search_type": "grid",
        },
    }
    return models

# ─── Main runner ─────────────────────────────────────────────────────

def run_supervised(csv_path: str, plots_dir: str = "phase2_plots") -> pd.DataFrame:
    """
    Full supervised learning pipeline.
    Returns a DataFrame of model metrics.
    """
    os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "=" * 68)
    print("  PHASE II · Supervised Learning")
    print("=" * 68)

    # ── Load data
    X, y, df = load_and_prepare(csv_path)
    print(f"\n  Dataset  : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Target   : {y.value_counts().to_dict()}")

    # ── Splits
    X_train, X_test, y_train, y_test = build_splits(X, y)
    print(f"  Train    : {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Scalers 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    feature_names = list(X.columns)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    model_defs = get_models_and_grids(X_train)

    all_metrics   = []
    trained_models = []   # [(name, fitted_model)]

    for name, cfg in model_defs.items():
        print(f"\n  {'─' * 60}")
        print(f"  Training: {name}")

        estimator = cfg["estimator"]
        grid      = cfg["param_grid"]
        scaling   = cfg["needs_scaling"]
        stype     = cfg["search_type"]

        Xtr = X_train_scaled if scaling else X_train.values
        Xte = X_test_scaled  if scaling else X_test.values

        # Subsample for slow algorithms (SVM)
        subsample = cfg.get("subsample", None)
        Xtr_fit, ytr_fit = Xtr, y_train
        if subsample and len(Xtr) > subsample:
            rng = np.random.RandomState(RANDOM_SEED)
            idx = rng.choice(len(Xtr), subsample, replace=False)
            Xtr_fit = Xtr[idx]
            ytr_fit = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
            print(f"    (subsampled to {subsample} for tuning speed)")


        if stype == "grid":
            searcher = GridSearchCV(
                estimator, grid, cv=cv5,
                scoring="f1", n_jobs=-1, refit=True, verbose=0
            )
        else:
            searcher = RandomizedSearchCV(
                estimator, grid, n_iter=20, cv=cv5,
                scoring="f1", n_jobs=-1, refit=True,
                random_state=RANDOM_SEED, verbose=0
            )

        searcher.fit(Xtr_fit, ytr_fit)
        best_model = searcher.best_estimator_

        best_model.fit(Xtr, y_train)
        print(f"    Best params : {searcher.best_params_}")
        print(f"    Best CV F1  : {searcher.best_score_:.4f}")

        metrics, fitted = evaluate_model(
            name, best_model, Xtr, Xte, y_train, y_test,
            cv=cv5, plots_dir=plots_dir
        )
        all_metrics.append(metrics)
        trained_models.append((name, fitted))

        plot_feature_importance(name, fitted, feature_names, plots_dir)

        try:
            plot_learning_curve(name, best_model, Xtr, y_train,
                                cv5, plots_dir)
        except Exception as e:
            print(f"    ⚠ Learning curve skipped: {e}")

       
        Yte_pred = fitted.predict(Xte)
        print(f"\n    Classification Report:\n")
        print(classification_report(y_test, Yte_pred,
                                    target_names=["Human", "Bot"],
                                    digits=4))

    metrics_df = pd.DataFrame([
        {k: v for k, v in m.items() if k != "CM"}
        for m in all_metrics
    ])
    plot_all_roc(trained_models, X_test_scaled, y_test, plots_dir)
    plot_metrics_comparison(metrics_df, plots_dir)

    metrics_csv = os.path.join(plots_dir, "supervised_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n  ✓ Metrics saved → {metrics_csv}")

    print("\n  " + "=" * 64)
    print("  SUPERVISED LEARNING SUMMARY")
    print("  " + "=" * 64)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    print(metrics_df[cols].sort_values("F1", ascending=False).to_string(index=False))

    best_row = metrics_df.loc[
        pd.to_numeric(metrics_df["F1"], errors="coerce").idxmax()
    ]
    print(f"\n    Best model by F1: {best_row['Model']}  "
          f"(F1={best_row['F1']})")

    return metrics_df, trained_models, X_test_scaled, y_test


if __name__ == "__main__":
    import sys, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV  = os.path.join(ROOT, "processed dataset", "processed_github.csv")
    OUT  = os.path.join(ROOT, "phase2_plots")
    run_supervised(CSV, OUT)