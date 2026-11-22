"""
Módulo B - Entrenamiento y Validación de Clasificadores Citológicos (SIPaKMeD)
Compara múltiples algoritmos supervisados sobre los descriptores morfométricos,
colorimétricos (CIELAB) y de textura (GLCM), exporta el mejor pipeline entrenado
y genera la matriz de confusión junto con la importancia de biomarcadores.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_features(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Carga la matriz de características y codifica las 5 categorías celulares."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró {csv_path}. Ejecuta primero src/cytology/features.py."
        )

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("image_id", "class")]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    le = LabelEncoder()
    y = le.fit_transform(df["class"])
    return X, y, le


def build_candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Define los pipelines de clasificación candidatos."""
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=4,
                random_state=random_state,
            )),
        ]),
        "SVM (RBF)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=3.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }


def compute_screening_triage_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    """
    Calcula sensibilidad y especificidad de tamizaje binario:
    - Normal / Benigno: Superficial-Intermediate, Parabasal, Metaplastic
    - Atipia / Lesión (LSIL/HSIL): Koilocytotic, Dyskeratotic
    """
    abnormal_classes = {"Koilocytotic", "Dyskeratotic"}
    abnormal_idx = {i for i, name in enumerate(class_names) if name in abnormal_classes}

    true_bin = np.array([1 if y in abnormal_idx else 0 for y in y_true])
    pred_bin = np.array([1 if y in abnormal_idx else 0 for y in y_pred])

    tp = int(((true_bin == 1) & (pred_bin == 1)).sum())
    tn = int(((true_bin == 0) & (pred_bin == 0)).sum())
    fp = int(((true_bin == 0) & (pred_bin == 1)).sum())
    fn = int(((true_bin == 1) & (pred_bin == 0)).sum())

    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {"screening_sensitivity": sensitivity, "screening_specificity": specificity}


def plot_evaluation_summary(
    cm: np.ndarray,
    class_names: list[str],
    feature_names: list[str],
    importances: np.ndarray | None,
    best_model_name: str,
    out_path: Path,
) -> None:
    """Genera figura combinada: Matriz de Confusión normalizada + Top 15 Biomarcadores."""
    short_names = [c.replace("Superficial-Intermediate", "Sup-Interm") for c in class_names]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    ncols = 2 if importances is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(14 if ncols == 2 else 7, 5.8))
    ax_cm = axes[0] if ncols == 2 else axes

    im = ax_cm.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_cm.set(
        xticks=np.arange(len(short_names)),
        yticks=np.arange(len(short_names)),
        xticklabels=short_names,
        yticklabels=short_names,
        ylabel="Clase Real (SIPaKMeD)",
        xlabel="Clase Predicha",
        title=f"Matriz de Confusión - {best_model_name}",
    )
    plt.setp(ax_cm.get_xticklabels(), rotation=25, ha="right")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax_cm.text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center", color=color, fontsize=8.5
            )

    if importances is not None:
        ax_imp = axes[1]
        top_k = min(15, len(feature_names))
        indices = np.argsort(importances)[-top_k:]
        top_names = [feature_names[i] for i in indices]
        top_vals = importances[indices]

        ax_imp.barh(range(top_k), top_vals, color="#2B5C8F", alpha=0.88)
        ax_imp.set_yticks(range(top_k))
        ax_imp.set_yticklabels(top_names, fontsize=9)
        ax_imp.set_xlabel("Importancia Relativa (Gini / Impureza)")
        ax_imp.set_title("Top 15 Biomarcadores Diagnósticos", fontweight="bold")
        ax_imp.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def train_and_evaluate(
    features_csv: Path, results_dir: Path, models_dir: Path
) -> pd.DataFrame:
    """Entrena, valida y exporta el mejor clasificador citológico."""
    X, y, le = load_features(features_csv)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_candidate_models(random_state=42)

    comparison_rows = []
    best_model_name = ""
    best_f1 = -1.0
    best_pipeline = None
    best_preds = None

    for name, pipe in models.items():
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        triage = compute_screening_triage_metrics(y_test, y_pred, class_names)

        comparison_rows.append({
            "model": name,
            "cv_f1_macro_mean": round(float(cv_f1.mean()), 4),
            "cv_f1_macro_std": round(float(cv_f1.std()), 4),
            "test_accuracy": round(float(acc), 4),
            "test_f1_macro": round(float(f1_macro), 4),
            "test_roc_auc_ovr": round(float(roc_auc), 4),
            "triage_sensitivity": round(triage["screening_sensitivity"], 4),
            "triage_specificity": round(triage["screening_specificity"], 4),
        })

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_pipeline = pipe
            best_preds = y_pred

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    comp_df = pd.DataFrame(comparison_rows).sort_values("test_f1_macro", ascending=False)
    comp_df.to_csv(results_dir / "cytology_model_comparison.csv", index=False)

    # Extrae importancias del modelo ganador (o usa Random Forest de referencia si gana SVM/LR)
    clf_step = best_pipeline.named_steps["clf"]
    if hasattr(clf_step, "feature_importances_"):
        importances = clf_step.feature_importances_
    else:
        rf_ref = models["Random Forest"].named_steps["clf"]
        importances = getattr(rf_ref, "feature_importances_", None)

    cm = confusion_matrix(y_test, best_preds)
    plot_evaluation_summary(
        cm=cm,
        class_names=class_names,
        feature_names=list(X.columns),
        importances=importances,
        best_model_name=best_model_name,
        out_path=results_dir / "cytology_confusion_matrix.png",
    )

    artifact = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "label_encoder": le,
        "feature_names": list(X.columns),
        "class_names": class_names,
    }
    joblib.dump(artifact, models_dir / "cytology_classifier.joblib")

    print("\n=== Comparativa de Modelos (SIPaKMeD - Test Set 20%) ===")
    print(comp_df.to_string(index=False))
    print(f"\nMejor modelo seleccionado: {best_model_name} (F1-Macro = {best_f1:.4f})")
    print("\nReporte detallado por categoría celular:")
    print(classification_report(y_test, best_preds, target_names=class_names, digits=3))
    print("Modelo guardado en: models/cytology_classifier.joblib")
    print("Gráfica guardada en: results/cytology_confusion_matrix.png")

    return comp_df


if __name__ == "__main__":
    csv_file = ROOT_DIR / "results" / "cytology_features.csv"
    out_results = ROOT_DIR / "results"
    out_models = ROOT_DIR / "models"

    print("Entrenando y evaluando clasificadores sobre biomarcadores celulares...")
    train_and_evaluate(csv_file, out_results, out_models)