"""Shared schema and evaluation rules for training and the demonstration app."""
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
    "ShapeFactor3", "ShapeFactor4",
]
TARGET = "Class"
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
MODEL_FILES = OrderedDict([
    ("Logistic Regression", "logistic_regression.joblib"),
    ("Decision Tree", "decision_tree.joblib"),
    ("KNN", "knn.joblib"),
    ("Naive Bayes", "naive_bayes.joblib"),
    ("Support Vector Machine", "svm.joblib"),
])


def validate_test_frame(frame):
    """Require labelled, finite numeric data with the original dataset schema."""
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")
    required = FEATURES + [TARGET]
    missing = [name for name in required if name not in frame.columns]
    extra = [name for name in frame.columns if name not in required]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if extra:
        raise ValueError("Unexpected columns: " + ", ".join(extra))
    try:
        features = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Every input feature must contain numeric values.") from exc
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Input features must not contain blanks, NaN or infinity.")
    labels = frame[TARGET]
    if labels.isna().any() or not labels.isin(CLASSES).all():
        raise ValueError("Class must use these exact labels: " + ", ".join(CLASSES))
    return features, labels


def evaluate_model(model, features, labels):
    """Macro metrics cover all seven classes; AUC needs every class present."""
    predicted = model.predict(features)
    classes = model.classes_
    auc = None
    if set(labels) == set(classes):
        auc = float(roc_auc_score(labels, model.predict_proba(features),
                                  labels=classes, multi_class="ovr", average="macro"))
    return {
        "Accuracy": float(accuracy_score(labels, predicted)),
        "AUC": auc,
        "Precision": float(precision_score(labels, predicted, labels=classes,
                                            average="macro", zero_division=0)),
        "Recall": float(recall_score(labels, predicted, labels=classes,
                                      average="macro", zero_division=0)),
        "F1": float(f1_score(labels, predicted, labels=classes,
                              average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, predicted)),
    }

