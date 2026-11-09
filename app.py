"""Evaluate saved Dry Bean classifiers on labelled test data."""
from io import BytesIO
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from common import FEATURES, MODEL_FILES, evaluate_model, validate_test_frame

ROOT = Path(__file__).resolve().parent
MODEL_DIRECTORY = ROOT / "model"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Bean Classifier | Model Workbench", page_icon="🌱", layout="wide")
st.markdown("""
<style>
.stApp { background: #f7f8f4; }
[data-testid="stMetric"] {
    background: white; border: 1px solid #dde4d8;
    padding: 14px 16px; border-radius: 10px;
}
[data-testid="stSidebar"] { background: #eaf0e5; }
.dataset-summary {
    background: linear-gradient(110deg, #ffffff, #edf5eb);
    border: 1px solid #cbdcc8;
    border-left: 5px solid #326846;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 0 0 20px;
    box-shadow: 0 4px 16px rgba(32, 70, 42, 0.06);
}
.dataset-summary__label {
    color: #53725c;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.dataset-summary__title {
    color: #173e27;
    font-size: 1.2rem;
    font-weight: 700;
    margin: 3px 0 16px;
}
.dataset-summary__stats { display: flex; flex-wrap: wrap; gap: 12px 32px; }
.dataset-summary__stat { display: flex; align-items: baseline; gap: 7px; }
.dataset-summary__value { color: #245b37; font-size: 1.35rem; font-weight: 750; }
.dataset-summary__name { color: #596d5d; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_models(file_stamps):
    """Invalidate the cache after model artifacts are replaced."""
    del file_stamps
    return {name: joblib.load(MODEL_DIRECTORY / filename) for name, filename in MODEL_FILES.items()}


def metric_text(value):
    if value is None or not np.isfinite(value):
        return "Unavailable"
    return f"{value:.4f}"


st.title("Dry Bean classification workbench")

with st.sidebar:
    selected_model = st.selectbox("Model", list(MODEL_FILES))
    uploaded_csv = st.file_uploader("Upload test CSV", type=["csv"])

sample_path = ROOT / "test_data.csv"

try:
    if uploaded_csv is None:
        frame = pd.read_csv(sample_path)
        source_label = "Supplied held-out test set"
    else:
        frame = pd.read_csv(BytesIO(uploaded_csv.getvalue()))
        source_label = "Uploaded test set"
    inputs, targets = validate_test_frame(frame)
except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(f"Test CSV could not be evaluated: {exc}")
    st.stop()

try:
    file_stamps = tuple((filename, (MODEL_DIRECTORY / filename).stat().st_mtime_ns)
                        for filename in MODEL_FILES.values())
    models = load_models(file_stamps)
except Exception as exc:
    st.error("The saved models could not be loaded. Keep the model folder beside app.py "
             "and install the versions in requirements.txt.")
    st.caption(f"Details: {exc}")
    st.stop()

st.markdown(f"""
<section class="dataset-summary" aria-label="Test set summary">
    <div class="dataset-summary__label">Evaluation data</div>
    <div class="dataset-summary__title">{source_label}</div>
    <div class="dataset-summary__stats">
        <div class="dataset-summary__stat"><span class="dataset-summary__value">{len(inputs):,}</span><span class="dataset-summary__name">Records</span></div>
        <div class="dataset-summary__stat"><span class="dataset-summary__value">{len(FEATURES)}</span><span class="dataset-summary__name">Features</span></div>
        <div class="dataset-summary__stat"><span class="dataset-summary__value">{targets.nunique()} / {len(models[selected_model].classes_)}</span><span class="dataset-summary__name">Classes present</span></div>
    </div>
</section>
""", unsafe_allow_html=True)

try:
    with st.spinner("Evaluating the saved classifiers..."):
        comparison = pd.DataFrame([
            {"Model": name, **evaluate_model(model, inputs, targets)}
            for name, model in models.items()
        ]).set_index("Model")[METRIC_NAMES]
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

st.subheader(selected_model)
selected_scores = comparison.loc[selected_model]
for column, metric_name in zip(st.columns(6), METRIC_NAMES):
    column.metric(metric_name, metric_text(selected_scores[metric_name]))
st.caption("Precision, recall and F1 use macro averages over all seven trained classes. AUC uses macro one-vs-rest "
           "probabilities. MCC is the multiclass Matthews correlation coefficient. "
           "All values are on a 0–1 scale except MCC, whose range is −1 to 1.")
if pd.isna(selected_scores["AUC"]):
    st.warning("AUC is unavailable because this test set does not contain every trained class.")

st.subheader("Same records, all five models")
st.dataframe(comparison.style.format("{:.4f}", na_rep="Unavailable"), width="stretch")

chosen_pipeline = models[selected_model]
predictions = chosen_pipeline.predict(inputs)
classes = chosen_pipeline.classes_
left, right = st.columns([1.1, 1])
with left:
    st.subheader("Confusion matrix")
    fig, axis = plt.subplots(figsize=(7.4, 6.1))
    ConfusionMatrixDisplay.from_predictions(targets, predictions, labels=classes,
        cmap="Greens", colorbar=False, ax=axis, xticks_rotation=45)
    axis.set_title(selected_model)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
with right:
    st.subheader("Classification report")
    report = pd.DataFrame(classification_report(targets, predictions, labels=classes,
                          output_dict=True, zero_division=0)).T
    st.dataframe(report.style.format("{:.4f}"), width="stretch")

manifest_path = MODEL_DIRECTORY / "manifest.json"
if manifest_path.is_file():
    with st.expander("Training record"):
        try:
            st.json(json.loads(manifest_path.read_text(encoding="utf-8")), expanded=False)
        except (ValueError, OSError):
            st.caption("Training metadata is unavailable. The classifiers were loaded above.")


