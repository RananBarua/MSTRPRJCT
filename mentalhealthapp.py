# app_with_dropdowns.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import re

st.set_page_config(page_title="Mental Health Risk Predictor", page_icon="🧠", layout="wide")

# =========================
# Artifact loading (cached)
# =========================
@st.cache_resource
def load_artifacts():
    files = {
        "model": "lgbm_model.joblib",
        "label_encoder": "label_encoder.joblib",
        "x_sample": "X_sample.joblib",
    }
    art = {}
    for k, fname in files.items():
        if not os.path.exists(fname):
            st.error(f"Missing {fname}. Please save it from your notebook.")
            st.stop()
        art[k] = joblib.load(fname)

    # Optional artifacts
    art["explainer"] = joblib.load("shap_explainer.joblib") if os.path.exists("shap_explainer.joblib") else None
    art["feature_names"] = joblib.load("feature_names.joblib") if os.path.exists("feature_names.joblib") else None
    art["class_names"] = joblib.load("class_names.joblib") if os.path.exists("class_names.joblib") else None
    art["preprocessor"] = joblib.load("preprocessor.joblib") if os.path.exists("preprocessor.joblib") else None
    art["ui_value_maps"] = joblib.load("ui_value_maps.joblib") if os.path.exists("ui_value_maps.joblib") else {}

    # Ensure X_sample is a DataFrame
    if isinstance(art["x_sample"], np.ndarray):
        if art["feature_names"] is None:
            st.error("X_sample.joblib is a numpy array but feature_names.joblib is missing.")
            st.stop()
        art["x_sample"] = pd.DataFrame(art["x_sample"], columns=art["feature_names"])
    return art

art = load_artifacts()
model = art["model"]
le_y = art["label_encoder"]
X_sample = art["x_sample"].copy()
explainer = art["explainer"]
feature_cols = art["feature_names"] or list(X_sample.columns)
class_names_art = art["class_names"] or list(le_y.classes_) if art.get("class_names") or le_y is not None else None
preprocessor = art["preprocessor"]
ui_value_maps = art["ui_value_maps"] or {}

# =========================
# Discover one-hot groups
# =========================
country_cols = [c for c in feature_cols if c.startswith("Country_")]
occupation_cols = [c for c in feature_cols if c.startswith("Occupation_")]
skip_cols = set(country_cols + occupation_cols)  # handled via dropdowns instead

def widget_for_column(col: str, series: pd.Series):
    if col in skip_cols:
        return None
    if col in ui_value_maps and ui_value_maps[col]:
        label_value_pairs = ui_value_maps[col]
        labels = [lv[0] for lv in label_value_pairs]
        values = [lv[1] for lv in label_value_pairs]
        default_idx = 0
        return float(values[st.sidebar.selectbox(col, options=list(range(len(labels))), format_func=lambda i: labels[i], index=default_idx)])
    
    s = series.dropna()
    uniq = sorted(s.unique().tolist()) if len(s) else []
    dtype = s.dtype

    if set(uniq).issubset({0, 1}) and len(uniq) <= 2:
        val = st.sidebar.checkbox(f"{col}", value=bool(int(np.median(s) if len(s) else 0)))
        return int(val)
    if np.issubdtype(dtype, np.integer) and 2 < len(uniq) <= 10:
        val = st.sidebar.selectbox(f"{col}", options=uniq, index=0)
        return int(val)
    if np.issubdtype(dtype, np.integer):
        mn = int(np.nanmin(s)) if len(s) else 0
        mx = int(np.nanmax(s)) if len(s) else 100
        default = int(np.median(s)) if len(s) else 0
        return st.sidebar.number_input(f"{col}", value=default, min_value=mn, max_value=max(mn+1, mx), step=1)
    if np.issubdtype(dtype, np.floating):
        mn = float(np.nanmin(s)) if len(s) else 0.0
        mx = float(np.nanmax(s)) if len(s) else 1.0
        default = float(np.median(s)) if len(s) else 0.0
        return st.sidebar.number_input(f"{col}", value=default, min_value=float(mn), max_value=float(max(mn+1e-6, mx)), step=0.1, format="%.4f")
    st.sidebar.warning(f"'{col}' appears non-numeric; using 0.0. Ensure preprocessing matches training.")
    return 0.0

def apply_onehot_choice(base_row: dict, group_cols: list, group_name: str, default_from_sample: bool = True):
    labels = [re.sub(rf"^{group_name}_", "", c) for c in group_cols]
    labels_with_base = ["(Baseline/Other)"] + labels
    default_idx = 0
    if default_from_sample and not X_sample.empty:
        sample_sum = int(X_sample[group_cols].iloc[0].sum())
        if sample_sum == 0:
            default_idx = 0
        else:
            for i, c in enumerate(group_cols, start=1):
                if int(X_sample[c].iloc[0]) == 1:
                    default_idx = i
                    break
    choice = st.sidebar.selectbox(group_name.replace("_", " "), options=list(range(len(labels_with_base))),
                                  format_func=lambda i: labels_with_base[i], index=default_idx)
    for c in group_cols:
        base_row[c] = 0.0
    if choice != 0:
        chosen_col = group_cols[choice - 1]
        base_row[chosen_col] = 1.0
    return base_row

# =========================
# UI Header
# =========================
st.title("🧠 Mental Health Risk Prediction")
st.caption("Predict risk and explain with SHAP • Dropdowns for Country/Occupation avoid one-hot mistakes.")

with st.expander("Artifacts & assumptions", expanded=False):
    st.markdown("""
    - X_sample.joblib is the **post-transform** numeric matrix (same columns & order as training).
    - Optional: ui_value_maps.joblib lets you present friendly labels (e.g., **'1–3 days'**) and convert to the numeric codes your model expects.
    - One-hot features (e.g., **Country**, **Occupation**) are handled by single dropdowns to avoid multi-select mistakes.
    """)

# =========================
# Build input row
# =========================
st.sidebar.header("🔧 Input Features")
user_values = {}
for col in feature_cols:
    if col not in X_sample.columns:
        user_values[col] = 0.0
    else:
        val = widget_for_column(col, X_sample[col])
        if val is not None:
            user_values[col] = val

if country_cols:
    user_values = apply_onehot_choice(user_values, country_cols, "Country")
if occupation_cols:
    user_values = apply_onehot_choice(user_values, occupation_cols, "Occupation")

raw_input_df = pd.DataFrame([user_values], columns=feature_cols)
st.subheader("📋 Your Input (pre-transform)")
st.dataframe(raw_input_df, use_container_width=True)

# =========================
# Preprocess (if provided)
# =========================
def apply_preprocess(df):
    if preprocessor is None:
        return df[feature_cols]
    X_num = preprocessor.transform(df)
    if isinstance(X_num, np.ndarray):
        cols = feature_cols
        if X_num.shape[1] != len(cols):
            st.error("Preprocessor output shape != expected feature count.")
            st.stop()
        return pd.DataFrame(X_num, columns=cols)
    return X_num

# =========================
# Predict helpers
# =========================
def predict_proba_safe(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)
    if hasattr(m, "decision_function"):
        scores = m.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    pred = m.predict(X)
    probs = np.zeros((len(pred), 3), dtype=float)
    probs[np.arange(len(pred)), pred] = 1.0
    return probs

def ensure_explainer(work_explainer, model, background_df):
    if work_explainer is not None:
        try:
            _ = work_explainer.shap_values(background_df.head(1))
            return work_explainer
        except Exception:
            pass
    return shap.TreeExplainer(model, data=background_df)

def pick_vector(V, cls_idx, n_features, n_classes):
    if hasattr(V, "values"):
        V = V.values
    if isinstance(V, list):
        return np.array(V[cls_idx][0])
    if not isinstance(V, np.ndarray):
        raise ValueError("Unsupported SHAP container type.")
    if V.ndim == 1:
        if V.shape[0] == n_features: return V
        raise ValueError(f"1D SHAP shape {V.shape} doesn't match n_features={n_features}")
    if V.ndim == 2:
        if V.shape == (1, n_features): return V[0, :]
        if V.shape == (n_classes, n_features): return V[cls_idx, :]
        if V.shape == (n_features, n_classes): return V[:, cls_idx]
        raise ValueError(f"Unexpected 2D SHAP shape: {V.shape}")
    if V.ndim == 3:
        s0, s1, s2 = V.shape
        if s0 != 1: raise ValueError(f"Unexpected 3D SHAP shape (n_samples != 1): {V.shape}")
        if s1 == n_classes and s2 == n_features: return V[0, cls_idx, :]
        if s1 == n_features and s2 == n_classes: return V[0, :, cls_idx]
        raise ValueError(f"Unexpected 3D SHAP shape: {V.shape}")
    raise ValueError(f"Unsupported SHAP array with ndim={V.ndim}")

# =========================
# Predict button
# =========================
predict_btn = st.button("🚀 Predict")

if predict_btn:
    with st.spinner("Preparing input..."):
        X_for_model = apply_preprocess(raw_input_df)

    st.subheader("🔢 Model Input (post-transform)")
    st.dataframe(X_for_model, use_container_width=True)

    with st.spinner("Running model..."):
        probs = predict_proba_safe(model, X_for_model)
        class_names = ['Low', 'Medium', 'High']
        pred_idx = int(np.argmax(probs[0]))
        pred_label = class_names[pred_idx]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Predicted Risk Level", f"{pred_label}")
    with c2:
        st.write("**Class Probabilities (Low → Medium → High)**")
        prob_df = pd.DataFrame([probs[0]], columns=class_names)
        st.bar_chart(prob_df.T)

    # =========================
    # Friendly advice messages
    # =========================
    highest_prob = probs[0][pred_idx]
    if pred_label == "Low":
        st.success(f"✅ Low Risk ({highest_prob:.2f} probability)\n\nYou seem to be doing well. Keep taking care of yourself!")
    elif pred_label == "Medium":
        st.warning(f"⚠️ Medium Risk ({highest_prob:.2f} probability)\n\nBe mindful of your mental health. Consider self-care and monitoring your stress.")
    else:
        st.error(f"❌ High Risk ({highest_prob:.2f} probability)\n\nCaution! Consider reaching out to a professional or support network.")

    st.divider()
    st.subheader("🔍 SHAP Explanation (Top 10 features for predicted class)")

    explainer = ensure_explainer(explainer, model, X_sample[feature_cols])

    try:
        shap_vals = explainer.shap_values(X_for_model[feature_cols])
        sv = pick_vector(shap_vals, pred_idx, n_features=len(feature_cols), n_classes=len(class_names))
        contrib = pd.Series(sv, index=feature_cols)
        topk = contrib.abs().sort_values(ascending=False).head(10).index
        plot_df = pd.DataFrame({"Feature": topk, "Contribution": contrib[topk].values}).sort_values("Contribution")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ["#C62828" if v > 0 else "#2E7D32" for v in plot_df["Contribution"].values]
        ax.barh(plot_df["Feature"].values, plot_df["Contribution"].values, color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("SHAP value (impact on log-odds/prob)")
        ax.set_ylabel("")
        ax.set_title("Top 10 feature contributions")
        st.pyplot(fig)

    except Exception as e:
        st.warning(
            "Could not render SHAP explanation.\n\n"
            f"Details: {e}\n\n"
            "Tips:\n"
            "- Ensure X_sample.joblib columns & order match the trained matrix.\n"
            "- If you changed models, re-save shap_explainer.joblib or delete it to rebuild automatically."
        )
