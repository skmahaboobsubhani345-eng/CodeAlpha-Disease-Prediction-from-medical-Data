"""
app.py — Streamlit interactive disease prediction application.

A premium, dark-themed web interface where users can:
1. Select a disease type (Heart Disease, Diabetes, Breast Cancer)
2. Enter patient data via intuitive form inputs
3. Get prediction, probability, and explainable feature contributions
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd

from src.predict import load_model, predict_from_array, DATASET_FEATURES

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Disease Prediction AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #a0aec0;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* Result cards */
    .prediction-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .prediction-positive {
        border-left: 4px solid #e74c3c;
    }
    .prediction-negative {
        border-left: 4px solid #2ecc71;
    }

    .result-label {
        color: #a0aec0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.3rem;
    }
    .result-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-positive { color: #e74c3c; }
    .result-negative { color: #2ecc71; }

    /* Probability bar */
    .prob-container {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .prob-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        height: 14px;
        overflow: hidden;
        margin-top: 0.8rem;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s ease;
    }

    /* Explanation cards */
    .explanation-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .explanation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .feature-name {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .feature-detail {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    .impact-up { color: #e74c3c; font-weight: 600; }
    .impact-down { color: #2ecc71; font-weight: 600; }

    /* Dataset info badge */
    .dataset-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Sidebar styling */
    .sidebar-info {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-info h4 {
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .sidebar-info p {
        color: #a0aec0;
        font-size: 0.85rem;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_disease_emoji(dataset_name: str) -> str:
    return {
        "heart_disease": "❤️",
        "diabetes": "🩸",
        "breast_cancer": "🎗️",
    }.get(dataset_name, "🏥")


def get_risk_color(probability: float) -> str:
    if probability >= 0.7:
        return "#e74c3c"
    elif probability >= 0.4:
        return "#f39c12"
    else:
        return "#2ecc71"


def render_probability_bar(probability: float, label: str = "Risk Probability") -> None:
    color = get_risk_color(probability)
    pct = probability * 100
    st.markdown(f"""
    <div class="prob-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="result-label">{label}</span>
            <span style="color: {color}; font-size: 1.4rem; font-weight: 700;">{pct:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width: {pct}%; background: linear-gradient(90deg, {color}88, {color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Form input definitions
# ---------------------------------------------------------------------------

def get_heart_disease_inputs() -> dict:
    """Render input form for Heart Disease and return values."""
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, help="Patient's age in years")
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"][x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                           format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=600, value=240)
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise-Induced Angina", options=[0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No")

    with col3:
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("ST Slope", options=[0, 1, 2],
                             format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("Major Vessels (fluoroscopy)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=[1, 2, 3],
                            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x - 1])

    return {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }


def get_diabetes_inputs() -> dict:
    """Render input form for Diabetes and return values."""
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, help="Number of pregnancies")
        glucose = st.number_input("Glucose (mg/dl)", min_value=0, max_value=300, value=120, help="Plasma glucose concentration")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, help="Diastolic blood pressure")
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness")

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80, help="2-hour serum insulin")
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, help="Genetic risk score")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, key="diabetes_age")

    return {
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
        "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }


def get_breast_cancer_inputs() -> dict:
    """Render input form for Breast Cancer and return values."""
    st.markdown("##### Mean Values")
    col1, col2, col3 = st.columns(3)

    with col1:
        mean_radius = st.number_input("Mean Radius", value=14.0, step=0.1, format="%.2f")
        mean_texture = st.number_input("Mean Texture", value=19.0, step=0.1, format="%.2f")
        mean_perimeter = st.number_input("Mean Perimeter", value=92.0, step=0.1, format="%.2f")
        mean_area = st.number_input("Mean Area", value=655.0, step=1.0, format="%.1f")
    with col2:
        mean_smoothness = st.number_input("Mean Smoothness", value=0.096, step=0.001, format="%.4f")
        mean_compactness = st.number_input("Mean Compactness", value=0.104, step=0.001, format="%.4f")
        mean_concavity = st.number_input("Mean Concavity", value=0.089, step=0.001, format="%.4f")
        mean_concave_points = st.number_input("Mean Concave Points", value=0.049, step=0.001, format="%.4f")
    with col3:
        mean_symmetry = st.number_input("Mean Symmetry", value=0.181, step=0.001, format="%.4f")
        mean_fractal = st.number_input("Mean Fractal Dim.", value=0.063, step=0.001, format="%.4f")

    st.markdown("##### Standard Error Values")
    col4, col5, col6 = st.columns(3)
    with col4:
        se_radius = st.number_input("Radius Error", value=0.405, step=0.01, format="%.3f")
        se_texture = st.number_input("Texture Error", value=1.22, step=0.01, format="%.3f")
        se_perimeter = st.number_input("Perimeter Error", value=2.87, step=0.01, format="%.3f")
        se_area = st.number_input("Area Error", value=40.3, step=0.1, format="%.1f")
    with col5:
        se_smoothness = st.number_input("Smoothness Error", value=0.007, step=0.001, format="%.4f")
        se_compactness = st.number_input("Compactness Error", value=0.025, step=0.001, format="%.4f")
        se_concavity = st.number_input("Concavity Error", value=0.032, step=0.001, format="%.4f")
        se_concave_points = st.number_input("Concave Points Err.", value=0.012, step=0.001, format="%.4f")
    with col6:
        se_symmetry = st.number_input("Symmetry Error", value=0.021, step=0.001, format="%.4f")
        se_fractal = st.number_input("Fractal Dim. Error", value=0.004, step=0.001, format="%.4f")

    st.markdown("##### Worst (Largest) Values")
    col7, col8, col9 = st.columns(3)
    with col7:
        worst_radius = st.number_input("Worst Radius", value=16.3, step=0.1, format="%.2f")
        worst_texture = st.number_input("Worst Texture", value=25.7, step=0.1, format="%.2f")
        worst_perimeter = st.number_input("Worst Perimeter", value=107.0, step=0.1, format="%.2f")
        worst_area = st.number_input("Worst Area", value=881.0, step=1.0, format="%.1f")
    with col8:
        worst_smoothness = st.number_input("Worst Smoothness", value=0.132, step=0.001, format="%.4f")
        worst_compactness = st.number_input("Worst Compactness", value=0.254, step=0.001, format="%.4f")
        worst_concavity = st.number_input("Worst Concavity", value=0.272, step=0.001, format="%.4f")
        worst_concave_points = st.number_input("Worst Concave Pts.", value=0.115, step=0.001, format="%.4f")
    with col9:
        worst_symmetry = st.number_input("Worst Symmetry", value=0.290, step=0.001, format="%.4f")
        worst_fractal = st.number_input("Worst Fractal Dim.", value=0.084, step=0.001, format="%.4f")

    return {
        "mean radius": mean_radius, "mean texture": mean_texture,
        "mean perimeter": mean_perimeter, "mean area": mean_area,
        "mean smoothness": mean_smoothness, "mean compactness": mean_compactness,
        "mean concavity": mean_concavity, "mean concave points": mean_concave_points,
        "mean symmetry": mean_symmetry, "mean fractal dimension": mean_fractal,
        "radius error": se_radius, "texture error": se_texture,
        "perimeter error": se_perimeter, "area error": se_area,
        "smoothness error": se_smoothness, "compactness error": se_compactness,
        "concavity error": se_concavity, "concave points error": se_concave_points,
        "symmetry error": se_symmetry, "fractal dimension error": se_fractal,
        "worst radius": worst_radius, "worst texture": worst_texture,
        "worst perimeter": worst_perimeter, "worst area": worst_area,
        "worst smoothness": worst_smoothness, "worst compactness": worst_compactness,
        "worst concavity": worst_concavity, "worst concave points": worst_concave_points,
        "worst symmetry": worst_symmetry, "worst fractal dimension": worst_fractal,
    }


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1>🏥 Disease Prediction AI</h1>
        <p>Interpretable machine learning predictions for Heart Disease, Diabetes & Breast Cancer</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔬 Disease Selection")

        dataset_name = st.selectbox(
            "Choose Disease Type",
            options=["heart_disease", "diabetes", "breast_cancer"],
            format_func=lambda x: {
                "heart_disease": "❤️ Heart Disease",
                "diabetes": "🩸 Diabetes",
                "breast_cancer": "🎗️ Breast Cancer",
            }[x],
            label_visibility="collapsed",
        )

        emoji = get_disease_emoji(dataset_name)
        display_name = dataset_name.replace("_", " ").title()

        st.markdown(f"""
        <div class="sidebar-info">
            <h4>{emoji} {display_name}</h4>
            <p>Enter patient data in the form and click <strong>Predict</strong> to get an AI-powered risk assessment with explainable factors.</p>
        </div>
        """, unsafe_allow_html=True)

        # Model info
        try:
            bundle = load_model(dataset_name)
            model_name = bundle["best_model_name"]
            metrics = bundle.get("metrics", {})

            st.markdown(f"""
            <div class="sidebar-info">
                <h4>📊 Model Info</h4>
                <p><strong>Algorithm:</strong> {model_name}</p>
                <p><strong>Accuracy:</strong> {metrics.get('accuracy', 0):.1%}</p>
                <p><strong>F1 Score:</strong> {metrics.get('f1', 0):.1%}</p>
                <p><strong>ROC-AUC:</strong> {metrics.get('roc_auc', 0):.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning("⚠️ No trained model found. Run `python -m src.train` first.")
            return

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-info">
            <h4>⚠️ Disclaimer</h4>
            <p>This tool is for <strong>educational purposes only</strong>.
            It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content: Input Form
    st.markdown(f"### {emoji} Enter Patient Data — {display_name}")

    # Render appropriate form
    if dataset_name == "heart_disease":
        patient_data = get_heart_disease_inputs()
    elif dataset_name == "diabetes":
        patient_data = get_diabetes_inputs()
    else:
        patient_data = get_breast_cancer_inputs()

    # Predict button
    st.markdown("")
    predict_clicked = st.button(
        "🔍  Analyze & Predict",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:
        with st.spinner("Analyzing patient data..."):
            try:
                # Get feature names in correct order
                ds_features = DATASET_FEATURES[dataset_name]["names"]
                features = np.array([
                    patient_data.get(name, 0.0) for name in ds_features
                ], dtype=np.float64)

                result = predict_from_array(dataset_name, features)

                # Results section
                st.markdown("---")
                st.markdown("### 📋 Prediction Results")

                col_result, col_prob = st.columns([1, 1])

                with col_result:
                    is_positive = result["prediction"] == 1
                    card_class = "prediction-positive" if is_positive else "prediction-negative"
                    result_class = "result-positive" if is_positive else "result-negative"
                    icon = "⚠️" if is_positive else "✅"

                    st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <div class="result-label">Prediction</div>
                        <div class="result-value {result_class}">
                            {icon} {result['prediction_label']}
                        </div>
                        <div style="color: #a0aec0; font-size: 0.9rem;">
                            Model: {result['model_name']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_prob:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="result-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                    render_probability_bar(result["probability"], "Disease Risk Probability")

                # Explanations
                st.markdown("### 🔍 Key Contributing Factors")
                st.markdown("*Understanding what drives this prediction — ranked by impact:*")

                for i, exp in enumerate(result["explanations"]):
                    impact_class = "impact-up" if exp["direction"] == "increases" else "impact-down"
                    arrow = "↑" if exp["direction"] == "increases" else "↓"
                    rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}."

                    st.markdown(f"""
                    <div class="explanation-card">
                        <div style="display: flex; align-items: center; gap: 0.8rem;">
                            <span style="font-size: 1.3rem;">{rank_emoji}</span>
                            <div style="flex: 1;">
                                <div class="feature-name">{exp['description']}</div>
                                <div class="feature-detail">
                                    Value: <strong>{exp['value']:.3f}</strong> &nbsp;|&nbsp;
                                    Impact: <span class="{impact_class}">{arrow} {exp['direction']} risk by {exp['importance']:.4f}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Input summary
                with st.expander("📊 View Input Data Summary"):
                    input_df = pd.DataFrame([patient_data])
                    st.dataframe(input_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
