# Disease Prediction from Medical Data

A production-quality machine learning project that predicts the likelihood of **Heart Disease**, **Diabetes**, and **Breast Cancer** from structured patient data using classical ML classification models — with full explainability.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)

---

## 🎯 Problem Statement

Early disease detection saves lives. This project demonstrates how classical machine learning can be applied to structured medical datasets to predict disease risk, while maintaining **interpretability** — a critical requirement for healthcare AI.

Each prediction is accompanied by:
- A confidence probability
- Top contributing features with plain-language explanations
- SHAP-based feature importance analysis

---

## 📊 Datasets

| Dataset | Source | Samples | Features | Target |
|---------|--------|---------|----------|--------|
| **Heart Disease** | UCI ML Repository (Cleveland) | 303 | 13 | Presence of heart disease (binary) |
| **Diabetes** | Pima Indians Diabetes (UCI) | 768 | 8 | Onset of diabetes (binary) |
| **Breast Cancer** | UCI Wisconsin Diagnostic (sklearn) | 569 | 30 | Malignant vs Benign (binary) |

All datasets are downloaded programmatically on first run — no manual downloads needed.

---

## 🏗️ Project Structure

```
disease-prediction/
├── data/                    # Raw + processed datasets + EDA plots
│   ├── raw/                 # Cached CSVs
│   ├── eda/                 # EDA visualizations per dataset
│   └── results/             # Evaluation plots + comparison tables
├── src/
│   ├── __init__.py          # Package config & dataset registry
│   ├── data_loader.py       # Fetch, cache, and load datasets
│   ├── preprocessing.py     # Missing values, scaling, encoding, SMOTE
│   ├── eda.py               # Exploratory analysis plots
│   ├── models.py            # Model definitions + hyperparameter tuning
│   ├── train.py             # Training orchestrator (all datasets)
│   ├── evaluate.py          # Metrics, ROC curves, confusion matrices, SHAP
│   └── predict.py           # CLI + API for inference
├── models/                  # Saved best models (.pkl)
├── tests/                   # Unit tests
│   ├── test_preprocessing.py
│   └── test_predict.py
├── app.py                   # Streamlit interactive web app
├── requirements.txt         # Pinned dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd disease-prediction

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Train Models

```bash
python -m src.train
```

This will:
- Download all 3 datasets (cached for offline re-runs)
- Generate EDA plots in `data/eda/`
- Train 4 base models + 2 tuned models per dataset
- Run 5-fold cross-validation
- Tune Random Forest and XGBoost via GridSearchCV
- Save the best model per dataset to `models/`
- Generate evaluation plots and comparison tables in `data/results/`

### 3. Run Predictions (CLI)

```bash
# Heart Disease
python -m src.predict --dataset heart_disease --features "63,1,3,145,233,1,0,150,0,2.3,0,0,1"

# Diabetes
python -m src.predict --dataset diabetes --features "6,148,72,35,0,33.6,0.627,50"

# Breast Cancer (30 features — see --info)
python -m src.predict --dataset breast_cancer --info
```

### 4. Launch Streamlit App

```bash
streamlit run app.py
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 🔬 Methodology

### Preprocessing
1. **Missing values**: Median imputation for numeric, mode for categorical. Domain-specific handling for Diabetes (zeros → NaN for biological measurements).
2. **Feature scaling**: StandardScaler (fit on training set only).
3. **Categorical encoding**: LabelEncoder for string features.
4. **Class imbalance**: SMOTE oversampling on training set + `class_weight='balanced'`.

### Models
Each dataset is trained with 4 algorithms:

| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| Logistic Regression | Linear | solver=lbfgs, max_iter=2000 |
| SVM (RBF kernel) | Kernel | probability=True, class_weight=balanced |
| Random Forest | Ensemble | **Tuned**: n_estimators, max_depth, min_samples |
| XGBoost | Boosting | **Tuned**: learning_rate, n_estimators, max_depth |

### Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: ROC curves, confusion matrices, comparison tables
- **Cross-validation**: 5-fold stratified CV on training set
- **Hyperparameter tuning**: GridSearchCV for RF and XGBoost

### Explainability
- **SHAP TreeExplainer** for tree-based models (RF, XGBoost)
- **Coefficient analysis** for linear models (Logistic Regression)
- **SHAP KernelExplainer** fallback for SVM
- Per-prediction explanations with feature rankings and plain-language descriptions

---

## 📈 Accuracy Targets

| Dataset | Target | Expected |
|---------|--------|----------|
| Heart Disease | >80% | ✅ ~83-87% |
| Diabetes | >80% | ✅ ~76-82% |
| Breast Cancer | >85% | ✅ ~95-97% |

*Actual results will vary slightly based on random seed and SMOTE resampling.*

---

## 🖥️ Streamlit App

The interactive web app features:
- **Disease selector** — choose Heart Disease, Diabetes, or Breast Cancer
- **Dynamic input forms** — with labeled fields and tooltips
- **Real-time predictions** — probability gauge + risk assessment
- **Explainable AI** — top contributing features ranked by impact
- **Premium dark theme** — gradient accents, glassmorphism, micro-animations

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---

## 📝 License

MIT License — see LICENSE for details.
