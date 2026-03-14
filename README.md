# 🏡 BlueStone Real Estate Business Intelligence Model

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-NdifrekeAnalytics-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/NdifrekeAnalytics)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ndifreke%20Ekanem-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ndifreke-ekanem-b479a027/)
[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://bluestone-real-estate-business-intelligence-model.streamlit.app)

<br/>

<a href="https://github.com/NdifrekeAnalytics" target="_blank">
  <img src="https://raw.githubusercontent.com/NdifrekeAnalytics/Bluestone-Real-Estate-Business-Intelligence-Model/main/assets/app_screenshot.png"
       alt="BlueStone Real Estate Intelligence App — Built by Ndifreke Ekanem | github.com/NdifrekeAnalytics"
       width="90%"
       style="border-radius: 12px; border: 2px solid #C8A85A;"/>
</a>

<sub>
  📌 <a href="https://github.com/NdifrekeAnalytics">github.com/NdifrekeAnalytics</a> &nbsp;|&nbsp;
  🔗 <a href="https://www.linkedin.com/in/ndifreke-ekanem-b479a027/">linkedin.com/in/ndifreke-ekanem-b479a027</a>
</sub>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives & Solutions](#-project-objectives--solutions)
- [Key Features](#-key-features-of-the-deployed-app)
- [Tech Stack](#-tech-stack)
- [Impact](#-impact)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Author](#-author)

---

## 🌐 Overview

**BlueStone Real Estate Business Intelligence Model** is a fully deployed, end-to-end data science and business intelligence solution designed to transform how real estate professionals manage property listings, predict market prices, and optimise lead conversion.

Built on a dataset of **61,520 property listings across 7 major US cities** — Chicago, Atlanta, Phoenix, Austin, Denver, Houston, and Charlotte — BlueStone integrates machine learning, interactive analytics, and a production-grade web application into a single centralised intelligence hub.

The app is live and publicly accessible at:
🔗 **[bluestone-real-estate-business-intelligence-model.streamlit.app](https://bluestone-real-estate-business-intelligence-model.streamlit.app)**

---

## 🔍 Problem Statement

Real estate businesses today are drowning in fragmented data — listings, inquiries, transactions, rental records — stored across disconnected systems with no unified view. Property managers, data analysts, and business leaders face four compounding challenges:

1. **No reliable property valuation tool** — Pricing decisions are made subjectively, leading to underpricing or overpricing that costs revenue.
2. **Poor lead qualification** — Sales teams waste time on inquiries unlikely to convert, with no data-driven prioritisation model.
3. **Reactive market analysis** — Without a live analytics dashboard, market trends are only visible in hindsight.
4. **No centralised intelligence** — ML insights, operational data, and business KPIs exist in silos with no integrated platform to act on them.

The result: lost revenue, inefficient sales processes, and strategic decisions made on gut feel rather than evidence.

---

## 🎯 Project Objectives & Solutions

| Objective | Solution Delivered |
|---|---|
| Build an accurate property price prediction model | Gradient Boosting ensemble achieving **R² = 0.9806**, RMSE = $124,563, MAPE = 7.84% |
| Predict inquiry-to-sale lead conversion | Classification model with **AUC-ROC = 0.9783**, F1 = 0.9013 |
| Provide explainable AI for every prediction | SHAP waterfall charts showing top feature drivers per prediction |
| Centralise business analytics in one place | 4-page interactive Streamlit app with live Power BI dashboard embed |
| Enable batch property valuation | CSV upload batch prediction engine for bulk portfolio scoring |
| Support operational data management | Full Listings & Inquiries CRUD interface with CSV export |
| Deploy as a production web application | Zero-cost Streamlit Community Cloud deployment with GitHub CI/CD |

---

## ✨ Key Features of the Deployed App

### 🏡 Property Predictor
- Real-time single property price prediction with 30+ input features
- Inquiry-to-sale lead conversion probability scoring
- **SHAP waterfall explainability chart** showing the top drivers behind each prediction
- Comparable property lookup from the BlueStone dataset
- Confidence scoring and market context

### 📦 Batch Prediction
- Upload a CSV of multiple properties for bulk price and conversion scoring
- Downloadable results with predicted price, conversion probability, and risk flags
- Summary statistics and distribution charts across the batch

### 📋 Listings & Inquiries
- Browse, filter, and search all 61,520 property listings
- View and manage customer inquiry records (252,792 inquiries)
- Add new listings and log new inquiries directly from the UI
- Export filtered data to CSV

### 📈 Analytics Dashboard
- Embedded live **Power BI dashboard** (Bluestone Real Estate Sale Transaction Analysis)
- City-level filters: Atlanta, Austin, Charlotte, Chicago, Denver, Houston, Phoenix
- KPIs: Total Sale Transactions, Average Market Price, Offer-to-List Ratio, Average Days on Market
- Sale Price by State, by Property Type, and Sale Transactions by State
- Year-over-year comparison (2024 vs 2025)

### 📊 Model Performance Dashboard
- Full regression metrics: Test R², RMSE, MAE, MAPE
- Candidate model comparison table (Gradient Boosting vs LightGBM vs Random Forest)
- Classification metrics: AUC-ROC, F1, Precision, Recall, Confusion Matrix
- Full hyperparameter configuration display
- SHAP global feature importance chart

### ℹ️ About & Deployment
- Full project documentation, data dictionary, and ML pipeline architecture
- Deployment checklist and technical configuration guide
- Required files inventory and artefact descriptions

---

## 🛠 Tech Stack

| Technology | Category | Role in the Project |
|---|---|---|
| **Python 3.11** | Core Language | Primary language for all data processing, ML, and app development |
| **Streamlit 1.45** | Web Framework | Builds and serves the entire multi-page interactive web application |
| **Pandas 2.2** | Data Engineering | Data loading, cleaning, transformation, and CSV operations |
| **NumPy 2.2** | Numerical Computing | Array operations, feature engineering, and numerical transformations |
| **Scikit-learn 1.5** | Machine Learning | Preprocessing pipelines, ensemble models, SMOTE, cross-validation |
| **Gradient Boosting (GBR)** | ML — Regression | Best regression model: R²=0.9806, RMSE=$124,563, MAPE=7.84% |
| **Gradient Boosting (CLF)** | ML — Classification | Best classification model: AUC-ROC=0.9783, F1=0.9013 |
| **LightGBM 4.5** | ML — Candidate Model | Candidate regressor evaluated during model selection |
| **XGBoost 2.1** | ML — Candidate Model | Candidate regressor evaluated during model selection |
| **SHAP 0.50** | Model Explainability | TreeExplainer waterfall charts — per-prediction feature attribution |
| **Optuna 3.6** | Hyperparameter Tuning | Bayesian optimisation for all candidate models |
| **Imbalanced-learn 0.12** | ML — Class Balancing | SMOTE oversampling for classification training set balance |
| **Scipy 1.13** | Statistical Computing | Statistical transformations and distance computations |
| **Matplotlib 3.9** | Visualisation | Static chart generation (SHAP importance plots, confusion matrix) |
| **Plotly 5.24** | Interactive Visualisation | Dynamic charts throughout the Streamlit app |
| **Joblib 1.4** | Model Serialisation | Saving and loading all `.pkl` model artefacts |
| **Power BI Service** | Business Intelligence | Embedded interactive analytics dashboard via Publish-to-Web iframe |
| **PyArrow 17.0** | Data Serialisation | Efficient DataFrame serialisation for Streamlit data caching |
| **Git & GitHub** | Version Control & CI/CD | Source control and automatic Streamlit Cloud deployment pipeline |
| **Streamlit Community Cloud** | Deployment Platform | Zero-cost production hosting with GitHub-integrated auto-deploy |

---

## 💥 Impact

### Model Performance
| Metric | Baseline (v1) | Deployed (v2) | Improvement |
|---|---|---|---|
| Regression R² | 0.8509 | **0.9806** | +15.2% |
| RMSE | $302,437 | **$124,563** | −58.8% |
| MAE | $73,424 | **$41,208** | −43.9% |
| MAPE | 24.51% | **7.84%** | −68.0% |
| Classification AUC-ROC | — | **0.9783** | — |
| Classification F1 | — | **0.9013** | — |

### Business Impact
- **Pricing accuracy** improved by 68% (MAPE reduction), eliminating systematic over/underpricing across a 61,520-listing portfolio
- **Lead qualification** at 97.83% AUC-ROC enables sales teams to focus effort on the highest-probability conversion prospects
- **Centralised intelligence** replaces 4+ disconnected data sources with a single deployed platform accessible from any browser
- **Batch prediction** capability enables portfolio-wide valuation in seconds — work that previously took days manually
- **Explainable AI** via SHAP gives non-technical stakeholders transparent, auditable reasoning behind every price and conversion prediction
- **Zero operational cost** — fully deployed on free-tier infrastructure (Streamlit Community Cloud + Power BI free)

---

## 📁 Project Structure

```
bluestone-real-estate-business-intelligence-model/
│
├── app.py                          # Main Streamlit application (6 pages)
├── bluestone_ml_pipeline.py        # Data merge & feature engineering (Blocks 1–3)
├── bluestone_ml_pipeline_v2.py     # Full ML pipeline — regression & classification (Blocks 4–10)
├── requirements.txt                # Pinned Python dependencies for Streamlit Cloud
├── Bluestone_data.csv              # Master dataset (61,520 listings × 97 columns)
│
├── bluestone_outputs/              # Trained model artefacts (generated by pipeline)
│   ├── best_regression_model.pkl
│   ├── best_classification_model.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── model_metadata.pkl
│   ├── shap_explainer_regression.pkl
│   ├── shap_explainer_classification.pkl
│   ├── target_encoding_maps.pkl
│   ├── label_encoders.pkl
│   ├── geo_cluster_model.pkl
│   ├── price_caps.pkl
│   ├── price_tiers.pkl
│   └── shap_importance_v2.png
│
└── assets/
    └── BlueStone_Logo.png
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Anaconda or virtualenv
- Git

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/NdifrekeAnalytics/Bluestone-Real-Estate-Business-Intelligence-Model.git
cd Bluestone-Real-Estate-Business-Intelligence-Model

# 2. Create and activate environment
conda create -n bluestone-app python=3.11
conda activate bluestone-app

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the ML pipeline to generate model artefacts (if not already present)
python bluestone_ml_pipeline.py       # Blocks 1–3: data prep
python bluestone_ml_pipeline_v2.py    # Blocks 4–10: training & saving pkl files

# 5. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Live Deployment
The app is deployed at:
🔗 **[bluestone-real-estate-business-intelligence-model.streamlit.app](https://bluestone-real-estate-business-intelligence-model.streamlit.app)**

---

## 👤 Author

<div align="center">

**Ndifreke Ekanem**
*Data Scientist | Business Intelligence Analyst*

[![GitHub](https://img.shields.io/badge/GitHub-NdifrekeAnalytics-181717?style=flat-square&logo=github)](https://github.com/NdifrekeAnalytics)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/ndifreke-ekanem-b479a027/)

*Built as part of the 10Alytics Data Science Programme — Amdari Internship Project*

</div>

---

<div align="center">
  <sub>⭐ If you found this project useful, please consider starring the repository</sub>
</div>
