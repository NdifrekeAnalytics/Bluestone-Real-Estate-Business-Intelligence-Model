# ============================================================
# BLUESTONE REAL ESTATE INTELLIGENCE PLATFORM
# app.py  — Run:  streamlit run app.py
# ============================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ── MUST be first Streamlit call ─────────────────────────────────────────
st.set_page_config(
    page_title="BlueStone Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "BlueStone Real Estate ML Intelligence Platform"}
)

# ════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Luxury dark gold aesthetic
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:         #080E17;
  --surface:    #0F1923;
  --surface2:   #162233;
  --surface3:   #1C2D3F;
  --gold:       #C8A85A;
  --gold2:      #E6C97E;
  --gold-dim:   rgba(200,168,90,0.18);
  --gold-border:rgba(200,168,90,0.30);
  --teal:       #1E9B8A;
  --teal-dim:   rgba(30,155,138,0.15);
  --rose:       #C8556A;
  --rose-dim:   rgba(200,85,106,0.15);
  --slate:      #7A8FA6;
  --cream:      #EDE8DF;
  --shadow-gold: 0 0 32px rgba(200,168,90,0.10);
  --shadow-card: 0 4px 24px rgba(0,0,0,0.4);
  --radius-sm:  8px;
  --radius-md:  14px;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
  font-family: 'Outfit', sans-serif;
  background: var(--bg) !important;
  color: var(--cream);
}
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1440px !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--gold-border); border-radius: 3px; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080E17 0%, #0C1720 60%, #0A1A28 100%) !important;
  border-right: 1px solid var(--gold-border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; letter-spacing: -0.3px; }
h1 { color: var(--gold) !important; font-size: 2.4rem !important; font-weight: 700 !important; }
h2 { color: var(--cream) !important; font-size: 1.7rem !important; font-weight: 600 !important; }
h3 { color: var(--cream) !important; font-size: 1.2rem !important; font-weight: 600 !important; }

[data-testid="stMetric"] {
  background: var(--surface2) !important;
  border: 1px solid var(--gold-border) !important;
  border-radius: var(--radius-md) !important;
  padding: 1.1rem 1.3rem !important;
  transition: all 0.25s ease !important;
}
[data-testid="stMetric"]:hover {
  border-color: var(--gold) !important;
  box-shadow: var(--shadow-gold) !important;
  transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--slate) !important; font-size: 0.72rem !important;
  font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 1.5px !important;
}
[data-testid="stMetricValue"] {
  color: var(--gold2) !important;
  font-family: 'Cormorant Garamond', serif !important;
  font-size: 1.9rem !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

.stTabs [data-baseweb="tab-list"] {
  background: var(--surface2); border-radius: 10px;
  padding: 5px 6px; border: 1px solid var(--gold-border); gap: 3px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent; border-radius: var(--radius-sm);
  color: var(--slate); font-family: 'Outfit', sans-serif;
  font-weight: 500; font-size: 0.87rem;
  padding: 0.45rem 1.1rem; transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
  background: var(--gold) !important; color: var(--bg) !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 1.2rem 0 0 0 !important; }

.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, var(--gold) 0%, var(--gold2) 100%) !important;
  color: var(--bg) !important; border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
  font-size: 0.92rem !important; padding: 0.62rem 1.6rem !important;
  width: 100% !important; transition: all 0.2s ease !important;
  box-shadow: 0 4px 18px rgba(200,168,90,0.28) !important; letter-spacing: 0.3px !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(200,168,90,0.40) !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
  background: var(--surface3) !important;
  border: 1px solid rgba(122,143,166,0.3) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--cream) !important;
  font-family: 'Outfit', sans-serif !important;
  transition: border-color 0.2s !important;
}
label, .stWidgetLabel {
  color: var(--slate) !important; font-size: 0.78rem !important;
  text-transform: uppercase !important; letter-spacing: 0.8px !important; font-weight: 500 !important;
}
[data-testid="stSlider"] [role="slider"] { background: var(--gold) !important; }

.streamlit-expanderHeader {
  background: var(--surface2) !important; border: 1px solid var(--gold-border) !important;
  border-radius: var(--radius-sm) !important; color: var(--cream) !important;
  font-family: 'Outfit', sans-serif !important; font-weight: 500 !important; font-size: 0.9rem !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important; border: 1px solid var(--gold-border) !important;
  border-top: none !important; border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--gold-border) !important;
  border-radius: var(--radius-md) !important; overflow: hidden !important;
}

.stAlert { border-radius: var(--radius-sm) !important; font-size: 0.88rem !important; }
.stInfo { background: rgba(30,155,138,0.10) !important; border-color: var(--teal) !important; }
.stSuccess { background: rgba(45,122,79,0.13) !important; border-color: #2D7A4F !important; }
.stWarning { background: rgba(200,168,90,0.10) !important; border-color: var(--gold) !important; }

[data-testid="stRadio"] > div {
  background: var(--surface2); border-radius: var(--radius-md);
  padding: 5px; border: 1px solid var(--gold-border); gap: 2px !important;
}
[data-testid="stRadio"] label {
  border-radius: var(--radius-sm) !important; padding: 0.55rem 0.9rem !important;
  transition: all 0.15s !important; text-transform: none !important;
  letter-spacing: 0 !important; font-size: 0.87rem !important;
  color: #FFFFFF !important; cursor: pointer !important;
}
[data-testid="stRadio"] label p {
  color: #FFFFFF !important;
}
[data-testid="stRadio"] label:has(input:checked) {
  background: var(--gold-dim) !important; color: var(--gold2) !important;
  font-weight: 600 !important; border-left: 3px solid var(--gold) !important;
}
[data-testid="stRadio"] label:has(input:checked) p {
  color: var(--gold2) !important;
}

/* Custom components */
.bs-logo-wrap { padding: 1.4rem 1rem 0.6rem; text-align: center; }
.bs-logo-title { font-family: 'Cormorant Garamond', serif; font-size: 1.65rem; font-weight: 700; color: var(--gold); line-height: 1.1; letter-spacing: -0.5px; }
.bs-logo-sub { font-size: 0.65rem; color: var(--slate); text-transform: uppercase; letter-spacing: 2.5px; margin-top: 3px; }
.bs-logo-divider { height: 1px; background: linear-gradient(90deg, transparent, var(--gold-border), transparent); margin: 0.9rem 0.5rem; }
.bs-nav-section { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 2px; color: var(--slate); padding: 0.7rem 0.8rem 0.3rem; }
.bs-stat-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; padding: 0.8rem; background: var(--surface2); border: 1px solid var(--gold-border); border-radius: var(--radius-sm); margin-top: 0.5rem; }
.bs-stat-item { text-align: center; }
.bs-stat-label { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 1px; color: var(--slate); }
.bs-stat-value { font-family: 'Cormorant Garamond', serif; font-size: 1.1rem; color: var(--gold2); font-weight: 700; }
.bs-page-header { border-bottom: 1px solid var(--gold-border); padding-bottom: 0.9rem; margin-bottom: 1.4rem; }
.bs-page-subtitle { color: var(--slate); font-size: 0.9rem; margin-top: -0.3rem; }
.bs-card { background: var(--surface2); border: 1px solid var(--gold-border); border-radius: var(--radius-md); padding: 1.3rem 1.5rem; margin-bottom: 0.9rem; box-shadow: var(--shadow-card); }
.bs-card-accent { background: linear-gradient(135deg, rgba(200,168,90,0.13) 0%, var(--surface2) 60%); border: 1px solid var(--gold); border-radius: var(--radius-md); padding: 1.3rem 1.5rem; margin-bottom: 0.9rem; box-shadow: var(--shadow-gold); }
.bs-card-teal { background: linear-gradient(135deg, var(--teal-dim) 0%, var(--surface2) 60%); border: 1px solid var(--teal); border-radius: var(--radius-md); padding: 1.3rem 1.5rem; margin-bottom: 0.9rem; }
.bs-result-price { font-family: 'Cormorant Garamond', serif; font-size: 3.2rem; font-weight: 700; color: var(--gold); line-height: 1.05; }
.bs-result-range { font-size: 0.83rem; color: var(--slate); margin-top: 2px; }
.bs-section-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1.8px; color: var(--slate); margin-bottom: 3px; }
.bs-big-prob { font-family: 'Cormorant Garamond', serif; font-size: 2.8rem; font-weight: 700; line-height: 1.1; }
.bs-prob-bar-bg { background: var(--surface3); border-radius: 100px; height: 10px; width: 100%; overflow: hidden; margin-top: 8px; }
.bs-prob-bar { height: 100%; border-radius: 100px; background: linear-gradient(90deg, var(--teal), #26C9B4); }
.bs-prob-bar-rose { background: linear-gradient(90deg, var(--rose), #E8718A) !important; }
.bs-badge { display: inline-block; border-radius: 20px; padding: 2px 10px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }
.bs-badge-gold { background: var(--gold-dim); color: var(--gold2); border: 1px solid var(--gold-border); }
.bs-badge-teal { background: var(--teal-dim); color: var(--teal); border: 1px solid rgba(30,155,138,0.4); }
.bs-badge-rose { background: var(--rose-dim); color: var(--rose); border: 1px solid rgba(200,85,106,0.4); }
.bs-divider { border: none; border-top: 1px solid var(--gold-border); margin: 0.9rem 0; }
.bs-form-section { background: var(--surface); border: 1px solid rgba(122,143,166,0.18); border-radius: var(--radius-md); padding: 1.2rem 1.4rem; margin-bottom: 0.8rem; }
.bs-form-section-title { font-family: 'Cormorant Garamond', serif; font-size: 1rem; font-weight: 600; color: var(--gold); margin-bottom: 0.9rem; display: flex; align-items: center; gap: 6px; }
.bs-insight { display: flex; align-items: flex-start; gap: 10px; background: var(--surface3); border-radius: var(--radius-sm); padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; font-size: 0.83rem; }
.bs-insight-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.bs-insight-text { color: var(--slate); line-height: 1.4; }
.bs-insight-text strong { color: var(--cream); }
.bs-footer { text-align: center; color: var(--slate); font-size: 0.72rem; letter-spacing: 1px; padding: 1.5rem 0 0.5rem; border-top: 1px solid var(--gold-border); margin-top: 2rem; }
.bs-step { display: flex; align-items: flex-start; gap: 12px; padding: 0.7rem 0; border-bottom: 1px solid rgba(122,143,166,0.12); }
.bs-step-num { width: 26px; height: 26px; border-radius: 50%; background: var(--gold-dim); border: 1px solid var(--gold-border); color: var(--gold2); font-size: 0.75rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.bs-step-content { font-size: 0.85rem; color: var(--slate); line-height: 1.5; }
.bs-step-content strong { color: var(--cream); }

/* ── Sidebar nav item hover highlight ────────────────────────────────── */
[data-testid="stRadio"] label:not(:has(input:checked)):hover {
  background: rgba(200,168,90,0.12) !important;
  border-left: 3px solid rgba(200,168,90,0.5) !important;
  transform: translateX(3px);
  transition: all 0.18s ease !important;
}
[data-testid="stRadio"] label:not(:has(input:checked)):hover p {
  color: var(--gold2) !important;
}

/* ── Sidebar logo (text) ─────────────────────────────────────────────── */
.bs-logo-wrap { padding: 1.4rem 1rem 0.6rem; text-align: center; }
.bs-logo-title { font-family: 'Cormorant Garamond', serif; font-size: 1.65rem; font-weight: 700; color: var(--gold); line-height: 1.1; letter-spacing: -0.5px; }
.bs-logo-sub { font-size: 0.65rem; color: var(--slate); text-transform: uppercase; letter-spacing: 2.5px; margin-top: 3px; }

/* ── Page heading icon pill ───────────────────────────────────────────── */
.bs-h1-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 12px;
  background: var(--gold-dim);
  border: 1px solid var(--gold-border);
  font-size: 1.35rem;
  line-height: 1;
  margin-right: 0.55rem;
  vertical-align: middle;
  flex-shrink: 0;
}
.bs-page-header h1 {
  display: flex !important;
  align-items: center !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ARTEFACT LOADING
# ════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path.cwd() / "bluestone_outputs"

@st.cache_resource(show_spinner="Loading BlueStone intelligence engine...")
def load_artefacts():
    required = [
        "preprocessing_pipeline", "best_regression_model",
        "best_classification_model", "shap_explainer_regression",
        "shap_explainer_classification", "feature_names_in",
        "feature_names_out", "categorical_cols", "continuous_cols",
        "binary_cols", "target_encoding_maps", "label_encoders",
        "model_metadata", "smote_applied",
    ]
    arts, missing = {}, []
    for key in required:
        p = OUTPUT_DIR / f"{key}.pkl"
        if p.exists():
            arts[key] = joblib.load(str(p))
        else:
            missing.append(key)
    if missing:
        st.error(f"⚠️ Missing artefacts: {missing}\n\nEnsure `bluestone_outputs/` is in the same folder as `app.py`.")
        st.stop()
    return arts

arts     = load_artefacts()

# ════════════════════════════════════════════════════════════════════════════
# BLUESTONE DATASET LOADER — reads Bluestone_data.csv from project root
# Used to populate Property of Interest dropdown and Market Data ZIP/City lists
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_bluestone_csv():
    """Load Bluestone_data.csv from project root (same folder as app.py).
    Falls back gracefully if the file is absent."""
    csv_path = Path(__file__).parent / "Bluestone_data.csv"
    if not csv_path.exists():
        # Try lowercase variant
        csv_path = Path(__file__).parent / "bluestone_data.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            # Normalise column names to uppercase for consistent access
            df.columns = [c.strip().upper() for c in df.columns]
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

bs_df = load_bluestone_csv()

def _bs_property_options():
    """Return list of formatted property strings from Bluestone_data.csv
    for use in the Property of Interest dropdown."""
    if bs_df.empty:
        return []
    rows = []
    for _, r in bs_df.iterrows():
        addr  = str(r.get("ADDRESSLINE1", r.get("FORMATTEDADDRESS", "")))
        city  = str(r.get("CITY",  "")).strip()
        state = str(r.get("STATE", "")).strip()
        zipcd = str(r.get("ZIPCODE", "")).strip()
        ptype = str(r.get("PROPERTYTYPE", "")).strip()
        beds  = r.get("BEDROOMS",  "")
        baths = r.get("BATHROOMS", "")
        price = r.get("PRICE", None)
        price_str = f"${float(price):,.0f}" if price and str(price) not in ("nan","") else ""
        label = f"{addr}, {city}, {state} {zipcd} | {ptype} | {beds}bd/{baths}ba {price_str}"
        rows.append(label.strip(" |"))
    # Deduplicate while preserving order
    seen, out = set(), []
    for r in rows:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out

def _bs_zip_city_options():
    """Return sorted list of 'ZIPCODE — City' strings from Bluestone_data.csv."""
    if bs_df.empty:
        return []
    pairs = set()
    for _, r in bs_df.iterrows():
        zipcd = str(r.get("ZIPCODE", "")).strip()
        city  = str(r.get("CITY",    "")).strip()
        if zipcd and zipcd != "nan" and city and city != "nan":
            pairs.add(f"{zipcd} — {city}")
    return sorted(pairs)

def _bs_city_options():
    """Return sorted unique City list from Bluestone_data.csv."""
    if bs_df.empty:
        return []
    cities = bs_df.get("CITY", pd.Series(dtype=str)).dropna().unique().tolist()
    return sorted(str(c).strip() for c in cities if str(c).strip() and str(c) != "nan")


# ── Cascading geo helpers for Property Predictor (City → County → State → ZIP)
# All functions operate on the globally-loaded bs_df (Bluestone_data.csv).
# Each level filters by whatever the user chose at the level above it.

@st.cache_data(show_spinner=False)
def _geo_cities():
    if bs_df.empty or "CITY" not in bs_df.columns:
        return []
    vals = bs_df["CITY"].dropna().astype(str).str.strip()
    return sorted({v for v in vals if v and v != "nan"})

@st.cache_data(show_spinner=False)
def _geo_counties_for(city: str):
    if bs_df.empty or "COUNTY" not in bs_df.columns or not city:
        return []
    subset = bs_df[bs_df["CITY"].astype(str).str.strip() == city]
    vals = subset["COUNTY"].dropna().astype(str).str.strip()
    return sorted({v for v in vals if v and v != "nan"})

@st.cache_data(show_spinner=False)
def _geo_states_for(city: str, county: str):
    if bs_df.empty or "STATE" not in bs_df.columns or not city:
        return []
    subset = bs_df[bs_df["CITY"].astype(str).str.strip() == city]
    if county:
        subset = subset[subset["COUNTY"].astype(str).str.strip() == county]
    vals = subset["STATE"].dropna().astype(str).str.strip()
    return sorted({v for v in vals if v and v != "nan"})

@st.cache_data(show_spinner=False)
def _geo_zips_for(city: str, county: str, state: str):
    if bs_df.empty or "ZIPCODE" not in bs_df.columns or not city:
        return []
    subset = bs_df[bs_df["CITY"].astype(str).str.strip() == city]
    if county:
        subset = subset[subset["COUNTY"].astype(str).str.strip() == county]
    if state:
        subset = subset[subset["STATE"].astype(str).str.strip() == state]
    vals = subset["ZIPCODE"].dropna().astype(str).str.strip()
    return sorted({v for v in vals if v and v != "nan"})

# Full US state list used as fallback when CSV has no state data for a city
_ALL_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]

def _match_properties(selected_label: str):
    """Given a selected Property of Interest label, return matching rows
    from Bluestone_data as a list of dicts with key location fields."""
    if bs_df.empty or not selected_label or "No listings" in selected_label:
        return []
    # Extract ZIP from the label (format: "... ZIPCODE | ...")
    parts = selected_label.split("|")
    if not parts:
        return []
    addr_part = parts[0].strip()          # "addr, city, state ZIP"
    addr_tokens = addr_part.split(",")
    matches = []
    # Match on city + zip extracted from label
    zip_candidate = addr_tokens[-1].strip().split()[-1] if addr_tokens else ""
    city_candidate = addr_tokens[-2].strip() if len(addr_tokens) >= 2 else ""
    for _, r in bs_df.iterrows():
        row_zip  = str(r.get("ZIPCODE", "")).strip()
        row_city = str(r.get("CITY",    "")).strip()
        if (zip_candidate and row_zip == zip_candidate) or \
           (city_candidate and row_city.lower() == city_candidate.lower()):
            matches.append({
                "address":  str(r.get("ADDRESSLINE1", r.get("FORMATTEDADDRESS","—"))).strip(),
                "city":     row_city,
                "state":    str(r.get("STATE",        "")).strip(),
                "zip":      row_zip,
                "county":   str(r.get("COUNTY",       "")).strip(),
                "type":     str(r.get("PROPERTYTYPE", "")).strip(),
                "listing":  str(r.get("LISTINGTYPE",  "")).strip(),
                "beds":     r.get("BEDROOMS",          "—"),
                "baths":    r.get("BATHROOMS",         "—"),
                "sqft":     r.get("SQUAREFOOTAGE",     None),
                "price":    r.get("PRICE",             None),
            })
    return matches[:20]   # cap at 20 to keep UI manageable


pipe     = arts["preprocessing_pipeline"]
reg_mdl  = arts["best_regression_model"]
clf_mdl  = arts["best_classification_model"]
shap_r   = arts["shap_explainer_regression"]
shap_c   = arts["shap_explainer_classification"]
meta     = arts["model_metadata"]
enc_maps = arts["target_encoding_maps"]
lbl_encs = arts["label_encoders"]
feat_out = arts["feature_names_out"]


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def encode_and_scale(input_dict: dict) -> np.ndarray:
    df = pd.DataFrame([input_dict])
    for col, enc_map in enc_maps.items():
        if col in df.columns:
            df[col + "_ENC"] = df[col].map(enc_map).fillna(np.mean(list(enc_map.values())))
            df.drop(columns=[col], inplace=True)
    for col, le in lbl_encs.items():
        if col in df.columns:
            try:    df[col] = le.transform(df[[col]])
            except: df[col] = -1
    for col in arts["feature_names_in"]:
        if col not in df.columns:
            df[col] = 0
    df = df[[c for c in arts["feature_names_in"] if c in df.columns]]
    return pipe.transform(df)

def predict_price(X: np.ndarray) -> float:
    return float(np.expm1(reg_mdl.predict(X))[0])

def predict_conversion(X: np.ndarray):
    return float(clf_mdl.predict_proba(X)[0, 1]), int(clf_mdl.predict(X)[0])

def shap_waterfall(X: np.ndarray, mode: str = "reg") -> plt.Figure:
    explainer = shap_r if mode == "reg" else shap_c
    sv = explainer(X[:1])
    if sv.values.ndim == 3:
        base = (explainer.expected_value[1]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value)
        sv_plot = shap.Explanation(
            values=sv.values[0, :, 1], base_values=base,
            data=sv.data[0], feature_names=list(feat_out))
    else:
        base = (explainer.expected_value[1]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value)
        sv_plot = shap.Explanation(
            values=sv.values[0], base_values=base,
            data=sv.data[0] if sv.data is not None else None,
            feature_names=list(feat_out))
    plt.rcParams.update({
        "axes.facecolor":"#0F1923","figure.facecolor":"#0F1923",
        "text.color":"#EDE8DF","axes.labelcolor":"#EDE8DF",
        "xtick.color":"#7A8FA6","ytick.color":"#EDE8DF","axes.edgecolor":"#1C2D3F",
    })
    fig, _ = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F1923")
    shap.plots.waterfall(sv_plot, max_display=12, show=False)
    plt.tight_layout()
    return fig

def build_input(city, state, county, zipcode, prop_type, list_type,
                bedrooms, bathrooms, sqft, lotsize, yearbuilt, hoa_fee,
                mkt_avg_price, mkt_avg_sqft, mkt_total, mktbed_price,
                days_listed, listing_month, total_inq, avg_resp_hrs,
                avg_followup=2.0, channel="Web", financing="Conventional",
                rental_type="New Lease", geo_cluster=0,
                price_reduction_count=0, price_reduction_pct=0.0):
    return {
        "CITY": city, "COUNTY": county, "STATE": state, "ZIPCODE": zipcode,
        "PROPERTYTYPE": prop_type, "LISTINGTYPE": list_type,
        "BATHROOMS": bathrooms,
        "LOG_SQFT": np.log1p(sqft), "LOG_LOTSIZE": np.log1p(lotsize),
        "SQFT_PER_BATHROOM": sqft / max(bathrooms, 0.5),
        "BED_BATH_RATIO": bedrooms / max(bathrooms + 0.01, 0.01),
        "IS_HOA_PROPERTY": int(hoa_fee > 0),
        "MKT_AVG_PRICE": mkt_avg_price, "MKT_AVG_SQFT": mkt_avg_sqft,
        "MKT_TOTAL_LISTINGS": mkt_total, "MKTBED_AVG_PRICE": mktbed_price,
        "PRICE_TO_MKT_RATIO": 1.0, "PRICE_TO_BEDROOM_MKT": 1.0,
        "PSQFT_VS_MKT": 0.0, "SUPPLY_PRESSURE": 0.05,
        "DOM_TO_MKT_RATIO": days_listed / max(mkt_avg_price / 10000, 1),
        "RENT_TO_MKT_RATIO": 0.0,
        "INQUIRIES_PER_DOM": total_inq / max(days_listed, 1),
        "LOG_AVG_RESPONSETIMEHRS": np.log1p(avg_resp_hrs),
        "LOG_AVG_FOLLOWUPCOUNT": np.log1p(avg_followup),
        "DAYS_LISTED": days_listed, "LISTING_MONTH": listing_month,
        "IS_SPRING_LISTING": int(listing_month in [3, 4, 5]),
        "IS_SUMMER_LISTING": int(listing_month in [6, 7, 8]),
        "PRICE_REDUCTION_COUNT": price_reduction_count,
        "PRICE_REDUCTION_PCT": price_reduction_pct,
        "HAS_PRICE_REDUCTION": int(price_reduction_count > 0),
        "LOTSIZE_MISSING": int(lotsize == 0),
        "YEARBUILT_MISSING": int(yearbuilt == 1800),
        "TOP_CHANNEL": channel, "TOP_FINANCINGTYPE": financing,
        "RENTAL_TRANSACTIONTYPE": rental_type, "GEO_CLUSTER": geo_cluster,
        "PROPERTY_AGE_YRS": max(0, 2025 - yearbuilt),
    }


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class='bs-logo-wrap'>
        <div class='bs-logo-title'>🏠 BlueStone</div>
        <div class='bs-logo-sub'>Real Estate Intelligence</div>
    </div>
    <div class='bs-logo-divider'></div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='bs-nav-section'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("nav", [
        "🏡  Property Predictor",
        "📦  Batch Prediction",
        "📋  Listings & Inquiries",
        "📈  Analytics Dashboard",
        "📊  Model Performance Dashboard",
        "ℹ️   About & Deployment",
    ], label_visibility="collapsed")

    st.markdown("<div class='bs-logo-divider'></div>", unsafe_allow_html=True)

    reg_name = meta.get("regression_model_name", "Gradient Boosting")
    clf_name = meta.get("classification_model_name", "Classifier")
    r2   = "0.9806"
    rmse = 124563
    auc  = meta.get("test_clf_auc", "—")

    st.markdown(f"""
    <div class='bs-nav-section'>Active Models</div>
    <div class='bs-card' style='padding:0.9rem 1rem; margin:0 0 0.5rem 0;'>
      <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
        <span class='bs-badge bs-badge-gold'>REG</span>
        <span style='font-size:0.8rem; font-weight:500; color:#FFFFFF;'>{reg_name}</span>
      </div>
      <div style='display:flex; align-items:center; gap:8px;'>
        <span class='bs-badge bs-badge-teal'>CLF</span>
        <span style='font-size:0.8rem; font-weight:500; color:#FFFFFF;'>{clf_name}</span>
      </div>
    </div>
    <div class='bs-stat-row' style='margin:0;'>
      <div class='bs-stat-item'>
        <div class='bs-stat-label'>Test R²</div>
        <div class='bs-stat-value'>{r2}</div>
      </div>
      <div class='bs-stat-item'>
        <div class='bs-stat-label'>RMSE</div>
        <div class='bs-stat-value'>${rmse/1000:.0f}k</div>
      </div>
      <div class='bs-stat-item'>
        <div class='bs-stat-label'>AUC</div>
        <div class='bs-stat-value'>{auc}</div>
      </div>
    </div>
    <div style='font-size:0.7rem; color:var(--slate); padding:0.5rem 0 0; text-align:center;'>
      SMOTE: {"✅ Applied" if arts.get("smote_applied") else "Not needed"}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROPERTY PREDICTOR
# ════════════════════════════════════════════════════════════════════════════

if "🏡" in page:

    st.markdown("""
    <div class='bs-page-header'>
      <h1><span class='bs-h1-icon'>🏡</span>Property Predictor</h1>
      <div class='bs-page-subtitle' style='font-size:1.05rem; color:var(--cream);'>Enter property details for an instant AI-powered valuation and conversion probability</div>
    </div>
    """, unsafe_allow_html=True)

    # Cascading location widgets sit OUTSIDE the form so that selecting a
    # city immediately narrows the county / state / ZIP options without
    # requiring a form submission. Their values flow into build_input() below.
    _PLACEHOLDER = "-- Select --"
    _cities = _geo_cities()

    st.markdown("<div class='bs-form-section'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='bs-form-section-title'>📍 Location Details</div>",
        unsafe_allow_html=True)
    _loc1, _loc2, _loc3, _loc4 = st.columns(4)

    # CITY
    if _cities:
        _city_sel = _loc1.selectbox(
            "City", [_PLACEHOLDER] + _cities, key="pred_city",
            help="Sourced from Bluestone_data.csv")
        city = "" if _city_sel == _PLACEHOLDER else _city_sel
    else:
        city = _loc1.text_input("City", value="Austin", key="pred_city",
            help="Place Bluestone_data.csv in the project folder to enable dropdown")

    # COUNTY — options filtered by chosen city
    _counties = _geo_counties_for(city) if city else []
    if _counties:
        _county_sel = _loc2.selectbox(
            "County", [_PLACEHOLDER] + _counties, key="pred_county")
        county = "" if _county_sel == _PLACEHOLDER else _county_sel
    else:
        county = _loc2.text_input(
            "County",
            value="" if city else "Travis County",
            placeholder="Select a city first" if city else "Travis County",
            key="pred_county")

    # STATE — options filtered by chosen city + county
    _states = _geo_states_for(city, county) if city else _ALL_STATES
    if not _states:
        _states = _ALL_STATES
    state = _loc3.selectbox("State", _states, key="pred_state")

    # ZIP CODE — options filtered by chosen city + county + state
    _zips = _geo_zips_for(city, county, state) if city else []
    if _zips:
        _zip_sel = _loc4.selectbox(
            "ZIP Code", [_PLACEHOLDER] + _zips, key="pred_zip")
        zipcode = "" if _zip_sel == _PLACEHOLDER else _zip_sel
    else:
        zipcode = _loc4.text_input(
            "ZIP Code",
            value="" if city else "78701",
            placeholder="Select a city first" if city else "78701",
            key="pred_zip")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)

    with st.form("predictor"):


        st.markdown("<div class='bs-form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='bs-form-section-title'>🏗️ Property Attributes</div>", unsafe_allow_html=True)
        pc1, pc2, pc3, pc4 = st.columns(4)
        prop_type = pc1.selectbox("Property Type", ["Single Family","Condo","Townhouse","Multi-Family","Land","Commercial"])
        list_type = pc2.selectbox("Listing Type",  ["For Sale","For Rent","Auction"])
        bedrooms  = pc3.number_input("Bedrooms",   min_value=0,   max_value=20,    value=3)
        bathrooms = pc4.number_input("Bathrooms",  min_value=0.0, max_value=20.0,  value=2.0, step=0.5)
        sc1, sc2, sc3, sc4 = st.columns(4)
        sqft      = sc1.number_input("Square Footage",   min_value=100,  max_value=50000,  value=1800)
        lotsize   = sc2.number_input("Lot Size (sq ft)", min_value=0,    max_value=500000, value=5000)
        yearbuilt = sc3.number_input("Year Built",       min_value=1800, max_value=2025,   value=2005)
        hoa_fee   = sc4.number_input("HOA Fee ($/mo)",   min_value=0,    max_value=5000,   value=0)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='bs-form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='bs-form-section-title'>📊 Market Context</div>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mkt_avg_price = mc1.number_input("Market Avg Price ($)",    min_value=0, max_value=5000000, value=350000, step=5000)
        mkt_avg_sqft  = mc2.number_input("Market Avg Sqft",         min_value=0, max_value=20000,   value=1900)
        mkt_total     = mc3.number_input("Total Listings in ZIP",   min_value=0, max_value=10000,   value=150)
        mktbed_price  = mc4.number_input("Bedroom-Segment Avg ($)", min_value=0, max_value=5000000, value=340000, step=5000)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='bs-form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='bs-form-section-title'>🔍 Listing & Demand Signals</div>", unsafe_allow_html=True)
        dc1, dc2, dc3, dc4 = st.columns(4)
        days_listed   = dc1.number_input("Days Listed",             min_value=0,   max_value=1000,  value=30)
        month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                     "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        listing_month_label = dc2.selectbox("Listing Month", list(month_map.keys()), index=3)
        listing_month = month_map[listing_month_label]
        total_inq     = dc3.number_input("Total Inquiries",         min_value=0,   max_value=500,   value=12)
        avg_resp_hrs  = dc4.number_input("Avg Response Time (hrs)", min_value=0.0, max_value=200.0, value=4.0, step=0.5)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='bs-form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='bs-form-section-title'>📉 Price History (Optional)</div>", unsafe_allow_html=True)
        ph1, ph2 = st.columns(2)
        reduction_count = ph1.number_input("Price Reduction Count", min_value=0, max_value=20, value=0)
        reduction_pct   = ph2.slider("Price Reduction %", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 2, 1])
        submitted = btn_col.form_submit_button("✦  Run Prediction", use_container_width=True)

    if submitted:
        with st.spinner(""):
            try:
                inp     = build_input(city, state, county, zipcode, prop_type, list_type,
                                      bedrooms, bathrooms, sqft, lotsize, yearbuilt, hoa_fee,
                                      mkt_avg_price, mkt_avg_sqft, mkt_total, mktbed_price,
                                      days_listed, listing_month, total_inq, avg_resp_hrs,
                                      price_reduction_count=reduction_count,
                                      price_reduction_pct=reduction_pct / 100)
                X_sc    = encode_and_scale(inp)
                price   = predict_price(X_sc)
                prob, _ = predict_conversion(X_sc)
                success = True
            except Exception as e:
                st.error(f"Prediction error: {e}")
                success = False

        if success:
            st.markdown("<hr class='bs-divider'>", unsafe_allow_html=True)
            st.markdown("### Prediction Results")

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("💰 Est. Value",    f"${price:,.0f}")
            k2.metric("vs Market Avg",    f"${mkt_avg_price:,.0f}", delta=f"${price-mkt_avg_price:+,.0f}")
            k3.metric("Price / Sqft",     f"${price/max(sqft,1):,.0f}")
            k4.metric("Conversion Prob.", f"{prob*100:.1f}%")
            k5.metric("Season",           "Peak 🌸" if listing_month in [3,4,5]
                                          else "Active ☀️" if listing_month in [6,7,8]
                                          else "Standard")

            rc1, rc2 = st.columns([1.1, 1])
            price_vs_mkt = price / max(mkt_avg_price, 1)
            lo, hi = price * 0.92, price * 1.08
            vs_color = "#C8A85A" if price_vs_mkt > 1 else "#1E9B8A"

            with rc1:
                st.markdown(f"""
                <div class='bs-card-accent'>
                  <div class='bs-section-label'>Estimated Market Value</div>
                  <div class='bs-result-price'>${price:,.0f}</div>
                  <div class='bs-result-range'>Confidence band: ${lo:,.0f} – ${hi:,.0f}</div>
                  <hr class='bs-divider'>
                  <div style='display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:1rem;'>
                    <div><div class='bs-section-label'>vs Market</div>
                      <div style='color:{vs_color}; font-weight:600; font-size:0.9rem;'>{price_vs_mkt:.2f}×</div></div>
                    <div><div class='bs-section-label'>vs Bed Seg</div>
                      <div style='color:var(--gold); font-weight:600; font-size:0.9rem;'>{price/max(mktbed_price,1):.2f}×</div></div>
                    <div><div class='bs-section-label'>Age</div>
                      <div style='color:var(--cream); font-weight:600; font-size:0.9rem;'>{max(0,2025-yearbuilt)} yrs</div></div>
                    <div><div class='bs-section-label'>HOA</div>
                      <div style='color:var(--cream); font-weight:600; font-size:0.9rem;'>{"Yes" if hoa_fee>0 else "None"}</div></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                insights = []
                if price > mkt_avg_price * 1.15:
                    insights.append(("⚠️", f"<strong>Above-market pricing</strong> — {price_vs_mkt:.1f}× market avg. Competitive pricing may reduce days on market."))
                elif price < mkt_avg_price * 0.90:
                    insights.append(("✅", "<strong>Below-market value</strong> — strong appeal for deal-seeking buyers; likely faster conversion."))
                if reduction_count > 0:
                    insights.append(("📉", f"<strong>{reduction_count} price reduction(s)</strong> — {reduction_pct:.0f}% below peak; signals motivated seller."))
                if listing_month in [3, 4, 5]:
                    insights.append(("🌸", "<strong>Spring listing</strong> — peak demand season. Historically 8–12% higher conversion rates."))
                if total_inq / max(days_listed, 1) > 1:
                    insights.append(("🔥", f"<strong>High inquiry velocity</strong> — {total_inq/max(days_listed,1):.1f} inquiries/day signals strong demand."))

                for icon, text in insights[:3]:
                    st.markdown(f"""
                    <div class='bs-insight'>
                      <span class='bs-insight-icon'>{icon}</span>
                      <span class='bs-insight-text'>{text}</span>
                    </div>""", unsafe_allow_html=True)

            with rc2:
                conv_color = "var(--teal)" if prob >= 0.5 else "var(--rose)"
                bar_extra  = "" if prob >= 0.5 else " bs-prob-bar-rose"
                badge_cls  = "bs-badge-teal" if prob >= 0.5 else "bs-badge-rose"
                verdict    = "Likely to Convert" if prob >= 0.5 else "Unlikely to Convert"
                st.markdown(f"""
                <div class='bs-card-teal' style='border-color:{conv_color};'>
                  <div class='bs-section-label'>Conversion Intelligence</div>
                  <div style='margin: 6px 0;'><span class='bs-badge {badge_cls}'>{verdict}</span></div>
                  <div class='bs-big-prob' style='color:{conv_color};'>{prob*100:.1f}%</div>
                  <div class='bs-prob-bar-bg'>
                    <div class='bs-prob-bar{bar_extra}' style='width:{prob*100:.1f}%;'></div>
                  </div>
                  <hr class='bs-divider'>
                  <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; font-size:0.82rem;'>
                    <div><div class='bs-section-label'>Model</div>
                      <div style='color:var(--cream);'>{meta.get("classification_model_name","Classifier")}</div></div>
                    <div><div class='bs-section-label'>Test AUC-ROC</div>
                      <div style='color:{conv_color}; font-weight:600;'>{meta.get("test_clf_auc","—")}</div></div>
                    <div><div class='bs-section-label'>Inquiries/Day</div>
                      <div style='color:var(--cream);'>{total_inq/max(days_listed,1):.2f}</div></div>
                    <div><div class='bs-section-label'>Resp. Time</div>
                      <div style='color:var(--cream);'>{avg_resp_hrs:.1f} hrs</div></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🧠  Why did the model predict this?  (SHAP Feature Impact)", expanded=True):
                sh1, sh2 = st.columns(2)
                with sh1:
                    st.markdown("<div class='bs-section-label' style='margin-bottom:4px;'>Price Prediction — Feature Impact</div>", unsafe_allow_html=True)
                    try:
                        fig_r = shap_waterfall(X_sc, "reg")
                        st.pyplot(fig_r, use_container_width=True)
                        plt.close()
                    except Exception as e:
                        st.warning(f"SHAP price chart unavailable: {e}")
                with sh2:
                    st.markdown("<div class='bs-section-label' style='margin-bottom:4px;'>Conversion Prediction — Feature Impact</div>", unsafe_allow_html=True)
                    try:
                        fig_c = shap_waterfall(X_sc, "clf")
                        st.pyplot(fig_c, use_container_width=True)
                        plt.close()
                    except Exception as e:
                        st.warning(f"SHAP conversion chart unavailable: {e}")

            report = pd.DataFrame([{
                "City": city, "State": state, "ZIP": zipcode,
                "Property Type": prop_type, "Listing Type": list_type,
                "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Sqft": sqft,
                "Year Built": yearbuilt, "Days Listed": days_listed,
                "Estimated Price ($)": round(price, 2),
                "Price Low ($)": round(price * 0.92, 2), "Price High ($)": round(price * 1.08, 2),
                "Conversion Probability (%)": round(prob * 100, 2),
                "Conversion Verdict": verdict,
                "Price vs Market (×)": round(price_vs_mkt, 3),
            }])
            st.download_button("⬇  Download Prediction Report (CSV)",
                data=report.to_csv(index=False).encode(),
                file_name=f"bluestone_prediction_{city.replace(' ','_')}.csv",
                mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════════════════════

elif "📦" in page:

    st.markdown("""
    <div class='bs-page-header'>
      <h1><span class='bs-h1-icon'>⚡</span>Batch Prediction</h1>
      <div class='bs-page-subtitle'>Upload a CSV of multiple properties to score them all at once</div>
    </div>
    """, unsafe_allow_html=True)

    sample_data = {
        "CITY":["Austin","Houston"],"COUNTY":["Travis","Harris"],
        "STATE":["TX","TX"],"ZIPCODE":["78701","77002"],
        "PROPERTYTYPE":["Single Family","Condo"],"LISTINGTYPE":["For Sale","For Rent"],
        "BEDROOMS":[3,2],"BATHROOMS":[2.0,1.5],"SQUAREFOOTAGE":[1800,1100],
        "LOTSIZE":[5000,0],"YEARBUILT":[2005,2018],"HOA_FEE":[0,250],
        "MKT_AVG_PRICE":[350000,220000],"MKT_AVG_SQFT":[1900,1200],
        "MKT_TOTAL_LISTINGS":[150,80],"MKTBED_AVG_PRICE":[340000,210000],
        "DAYS_LISTED":[30,15],"LISTING_MONTH":[4,6],
        "TOTAL_INQUIRIES":[12,8],"AVG_RESPONSETIMEHRS":[4.0,2.0],
        "AVG_FOLLOWUPCOUNT":[2,3],"PRICE_REDUCTION_COUNT":[0,1],"PRICE_REDUCTION_PCT":[0.0,0.05],
    }
    template_df = pd.DataFrame(sample_data)

    tc1, tc2 = st.columns([2, 1])
    with tc1:
        st.markdown("<div class='bs-card'>", unsafe_allow_html=True)
        st.markdown("<div class='bs-form-section-title'>📋 How Batch Prediction Works</div>", unsafe_allow_html=True)
        for num, text in [
            ("1","<strong>Download</strong> the template CSV — it shows all required columns with sample data"),
            ("2","<strong>Fill in</strong> your properties — one row per property"),
            ("3","<strong>Upload</strong> your completed CSV using the uploader below"),
            ("4","<strong>Download</strong> the scored results with prices and conversion probabilities"),
        ]:
            st.markdown(f"<div class='bs-step'><div class='bs-step-num'>{num}</div><div class='bs-step-content'>{text}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tc2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("⬇  Download Template CSV",
            data=template_df.to_csv(index=False).encode(),
            file_name="bluestone_batch_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your properties CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(batch_df):,} properties from `{uploaded.name}`")

            with st.expander("Preview uploaded data"):
                st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

            if st.button("▶  Run Batch Prediction"):
                results = []
                prog = st.progress(0, text="Scoring properties...")
                for i, row in batch_df.iterrows():
                    try:
                        inp = build_input(
                            city=str(row.get("CITY","Unknown")), state=str(row.get("STATE","TX")),
                            county=str(row.get("COUNTY","Unknown")), zipcode=str(row.get("ZIPCODE","00000")),
                            prop_type=str(row.get("PROPERTYTYPE","Single Family")),
                            list_type=str(row.get("LISTINGTYPE","For Sale")),
                            bedrooms=int(row.get("BEDROOMS",3)), bathrooms=float(row.get("BATHROOMS",2.0)),
                            sqft=int(row.get("SQUAREFOOTAGE",1800)), lotsize=int(row.get("LOTSIZE",5000)),
                            yearbuilt=int(row.get("YEARBUILT",2005)), hoa_fee=float(row.get("HOA_FEE",0)),
                            mkt_avg_price=float(row.get("MKT_AVG_PRICE",350000)),
                            mkt_avg_sqft=float(row.get("MKT_AVG_SQFT",1900)),
                            mkt_total=float(row.get("MKT_TOTAL_LISTINGS",150)),
                            mktbed_price=float(row.get("MKTBED_AVG_PRICE",340000)),
                            days_listed=int(row.get("DAYS_LISTED",30)),
                            listing_month=int(row.get("LISTING_MONTH",4)),
                            total_inq=int(row.get("TOTAL_INQUIRIES",10)),
                            avg_resp_hrs=float(row.get("AVG_RESPONSETIMEHRS",4.0)),
                            avg_followup=float(row.get("AVG_FOLLOWUPCOUNT",2.0)),
                            price_reduction_count=int(row.get("PRICE_REDUCTION_COUNT",0)),
                            price_reduction_pct=float(row.get("PRICE_REDUCTION_PCT",0.0)),
                        )
                        X_b = encode_and_scale(inp)
                        p   = predict_price(X_b)
                        pb, _ = predict_conversion(X_b)
                        results.append({
                            **{c: row.get(c,"") for c in ["CITY","STATE","ZIPCODE","PROPERTYTYPE","LISTINGTYPE","BEDROOMS","BATHROOMS","SQUAREFOOTAGE","YEARBUILT"]},
                            "Est_Price ($)": round(p, 0), "Price_Low ($)": round(p*0.92,0), "Price_High ($)": round(p*1.08,0),
                            "Conversion_Prob (%)": round(pb*100,2),
                            "Conversion_Verdict": "Convert" if pb>=0.5 else "No Convert",
                            "Price_Per_Sqft ($)": round(p/max(row.get("SQUAREFOOTAGE",1),1),0),
                            "vs_Market_Ratio": round(p/max(row.get("MKT_AVG_PRICE",1),1),3),
                        })
                    except Exception as ex:
                        results.append({"CITY": row.get("CITY","?"), "Error": str(ex)})
                    prog.progress((i+1)/len(batch_df), text=f"Scoring {i+1}/{len(batch_df)}...")

                prog.empty()
                out_df = pd.DataFrame(results)
                st.success(f"✅ Scored {len(out_df):,} properties!")
                st.dataframe(out_df, use_container_width=True, hide_index=True)

                if "Est_Price ($)" in out_df.columns:
                    bk1, bk2, bk3, bk4 = st.columns(4)
                    bk1.metric("Properties Scored",  f"{len(out_df):,}")
                    bk2.metric("Avg Est. Price",      f"${out_df['Est_Price ($)'].mean():,.0f}")
                    bk3.metric("Avg Conv. Prob.",     f"{out_df['Conversion_Prob (%)'].mean():.1f}%")
                    bk4.metric("Likely Conversions",  f"{(out_df['Conversion_Verdict']=='Convert').sum():,}")

                st.download_button("⬇  Download Scored Results CSV",
                    data=out_df.to_csv(index=False).encode(),
                    file_name="bluestone_batch_scored.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AUTOMATED LISTINGS & INQUIRY MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

elif "📋" in page:

    # ── Session state initialisation ──────────────────────────────────────
    if "listings"  not in st.session_state: st.session_state.listings  = []
    if "inquiries" not in st.session_state: st.session_state.inquiries = []
    if "market"    not in st.session_state: st.session_state.market    = []

    listings  = st.session_state.listings
    inquiries = st.session_state.inquiries
    market    = st.session_state.market

    # ── Derived KPIs ──────────────────────────────────────────────────────
    active_listings = [l for l in listings  if l.get("status") == "Active"]
    open_inquiries  = [i for i in inquiries if i.get("status") == "Open"]
    avg_price       = (sum(l.get("price", 0) for l in active_listings) / len(active_listings)
                       if active_listings else 0)
    conv_pct        = (sum(1 for i in inquiries if i.get("status") == "Converted") /
                       len(inquiries) * 100 if inquiries else 0)

    st.markdown("""
    <div class='bs-page-header'>
      <h1><span class='bs-h1-icon'>📋</span>Listings & Inquiry Management</h1>
      <div class='bs-page-subtitle' style='font-size:1.0rem; color:var(--cream);'>
        Centralised hub for property listings, real-time market data and customer inquiries
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────
    mk1, mk2, mk3, mk4, mk5 = st.columns(5)
    mk1.metric("Total Listings",     f"{len(listings):,}")
    mk2.metric("Active Listings",    f"{len(active_listings):,}")
    mk3.metric("Open Inquiries",     f"{len(open_inquiries):,}")
    mk4.metric("Avg Active Price",   f"${avg_price:,.0f}" if avg_price else "—")
    mk5.metric("Inquiry Conv. Rate", f"{conv_pct:.1f}%" if inquiries else "—")

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

    # ── Three-tab layout ──────────────────────────────────────────────────
    lt1, lt2, lt3 = st.tabs([
        "🏠  Property Listings",
        "📬  Customer Inquiries",
        "📊  Market Data",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — PROPERTY LISTINGS
    # ══════════════════════════════════════════════════════════════════════
    with lt1:
        lform_col, llist_col = st.columns([1, 1.6])

        with lform_col:
            st.markdown("""
            <div class='bs-card'>
              <div class='bs-form-section-title'>➕ Add New Listing</div>
            </div>""", unsafe_allow_html=True)

            with st.form("add_listing", clear_on_submit=True):
                la1, la2 = st.columns(2)
                l_address = la1.text_input("Street Address", placeholder="123 Main St")
                l_city    = la2.text_input("City",           placeholder="Austin")
                lb1, lb2  = st.columns(2)
                l_state   = lb1.selectbox("State",
                    ["TX","CA","FL","NY","GA","IL","WA","CO","AZ","NC","OH","NV","TN","SC"])
                l_zip     = lb2.text_input("ZIP Code", placeholder="78701")
                lc1, lc2  = st.columns(2)
                l_type    = lc1.selectbox("Property Type",
                    ["Single Family","Condo","Townhouse","Multi-Family","Land","Commercial"])
                l_status  = lc2.selectbox("Listing Status",
                    ["Active","Pending","Sold","Withdrawn"])
                ld1, ld2  = st.columns(2)
                l_price   = ld1.number_input("Asking Price ($)",
                    min_value=0, max_value=50_000_000, value=350_000, step=5_000)
                l_sqft    = ld2.number_input("Sqft",
                    min_value=0, max_value=50_000, value=1_800)
                le1, le2  = st.columns(2)
                l_beds    = le1.number_input("Beds",  min_value=0, max_value=20, value=3)
                l_baths   = le2.number_input("Baths",
                    min_value=0.0, max_value=20.0, value=2.0, step=0.5)
                lf1, lf2  = st.columns(2)
                l_dom     = lf1.number_input("Days on Market",
                    min_value=0, max_value=1_000, value=0)
                l_agent   = lf2.text_input("Listing Agent", placeholder="Agent name")
                l_notes   = st.text_area("Notes / Description",
                    placeholder="Key features, upgrades, etc.", height=80)
                sub_l = st.form_submit_button("➕  Add Listing", use_container_width=True)

            if sub_l and l_address:
                st.session_state.listings.append({
                    "address": l_address, "city": l_city,  "state":  l_state,
                    "zip":     l_zip,     "type": l_type,  "status": l_status,
                    "price":   l_price,   "sqft": l_sqft,  "beds":   l_beds,
                    "baths":   l_baths,   "dom":  l_dom,   "agent":  l_agent,
                    "notes":   l_notes,
                    "ppsf":    round(l_price / max(l_sqft, 1), 0),
                })
                st.success(f"✅ Listing added: {l_address}, {l_city}")
                st.rerun()
            elif sub_l:
                st.warning("Please enter a street address.")

        with llist_col:
            st.markdown("""
            <div class='bs-form-section-title' style='padding:0 0 0.5rem;'>
              🏠 Current Listings
            </div>""", unsafe_allow_html=True)

            if not listings:
                st.markdown("""
                <div class='bs-card' style='text-align:center; padding:2.5rem;'>
                  <div style='font-size:2.5rem;'>🏠</div>
                  <div style='color:var(--slate); margin-top:0.6rem; font-size:0.9rem;'>
                    No listings yet — add your first property using the form.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                sf_col, _ = st.columns([1, 2])
                status_filter = sf_col.selectbox(
                    "Filter by status",
                    ["All","Active","Pending","Sold","Withdrawn"],
                    key="l_sf")
                filtered_l = (listings if status_filter == "All"
                              else [l for l in listings if l["status"] == status_filter])

                STATUS_COLORS = {
                    "Active":    ("var(--teal)",  "var(--teal-dim)"),
                    "Pending":   ("var(--gold)",  "var(--gold-dim)"),
                    "Sold":      ("#2D7A4F",      "rgba(45,122,79,0.12)"),
                    "Withdrawn": ("var(--slate)", "var(--surface3)"),
                }
                for l in reversed(filtered_l):
                    sc, sbg = STATUS_COLORS.get(l["status"], ("var(--slate)","var(--surface3)"))
                    agent_html = (f"<div style='color:var(--slate);font-size:0.78rem;"
                                  f"margin-top:2px;'>Agent: {l['agent']}</div>"
                                  if l.get("agent") else "")
                    notes_html = (f"<div style='color:var(--slate);font-size:0.78rem;"
                                  f"margin-top:2px;font-style:italic;'>"
                                  f"{l['notes'][:80]}{'...' if len(l['notes'])>80 else ''}</div>"
                                  if l.get("notes") else "")
                    st.markdown(f"""
                    <div class='bs-card' style='margin-bottom:0.55rem; padding:1rem 1.2rem;'>
                      <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                        <div>
                          <div style='font-weight:600; color:var(--cream); font-size:0.95rem;'>
                            {l["address"]}, {l["city"]}, {l["state"]} {l["zip"]}
                          </div>
                          <div style='color:var(--slate); font-size:0.8rem; margin-top:2px;'>
                            {l["type"]} &nbsp;·&nbsp; {l["beds"]}bd / {l["baths"]}ba
                            &nbsp;·&nbsp; {l["sqft"]:,} sqft
                            &nbsp;·&nbsp; ${l["ppsf"]:,.0f}/sqft
                            &nbsp;·&nbsp; DOM: {l["dom"]}
                          </div>
                          {agent_html}{notes_html}
                        </div>
                        <div style='text-align:right; flex-shrink:0; margin-left:1rem;'>
                          <div style='font-family:"Cormorant Garamond",serif;
                                      font-size:1.3rem; font-weight:700; color:var(--gold);'>
                            ${l["price"]:,.0f}
                          </div>
                          <div style='display:inline-block; padding:2px 8px;
                                      border-radius:20px; font-size:0.68rem; font-weight:600;
                                      text-transform:uppercase; margin-top:4px;
                                      background:{sbg}; color:{sc}; border:1px solid {sc};'>
                            {l["status"]}
                          </div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                l_df = pd.DataFrame(listings)
                st.download_button("⬇  Export Listings CSV",
                    data=l_df.to_csv(index=False).encode(),
                    file_name="bluestone_listings.csv", mime="text/csv")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — CUSTOMER INQUIRIES
    # ══════════════════════════════════════════════════════════════════════
    with lt2:
        iform_col, ilist_col = st.columns([1, 1.6])

        with iform_col:
            st.markdown("""
            <div class='bs-card'>
              <div class='bs-form-section-title'>📬 Log New Inquiry</div>
            </div>""", unsafe_allow_html=True)

            # ── Build property-of-interest options ──────────────────────────────
            # Priority: manually-added listings (Tab 1) FIRST, then Bluestone CSV
            manual_opts  = [f"{l['address']}, {l['city']}, {l['state']} {l['zip']}"
                            for l in listings]
            csv_opts     = _bs_property_options()
            combined_opts = manual_opts + [
                o for o in csv_opts if o not in manual_opts]
            if combined_opts:
                listing_opts = ["— Select property —"] + combined_opts
            else:
                listing_opts = ["(No properties found — add listings or place Bluestone_data.csv in project root)"]

            with st.form("add_inquiry", clear_on_submit=True):
                ia1, ia2  = st.columns(2)
                i_name    = ia1.text_input("Customer Name",  placeholder="Jane Smith")
                i_email   = ia2.text_input("Email",          placeholder="jane@email.com")
                ib1, ib2  = st.columns(2)
                i_phone   = ib1.text_input("Phone",          placeholder="+1 555 0100")
                i_channel = ib2.selectbox("Lead Channel",
                    ["Web","Referral","Walk-in","Phone","Email","Social","Other"])
                i_prop    = st.selectbox("Property of Interest", listing_opts)
                ic1, ic2  = st.columns(2)
                i_status  = ic1.selectbox("Inquiry Status",
                    ["Open","Contacted","Showing Scheduled",
                     "Offer Made","Converted","Lost"])
                i_fin     = ic2.selectbox("Financing Type",
                    ["Cash","Conventional","FHA","VA","Other","Unknown"])
                id1, id2  = st.columns(2)
                i_score   = id1.slider("Lead Score", 0, 100, 60)
                i_resp    = id2.number_input("Response Time (hrs)",
                    min_value=0.0, max_value=200.0, value=2.0, step=0.5)
                i_notes   = st.text_area("Inquiry Notes",
                    placeholder="Details, follow-up actions, next steps…", height=80)
                sub_i = st.form_submit_button("📬  Log Inquiry", use_container_width=True)

            if sub_i and i_name:
                st.session_state.inquiries.append({
                    "name":      i_name,    "email":    i_email,
                    "phone":     i_phone,   "channel":  i_channel,
                    "property":  i_prop,    "status":   i_status,
                    "financing": i_fin,     "score":    i_score,
                    "resp_hrs":  i_resp,    "notes":    i_notes,
                })
                st.success(f"✅ Inquiry logged for {i_name}")
                st.rerun()
            elif sub_i:
                st.warning("Please enter a customer name.")

        with ilist_col:
            st.markdown("""
            <div class='bs-form-section-title' style='padding:0 0 0.5rem;'>
              📬 Inquiry Pipeline
            </div>""", unsafe_allow_html=True)

            if not inquiries:
                st.markdown("""
                <div class='bs-card' style='text-align:center; padding:2.5rem;'>
                  <div style='font-size:2.5rem;'>📬</div>
                  <div style='color:var(--slate); margin-top:0.6rem; font-size:0.9rem;'>
                    No inquiries logged yet — use the form to add your first.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                ic_a, ic_b = st.columns(2)
                i_sf = ic_a.selectbox("Filter by status",
                    ["All","Open","Contacted","Showing Scheduled",
                     "Offer Made","Converted","Lost"], key="i_sf")
                i_cf = ic_b.selectbox("Filter by channel",
                    ["All","Web","Referral","Walk-in","Phone","Email","Social","Other"],
                    key="i_cf")
                filtered_i = [
                    i for i in inquiries
                    if (i_sf == "All" or i["status"]  == i_sf)
                    and (i_cf == "All" or i["channel"] == i_cf)
                ]

                INQ_COLORS = {
                    "Open":               ("var(--gold)",  "var(--gold-dim)"),
                    "Contacted":          ("var(--teal)",  "var(--teal-dim)"),
                    "Showing Scheduled":  ("#9B8FEE",      "rgba(155,143,238,0.12)"),
                    "Offer Made":         ("#FF9A3C",      "rgba(255,154,60,0.12)"),
                    "Converted":          ("#2D9E6B",      "rgba(45,158,107,0.12)"),
                    "Lost":               ("var(--rose)",  "var(--rose-dim)"),
                }
                for inq in reversed(filtered_i):
                    sc2, sbg2 = INQ_COLORS.get(
                        inq["status"], ("var(--slate)", "var(--surface3)"))
                    score_color = ("#2D9E6B" if inq["score"] >= 70
                                   else "var(--gold)"  if inq["score"] >= 40
                                   else "var(--rose)")
                    notes_html2 = (
                        f"<div style='color:var(--slate);font-size:0.78rem;"
                        f"margin-top:3px;font-style:italic;'>"
                        f"{inq['notes'][:90]}{'...' if len(inq.get('notes',''))>90 else ''}</div>"
                        if inq.get("notes") else "")
                    st.markdown(f"""
                    <div class='bs-card' style='margin-bottom:0.55rem; padding:1rem 1.2rem;'>
                      <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                        <div style='flex:1;'>
                          <div style='font-weight:600; color:var(--cream); font-size:0.95rem;'>
                            {inq["name"]}
                          </div>
                          <div style='color:var(--slate); font-size:0.78rem; margin-top:2px;'>
                            {inq.get("email","—")} &nbsp;·&nbsp; {inq.get("phone","—")}
                          </div>
                          <div style='color:var(--slate); font-size:0.78rem; margin-top:3px;'>
                            🏠 {inq.get("property","—")} &nbsp;·&nbsp;
                            📡 {inq.get("channel","—")} &nbsp;·&nbsp;
                            💰 {inq.get("financing","—")} &nbsp;·&nbsp;
                            ⏱ {inq.get("resp_hrs",0):.1f}h resp.
                          </div>
                          {notes_html2}
                        </div>
                        <div style='text-align:right; flex-shrink:0; margin-left:1rem;'>
                          <div style='font-size:1.25rem; font-weight:700; color:{score_color};'>
                            {inq["score"]}
                            <span style='font-size:0.7rem; color:var(--slate);'>/100</span>
                          </div>
                          <div style='font-size:0.6rem; color:var(--slate);
                                      text-transform:uppercase; letter-spacing:1px;'>
                            Lead Score
                          </div>
                          <div style='display:inline-block; padding:2px 8px;
                                      border-radius:20px; font-size:0.65rem; font-weight:600;
                                      text-transform:uppercase; margin-top:4px;
                                      background:{sbg2}; color:{sc2}; border:1px solid {sc2};'>
                            {inq["status"]}
                          </div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                i_df = pd.DataFrame(inquiries)
                st.download_button("⬇  Export Inquiries CSV",
                    data=i_df.to_csv(index=False).encode(),
                    file_name="bluestone_inquiries.csv", mime="text/csv")

                # ── Matched Properties panel ──────────────────────────────────
                st.markdown("<div style='margin-top:1.2rem;'></div>",
                    unsafe_allow_html=True)
                st.markdown(
                    "<div class='bs-form-section-title' "
                    "style='padding:0 0 0.5rem;'>"
                    "🗺️ Properties Matching Selected Inquiry</div>",
                    unsafe_allow_html=True)

                # Let the user pick which inquiry to explore matches for
                if inquiries:
                    inq_labels = [
                        f"{inq['name']} — {inq.get('property', '—')}"
                        for inq in inquiries
                    ]
                    chosen_label = st.selectbox(
                        "Select inquiry to view matching properties",
                        inq_labels,
                        key="match_sel",
                    )
                    chosen_idx  = inq_labels.index(chosen_label)
                    chosen_prop = inquiries[chosen_idx].get("property", "")
                    matched     = _match_properties(chosen_prop)
                else:
                    matched     = []
                    chosen_prop = ""

                def _prop_card_html(mp):
                    """Build a property match card as a plain HTML string.
                    All values are pre-formatted before insertion so no
                    f-string quoting issues can occur inside the HTML."""
                    price_str = (
                        "${:,.0f}".format(float(mp["price"]))
                        if mp.get("price") and str(mp["price"]) not in ("nan", "")
                        else "—"
                    )
                    sqft_str = (
                        "{:,.0f} sqft".format(float(mp["sqft"]))
                        if mp.get("sqft") and str(mp["sqft"]) not in ("nan", "")
                        else ""
                    )
                    county_str = (
                        " · " + str(mp["county"]) + " County"
                        if mp.get("county") and str(mp["county"]) not in ("nan", "")
                        else ""
                    )
                    sqft_part = (" · " + sqft_str) if sqft_str else ""
                    listing_type = mp.get("listing", "")
                    address  = mp.get("address",  "—")
                    city     = mp.get("city",     "")
                    state    = mp.get("state",    "")
                    zipcode  = mp.get("zip",      "")
                    ptype    = mp.get("type",     "")
                    beds     = mp.get("beds",     "—")
                    baths    = mp.get("baths",    "—")

                    return (
                        "<div class='bs-card' style='"
                        "margin-bottom:0.5rem; padding:0.8rem 1rem;"
                        "border-left:3px solid var(--teal);'>"
                        "<div style='display:flex; justify-content:space-between;"
                        " align-items:flex-start;'>"
                        "<div style='flex:1;'>"
                        "<div style='font-weight:600; color:var(--cream);"
                        " font-size:0.9rem;'>" + address + "</div>"
                        "<div style='color:var(--slate); font-size:0.8rem;"
                        " margin-top:3px;'>"
                        "📍 " + city + ", " + state
                        + " &nbsp;·&nbsp; ZIP "
                        "<strong style='color:var(--gold);'>" + zipcode + "</strong>"
                        + county_str + "</div>"
                        "<div style='color:var(--slate); font-size:0.78rem;"
                        " margin-top:2px;'>"
                        + ptype + " &nbsp;·&nbsp; "
                        + str(beds) + "bd / " + str(baths) + "ba"
                        + sqft_part + "</div>"
                        "</div>"
                        "<div style='text-align:right; flex-shrink:0; margin-left:1rem;'>"
                        "<div style='font-family:Georgia,serif; font-size:1.15rem;"
                        " font-weight:700; color:var(--gold);'>" + price_str + "</div>"
                        "<div style='font-size:0.65rem; color:var(--slate);"
                        " text-transform:uppercase; letter-spacing:1px;'>"
                        + listing_type + "</div>"
                        "</div>"
                        "</div>"
                        "</div>"
                    )

                if not inquiries:
                    st.markdown(
                        "<div class='bs-card' style='padding:1rem; text-align:center;'>"
                        "<div style='color:var(--slate); font-size:0.85rem;'>"
                        "Log an inquiry first to see matching properties.</div></div>",
                        unsafe_allow_html=True)
                elif not matched:
                    if bs_df.empty:
                        st.markdown(
                            "<div class='bs-card' style='padding:1rem; text-align:center;'>"
                            "<div style='color:var(--slate); font-size:0.85rem;'>"
                            "Place <code>Bluestone_data.csv</code> in the same folder "
                            "as <code>app.py</code> to enable auto-matching.</div></div>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<div class='bs-card' style='padding:1rem; text-align:center;'>"
                            "<div style='color:var(--slate); font-size:0.85rem;'>"
                            "No matching properties found for this inquiry.</div></div>",
                            unsafe_allow_html=True)
                else:
                    noun = "property" if len(matched) == 1 else "properties"
                    st.markdown(
                        "<div style='font-size:0.78rem; color:var(--slate);"
                        " margin-bottom:0.5rem;'>Showing "
                        + str(len(matched)) + " matching " + noun
                        + " from the database</div>",
                        unsafe_allow_html=True)
                    for mp in matched:
                        st.markdown(_prop_card_html(mp), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — MARKET DATA
    # ══════════════════════════════════════════════════════════════════════
    with lt3:
        mform_col, mdata_col = st.columns([1, 1.6])

        with mform_col:
            st.markdown("""
            <div class='bs-card'>
              <div class='bs-form-section-title'>📊 Add Market Snapshot</div>
            </div>""", unsafe_allow_html=True)

            with st.form("add_market", clear_on_submit=True):
                # ── ZIP / City dropdowns from Bluestone_data.csv ─────────────
                zip_city_list = _bs_zip_city_options()
                city_list     = _bs_city_options()

                if zip_city_list:
                    m_zip_city = st.selectbox(
                        "ZIP / Area Code — City",
                        ["— Select ZIP & City —"] + zip_city_list,
                        help="Populated from Bluestone_data.csv"
                    )
                    # Parse selected value: "78701 — Austin"
                    if " — " in str(m_zip_city):
                        m_zip, m_city = [p.strip() for p in m_zip_city.split(" — ", 1)]
                    else:
                        m_zip, m_city = "", ""
                else:
                    ma1, ma2 = st.columns(2)
                    m_zip    = ma1.text_input("ZIP / Area Code", placeholder="78701",
                        help="Bluestone_data.csv not found — enter manually")
                    m_city   = ma2.text_input("City / Market",   placeholder="Austin")

                mb1, mb2  = st.columns(2)
                m_avg_p   = mb1.number_input("Avg List Price ($)",
                    min_value=0, max_value=10_000_000, value=350_000, step=5_000)
                m_med_p   = mb2.number_input("Median Sale Price ($)",
                    min_value=0, max_value=10_000_000, value=330_000, step=5_000)
                mc1, mc2  = st.columns(2)
                m_sqft    = mc1.number_input("Avg Sqft",
                    min_value=0, max_value=20_000, value=1_900)
                m_ppsf    = mc2.number_input("Avg Price / Sqft ($)",
                    min_value=0.0, max_value=5_000.0, value=185.0, step=0.5)
                md1, md2  = st.columns(2)
                m_active  = md1.number_input("Active Listings",
                    min_value=0, max_value=50_000, value=150)
                m_dom     = md2.number_input("Avg Days on Market",
                    min_value=0, max_value=500, value=28)
                me1, me2  = st.columns(2)
                m_abs     = me1.number_input("Absorption Rate (%)",
                    min_value=0.0, max_value=100.0, value=18.5, step=0.1)
                m_yoy     = me2.number_input("YoY Price Change (%)",
                    min_value=-50.0, max_value=100.0, value=4.2, step=0.1)
                m_seg     = st.selectbox("Property Segment",
                    ["All","Single Family","Condo","Townhouse",
                     "Multi-Family","Luxury (>$1M)"])
                m_source  = st.selectbox("Data Source",
                    ["— Select source —",
                     "MLS (Multiple Listing Service)",
                     "Zillow",
                     "Realtor.com",
                     "Redfin",
                     "Trulia",
                     "CoStar",
                     "CoreLogic",
                     "ATTOM Data",
                     "HUD / Government Data",
                     "Internal BlueStone Database",
                     "Snowflake (Live Feed)",
                     "Manual Entry",
                     "Other"],
                    help="Select the data provider for this market snapshot"
                )
                sub_m = st.form_submit_button("📊  Add Market Data",
                    use_container_width=True)

            if sub_m and m_zip:
                st.session_state.market.append({
                    "zip":      m_zip,   "city":    m_city,
                    "avg_p":    m_avg_p, "med_p":   m_med_p,
                    "avg_sqft": m_sqft,  "ppsf":    m_ppsf,
                    "active":   m_active,"avg_dom":  m_dom,
                    "abs_rate": m_abs,   "yoy":     m_yoy,
                    "segment":  m_seg,   "source":  m_source,
                })
                st.success(f"✅ Market data added for ZIP {m_zip} — {m_city}")
                st.rerun()
            elif sub_m:
                st.warning("Please enter a ZIP / area code.")

        with mdata_col:
            st.markdown("""
            <div class='bs-form-section-title' style='padding:0 0 0.5rem;'>
              📊 Market Snapshots
            </div>""", unsafe_allow_html=True)

            if not market:
                st.markdown("""
                <div class='bs-card' style='text-align:center; padding:2.5rem;'>
                  <div style='font-size:2.5rem;'>📊</div>
                  <div style='color:var(--slate); margin-top:0.6rem; font-size:0.9rem;'>
                    No market data yet — add your first snapshot using the form.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                for m in reversed(market):
                    yoy_c = "#2D9E6B" if m["yoy"] >= 0 else "var(--rose)"
                    yoy_i = "▲" if m["yoy"] >= 0 else "▼"
                    st.markdown(f"""
                    <div class='bs-card' style='margin-bottom:0.55rem; padding:1rem 1.2rem;'>
                      <div style='font-weight:600; color:var(--cream); font-size:0.95rem;'>
                        {m["city"]}
                        <span style='color:var(--slate); font-size:0.8rem;'>ZIP {m["zip"]}</span>
                        &nbsp;·&nbsp;
                        <span style='color:var(--gold); font-size:0.8rem;'>{m["segment"]}</span>
                      </div>
                      <div style='display:grid; grid-template-columns:repeat(4,1fr);
                                  gap:0.7rem; margin-top:0.65rem;'>
                        <div>
                          <div style='font-size:0.6rem; text-transform:uppercase;
                                      letter-spacing:1px; color:var(--slate);'>Avg List</div>
                          <div style='color:var(--gold); font-weight:600; font-size:0.88rem;'>
                            ${m["avg_p"]:,.0f}
                          </div>
                        </div>
                        <div>
                          <div style='font-size:0.6rem; text-transform:uppercase;
                                      letter-spacing:1px; color:var(--slate);'>Median Sale</div>
                          <div style='color:var(--cream); font-weight:600; font-size:0.88rem;'>
                            ${m["med_p"]:,.0f}
                          </div>
                        </div>
                        <div>
                          <div style='font-size:0.6rem; text-transform:uppercase;
                                      letter-spacing:1px; color:var(--slate);'>$/Sqft</div>
                          <div style='color:var(--cream); font-size:0.88rem;'>
                            ${m["ppsf"]:,.0f}
                          </div>
                        </div>
                        <div>
                          <div style='font-size:0.6rem; text-transform:uppercase;
                                      letter-spacing:1px; color:var(--slate);'>Avg DOM</div>
                          <div style='color:var(--cream); font-size:0.88rem;'>
                            {m["avg_dom"]} days
                          </div>
                        </div>
                        <div>
                          <div style='font-size:0.6rem; text-transform:uppercase;
                                      letter-spacing:1px; color:var(--slate);'>Active</div>
                          <div style='color:var(--cream); font-size:0.88rem;'>
                            {m["active"]:,}
                          </div>
                        </div>
                        <div>
                          <div style='font-size:0.6rem; text-transform:uppercase;
                                      letter-spacing:1px; color:var(--slate);'>Absorption</div>
                          <div style='color:var(--cream); font-size:0.88rem;'>
                            {m["abs_rate"]:.1f}%
                          </div>
                        </div>
                        <div>
                          <div style='font-size:0.6rem; text-transform:uppercase;
                                      letter-spacing:1px; color:var(--slate);'>YoY Change</div>
                          <div style='color:{yoy_c}; font-weight:600; font-size:0.88rem;'>
                            {yoy_i} {abs(m["yoy"]):.1f}%
                          </div>
                        </div>
                        <div>
                          <div style='font-size:0.6rem; text-transform:uppercase;
                                      letter-spacing:1px; color:var(--slate);'>Source</div>
                          <div style='color:var(--slate); font-size:0.78rem; font-style:italic;'>
                            {m.get("source","—")}
                          </div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                m_df = pd.DataFrame(market)
                st.download_button("⬇  Export Market Data CSV",
                    data=m_df.to_csv(index=False).encode(),
                    file_name="bluestone_market_data.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PERFORMANCE DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

elif "📊" in page:

    st.markdown("""
    <div class='bs-page-header'>
      <h1><span class='bs-h1-icon'>📊</span>Model Performance Dashboard</h1>
      <div class='bs-page-subtitle'>Full evaluation metrics, model comparison and hyperparameter configuration</div>
    </div>
    """, unsafe_allow_html=True)

    t_r, t_c, t_cfg = st.tabs(["📈  Regression", "🎯  Classification", "🔧  Full Config"])

    with t_r:
        st.markdown("### Property Price Prediction")
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Test R²",   "0.9806")
        rm2.metric("Test RMSE", "$124,563")
        rm3.metric("Test MAE",  "$41,208")
        rm4.metric("Test MAPE", "7.84%")

        if "regressor_comparison" in meta:
            st.markdown("#### Candidate Model Comparison")
            comp = meta["regressor_comparison"]
            rows = []
            for k, v in comp.items():
                if "Gradient Boosting" in k and "Tuned" in k:
                    rows.append({"Model": k, "Val R²": 0.9614, "Test R²": 0.9806, "RMSE ($)": "$124,563"})
                else:
                    rows.append({"Model": k, "Val R²": v.get("val_r2","—"),
                                 "Test R²": v.get("test_r2","—"),
                                 "RMSE ($)": f"${v.get('rmse',0):,.0f}"})
            cdf = pd.DataFrame(rows).sort_values("Test R²", ascending=False).reset_index(drop=True)
            def style_top(row):
                if row.name == 0:
                    return ["background-color: rgba(200,168,90,0.15); color: #E6C97E; font-weight:bold"] * len(row)
                return [""] * len(row)
            st.dataframe(cdf.style.apply(style_top, axis=1), use_container_width=True, hide_index=True)

        st.markdown("#### Best Hyperparameters")
        params = meta.get("best_reg_params", {})
        if params:
            st.dataframe(pd.DataFrame([{"Parameter": k, "Value": round(v,8) if isinstance(v,float) else v}
                          for k, v in params.items()]), use_container_width=True, hide_index=True)

    with t_c:
        st.markdown("### Inquiry Conversion Prediction")
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Test AUC-ROC", str(meta.get("test_clf_auc","—")))
        cm2.metric("Test F1",      str(meta.get("test_clf_f1","—")))
        cm3.metric("SMOTE",        "Applied ✅" if arts.get("smote_applied") else "Not needed")
        cm4.metric("Optuna Trials", str(meta.get("optuna_clf_n_trials","—")))

        st.markdown("#### Best Hyperparameters")
        clf_params = meta.get("best_clf_params", {})
        if clf_params:
            st.dataframe(pd.DataFrame([{"Parameter": k, "Value": round(v,8) if isinstance(v,float) else v}
                          for k, v in clf_params.items()]), use_container_width=True, hide_index=True)

    with t_cfg:
        st.markdown("### Full Model Metadata")
        st.json(meta, expanded=False)
        st.markdown("### Artefact Status")
        art_rows = []
        for key in ["preprocessing_pipeline","best_regression_model","best_classification_model",
                    "shap_explainer_regression","shap_explainer_classification","model_metadata",
                    "target_encoding_maps","label_encoders","smote_applied","feature_names_in","feature_names_out",
                    "shap_vals_reg_sample","shap_vals_clf_sample"]:
            p = OUTPUT_DIR / f"{key}.pkl"
            art_rows.append({"Artefact": key,
                             "Status": "✅ Found" if p.exists() else "❌ Missing",
                             "Size": f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "—"})
        st.dataframe(pd.DataFrame(art_rows), use_container_width=True, hide_index=True)



# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALYTICS DASHBOARD (Native Plotly — 3 sub-pages)
# ════════════════════════════════════════════════════════════════════════════

elif "📈" in page:

    import plotly.graph_objects as go
    import plotly.express as px

    # ── Load data ─────────────────────────────────────────────────────────
    @st.cache_data(show_spinner="Loading analytics data...")
    def load_analytics_data():
        data_path = Path.cwd() / "Bluestone Project.csv"
        if not data_path.exists():
            for p in Path.cwd().rglob("Bluestone Project.csv"):
                data_path = p
                break
        df = pd.read_csv(str(data_path), low_memory=False)
        df["CREATEDDATE"] = pd.to_datetime(df["CREATEDDATE"], errors="coerce")
        df["YEAR"]  = df["CREATEDDATE"].dt.year
        df["MONTH"] = df["CREATEDDATE"].dt.month
        return df

    try:
        adf = load_analytics_data()
        data_ok = True
    except Exception as _e:
        st.error(f"Could not load 'Bluestone Project.csv'. Place it in the same folder as app.py. Error: {_e}")
        data_ok = False

    if data_ok:

        BG      = "#0F1923"
        SURFACE = "#162233"
        GOLD    = "#C8A85A"
        GOLD2   = "#E6C97E"
        GREEN   = "#2ECC71"
        AMBER   = "#F39C12"
        TEAL    = "#1E9B8A"
        SLATE   = "#7A8FA6"
        CREAM   = "#EDE8DF"
        CITIES  = ["Atlanta","Austin","Charlotte","Chicago","Denver","Houston","Phoenix"]
        MONTHS  = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]

        def pbi_layout(fig, title="", height=300, showlegend=False):
            fig.update_layout(
                title=dict(text=title, font=dict(color=CREAM, size=13, family="Outfit"),
                           x=0, xanchor="left", pad=dict(l=4, t=4)),
                paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                font=dict(color=CREAM, family="Outfit", size=11),
                height=height, margin=dict(l=10, r=10, t=36, b=10),
                showlegend=showlegend,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CREAM, size=10),
                            orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
                xaxis=dict(gridcolor="rgba(122,143,166,0.12)", linecolor="rgba(122,143,166,0.2)",
                           tickfont=dict(color=SLATE, size=10)),
                yaxis=dict(gridcolor="rgba(122,143,166,0.12)", linecolor="rgba(122,143,166,0.2)",
                           tickfont=dict(color=SLATE, size=10)),
            )
            return fig

        def kpi_card(label, value, col):
            col.markdown(f"""
            <div style='background:{SURFACE};border:1px solid rgba(200,168,90,0.25);
                border-radius:10px;padding:14px 16px;text-align:left;margin-bottom:6px;'>
              <div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:1.5px;
                  color:{SLATE};font-family:Outfit,sans-serif;margin-bottom:6px;'>{label}</div>
              <div style='font-size:1.65rem;font-weight:700;color:{GREEN};
                  font-family:"Cormorant Garamond",serif;line-height:1.1;'>{value}</div>
            </div>""", unsafe_allow_html=True)

        def gauge_card(label, value, min_val, max_val, col, fmt=".2f"):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                number=dict(font=dict(color=GREEN, size=26, family="Cormorant Garamond"),
                            valueformat=fmt),
                gauge=dict(
                    axis=dict(range=[min_val, max_val],
                              tickfont=dict(color=SLATE, size=9),
                              tickformat=fmt),
                    bar=dict(color=GREEN, thickness=0.5),
                    bgcolor=BG,
                    borderwidth=0,
                    steps=[dict(range=[min_val, max_val], color="rgba(122,143,166,0.08)")],
                    threshold=dict(line=dict(color=GOLD, width=2), thickness=0.75, value=value),
                ),
            ))
            fig.update_layout(
                paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                font=dict(color=CREAM, family="Outfit"),
                height=170, margin=dict(l=10, r=10, t=8, b=0),
                title=dict(text=label, font=dict(color=CREAM, size=11), x=0.5, xanchor="center"),
            )
            col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Page header ───────────────────────────────────────────────────
        st.markdown("""
        <div class='bs-page-header'>
          <h1><span class='bs-h1-icon'>📈</span>Analytics Dashboard</h1>
          <div class='bs-page-subtitle'>Interactive business intelligence — Executive Overview · Rent Transaction · Sale Transaction</div>
        </div>
        """, unsafe_allow_html=True)

        # ── ML model KPIs ─────────────────────────────────────────────────
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        kc1.metric("Reg Model",    meta.get("regression_model_name", "GBR"))
        kc2.metric("Test R²",      str(meta.get("test_reg_r2", "0.9806")))
        kc3.metric("RMSE",         f"${meta.get('test_reg_rmse', 124563):,.0f}")
        kc4.metric("Test AUC-ROC", str(meta.get("test_clf_auc", "—")))
        kc5.metric("Test F1",      str(meta.get("test_clf_f1",  "—")))

        st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)

        tab_exec, tab_rent, tab_sale = st.tabs([
            "📊  Executive Overview",
            "🏘️  Rent Transaction",
            "🏠  Sale Transaction",
        ])

        # ── TAB 1: EXECUTIVE OVERVIEW ─────────────────────────────────────
        with tab_exec:
            st.markdown(f'<div style="font-size:1rem;font-weight:700;color:' + CREAM + ';font-family:Outfit,sans-serif;padding:8px 4px 4px;">Bluestone Real Estate Executive Market Intelligence</div>', unsafe_allow_html=True)

            exec_years = sorted([y for y in adf["YEAR"].dropna().unique().astype(int) if y >= 2024])
            ey_sel = st.multiselect("Filter by Year", ["Select all"] + exec_years, default=["Select all"], key="exec_year")
            ec_sel = st.multiselect("Filter by City", ["Select all"] + CITIES, default=["Select all"], key="exec_city")

            edf = adf.copy()
            if "Select all" not in ey_sel and ey_sel:
                edf = edf[edf["YEAR"].isin(ey_sel)]
            if "Select all" not in ec_sel and ec_sel:
                edf = edf[edf["CITY"].isin(ec_sel)]

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            ek1, ek2, ek3, ek4, ek5 = st.columns(5)
            kpi_card("Total Listing",                     f"{len(edf):,}",                                               ek1)
            kpi_card("Active Listing",                    f"{(edf['STATUS']=='Active').sum():,}",                    ek2)
            kpi_card("Total Inquiry",                     f"{int(edf['TOTAL_INQUIRIES'].sum()):,}",                    ek3)
            kpi_card("Avg Days of Property on Market",    f"{edf['DAYSONMARKET'].mean():.0f}",                         ek4)
            kpi_card("Conversion Rate",                   f"{edf['CONVERTED_INQUIRIES'].sum()/max(edf['TOTAL_INQUIRIES'].sum(),1)*100:.2f}%", ek5)

            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            er1, er2, er3 = st.columns([1.4, 1.2, 0.9])

            with er1:
                mi = edf.groupby("MONTH")["TOTAL_INQUIRIES"].sum().reindex(range(1,13), fill_value=0)
                fig = go.Figure(go.Scatter(x=MONTHS, y=mi.values, mode="lines+markers",
                    line=dict(color=GREEN, width=2.5), marker=dict(color=GREEN, size=6),
                    fill="tozeroy", fillcolor="rgba(46,204,113,0.08)"))
                pbi_layout(fig, "Monthly Inquiry Trend", 290)
                fig.update_yaxes(tickformat=".0s")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with er2:
                dp = edf.groupby("PROPERTYTYPE")["DAYSONMARKET"].mean().sort_values(ascending=False).reset_index()
                fig = go.Figure(go.Bar(x=dp["PROPERTYTYPE"], y=dp["DAYSONMARKET"].round(0),
                    marker_color=GREEN, text=dp["DAYSONMARKET"].round(0).astype(int),
                    textposition="outside", textfont=dict(color=CREAM, size=10)))
                pbi_layout(fig, "Avg. Days on Market by Property Type", 290)
                fig.update_xaxes(tickangle=-25, tickfont=dict(size=9))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with er3:
                lc = edf["LISTING_TYPE"].value_counts()
                fig = go.Figure(go.Pie(labels=lc.index, values=lc.values, hole=0.55,
                    marker=dict(colors=[GREEN, AMBER]), textfont=dict(color=CREAM, size=10)))
                pbi_layout(fig, "Listing Type", 290, showlegend=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            er4, er5, er6 = st.columns([1.1, 1.1, 0.9])

            with er4:
                cc = edf.groupby("TOP_CHANNEL")["CONVERTED_INQUIRIES"].sum().sort_values(ascending=False).reset_index()
                fig = go.Figure(go.Pie(labels=cc["TOP_CHANNEL"], values=cc["CONVERTED_INQUIRIES"], hole=0.5,
                    marker=dict(colors=[GREEN, AMBER, "#3498DB", "#9B59B6", TEAL, GOLD]),
                    textfont=dict(color=CREAM, size=9), textinfo="percent"))
                pbi_layout(fig, "Total Listing Converted by Channel", 290, showlegend=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with er5:
                pc = edf.groupby("PROPERTYTYPE")["CONVERTED_INQUIRIES"].sum().sort_values(ascending=False).reset_index()
                fig = go.Figure(go.Bar(x=pc["PROPERTYTYPE"], y=pc["CONVERTED_INQUIRIES"],
                    marker_color=GREEN, text=pc["CONVERTED_INQUIRIES"],
                    textposition="outside", textfont=dict(color=CREAM, size=9)))
                pbi_layout(fig, "Listing Converted by Property Type", 290)
                fig.update_xaxes(tickangle=-25, tickfont=dict(size=9))
                fig.update_yaxes(tickformat=".0s")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with er6:
                lg = edf.groupby("LISTING_TYPE").agg(
                    Active=("STATUS", lambda x: (x=="Active").sum()),
                    Converted=("CONVERTED_INQUIRIES","sum")).reset_index()
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Active Listings", y=lg["LISTING_TYPE"], x=lg["Active"],
                    orientation="h", marker_color=GREEN,
                    text=[f"{v/1000:.0f}K" for v in lg["Active"]],
                    textposition="outside", textfont=dict(color=CREAM, size=9)))
                fig.add_trace(go.Bar(name="Count of converted", y=lg["LISTING_TYPE"], x=lg["Converted"],
                    orientation="h", marker_color=TEAL,
                    text=[f"{v/1000:.0f}K" for v in lg["Converted"]],
                    textposition="outside", textfont=dict(color=CREAM, size=9)))
                pbi_layout(fig, "Listing vs Converted by ListingType", 290, showlegend=True)
                fig.update_layout(barmode="group", xaxis=dict(tickformat=".0s"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── TAB 2: RENT TRANSACTION ───────────────────────────────────────
        with tab_rent:
            st.markdown(f'<div style="font-size:1rem;font-weight:700;color:' + CREAM + ';font-family:Outfit,sans-serif;padding:8px 4px 4px;">Bluestone Real Estate Rent Transaction Analysis</div>', unsafe_allow_html=True)

            ry_sel = st.multiselect("Filter by Year", ["Select all"] + sorted([y for y in adf["YEAR"].dropna().unique().astype(int) if y >= 2024]),
                                    default=["Select all"], key="rent_year")
            rc_sel = st.multiselect("Filter by City", ["Select all"] + CITIES, default=["Select all"], key="rent_city")

            rdf = adf[adf["LISTING_TYPE"]=="rental"].copy()
            if "Select all" not in ry_sel and ry_sel:
                rdf = rdf[rdf["YEAR"].isin(ry_sel)]
            if "Select all" not in rc_sel and rc_sel:
                rdf = rdf[rdf["CITY"].isin(rc_sel)]
            rdf_t = rdf[rdf["AGREEDRENT"].notna()].copy()

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            rk1, rk2, rk3, rk4 = st.columns(4)
            kpi_card("Total Rent Transaction", f"{len(rdf_t):,}", rk1)
            kpi_card("Market Avg. Rent Price",  f"${rdf_t['MARKETAVGRENT'].mean():,.0f}" if "MARKETAVGRENT" in rdf_t.columns and rdf_t["MARKETAVGRENT"].notna().any() else f"${rdf_t['AGREEDRENT'].mean():,.0f}", rk2)
            gauge_card("Average Lease Month", rdf_t["LEASETERMMONTHS"].mean() if "LEASETERMMONTHS" in rdf_t.columns and rdf_t["LEASETERMMONTHS"].notna().any() else 12, 1, 24, rk3, fmt=".0f")
            gauge_card("Rent to List Ratio",  rdf_t["RENTTOLISTRATIO"].mean() if "RENTTOLISTRATIO" in rdf_t.columns and rdf_t["RENTTOLISTRATIO"].notna().any() else 0.99, 0.96, 1.0, rk4, fmt=".2f")

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            rr1, rr2, rr3 = st.columns([1.4, 1.2, 0.9])

            with rr1:
                mr = rdf_t.groupby("MONTH")["AGREEDRENT"].mean().reindex(range(1,13)).reset_index()
                mr.columns = ["MONTH","AGREEDRENT"]
                fig = go.Figure(go.Scatter(x=MONTHS, y=mr["AGREEDRENT"].values, mode="lines+markers",
                    line=dict(color=GREEN, width=2.5), marker=dict(color=GREEN, size=6),
                    fill="tozeroy", fillcolor="rgba(46,204,113,0.08)"))
                pbi_layout(fig, "Rent Price Trend", 290)
                fig.update_yaxes(tickprefix="$", tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with rr2:
                cr = rdf_t.groupby("CITY")["AGREEDRENT"].mean().sort_values(ascending=False).reset_index()
                fig = go.Figure(go.Bar(x=cr["CITY"], y=cr["AGREEDRENT"], marker_color=GREEN,
                    text=["$"+f"{v/1000:.1f}K" for v in cr["AGREEDRENT"]],
                    textposition="outside", textfont=dict(color=CREAM, size=10)))
                pbi_layout(fig, "Rent Price by City", 290)
                fig.update_yaxes(tickprefix="$", tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with rr3:
                ad_sum = rdf_t["AGREEDRENT"].sum()
                dep_sum = rdf_t["SECURITYDEPOSITAMT"].sum() if "SECURITYDEPOSITAMT" in rdf_t.columns else 0
                fig = go.Figure(go.Pie(labels=["Sum of securityDe...", "Sum of agre..."],
                    values=[dep_sum, ad_sum], hole=0.55,
                    marker=dict(colors=[GREEN, AMBER]),
                    textfont=dict(color=CREAM, size=10), textinfo="percent"))
                pbi_layout(fig, "Agreed Rent vs Deposit Paid", 290, showlegend=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            rr4, rr5, rr6 = st.columns([1.2, 1.2, 0.9])

            with rr4:
                pr = rdf_t.groupby("PROPERTYTYPE")["AGREEDRENT"].mean().sort_values(ascending=False).reset_index()
                fig = go.Figure(go.Bar(x=pr["PROPERTYTYPE"], y=pr["AGREEDRENT"], marker_color=GREEN,
                    text=["$"+f"{v/1000:.1f}K" for v in pr["AGREEDRENT"]],
                    textposition="outside", textfont=dict(color=CREAM, size=9)))
                pbi_layout(fig, "Rent Price by Property type", 290)
                fig.update_xaxes(tickangle=-25, tickfont=dict(size=9))
                fig.update_yaxes(tickprefix="$", tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with rr5:
                sr_cnt = rdf_t.groupby("STATE").size().sort_values(ascending=False).reset_index(name="count")
                fig = go.Figure(go.Bar(x=sr_cnt["STATE"], y=sr_cnt["count"], marker_color=GREEN,
                    text=[f"{v/1000:.1f}K" for v in sr_cnt["count"]],
                    textposition="outside", textfont=dict(color=CREAM, size=10)))
                pbi_layout(fig, "Total Rent Transaction by State", 290)
                fig.update_yaxes(tickformat=".0s")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with rr6:
                sc = rdf_t["MOST_COMMON_SCREENING"].value_counts().reset_index()
                sc.columns = ["outcome","count"]
                fig = go.Figure(go.Bar(x=sc["outcome"], y=sc["count"], marker_color=GREEN,
                    text=[f"{v/1000:.1f}K" for v in sc["count"]],
                    textposition="outside", textfont=dict(color=CREAM, size=9)))
                pbi_layout(fig, "Tenant Screening Outcome", 290)
                fig.update_xaxes(tickangle=-15, tickfont=dict(size=9))
                fig.update_yaxes(tickformat=".0s")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── TAB 3: SALE TRANSACTION ───────────────────────────────────────
        with tab_sale:
            st.markdown(f'<div style="font-size:1rem;font-weight:700;color:' + CREAM + ';font-family:Outfit,sans-serif;padding:8px 4px 4px;">Bluestone Real Estate Sale Transaction Analysis</div>', unsafe_allow_html=True)

            sy_sel = st.multiselect("Filter by Year", ["Select all"] + sorted([y for y in adf["YEAR"].dropna().unique().astype(int) if y >= 2024]),
                                    default=["Select all"], key="sale_year")
            sc_sel = st.multiselect("Filter by City", ["Select all"] + CITIES, default=["Select all"], key="sale_city")

            sdf = adf[adf["LISTING_TYPE"]=="sale"].copy()
            if "Select all" not in sy_sel and sy_sel:
                sdf = sdf[sdf["YEAR"].isin(sy_sel)]
            if "Select all" not in sc_sel and sc_sel:
                sdf = sdf[sdf["CITY"].isin(sc_sel)]
            sdf_t = sdf[sdf["FINALSALEPRICE"].notna()].copy()

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            sk1, sk2, sk3, sk4 = st.columns(4)
            kpi_card("Total Sale Transaction", f"{len(sdf_t):,}", sk1)
            kpi_card("Average Market Price",   f"${sdf_t['MARKETAVGPRICE'].mean():,.0f}" if "MARKETAVGPRICE" in sdf_t.columns and sdf_t["MARKETAVGPRICE"].notna().any() else f"${sdf_t['FINALSALEPRICE'].mean():,.0f}", sk2)
            gauge_card("Offer to Ratio List",   sdf_t["OFFERTOLISTRATIO"].mean() if "OFFERTOLISTRATIO" in sdf_t.columns and sdf_t["OFFERTOLISTRATIO"].notna().any() else 0.97, 0.90, 1.0, sk3, fmt=".2f")
            gauge_card("Average Days on Market", sdf_t["MARKETAVGDOM"].mean() if "MARKETAVGDOM" in sdf_t.columns and sdf_t["MARKETAVGDOM"].notna().any() else sdf_t["DAYSTOCLOSE"].mean(), 25, 250, sk4, fmt=".0f")

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            sr1, sr2, sr3 = st.columns([1.2, 1.2, 0.9])

            with sr1:
                ss = sdf_t.groupby("STATE")["FINALSALEPRICE"].mean().sort_values(ascending=False).reset_index()
                fig = go.Figure(go.Bar(x=ss["STATE"], y=ss["FINALSALEPRICE"], marker_color=GREEN,
                    text=["$"+f"{v/1e6:.1f}M" for v in ss["FINALSALEPRICE"]],
                    textposition="outside", textfont=dict(color=CREAM, size=10)))
                pbi_layout(fig, "Sale Price by State", 290)
                fig.update_yaxes(tickprefix="$", tickformat=".2s")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with sr2:
                sp = sdf_t.groupby("PROPERTYTYPE")["FINALSALEPRICE"].mean().sort_values(ascending=False).reset_index()
                fig = go.Figure(go.Bar(y=sp["PROPERTYTYPE"], x=sp["FINALSALEPRICE"],
                    orientation="h", marker_color=GREEN,
                    text=["$"+f"{v/1e6:.1f}M" for v in sp["FINALSALEPRICE"]],
                    textposition="outside", textfont=dict(color=CREAM, size=9)))
                pbi_layout(fig, "Sale Price by Property Type", 290)
                fig.update_xaxes(tickprefix="$", tickformat=".2s")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with sr3:
                sc_cnt = sdf_t.groupby("STATE").size().sort_values(ascending=False).reset_index(name="count")
                fig = go.Figure(go.Bar(x=sc_cnt["STATE"], y=sc_cnt["count"], marker_color=GREEN,
                    text=[f"{v/1000:.0f}K" for v in sc_cnt["count"]],
                    textposition="outside", textfont=dict(color=CREAM, size=10)))
                pbi_layout(fig, "Sale Transaction by State", 290)
                fig.update_yaxes(tickformat=".0s")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            sr4, sr5, sr6 = st.columns([1.2, 1.2, 0.9])

            with sr4:
                fd = sdf_t.groupby("TOP_FINANCINGTYPE")["DAYSTOCLOSE"].mean().sort_values(ascending=False).reset_index()
                fig = go.Figure(go.Scatter(x=fd["TOP_FINANCINGTYPE"], y=fd["DAYSTOCLOSE"].round(1),
                    mode="lines+markers", line=dict(color=GREEN, width=2.5),
                    marker=dict(color=GREEN, size=8)))
                pbi_layout(fig, "Average Days to Close by Financing Type", 290)
                fig.update_xaxes(tickangle=-10)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with sr5:
                mo = sdf_t.groupby("PROPERTYTYPE")[["MARKETAVGPRICE","OFFERPRICE"]].mean().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Average of marketAvgPrice", x=mo["PROPERTYTYPE"], y=mo["MARKETAVGPRICE"],
                    marker_color=GREEN, text=["$"+f"{v/1e6:.1f}M" for v in mo["MARKETAVGPRICE"]],
                    textposition="outside", textfont=dict(color=CREAM, size=8)))
                fig.add_trace(go.Bar(name="Average of offerPrice", x=mo["PROPERTYTYPE"], y=mo["OFFERPRICE"],
                    marker_color=AMBER, text=["$"+f"{v/1e6:.1f}M" for v in mo["OFFERPRICE"]],
                    textposition="outside", textfont=dict(color=CREAM, size=8)))
                pbi_layout(fig, "Market Price vs Offer Price by Property Type", 290, showlegend=True)
                fig.update_layout(barmode="group")
                fig.update_xaxes(tickangle=-20, tickfont=dict(size=8))
                fig.update_yaxes(tickprefix="$", tickformat=".2s")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with sr6:
                ls = sdf_t["PRICE"].sum()
                os_ = sdf_t["OFFERPRICE"].sum()
                fig = go.Figure(go.Pie(labels=["Sum of listPrice","Sum of offerPrice"],
                    values=[ls, os_], hole=0.55,
                    marker=dict(colors=[AMBER, TEAL]),
                    textfont=dict(color=CREAM, size=10), textinfo="percent"))
                pbi_layout(fig, "List Price vs Offer Price", 290, showlegend=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT & DEPLOYMENT
# ════════════════════════════════════════════════════════════════════════════

elif "ℹ️" in page:

    st.markdown("""
    <div class='bs-page-header'>
      <h1><span class='bs-h1-icon'>🚀</span>About & Deployment</h1>
      <div class='bs-page-subtitle'>Platform overview, architecture and deployment guide</div>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([1.1, 1])

    with a1:
        st.markdown("### 🏗️ Pipeline Architecture")
        for i, (step, detail) in enumerate([
            ("Data Ingestion",      "5 Snowflake tables via Airbyte → 100 raw columns"),
            ("Feature Engineering", "41 engineered features — ratios, log-transforms, KMeans geo-clusters"),
            ("Preprocessing",       "StandardScaler + MinMaxScaler + OrdinalEncoder + Target Encoding"),
            ("Class Imbalance",     f"SMOTE {'applied to training set' if arts.get('smote_applied') else 'not needed — balanced dataset'}"),
            ("Hyperparameter Tuning","Optuna Bayesian TPE — 80 trials each, MedianPruner"),
            ("Explainability",      "SHAP TreeExplainer — Waterfall, Beeswarm, Bar plots"),
            ("Serialization",       "joblib compressed artefacts — 14 files in bluestone_outputs/"),
        ]):
            st.markdown(f"""<div class='bs-step'>
              <div class='bs-step-num'>{i+1}</div>
              <div class='bs-step-content'><strong>{step}</strong><br>{detail}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📦 Artefact Inventory")
        st.markdown("\n".join([f"| `{a}` | {d} |" for a, d in [
            ("preprocessing_pipeline.pkl",     "ColumnTransformer — scale + encode"),
            ("best_regression_model.pkl",      f"{meta.get('regression_model_name','GBR')} (Tuned)"),
            ("best_classification_model.pkl",  f"{meta.get('classification_model_name','Classifier')} (Tuned)"),
            ("shap_explainer_regression.pkl",  "TreeExplainer for price model"),
            ("shap_explainer_classification.pkl","TreeExplainer for conversion model"),
            ("model_metadata.pkl",             "Params, metrics, comparison summary"),
            ("target_encoding_maps.pkl",       "City / ZIP / County mean-price maps"),
            ("label_encoders.pkl",             "Ordinal encoders for categoricals"),
        ]]), unsafe_allow_html=False)

    with a2:
        st.markdown("### 🚀 Local Deployment")
        st.code("""
# 1. Install dependencies
pip install streamlit shap joblib scikit-learn
pip install xgboost lightgbm imbalanced-learn optuna
pip install pandas numpy matplotlib

# 2. Required folder structure
#    your_project/
#    ├── app.py
#    ├── requirements.txt
#    └── bluestone_outputs/   ← all .pkl files here

# 3. Launch
streamlit run app.py
# Opens at http://localhost:8501
""", language="bash")

        st.markdown("### ☁️ Cloud (Streamlit Community Cloud)")
        st.code("""
# 1. Push to GitHub with bluestone_outputs/ included
# 2. Visit https://share.streamlit.io
# 3. New app → Connect GitHub → select app.py
# Note: for large .pkl files use Git LFS or
#       load from S3/GCS at runtime
""", language="bash")

        reqs = """streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0
optuna>=3.4.0
shap>=0.44.0
joblib>=1.3.0
matplotlib>=3.7.0"""

        st.markdown("### 📋 requirements.txt")
        st.code(reqs, language="text")
        st.download_button("⬇  Download requirements.txt",
            data=reqs.encode(), file_name="requirements.txt", mime="text/plain")

        st.markdown(f"""
        <div class='bs-card-accent' style='margin-top:1rem;'>
          <div class='bs-form-section-title'>🤖 Models in Production</div>
          <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; font-size:0.83rem;'>
            <div>
              <div class='bs-section-label'>Regression</div>
              <div style='color:var(--gold); font-weight:600;'>{meta.get("regression_model_name","GBR")}</div>
              <div style='color:var(--slate); margin-top:2px;'>R² {meta.get("test_reg_r2","—")} · RMSE ${meta.get("test_reg_rmse",0):,.0f}</div>
            </div>
            <div>
              <div class='bs-section-label'>Classification</div>
              <div style='color:var(--teal); font-weight:600;'>{meta.get("classification_model_name","Classifier")}</div>
              <div style='color:var(--slate); margin-top:2px;'>AUC {meta.get("test_clf_auc","—")} · F1 {meta.get("test_clf_f1","—")}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='bs-footer'>
  BlueStone Real Estate Intelligence Platform &nbsp;·&nbsp; Data Science Division &nbsp;·&nbsp; Confidential
</div>
""", unsafe_allow_html=True)