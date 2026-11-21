# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="Chubb Loyalty Navigator", page_icon="", layout="wide")

# --- OPTIMIZED Feature Engineering Function ---
@st.cache_data
def create_features_optimized(_df):
    df = _df.copy()
    
    # --- Date and Household Features (already efficient) ---
    df['cust_orig_date'] = pd.to_datetime(df['cust_orig_date'], errors='coerce')
    df['orig_year'] = df['cust_orig_date'].dt.year
    df['orig_month'] = df['cust_orig_date'].dt.month
    
    # A slightly more efficient way to calculate household size
    df['household_size'] = df.groupby('address_id')['address_id'].transform('count')
    
    # --- Vectorized Haversine Distance Calculation ---
    def vectorized_haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    dallas_center = (32.7767, -96.7970)
    fort_worth_center = (32.7555, -97.3308)
    
    # Apply the vectorized function to the entire columns at once
    df['distance_to_dallas'] = vectorized_haversine(df['latitude'], df['longitude'], dallas_center[0], dallas_center[1])
    df['distance_to_fort_worth'] = vectorized_haversine(df['latitude'], df['longitude'], fort_worth_center[0], fort_worth_center[1])
    
    # --- Other Engineered Features ---
    df['tenure_in_years'] = df['days_tenure'] / 365.25
    df['income'] = df['income'].replace(0, 1)
    df['premium_to_income_ratio'] = df['curr_ann_amt'] / df['income']
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

@st.cache_resource
def load_and_prep_data(sample_mode=False):
    model = joblib.load('artifacts/xgb_model.joblib')
    preprocessor = joblib.load('artifacts/preprocessor.joblib')
    explainer = joblib.load('artifacts/shap_explainer.joblib')
    
    if sample_mode:
        raw_data = pd.read_csv('../archive/autoinsurance_churn.csv', nrows=10000)
    else:
        raw_data = pd.read_csv('../archive/autoinsurance_churn.csv')
        
    raw_data['Churn'] = raw_data['acct_suspd_date'].notna().astype(int)
    
    # Call the new, optimized function
    featured_data = create_features_optimized(raw_data)
    
    feature_cols = preprocessor.feature_names_in_
    data_to_transform = featured_data.loc[:, feature_cols]
    
    X_processed = preprocessor.transform(data_to_transform)
    churn_probs = model.predict_proba(X_processed)[:, 1]
    featured_data['churn_probability'] = churn_probs
    
    return model, preprocessor, explainer, featured_data

# --- Main App Loading Logic ---
if 'artifacts_loaded' not in st.session_state:
    with st.spinner("Loading models and preparing data... This will be much faster now."):
        # Set sample_mode=False to see the speed improvement on the full dataset
        model, preprocessor, explainer, featured_data = load_and_prep_data(sample_mode=False)
        st.session_state['model'] = model
        st.session_state['preprocessor'] = preprocessor
        st.session_state['explainer'] = explainer
        st.session_state['featured_data'] = featured_data
        st.session_state['artifacts_loaded'] = True

st.title("Welcome to Chubb Loyalty Navigator")
st.sidebar.success("Select a page to begin.")