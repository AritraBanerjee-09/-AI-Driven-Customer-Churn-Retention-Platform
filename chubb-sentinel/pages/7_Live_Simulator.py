# pages/7_Live_Simulator.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from transformers import pipeline # NEW import

st.set_page_config(page_title="Live Simulator", layout="wide")
st.title("Live Churn Prediction Simulator")
st.markdown("Create a hypothetical customer profile and adjust their service experience to see a real-time churn prediction.")


with st.expander("View Key Assumptions & Placeholders for this Prototype"):
    st.caption("""
    This prototype makes several assumptions due to the absence of complete, real-world operational data. For a production system, these placeholders would be replaced with live data integrations.

    **Behavioral Model (in Live Simulator):**
    - The **"Effort Penalty"** is a heuristic model. The impact of claim duration, support calls, and sentiment on the final churn score is based on assumed weights, not a trained model.
    """)


# --- NEW: Sentiment Analysis Function ---
# This function loads a pre-trained model from Hugging Face.
# @st.cache_resource ensures the model is loaded only once.
@st.cache_resource
def load_sentiment_pipeline():
    """Loads the sentiment analysis pipeline from Hugging Face."""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_pipeline()

def get_sentiment_penalty(text):
    """Analyzes text and returns a penalty/bonus for the churn score."""
    if not text.strip():
        return 0.0, "NEUTRAL"
    
    result = sentiment_pipeline(text)[0]
    sentiment = result['label']
    score = result['score']
    
    if sentiment == 'NEGATIVE' and score > 0.7:
        return 0.20, f"NEGATIVE ({score:.0%})" # Add a 20% penalty for strong negative sentiment
    elif sentiment == 'NEGATIVE':
        return 0.10, f"NEGATIVE ({score:.0%})" # Add a 10% penalty for mild negative sentiment
    elif sentiment == 'POSITIVE' and score > 0.9:
        return -0.10, f"POSITIVE ({score:.0%})" # Give a 10% bonus for strong positive sentiment
    else:
        return 0.0, f"POSITIVE ({score:.0%})"

# --- Load artifacts from session state ---
if 'model' not in st.session_state:
    st.error("Model artifacts not loaded. Please start from the main 'app.py' page.")
    st.stop()
model, preprocessor, explainer, featured_data = st.session_state['model'], st.session_state['preprocessor'], st.session_state['explainer'], st.session_state['featured_data']

# --- Create Input Widgets ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Profile & History")
    days_tenure = st.slider("Days as Customer (Tenure)", 0, 8000, 365)
    curr_ann_amt = st.slider("Annual Premium ($)", 500, 2500, 950)
    age_in_years = st.slider("Age", 18, 90, 45)
    marital_status = st.selectbox("Marital Status", featured_data['marital_status'].unique())

with col2:
    st.subheader("Financial & Service Experience")
    income = st.slider("Annual Income ($)", 10000, 200000, 60000, step=5000)
    claim_duration = st.slider("Days to Resolve Last Claim", 0, 60, 5)
    support_calls = st.slider("Number of Support Calls (Last 90 Days)", 0, 10, 1)

# --- NEW: Unstructured Text Input ---
st.subheader("Unstructured Data: Last Customer Interaction")
interaction_text = st.text_area("Paste a hypothetical customer email or chat message here:", 
                                value="The website is broken again, I can't pay my bill and nobody is answering the phone. This is the third time this month. I am extremely frustrated.")

# --- Prediction Logic ---
if st.button("Predict Churn Probability", type="primary"):
    # Create hypothetical customer DataFrame
    hypothetical_customer = featured_data.iloc[0:1].copy()
    hypothetical_customer['days_tenure'] = days_tenure
    hypothetical_customer['curr_ann_amt'] = curr_ann_amt
    hypothetical_customer['age_in_years'] = age_in_years
    hypothetical_customer['marital_status'] = marital_status
    hypothetical_customer['income'] = income
    for col in preprocessor.feature_names_in_:
        if col not in hypothetical_customer.columns:
            hypothetical_customer[col] = featured_data[col].median()
    
    # Base prediction from the structured data model
    feature_cols = preprocessor.feature_names_in_
    processed_customer = preprocessor.transform(hypothetical_customer[feature_cols])
    base_prediction_proba = model.predict_proba(processed_customer)[0][1]

    # Calculate penalties from effort and sentiment
    effort_penalty = (claim_duration / 60) * 0.15 + (support_calls / 10) * 0.10
    sentiment_penalty, sentiment_label = get_sentiment_penalty(interaction_text)
    
    # Calculate the final, blended probability
    adjusted_proba = min(base_prediction_proba + effort_penalty + sentiment_penalty, 0.99)

    st.divider()
    st.subheader("Prediction Results")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Model's Base Prediction (Structured Data)", f"{base_prediction_proba:.2%}")
    res_col2.metric("Interaction Sentiment", sentiment_label)
    res_col3.metric("Final Adjusted Prediction", f"{adjusted_proba:.2%}", 
                    delta=f"{(adjusted_proba-base_prediction_proba):.2%} Adjustment")
    
    # Generate and display SHAP plot for the base prediction
    with st.spinner("Generating explanation for the base prediction..."):
        shap_values = explainer(processed_customer)
        st.subheader("Explanation for the Base Prediction (from Structured Data)")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)