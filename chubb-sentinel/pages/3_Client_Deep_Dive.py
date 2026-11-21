# # pages/3_Client_Deep_Dive.py
# import streamlit as st
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# import google.generativeai as genai
# import json
# import re

# st.set_page_config(page_title="Client Deep Dive", layout="wide")
# st.title("🧑‍🔬 Client Deep Dive & AI Action Plan")

# # --- Helper Functions ---

# def generate_xai_recommendation(shap_values, customer_data):
#     """Analyzes SHAP values and customer data to generate rule-based recommendations."""
#     shap_df = pd.DataFrame({'feature': shap_values.feature_names, 'value': shap_values.values}).sort_values(by='value', ascending=False)
#     top_driver = shap_df.iloc[0]['feature']
    
#     # Special Rule for Loyal Customers
#     if customer_data['days_tenure'].iloc[0] > 5000:
#         return {
#             "strategy": "🚨 High-Priority: At-Risk Loyal Customer",
#             "talking_points": [
#                 "This is a long-term client who is now a churn risk, requiring immediate, senior-level attention.",
#                 "Action: The assigned Relationship Manager must place a personal call within 24 hours.",
#                 "Goal: Discover the root cause of the recent dissatisfaction (e.g., a poorly handled claim, price increase).",
#                 "Incentive: Be prepared to offer a significant loyalty bonus or concession to save this relationship."
#             ]
#         }
    
#     # Standard rules
#     if 'days_tenure' in top_driver or 'orig_year' in top_driver:
#         return { "strategy": "Onboarding & Value Reinforcement (New Customer Risk)", "talking_points": ["This is a new client at high risk.", "Action: Schedule a 'New Client Check-up' call.", "Goal: Ensure they understand their policy benefits and confirm a smooth onboarding."] }
#     elif 'income' in top_driver or 'home_market_value' in top_driver:
#         return { "strategy": "Cost Optimization (Financial Strain Risk)", "talking_points": ["Client's financial profile is a key risk factor.", "Action: Offer a policy review focused on cost optimization.", "Incentive: Suggest switching to a monthly payment plan or explore different deductible options."] }
#     elif 'city' in top_driver or 'distance_to' in top_driver:
#          return { "strategy": "Geographic Risk Mitigation (Location-Based Risk)", "talking_points": ["Client's location is a significant factor.", "Action: Proactively communicate how Chubb manages location-specific risks.", "Talking Point: Frame their premium as a reflection of tailored, superior local protection."] }
    
#     return { "strategy": "General Review", "talking_points": ["Proactively schedule a policy review call.", "Reiterate the value of Chubb's claims service."] }

# def get_ai_summary(customer_info, churn_prob, shap_summary, api_key):
#     """Generates a concise executive summary using Gemini."""
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel('gemini-flash-latest')
#     prompt = f"""
#     You are an analyst providing a concise executive summary for a busy insurance agent.
#     Based on the following data, write a 2-3 sentence summary of the customer's situation.
#     - Customer Profile: {customer_info}
#     - Predicted Churn Probability: {churn_prob:.0%}
#     - Key Churn Drivers: {list(shap_summary.keys())}
#     Synthesize this into a brief, professional summary.
#     """
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception: return "Could not generate AI summary."

# def get_ai_action_plan(customer_info, shap_summary, api_key):
#     """Generates a creative retention plan and email draft using Gemini."""
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel('gemini-flash-latest')
#     prompt = f"""
#     You are a world-class retention expert for Chubb. A customer is at risk of churning.
#     CUSTOMER PROFILE: {customer_info}
#     TOP REASONS FOR CHURN RISK: {shap_summary}
#     Your task is to provide a comprehensive retention plan. Output ONLY in a clean JSON format:
#     {{
#       "suggested_action": "Your suggested action here...",
#       "email_draft": "Your email draft here..."
#     }}
#     """
#     try:
#         response = model.generate_content(prompt)
#         match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
#         return json.loads(match.group(1) if match else response.text)
#     except Exception: return None

# # --- Main App ---
# if 'featured_data' not in st.session_state: st.error("Application data not loaded..."); st.stop()

# # Load all artifacts
# model, preprocessor, explainer, featured_data = st.session_state['model'], st.session_state['preprocessor'], st.session_state['explainer'], st.session_state['featured_data']
# gemini_api_key = st.secrets.get("GEMINI_API_KEY")

# high_risk_ids = featured_data[featured_data['churn_probability'] > 0.5]['individual_id'].tolist()
# client_id = st.selectbox("Select a High-Risk Client ID to Analyze", high_risk_ids)

# if client_id:
#     client_data = featured_data[featured_data['individual_id'] == client_id]
#     feature_cols = preprocessor.feature_names_in_
#     client_features_processed = preprocessor.transform(client_data[feature_cols])
#     shap_values = explainer(client_features_processed)
    
#     st.subheader(f"Churn Analysis for Client {int(client_id)}")
    
#     # --- NEW: Executive Summary Section ---
#     with st.spinner("Generating AI summary..."):
#         customer_profile_summary = client_data[['age_in_years', 'days_tenure', 'income']].iloc[0].to_dict()
#         churn_prob_summary = client_data['churn_probability'].iloc[0]
#         top_drivers_summary = pd.Series(shap_values[0].values, index=shap_values[0].feature_names).sort_values(ascending=False)[:3].to_dict()
#         summary_text = get_ai_summary(customer_profile_summary, churn_prob_summary, top_drivers_summary, gemini_api_key)
#         st.info(f"**Executive Summary:** {summary_text}")
    
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st.metric("Churn Probability", f"{client_data['churn_probability'].iloc[0]:.2%}")
#         st.write("**Client Snapshot**")
#         display_data = client_data[['age_in_years', 'income', 'marital_status', 'days_tenure', 'curr_ann_amt']].T.rename(columns={client_data.index[0]: "Value"})
#         st.dataframe(display_data.astype(str))

#     with col2:
#         st.write("**Top Factors Influencing Churn Risk (SHAP)**")
#         fig, ax = plt.subplots(); shap.plots.waterfall(shap_values[0], max_display=10, show=False); plt.tight_layout(); st.pyplot(fig)
    
#     st.divider()

#     st.subheader("✅ XAI-Driven Recommendations")
#     xai_rec = generate_xai_recommendation(shap_values[0], client_data)
#     st.success(f"**Suggested Strategy:** {xai_rec['strategy']}")
#     for point in xai_rec['talking_points']:
#         st.write(f"- {point}")
        
#     st.divider()
    
#     st.subheader("🤖 AI-Generated Creative Action Plan (Optional)")
#     if st.button("Generate AI Email & Action Plan"):
#         with st.spinner("Generating personalized retention strategy with Gemini AI..."):
#             customer_profile = client_data[['age_in_years', 'days_tenure', 'income', 'marital_status']].iloc[0].to_dict()
#             action_plan = get_ai_action_plan(customer_profile, top_drivers_summary, gemini_api_key)
#             if action_plan:
#                 st.success("**Suggested Action:**")
#                 st.write(action_plan.get('suggested_action'))
#                 st.info("**Email Draft for Agent:**")
#                 st.text_area("", value=action_plan.get('email_draft'), height=250)
#             else:
#                 st.error("Failed to generate an AI action plan.")

# pages/3_Client_Deep_Dive.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai
import json
import re

st.set_page_config(page_title="Client Deep Dive", layout="wide")
st.title("🧑‍🔬 Client Deep Dive & Action Plan")

# --- Helper Functions (No changes needed) ---
def generate_xai_recommendation(shap_values, customer_data):
    """Analyzes SHAP values and customer data to generate a rule-based strategy."""
    shap_df = pd.DataFrame({'feature': shap_values.feature_names, 'value': shap_values.values}).sort_values(by='value', ascending=False)
    top_driver = shap_df.iloc[0]['feature']
    
    if customer_data['days_tenure'].iloc[0] > 5000:
        return { "strategy": "High-Priority: At-Risk Loyal Customer", "talking_points": ["This is a long-term client now at risk. Requires immediate, senior-level attention.", "Action: Personal call within 24 hours to discover the root cause.", "Incentive: Be prepared to offer a significant loyalty bonus or concession."] }
    if 'days_tenure' in top_driver or 'orig_year' in top_driver:
        return { "strategy": "Onboarding & Value Reinforcement (New Customer Risk)", "talking_points": ["This is a new client at high risk.", "Action: Schedule a 'New Client Check-up' call.", "Goal: Ensure they understand policy benefits and confirm a smooth onboarding."] }
    elif 'income' in top_driver or 'home_market_value' in top_driver:
        return { "strategy": "Cost Optimization (Financial Strain Risk)", "talking_points": ["Client's financial profile is a key risk factor.", "Action: Offer a policy review focused on cost optimization.", "Incentive: Suggest switching to a monthly payment plan."] }
    # Add other rules as needed...
    return { "strategy": "General Review", "talking_points": ["Proactively schedule a policy review call."] }


def get_ai_churn_rationale(customer_info, churn_prob, shap_summary, api_key):
    """Generates a human-readable explanation of churn drivers using Gemini."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    # --- NEW: More detailed prompt ---
    prompt = f"""
    You are an expert insurance analyst. Your task is to explain in clear, simple business terms WHY a customer is a churn risk.
    
    DATA:
    - Customer Profile: {customer_info}
    - Predicted Churn Probability: {churn_prob:.0%}
    - Top Technical Churn Drivers from our model: {list(shap_summary.keys())}

    Based on the data, write a 2-3 bullet point "Churn Rationale" that a busy agent can immediately understand.
    Translate the technical driver names into plain English. For example, 'days_tenure' or 'orig_year' means the customer is new.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception: return "Could not generate AI churn rationale."


def get_ai_action_plan(customer_info, shap_summary, xai_strategy, api_key):
    """Generates a creative action plan and email draft based on the XAI strategy."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    prompt = f"""
    As a retention expert for Chubb, you have been given the following information about an at-risk client:
    - CUSTOMER PROFILE: {customer_info}
    - KEY CHURN DRIVERS: {shap_summary}
    - RECOMMENDED STRATEGY: {xai_strategy}

    Your task is to expand on this strategy. Create a detailed, creative action plan and a personalized email draft for an agent to send. The tone should be empathetic and aligned with Chubb's premium brand.
    
    Provide your final output ONLY in a clean JSON format:
    {{
      "detailed_action_steps": "A few bullet points detailing the plan...",
      "email_draft": "Your personalized email draft here..."
    }}
    """
    try:
        response = model.generate_content(prompt)
        match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
        return json.loads(match.group(1) if match else response.text)
    except Exception: return None

# --- Main App ---

if 'featured_data' not in st.session_state: st.error("Application data not loaded..."); st.stop()

model, preprocessor, explainer, featured_data = st.session_state['model'], st.session_state['preprocessor'], st.session_state['explainer'], st.session_state['featured_data']
gemini_api_key = st.secrets.get("GEMINI_API_KEY")

high_risk_ids = featured_data[featured_data['churn_probability'] > 0.5]['individual_id'].tolist()
client_id = st.selectbox("Select a High-Risk Client ID to Analyze", high_risk_ids)

if client_id:
    client_data = featured_data[featured_data['individual_id'] == client_id]
    feature_cols = preprocessor.feature_names_in_
    client_features_processed = preprocessor.transform(client_data[feature_cols])
    shap_values = explainer(client_features_processed)
    
    st.subheader(f"Diagnosis for Client {int(client_id)}")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Churn Probability", f"{client_data['churn_probability'].iloc[0]:.2%}")
        st.write("**Client Snapshot**")
        display_data = client_data[['age_in_years', 'income', 'marital_status', 'days_tenure', 'curr_ann_amt']].T.rename(columns={client_data.index[0]: "Value"})
        st.dataframe(display_data.astype(str))

    with col2:
        st.write("**Top Factors Influencing Churn Risk (SHAP)**")
        fig, ax = plt.subplots(); shap.plots.waterfall(shap_values[0], max_display=10, show=False); plt.tight_layout(); st.pyplot(fig)
    
    st.divider()

    # --- NEW: AI-Powered Rationale Section ---
    st.subheader("AI-Powered Churn Rationale")
    st.markdown("A plain-English explanation of the key risk factors.")
    with st.spinner("Generating AI rationale..."):
        customer_profile_summary = client_data[['age_in_years', 'days_tenure', 'income']].iloc[0].to_dict()
        churn_prob_summary = client_data['churn_probability'].iloc[0]
        top_drivers_summary = pd.Series(shap_values[0].values, index=shap_values[0].feature_names).sort_values(ascending=False)[:3].to_dict()
        
        rationale_text = get_ai_churn_rationale(customer_profile_summary, churn_prob_summary, top_drivers_summary, gemini_api_key)
        st.info(rationale_text)
        
    st.divider()

    st.subheader("Recommended Action Plan")
    xai_rec = generate_xai_recommendation(shap_values[0], client_data)
    st.success(f"**XAI-Driven Strategy:** {xai_rec['strategy']}")
    
    # The creative AI action plan and email draft functionality can be triggered by a button if desired
    # For simplicity, we'll just show the XAI talking points here.
    for point in xai_rec['talking_points']:
        st.write(f"- {point}")