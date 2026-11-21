# pages/8_Chatbot_Analysis.py
import streamlit as st
import pandas as pd
import shap
import re
import google.generativeai as genai
import json

st.set_page_config(page_title="Chatbot Analysis", layout="wide")
st.title("💬 Conversational Customer Analyst")

# --- Helper Functions ---
def get_ai_response(prompt, api_key):
    """A general function to get a response from the Gemini model."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with AI model: {e}"

def parse_json_from_text(text):
    """Extracts a JSON object from a string, even with surrounding text."""
    if text is None: return None
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match: text = match.group(1)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

# --- Main App ---
if 'artifacts_loaded' not in st.session_state:
    st.error("Application artifacts not loaded. Please start from the main 'app.py' page.")
    st.stop()

# Load all necessary objects from the session state
model, preprocessor, explainer, featured_data = st.session_state['model'], st.session_state['preprocessor'], st.session_state['explainer'], st.session_state['featured_data']
gemini_api_key = st.secrets.get("GEMINI_API_KEY")

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI-powered churn analysis assistant. Please provide a Customer ID to begin."}]
if "stage" not in st.session_state:
    st.session_state.stage = "get_customer_id"
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
if prompt := st.chat_input("Your response..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Universal Check: See if the user wants to start over with a new ID
        is_new_id_request = False
        try:
            customer_id_test = float(prompt)
            if customer_id_test in featured_data['individual_id'].values:
                is_new_id_request = True
                st.session_state.stage = "get_customer_id"
        except ValueError:
            pass

        # --- State Machine for Conversation ---
        # STAGE 1: GET AND ANALYZE CUSTOMER ID
        if st.session_state.stage == "get_customer_id":
            if not is_new_id_request:
                response_text = "Invalid input. Please enter a numerical Customer ID."
                st.error(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()

            with st.spinner(f"Performing a comprehensive analysis for customer {int(customer_id_test)}..."):
                client_data = featured_data[featured_data['individual_id'] == customer_id_test]
                churn_prob = client_data['churn_probability'].iloc[0]
                feature_cols = preprocessor.feature_names_in_
                client_features_processed = preprocessor.transform(client_data[feature_cols])
                shap_values = explainer(client_features_processed)
                top_drivers_series = pd.Series(shap_values.values[0], index=shap_values.feature_names).sort_values(ascending=False)
                top_3_drivers = {k.replace("num__", "").replace("_", " ").title(): f"{v:.2f}" for k, v in top_drivers_series[~top_drivers_series.index.str.contains("cat__")][:3].to_dict().items()}
                
                customer_profile = client_data.iloc[0].to_dict()
                action_plan_prompt = f"""As a retention expert for Chubb, a customer with profile {customer_profile} has a {churn_prob:.0%} churn risk. The top reasons are {list(top_3_drivers.keys())}. Suggest a concrete retention action and draft a short, empathetic email. Provide output ONLY in JSON: {{"suggested_action": "...", "email_draft": "..."}}"""
                action_plan = parse_json_from_text(get_ai_response(action_plan_prompt, gemini_api_key))

                st.session_state.analysis_context = {
                    "customer_id": int(customer_id_test),
                    "churn_probability": f"{churn_prob:.2%}",
                    "top_churn_drivers": top_3_drivers,
                    "customer_profile": customer_profile,
                    "retention_plan": action_plan if action_plan else {}
                }
                
                response_text = f"**Analysis for Customer {int(customer_id_test)} is complete.**\n\n"
                response_text += f"- **Churn Probability:** {st.session_state.analysis_context['churn_probability']}\n"
                response_text += f"- **Top Churn Drivers:** {', '.join(st.session_state.analysis_context['top_churn_drivers'].keys())}\n\n"
                if action_plan:
                     response_text += f"**Suggested Action:** {action_plan.get('suggested_action')}\n\n"
                response_text += "---\n\n"
                response_text += "Would you like to calculate the potential ROI for retaining this customer?"

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.stage = "ask_for_roi"
                st.rerun()

        # STAGE 2: HANDLE ROI DECISION
        elif st.session_state.stage == "ask_for_roi":
            parser_prompt = f"User was asked 'yes' or 'no' about calculating ROI. Their response was: '{prompt}'. Is the intent 'yes' or 'no'? Respond ONLY in JSON: {{\"decision\": \"yes\"}} or {{\"decision\": \"no\"}}"
            parsed_input = parse_json_from_text(get_ai_response(parser_prompt, gemini_api_key))
            if parsed_input and parsed_input.get("decision") == "yes":
                response_text = "Great. Please provide the acquisition (CAC) and retention (COR) costs. (e.g., 'CAC is 500 and the action costs 100')"
                st.info(response_text); st.session_state.stage = "get_roi_costs"
            else:
                response_text = "No problem. Feel free to ask any follow-up questions about the analysis, or enter a new Customer ID to start over."
                st.info(response_text); st.session_state.stage = "follow_up_qna"
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()

        # STAGE 3: GET COSTS AND CALCULATE ROI
        elif st.session_state.stage == "get_roi_costs":
            parser_prompt = f"User was asked for a CAC and COR. Their response was: '{prompt}'. Extract the two numbers. Respond ONLY in JSON: {{\"cac\": <number_or_null>, \"cor\": <number_or_null>}}"
            parsed_input = parse_json_from_text(get_ai_response(parser_prompt, gemini_api_key))

            if parsed_input and parsed_input.get("cac") is not None and parsed_input.get("cor") is not None:
                cac, cor = parsed_input["cac"], parsed_input["cor"]
                cav = st.session_state.analysis_context['customer_profile']['curr_ann_amt']
                total_value = (cav - cor) + cac
                roi = ((total_value - cor) / cor) * 100 if cor > 0 else 0
                st.session_state.analysis_context['roi_analysis'] = {'CAC': cac, 'COR': cor, 'ROI': f"{roi:.1f}%"}
                
                response_text = f"**ROI Analysis Complete:** The ROI for this action is **{roi:.1f}%**. What are your follow-up questions? You can also enter a new Customer ID to start over."
                st.success(response_text)
                st.session_state.stage = "follow_up_qna"
            else:
                response_text = "I had trouble finding the two cost numbers. Please try again. (e.g., 'CAC is 500, retention cost is 100')"
                st.error(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()
        
        # STAGE 4: HANDLE FOLLOW-UP QUESTIONS
        elif st.session_state.stage == "follow_up_qna":
            with st.spinner("Thinking..."):
                context_for_json = st.session_state.analysis_context.copy()
                if 'customer_profile' in context_for_json:
                    profile = context_for_json['customer_profile']
                    serializable_profile = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in profile.items()}
                    context_for_json['customer_profile'] = serializable_profile
                json_context = json.dumps(context_for_json, indent=2, default=str)
                
                conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                
                follow_up_prompt = f"""You are 'Chubb Sentinel', an AI analyst. Your task is to have a conversation and answer questions based on the full context.
                ANALYSIS DATA: {json_context}
                CONVERSATION HISTORY: {conversation_history}
                Based on ALL of the above, provide a concise and helpful response to the user's latest message: "{prompt}"
                """
                response_text = get_ai_response(follow_up_prompt, gemini_api_key)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()