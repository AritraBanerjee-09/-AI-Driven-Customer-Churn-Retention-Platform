# pages/4_Quote_Analyzer.py
import streamlit as st
from PIL import Image
import pytesseract
import google.generativeai as genai
import json
import pandas as pd
import re

st.set_page_config(page_title="AI Quote Analyzer", layout="wide")
st.title("AI-Powered Competitor Quote Analyzer")


with st.expander("View Key Assumptions & Placeholders for this Prototype"):
    st.caption("""
    This prototype makes several assumptions due to the absence of complete, real-world operational data. For a production system, these placeholders would be replaced with live data integrations.

    **Policy Details (in Quote Analyzer):**
    - While the customer's **Premium** is from the real dataset, the following are assumed for comparison purposes:
    - **Bodily Injury Liability:** Assumed to be **$100,000**.
    - **Comprehensive Deductible:** Assumed to be **$500**.
 """)


# --- FIX: Check for and load the correct 'featured_data' object ---
if 'featured_data' not in st.session_state:
    st.error("Customer data not loaded. Please start from the main app page.")
    st.stop()

featured_data = st.session_state['featured_data']

st.markdown("First, select a customer from your portfolio. Their data will be used as the 'current Chubb policy' for the comparison.")
# FIX: Use 'featured_data' to populate the customer list
customer_ids = featured_data['individual_id'].unique()
selected_id = st.selectbox("Select a Customer ID", options=customer_ids)

if selected_id:
    # FIX: Use 'featured_data' to look up the selected customer
    customer_data = featured_data[featured_data['individual_id'] == selected_id].iloc[0]
    
    # This part is now correct as it uses the data from the line above
    chubb_policy_data = {
        "premium": customer_data['curr_ann_amt'],
        "liability_bodily_injury": 100000,
        "deductible_comprehensive": 500
    }
    st.info(f"Comparing against Customer **{int(selected_id)}**'s current policy (Premium: **${chubb_policy_data['premium']:.2f}**).")
else:
    st.warning("Please select a customer to proceed.")
    st.stop()

def get_gemini_analysis(text, api_key, current_policy):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    prompt = f"""
    You are an expert insurance analyst for Chubb. Your task is to analyze OCR text from a competitor's quote and generate talking points for an agent.
    First, extract these details from the competitor's quote text: premium, liability_bodily_injury, and deductible_comprehensive.
    Second, compare these against the customer's current Chubb policy, which is:
    - Premium: ${current_policy['premium']:.2f}
    - Bodily Injury Liability: ${current_policy['liability_bodily_injury']:,}
    - Comprehensive Deductible: ${current_policy['deductible_comprehensive']}
    Third, generate 2-3 concise talking points for the agent to highlight Chubb's value.
    Here is the OCR text:
    ---
    {text}
    ---
    Provide your final output ONLY in a clean JSON format:
    {{
      "analysis": {{
        "competitor_premium": <number or "Not Found">,
        "competitor_liability": <number or "Not Found">,
        "competitor_deductible": <number or "Not Found">
      }},
      "talking_points": [
        "First talking point...",
        "Second talking point..."
      ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
        json_str = match.group(1) if match else response.text
        return json.loads(json_str)
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        return None

uploaded_file = st.file_uploader("Now, upload an image of the competitor's quote", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Quote", width=300)
    
    with st.spinner("Step 1: Extracting text with OCR..."):
        extracted_text = pytesseract.image_to_string(Image.open(uploaded_file))
        st.success("Text extracted!")

    if extracted_text:
        with st.spinner("Step 2: Sending to Gemini AI for analysis..."):
            gemini_api_key = st.secrets.get("GEMINI_API_KEY")
            ai_result = get_gemini_analysis(extracted_text, gemini_api_key, chubb_policy_data)
            if ai_result: st.success("AI analysis complete!")

        if ai_result:
            st.subheader("AI-Powered Competitive Analysis")
            st.markdown("#### 💡 Talking Points for Agent:")
            for point in ai_result.get("talking_points", []):
                st.info(point)
            
            st.markdown("#### Data Comparison:")
            comp_data = ai_result.get("analysis", {})
            
            comparison_df = pd.DataFrame({
                "Metric": ["Premium", "Bodily Injury Liability", "Comprehensive Deductible"],
                "Competitor": [
                    f"${comp_data.get('competitor_premium')}" if isinstance(comp_data.get('competitor_premium'), (int, float)) else "Not Found",
                    f"${comp_data.get('competitor_liability'):,}" if isinstance(comp_data.get('competitor_liability'), (int, float)) else "Not Found",
                    f"${comp_data.get('competitor_deductible')}" if isinstance(comp_data.get('competitor_deductible'), (int, float)) else "Not Found"
                ],
                "Chubb (Current Policy)": [
                    f"${chubb_policy_data.get('premium'):.2f}",
                    f"${chubb_policy_data.get('liability_bodily_injury'):,}",
                    f"${chubb_policy_data.get('deductible_comprehensive')}"
                ]
            })
            st.dataframe(comparison_df.astype(str), hide_index=True)