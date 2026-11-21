# # pages/2_Triage_List.py
# import streamlit as st
# import pandas as pd

# st.set_page_config(page_title="Triage List", layout="wide")
# st.title("🎯 At-Risk Client Triage List")

# if 'raw_data' not in st.session_state:
#     st.error("Data not loaded. Please go to the main app page first.")
#     st.stop()

# df = st.session_state['raw_data'].copy()

# # Ensure probabilities are calculated
# if 'churn_probability' not in df.columns:
#     st.warning("Churn probabilities not calculated. Please visit the Dashboard page first.")
#     st.stop()

# # Create a priority score and filter for high-risk clients
# df['priority_score'] = df['churn_probability'] * df['curr_ann_amt']
# high_risk_df = df[df['churn_probability'] > 0.5].sort_values(by='priority_score', ascending=False)

# st.markdown("This is a prioritized list of customers with a >50% predicted likelihood of churning. Use the filters to narrow down the list.")

# # Display the dataframe
# st.dataframe(
#     high_risk_df[[
#         'individual_id', 
#         'churn_probability', 
#         'curr_ann_amt', 
#         'priority_score', 
#         'days_tenure', 
#         'city', 
#         'county'
#     ]].rename(columns={
#         'individual_id': 'Client ID',
#         'churn_probability': 'Churn Probability',
#         'curr_ann_amt': 'Annual Premium',
#         'priority_score': 'Priority Score',
#         'days_tenure': 'Tenure (Days)',
#         'city': 'City',
#         'county': 'County'
#     }),
#     use_container_width=True,
#     column_config={
#         "Churn Probability": st.column_config.ProgressColumn(
#             "Churn Probability",
#             format="%.2f%%",
#             min_value=0,
#             max_value=1,
#         ),
#     }
# )

# pages/2_Triage_List.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Triage List", layout="wide")
st.title("At-Risk Client List")

if 'featured_data' not in st.session_state:
    st.error("Data not loaded. Please go to the main app page first.")
    st.stop()

df = st.session_state['featured_data'].copy()

# Create a priority score and filter for high-risk clients
df['priority_score'] = df['churn_probability'] * df['curr_ann_amt']
high_risk_df = df[df['churn_probability'] > 0.5].sort_values(by='priority_score', ascending=False)

st.markdown("This is a prioritized list of customers with a >50% predicted likelihood of churning. Use the filters to narrow down the list.")

# --- ENHANCED DISPLAY ---
# Display the dataframe with cleaner headers and a progress bar
st.dataframe(
    high_risk_df[[
        'individual_id', 
        'churn_probability', 
        'curr_ann_amt', 
        'priority_score', 
        'days_tenure', 
        'city', 
        'county'
    ]].rename(columns={
        'individual_id': 'Client ID',
        'churn_probability': 'Churn Probability',
        'curr_ann_amt': 'Annual Premium',
        'priority_score': 'Priority Score',
        'days_tenure': 'Tenure (Days)',
        'city': 'City',
        'county': 'County'
    }),
    use_container_width=True,
    column_config={
        "Churn Probability": st.column_config.ProgressColumn(
            "Churn Probability",
            help="The model's predicted likelihood that this customer will churn.",
            format="%.0f%%", # Display as a percentage
            min_value=0,
            max_value=1,
        ),
        "Annual Premium": st.column_config.NumberColumn(
            format="$ %d"
        ),
         "Priority Score": st.column_config.NumberColumn(
            format="%.0f"
        )
    }
)

# --- EXPORT FEATURE (No changes needed) ---
st.divider()


# Convert the high-risk dataframe to CSV format in memory
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(high_risk_df)

# Create the download button
st.download_button(
   label="Export High-Risk List to CSV",
   data=csv_data,
   file_name='high_risk_customers.csv',
   mime='text/csv',
)