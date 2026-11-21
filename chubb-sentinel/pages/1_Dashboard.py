# pages/1_Dashboard.py
import streamlit as st
import plotly.express as px
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("📈 Portfolio Health Dashboard")

if 'featured_data' not in st.session_state:
    st.error("Application data not loaded. Please start from the main 'app.py' page.")
    st.stop()

df = st.session_state['featured_data']
preprocessor = st.session_state['preprocessor']
explainer = st.session_state['explainer']

st.markdown("### Key Performance Indicators")
col1, col2, col3 = st.columns(3)
overall_churn_risk = df['churn_probability'].mean() * 100
premium_at_risk = df[df['churn_probability'] > 0.5]['curr_ann_amt'].sum()
col1.metric("Overall Churn Risk", f"{overall_churn_risk:.2f}%")
col2.metric("Total Premium at Risk", f"${premium_at_risk:,.0f}")
col3.metric("High-Risk Clients (>70%)", df[df['churn_probability'] > 0.7].shape[0])

st.markdown("### Regional Churn Hotspots (Dallas-Fort Worth)")
viz_sample = df.sample(n=5000 if len(df) > 5000 else len(df), random_state=42)
viz_sample['color'] = viz_sample['Churn'].apply(lambda x: 'Churned' if x == 1 else 'Active')
fig_map = px.scatter_map(viz_sample, lat="latitude", lon="longitude", color="color", color_discrete_map={'Active': '#1f77b4', 'Churned': '#d62728'}, zoom=8.5, height=600, hover_name="individual_id", hover_data={"churn_probability": ":.2%", "city": True})
fig_map.update_layout(mapbox_style="carto-positron")
st.plotly_chart(fig_map, use_container_width=True)

# st.markdown("### Top Global Churn Drivers")
# with st.spinner("Generating global feature importance plot..."):
#     X_test = pd.read_csv('data/X_test.csv')
#     X_test_processed = preprocessor.transform(X_test)
#     sample_size = min(500, len(X_test_processed))
#     shap_values = explainer(X_test_processed[:sample_size])
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, plot_type="bar", max_display=10, show=False)
#     plt.tight_layout()
#     st.pyplot(fig)