# pages/5_Segment_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segment Analysis", layout="wide")
st.title("Customer Segment Analysis")

if 'featured_data' not in st.session_state:
    st.error("Application data not loaded. Please restart the app from the main page.")
    st.stop()

df = st.session_state['featured_data'].copy()
high_risk_df = df[df['churn_probability'] > 0.5].copy()

# --- Create Tabs for Each Analysis Type ---
tab1, tab2 = st.tabs(["Rule-Based Segments", "AI-Discovered Segments (Clustering)"])

# --- TAB 1: DYNAMIC RULE-BASED SEGMENTS ---
with tab1:
    st.header("Interactive Rule-Based Segmentation")
    st.markdown("Use the sliders to dynamically change the definition of each segment and see the impact in real-time.")

    # --- NEW: Dynamic Controls ---
    col_settings, col_kpis = st.columns([1, 2])
    with col_settings:
        st.subheader("Segment Controls")
        new_customer_threshold = st.slider("'New Customer' Tenure (Days)", 30, 730, 365)
        veteran_threshold = st.slider("'Veteran Customer' Tenure (Days)", 731, 10000, 1825)
        high_value_quantile = st.slider("'High-Value' Premium Quantile", 0.5, 1.0, 0.75, 0.05)

    # --- Define and Analyze Segments based on slider values ---
    high_value_threshold = high_risk_df['curr_ann_amt'].quantile(high_value_quantile)
    high_value_segment = high_risk_df[high_risk_df['curr_ann_amt'] >= high_value_threshold]
    new_unstable_segment = high_risk_df[high_risk_df['days_tenure'] <= new_customer_threshold]
    disengaged_veteran_segment = high_risk_df[high_risk_df['days_tenure'] >= veteran_threshold]

    with col_kpis:
        st.subheader("Segment Impact")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("High-Value At-Risk", f"{len(high_value_segment)}")
        kpi2.metric("New & Unstable", f"{len(new_unstable_segment)}")
        kpi3.metric("Disengaged Veterans", f"{len(disengaged_veteran_segment)}")
    
    # Visualization
    segment_data = pd.DataFrame({
        'Segment': ['High-Value At-Risk', 'New & Unstable', 'Disengaged Veterans'],
        'PremiumAtRisk': [high_value_segment['curr_ann_amt'].sum(), new_unstable_segment['curr_ann_amt'].sum(), disengaged_veteran_segment['curr_ann_amt'].sum()]
    })
    fig = px.bar(segment_data, x='Segment', y='PremiumAtRisk', title="Total Premium At-Risk by Segment", text_auto='.2s')
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: AI-DISCOVERED SEGMENTS (CLUSTERING) ---
with tab2:
    st.header("Automated Segment Discovery with K-Means Clustering")
    st.markdown("This analysis uses an unsupervised AI model to find natural groupings (or 'personas') within your high-risk customers based on their behavior and profile.")

    # --- Caching ensures this slow computation runs only once ---
    @st.cache_data
    def run_clustering(df_input):
        features_for_clustering = ['age_in_years', 'days_tenure', 'income', 'premium_to_income_ratio', 'household_size']
        
        # Prepare the data
        cluster_data = df_input[features_for_clustering].copy()
        cluster_data.fillna(cluster_data.median(), inplace=True)
        
        # Scale the data (important for clustering)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Run K-Means
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(scaled_data)
        
        df_input['Cluster'] = [f"Persona {c+1}" for c in clusters]
        return df_input

    with st.spinner("Running AI clustering to discover personas... (This runs only once)"):
        clustered_df = run_clustering(high_risk_df)

    st.success("AI has identified 4 distinct high-risk personas.")

    # --- Visualize the discovered clusters ---
    fig_cluster = px.scatter(
        clustered_df,
        x="days_tenure",
        y="curr_ann_amt",
        color="Cluster",
        size="churn_probability",
        hover_name="individual_id",
        title="AI-Discovered High-Risk Personas",
        labels={'days_tenure': 'Customer Tenure (Days)', 'curr_ann_amt': 'Annual Premium ($)'}
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("#### Persona Characteristics:")
    st.dataframe(clustered_df.groupby('Cluster')[['age_in_years', 'days_tenure', 'income', 'curr_ann_amt']].mean())
    st.markdown("""
    **How to read this:**
    - **Persona 1** might be your 'New & High-Value' customers.
    - **Persona 2** might be 'Established but Price-Sensitive' customers.
    
    This automated analysis helps you discover segments you might not have thought to look for.
    """)