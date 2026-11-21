# pages/6_Exploratory_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Exploratory Analysis", layout="wide")
st.title("🔬 Exploratory Churn Analysis")

if 'featured_data' not in st.session_state:
    st.error("Customer data not loaded. Please start from the main app page.")
    st.stop()

# Load the main dataframe once
df = st.session_state['featured_data']

st.markdown("This page explores the relationship between various customer attributes and churn rate.")

# --- CACHED ANALYSIS FUNCTIONS ---

@st.cache_data
def analyze_income(_df):
    """Calculates churn rate per income group."""
    df_copy = _df.copy()
    income_bins = [0, 30000, 50000, 75000, 100000, 150000, float('inf')]
    income_labels = ['< $30k', '$30k-$50k', '$50k-$75k', '$75k-$100k', '$100k-$150k', '$150k+']
    df_copy['income_group'] = pd.cut(df_copy['income'], bins=income_bins, labels=income_labels, right=False)
    return df_copy.groupby('income_group', observed=False)['Churn'].mean().reset_index()

@st.cache_data
def analyze_housing(_df):
    """Calculates churn rate for home ownership and length of residence."""
    df_copy = _df.copy()
    home_owner_churn = df_copy.groupby('home_owner')['Churn'].mean().reset_index()
    home_owner_churn['home_owner'] = home_owner_churn['home_owner'].map({1: 'Home Owner', 0: 'Renter'})
    
    residence_bins = [0, 1, 5, 10, 20, float('inf')]
    residence_labels = ['< 1 Year', '1-5 Years', '5-10 Years', '10-20 Years', '20+ Years']
    df_copy['residence_group'] = pd.cut(df_copy['length_of_residence'], bins=residence_bins, labels=residence_labels, right=False)
    residence_churn = df_copy.groupby('residence_group', observed=False)['Churn'].mean().reset_index()
    return home_owner_churn, residence_churn

@st.cache_data
def analyze_family(_df):
    """Calculates churn rate by marital status and presence of children."""
    df_copy = _df.copy()
    marital_churn = df_copy.groupby('marital_status')['Churn'].mean().reset_index()
    children_churn = df_copy.groupby('has_children')['Churn'].mean().reset_index()
    children_churn['has_children'] = children_churn['has_children'].map({1: 'Has Children', 0: 'No Children'})
    return marital_churn, children_churn

@st.cache_data
def analyze_socioeconomic(_df):
    """Calculates churn rate by credit, education, and age."""
    df_copy = _df.copy()
    credit_churn = df_copy.groupby('good_credit')['Churn'].mean().reset_index()
    credit_churn['good_credit'] = credit_churn['good_credit'].map({1: 'Good Credit', 0: 'Not Good Credit'})
    
    degree_churn = df_copy.groupby('college_degree')['Churn'].mean().reset_index()
    degree_churn['college_degree'] = degree_churn['college_degree'].map({1: 'Has Degree', 0: 'No Degree'})
    
    age_bins = [18, 25, 35, 45, 55, 65, float('inf')]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df_copy['age_group'] = pd.cut(df_copy['age_in_years'], bins=age_bins, labels=age_labels, right=False)
    age_churn = df_copy.groupby('age_group', observed=False)['Churn'].mean().reset_index()
    return credit_churn, degree_churn, age_churn

@st.cache_data
def analyze_customer_value(_df):
    """Calculates churn rate in a pivot table of premium vs. tenure."""
    df_copy = _df.copy()
    df_copy['premium_group'] = pd.qcut(df_copy['curr_ann_amt'], q=4, labels=['Lowest', 'Low', 'High', 'Highest'])
    df_copy['tenure_group'] = pd.qcut(df_copy['days_tenure'], q=4, labels=['Newest', 'New', 'Established', 'Most Loyal'])
    return df_copy.pivot_table(values='Churn', index='premium_group', columns='tenure_group', aggfunc='mean', observed=False)

# --- Create Tabs for Each Analysis ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Income", "Housing Stability", "Family Structure", "Socioeconomic Profile", "Customer Value"])

with tab1:
    st.header("Income Level and Churn")
    income_churn_data = analyze_income(df)
    fig = px.bar(income_churn_data, x='income_group', y='Churn', title='Churn Rate by Income Bracket', labels={'income_group': 'Income Bracket', 'Churn': 'Churn Rate'}, text_auto='.2%')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Housing Stability and Churn")
    home_owner_churn_data, residence_churn_data = analyze_housing(df)
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(home_owner_churn_data, names='home_owner', values='Churn', title='Churn Rate: Home Owners vs. Renters', hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(residence_churn_data, x='residence_group', y='Churn', title='Churn Rate by Length of Residence', text_auto='.2%')
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Family Structure and Churn")
    marital_churn_data, children_churn_data = analyze_family(df)
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(marital_churn_data, x='marital_status', y='Churn', title='Churn Rate by Marital Status', color='marital_status', text_auto='.2%')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(children_churn_data, x='has_children', y='Churn', title='Churn Rate by Presence of Children', color='has_children', text_auto='.2%')
        st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.header("Socioeconomic & Loyalty Profile vs. Churn")
    credit_churn_data, degree_churn_data, age_churn_data = analyze_socioeconomic(df)
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(credit_churn_data, x='good_credit', y='Churn', title='Churn Rate by Credit Standing', text_auto='.2%')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(degree_churn_data, x='college_degree', y='Churn', title='Churn Rate by Education Level', text_auto='.2%')
        st.plotly_chart(fig2, use_container_width=True)
    
    fig_age = px.line(age_churn_data, x='age_group', y='Churn', title='Churn Rate by Age Group', markers=True)
    fig_age.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig_age, use_container_width=True)

with tab5:
    st.header("Customer Value (Premium vs. Tenure) and Churn")
    value_churn_pivot_data = analyze_customer_value(df)
    fig = px.imshow(value_churn_pivot_data, text_auto='.2%', title='Churn Rate by Customer Value Segment', labels=dict(x="Customer Tenure", y="Premium Level", color="Churn Rate"), color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)