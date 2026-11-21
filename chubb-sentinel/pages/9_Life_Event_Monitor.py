# pages/9_Life_Event_Monitor.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Life Event Monitor", layout="wide")
st.title("Proactive Life Event Monitor")

if 'featured_data' not in st.session_state:
    st.error("Customer data not loaded. Please start from the main app page.")
    st.stop()

df = st.session_state['featured_data'].copy()

st.markdown("This tool proactively identifies customers likely experiencing major life events, turning moments of churn risk into opportunities for relationship building.")

# --- Define Life Event Segments ---
# Proxy for "Recently Moved"
recent_movers = df[df['length_of_residence'] <= 1]

# Proxy for "New Families" (can be refined with more data)
new_families = df[(df['has_children'] == 1) & (df['age_in_years'].between(28, 40))]

# --- Display Segments and Recommended Actions ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Recently Moved ({len(recent_movers)} Customers)")
    st.markdown("**Opportunity:** These customers are settling into a new home and are open to re-evaluating their policies.")
    st.success("**Proactive Action:** Initiate a 'Welcome to the Neighborhood' call. Offer a complimentary policy review to ensure their coverage matches their new property and check for new local discounts (e.g., new security system). This is a perfect time to discuss bundling home and auto insurance.")
    st.dataframe(recent_movers[['individual_id', 'city', 'county', 'length_of_residence']].head(), use_container_width=True)

with col2:
    st.subheader(f"New Families ({len(new_families)} Customers)")
    st.markdown("**Opportunity:** New parents are highly motivated to ensure their family's financial security.")
    st.success("**Proactive Action:** Send a congratulatory message. Offer a policy review to discuss increasing liability limits ('umbrella' coverage) and ensuring their vehicles are properly covered for family use. This demonstrates that Chubb understands and grows with their life stages.")
    st.dataframe(new_families[['individual_id', 'city', 'marital_status', 'age_in_years']].head(), use_container_width=True)