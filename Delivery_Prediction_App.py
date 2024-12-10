import streamlit as st
import pandas as pd
from joblib import load
from common_function import common_functions

# Load model
model = load('random_forest_model.joblib')
comm_fun = common_functions()

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Header and Title
st.title("📊 Customer Churn Prediction App")
st.write("### Predict the likelihood of a customer churning based on various factors")

# Sidebar
st.sidebar.title("Customer Information")
st.sidebar.write("Enter the customer details below to predict if they will churn.")

# Tabs for input organization
tab1, tab2 = st.tabs(["Customer Info", "Prediction Result"])

# Variable to track if prediction is made
prediction_made = False

# Use session state to manage the selected tab
if 'tab' not in st.session_state:
    st.session_state.tab = 0 

# Change tab button is clicked
def change_tab():
    st.session_state.tab = 1  

with tab1:
    # Input fields for the new dataset
    st.subheader("Enter Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        warehouse_block = st.selectbox("Select Warehouse Block", ("A", "B", "C", "D", "F"))
        mode_of_shipment = st.selectbox("Select Mode of Shipment", ("Flight", "Ship", "Road"))
        customer_care_calls = st.number_input("Number of Customer Care Calls", min_value=0, max_value=10, value=0)
        customer_rating = st.slider("Customer Rating", min_value=1, max_value=5, value=3)
        cost_of_product = st.number_input("Cost of Product (€)", min_value=0, max_value=1000, value=200)

    with col2:
        prior_purchases = st.number_input("Prior Purchases", min_value=0, max_value=10, value=0)
        product_importance = st.selectbox("Select Product Importance", ("low", "medium", "high"))
        gender = st.selectbox("Select Gender", ("M", "F"))
        discount_offered = st.number_input("Discount Offered (%)", min_value=0, max_value=100, value=10)
        weight_in_gms = st.number_input("Weight of Product (grams)", min_value=100, max_value=5000, value=1000)

    # Encoding features
    encoded_warehouse_block = comm_fun.label_encode(data=pd.DataFrame({'Warehouse_block': [warehouse_block]}), column="Warehouse_block").iloc[0]
    encoded_mode_of_shipment = comm_fun.label_encode(data=pd.DataFrame({'Mode_of_Shipment': [mode_of_shipment]}), column="Mode_of_Shipment").iloc[0]
    encoded_product_importance = comm_fun.label_encode(data=pd.DataFrame({'Product_importance': [product_importance]}), column="Product_importance").iloc[0]
    encoded_gender = comm_fun.label_encode(data=pd.DataFrame({'Gender': [gender]}), column="Gender").iloc[0]

    # List of selected features
    selected_features = [
        encoded_warehouse_block, encoded_mode_of_shipment, customer_care_calls,
        customer_rating, cost_of_product, prior_purchases, encoded_product_importance,
        encoded_gender, discount_offered, weight_in_gms
    ]

    if st.button("Predict Churn"):
        if None in selected_features or '' in selected_features:
            st.error("Please fill all fields before making a prediction.")
        else:
            predicted_churn = model.predict([selected_features])[0]
            st.session_state.predicted_churn = predicted_churn

            change_tab()

with tab2:
    if 'predicted_churn' in st.session_state:
        st.subheader("Prediction Result")
        predicted_churn = st.session_state.predicted_churn

        if predicted_churn == 1:
            st.markdown('<h3 style="color:green;">The customer is likely to churn (Not reached on time).</h3>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 style="color:red;">The customer is unlikely to churn (Reached on time).</h3>', unsafe_allow_html=True)
    else:
        st.error("Prediction not yet made. Please fill in all fields and click 'Predict Churn'.")