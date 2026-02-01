import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation & Prediction",
    layout="centered"
)

st.title("🛍️ E-commerce Customer Segmentation App")
st.write("Enter customer RFM values to predict their segment")

# -----------------------------
# Load Models
# -----------------------------
scaler = joblib.load("scaler_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
rf_model = joblib.load("rf_classifier_model.pkl")

# Cluster name mapping (same as main.py)
cluster_names = {
    0: "High-Value Customers",
    1: "Regular Customers",
    2: "VIP Customers",
    3: "Low-Value Customers"
}

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📊 Enter Customer RFM Values")

recency = st.number_input(
    "Recency (Days since last purchase)",
    min_value=0,
    value=30
)

frequency = st.number_input(
    "Frequency (Number of purchases)",
    min_value=1,
    value=5
)

monetary = st.number_input(
    "Monetary (Total spend)",
    min_value=0.0,
    value=1000.0
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Customer Segment"):
    
    # Prepare input
    input_data = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_data)

    # ---- KMeans Prediction ----
    kmeans_cluster = kmeans.predict(input_scaled)[0]
    kmeans_segment = cluster_names[kmeans_cluster]

    # ---- Random Forest Prediction ----
    rf_prediction = rf_model.predict(input_data)[0]
    rf_segment = rf_prediction

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("📌 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("KMeans Cluster", kmeans_cluster)
        st.write(f"**Segment:** {kmeans_segment}")

    with col2:
        st.metric("RF Predicted Segment", rf_prediction)
        st.write(f"**Segment:** {rf_segment}")

    # -----------------------------
    # Business Interpretation
    # -----------------------------
    st.subheader("💡 Business Insight")

    if rf_segment == "VIP Customers":
        st.success("High spending & highly frequent customers. Offer loyalty rewards 🎁")
    elif rf_segment == "High-Value Customers":
        st.info("Strong customers with good spending. Upsell premium products 🚀")
    elif rf_segment == "Regular Customers":
        st.warning("Average engagement. Target with promotions 📣")
    else:
        st.error("Low engagement & spending. Risk of churn ⚠️")
