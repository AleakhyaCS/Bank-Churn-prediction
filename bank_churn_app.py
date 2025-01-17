import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
#test comment
st.write("""
# Bank Churn Prediction and Risk Indicator

This app predicts whether a customer will churn and assigns them to a risk cluster.

""")

st.image(r'C:\Users\DELL\ML_proj\pr3_0A4123BF0F869513CFAC.png')

st.sidebar.header("Customer Banking Info")

def user_ip_features():
    credit_score = st.sidebar.slider("Credit Score",300, 850,650)
    geography = st.sidebar.selectbox("Geography", ["France","Germany","Spain"])
    gender = st.sidebar.selectbox("Gender",["Female","Male"])
    age = st.sidebar.slider("Age",18,100,40)
    tenure = st.sidebar.slider("Tenure(yrs)",0,10,5)
    balance = st.sidebar.slider("Balance",0.0,300000.0,50000.0)
    num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.sidebar.selectbox("Owns Credit Card", ["Yes", "No"])
    is_active_member = st.sidebar.selectbox("Is Active Member", ["Yes", "No"])
    estimated_salary = st.sidebar.slider("Estimated Salary", 0.0, 200000.0, 50000.0)

    geography_enc = {"France": 0, "Germany": 1, "Spain": 2}[geography]
    gender_enc = {"Male": 1, "Female": 0}[gender]
    has_cr_card_enc = {"Yes": 1, "No": 0}[has_cr_card]
    is_active_member_enc = {"Yes": 1, "No": 0}[is_active_member]

    
    inpdata = {
        "CreditScore": credit_score,
        "Geography": geography_enc,
        "Gender": gender_enc,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card_enc,
        "IsActiveMember": is_active_member_enc,
        "EstimatedSalary": estimated_salary,
    }
    inpfeat = pd.DataFrame(inpdata, index=[0])
    return inpfeat

input_df= user_ip_features()

rf_model = pickle.load(open("churnpred_rf.pkl", "rb"))  
cluster_model = pickle.load(open("clustering.pkl", "rb"))  
scaler_c = pickle.load(open("scaler.pkl", "rb"))  


if st.button("Enter"):
    # Scale the input data
    scaled_input = scaler_c.transform(input_df)
    
    # Predict churn using the random forest model
    prediction = rf_model.predict(input_df)
    
    # Predict risk cluster using the clustering model
    cluster = cluster_model.predict(scaled_input)

    st.subheader("Prediction")
    churn_status = np.array(["Retained", "Exited"])[prediction][0]
    cluster_mapping = {2: "Low Risk", 0: "Moderate Risk", 1: "High Risk"}
    risk_level = cluster_mapping[cluster[0]]
    
    st.write(f"Churn Prediction: **{churn_status}**")
    st.write(f"Assigned Risk Cluster: **{risk_level}**")





    
