### Bank Customer Churn Prediction and Risk Clustering  

This project focuses on predicting **customer churn** in banks using **machine learning** techniques and categorizing customers into risk levels to enable better retention strategies. Customer churn refers to customers leaving the bank by closing accounts, ceasing services, or switching to competitors, which negatively impacts the bank's revenue and growth.  

#### **Key Highlights**  
1. **Objective**:  
   The project aims to build a predictive model using a **Random Forest Classifier** to forecast whether a customer is likely to churn and classify them into risk levels using **K-Means clustering**.  

2. **Data Preprocessing**:  
   - Handled class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.  
   - Standardized features for clustering using **StandardScaler**.  
   - Encoded categorical variables for model compatibility.  

3. **Model Building**:  
   - **Random Forest Classifier**: Utilized an ensemble of decision trees for accurate predictions, leveraging bagging for robust results.  
   - **K-Means Clustering**: Grouped customers into risk levels (Low, Moderate, High) based on their features.  

4. **Results and Analysis**:  
   - The Random Forest model predicts customer churn with high accuracy.  
   - Clustering highlights customer risk levels, enabling targeted retention strategies.  

5. **Application**:  
   A user-friendly **Streamlit app** was developed for real-time predictions and risk assessment. Users can input customer details to receive a churn prediction and assigned risk cluster.  

#### **Technologies and Tools**  
- **Python**: Pandas, NumPy, Scikit-learn, Streamlit  
- **Machine Learning**: Random Forest, K-Means  
- **Data Visualization**: Matplotlib, Seaborn  

#### *Conclusion*  
This project demonstrates the power of machine learning in addressing business-critical problems like customer churn. By identifying at-risk customers early, banks can implement proactive measures to enhance customer satisfaction, reduce churn, and improve long-term profitability.  

