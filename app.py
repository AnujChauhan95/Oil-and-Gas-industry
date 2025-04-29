
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression

st.title("Revenue Prediction App")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    st.subheader("Initial Data Overview")
    st.write(df.head())
    
    df = df[df['Product'].notna()]
    df = df.drop(columns=['Offshore Region','Calendar Year','FIPS Code'], errors='ignore')
    
    # Handling missing values
    df = df.apply(lambda col: col.fillna(col.mode()[0]) if col.dtype == 'object' else col.fillna(col.mean()))
    
    # Encoding categorical columns
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    st.subheader("Processed Data")
    st.write(df.head())
    
    # Feature Selection and Model Training
    if 'Revenue' in df.columns:
        X = df.drop(columns=['Revenue'])
        y = df['Revenue']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        st.subheader("Model Evaluation Metrics")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
        
        st.subheader("Feature Importances")
        importance_df = pd.DataFrame({
            'Feature': df.drop(columns=['Revenue']).columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        st.write(importance_df)
        st.bar_chart(importance_df.set_index('Feature'))
    else:
        st.warning("'Revenue' column not found in the uploaded data.")
else:
    st.info("Please upload an Excel file to proceed.")
