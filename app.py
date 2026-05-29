import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("AI Employee Attrition Prediction System")

st.write("Predict whether an employee is likely to leave the company.")

# Inputs
age = st.number_input("Age", 18, 60, 35)
monthly_income = st.number_input("Monthly Income", 1000, 50000, 4500)
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
total_working_years = st.number_input("Total Working Years", 0, 40, 10)
distance_from_home = st.number_input("Distance From Home", 1, 50, 5)

if st.button("Predict Attrition"):

    # Create input row
    sample = pd.DataFrame([{
        "Age": age,
        "BusinessTravel": 1,
        "DailyRate": 1000,
        "Department": 2,
        "DistanceFromHome": distance_from_home,
        "Education": 3,
        "EducationField": 1,
        "EmployeeCount": 1,
        "EmployeeNumber": 1001,
        "EnvironmentSatisfaction": 3,
        "Gender": 1,
        "HourlyRate": 60,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": 4,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": 1,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "Over18": 0,
        "OverTime": 1,
        "PercentSalaryHike": 12,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StandardHours": 80,
        "StockOptionLevel": 1,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 6,
        "YearsInCurrentRole": 4,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 5
    }])

    sample_scaled = scaler.transform(sample)

    probability = model.predict_proba(sample_scaled)[0][1]

    if probability > 0.7:
        risk = "HIGH"
    elif probability > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    st.success(
        f"Attrition Probability: {round(probability*100,2)}%"
    )

    st.warning(
        f"Risk Level: {risk}"
    )