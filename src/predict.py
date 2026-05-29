import pandas as pd
import joblib
import os

# =====================================
# Paths
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

# =====================================
# Load Model & Scaler
# =====================================

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "random_forest.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

# =====================================
# Example Employee Data
# =====================================

employee = pd.DataFrame([{
    "Age": 35,
    "BusinessTravel": 1,
    "DailyRate": 1100,
    "Department": 2,
    "DistanceFromHome": 5,
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
    "JobSatisfaction": 2,
    "MaritalStatus": 1,
    "MonthlyIncome": 4500,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "Over18": 0,
    "OverTime": 1,
    "PercentSalaryHike": 12,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StandardHours": 80,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 6,
    "YearsInCurrentRole": 4,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 5
}])

# =====================================
# Scale
# =====================================

employee_scaled = scaler.transform(employee)

# =====================================
# Prediction
# =====================================

prediction = model.predict(employee_scaled)[0]

probability = model.predict_proba(
    employee_scaled
)[0][1]

# =====================================
# Risk Level
# =====================================

if probability >= 0.7:
    risk = "HIGH"
elif probability >= 0.4:
    risk = "MEDIUM"
else:
    risk = "LOW"

print("\nPrediction Results")
print("-" * 30)

print(
    "Attrition Probability:",
    round(probability * 100, 2),
    "%"
)

print(
    "Risk Level:",
    risk
)