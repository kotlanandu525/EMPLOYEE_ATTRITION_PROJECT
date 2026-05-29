# ==========================================
# Employee Attrition Analysis (EDA)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------
# Create reports folder if not exists
# ------------------------------------------

os.makedirs("reports", exist_ok=True)

# ------------------------------------------
# Load Dataset
# ------------------------------------------
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(
    current_dir,
    "..",
    "data",
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

df = pd.read_csv(dataset_path)

print("\n========== DATASET LOADED ==========\n")

# ------------------------------------------
# Basic Information
# ------------------------------------------

print("Shape of Dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

# ------------------------------------------
# Missing Values
# ------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# 1. Attrition Rate
# ==========================================

print("\n========== ATTRITION RATE ==========\n")

attrition_rate = (
    df["Attrition"]
    .value_counts(normalize=True)
    * 100
)

print(attrition_rate)

plt.figure(figsize=(6,4))
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Rate")
plt.savefig("reports/attrition_rate.png")
plt.show()

# ==========================================
# 2. Department Wise Attrition
# ==========================================

print("\n========== DEPARTMENT WISE ATTRITION ==========\n")

dept_attrition = pd.crosstab(
    df["Department"],
    df["Attrition"]
)

print(dept_attrition)

dept_attrition.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Department Wise Attrition")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig("reports/department_attrition.png")
plt.show()

# ==========================================
# 3. Gender Wise Attrition
# ==========================================

print("\n========== GENDER WISE ATTRITION ==========\n")

gender_attrition = pd.crosstab(
    df["Gender"],
    df["Attrition"]
)

print(gender_attrition)

plt.figure(figsize=(6,4))

sns.countplot(
    x="Gender",
    hue="Attrition",
    data=df
)

plt.title("Gender Wise Attrition")
plt.savefig("reports/gender_attrition.png")
plt.show()

# ==========================================
# 4. Salary Impact
# ==========================================

print("\n========== SALARY IMPACT ==========\n")

salary_stats = df.groupby(
    "Attrition"
)["MonthlyIncome"].describe()

print(salary_stats)

plt.figure(figsize=(7,5))

sns.boxplot(
    x="Attrition",
    y="MonthlyIncome",
    data=df
)

plt.title("Salary Impact on Attrition")
plt.savefig("reports/salary_impact.png")
plt.show()

# ==========================================
# 5. Experience Impact
# ==========================================

print("\n========== EXPERIENCE IMPACT ==========\n")

experience_stats = df.groupby(
    "Attrition"
)["TotalWorkingYears"].describe()

print(experience_stats)

plt.figure(figsize=(7,5))

sns.boxplot(
    x="Attrition",
    y="TotalWorkingYears",
    data=df
)

plt.title("Experience Impact on Attrition")
plt.savefig("reports/experience_impact.png")
plt.show()

# ==========================================
# Correlation Analysis
# ==========================================

print("\n========== CORRELATION ANALYSIS ==========\n")

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("reports/correlation_heatmap.png")
plt.show()

# ==========================================
# Summary Report
# ==========================================

attrition_percent = round(
    (df["Attrition"] == "Yes").mean() * 100,
    2
)

report = f"""
Employee Attrition Analysis Report

Total Employees : {len(df)}
Attrition Rate  : {attrition_percent}%

Key Findings:
-------------
1. Attrition Rate = {attrition_percent}%

2. Department-wise attrition analyzed.

3. Gender-wise attrition analyzed.

4. Salary impact analyzed using MonthlyIncome.

5. Experience impact analyzed using TotalWorkingYears.

6. Correlation heatmap generated.

Generated Successfully.
"""

with open(
    "reports/findings.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("\nReport saved to reports/findings.txt")

print("\n========== EDA COMPLETED SUCCESSFULLY ==========\n")