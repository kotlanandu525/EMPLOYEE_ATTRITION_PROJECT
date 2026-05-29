import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# ====================================
# Paths
# ====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ====================================
# Load Dataset
# ====================================

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded")
print(df.shape)

# ====================================
# Encode Target
# ====================================

df["Attrition"] = df["Attrition"].map(
    {"Yes": 1, "No": 0}
)

# ====================================
# Encode Categorical Features
# ====================================

le = LabelEncoder()

for col in df.columns:

    if df[col].dtype == "object":

        df[col] = le.fit_transform(df[col])

# ====================================
# Split Features & Target
# ====================================

X = df.drop("Attrition", axis=1)

y = df["Attrition"]

# ====================================
# Train Test Split
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ====================================
# Scaling
# ====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

# ====================================
# Logistic Regression
# ====================================

lr = LogisticRegression(
    max_iter=2000
)

lr.fit(
    X_train,
    y_train
)

lr_pred = lr.predict(X_test)

print("\nLogistic Regression")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            lr_pred
        ) * 100,
        2
    ),
    "%"
)

joblib.dump(
    lr,
    os.path.join(
        MODEL_DIR,
        "logistic_regression.pkl"
    )
)

# ====================================
# Decision Tree
# ====================================

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

dt_pred = dt.predict(X_test)

print("\nDecision Tree")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            dt_pred
        ) * 100,
        2
    ),
    "%"
)

joblib.dump(
    dt,
    os.path.join(
        MODEL_DIR,
        "decision_tree.pkl"
    )
)

# ====================================
# Random Forest
# ====================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(X_test)

print("\nRandom Forest")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            rf_pred
        ) * 100,
        2
    ),
    "%"
)

joblib.dump(
    rf,
    os.path.join(
        MODEL_DIR,
        "random_forest.pkl"
    )
)

print("\nModels Saved Successfully")