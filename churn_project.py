
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, classification_report)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv("telco.csv")
print("Shape:", df.shape)

# ---------------------------------------------------------------- 2. CLEAN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])  # new customers
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print("Churn rate: %.1f%%" % (df["Churn"].mean() * 100))

# ---------------------------------------------------------------- 3. EDA
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().plot.pie(
    autopct="%1.1f%%", ax=ax[0, 0], colors=["#4c9be8", "#e8654c"])
ax[0, 0].set_title("Churn Distribution"); ax[0, 0].set_ylabel("")
for a, col in zip([ax[0, 1], ax[0, 2], ax[1, 0]], ["Contract", "InternetService", "PaymentMethod"]):
    (df.groupby(col)["Churn"].mean() * 100).sort_values().plot.barh(ax=a, color="#e8654c")
    a.set_title(f"Churn % by {col}"); a.set_xlabel("Churn %")
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, ax=ax[1, 1], multiple="stack")
ax[1, 1].set_title("Tenure vs Churn")
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax[1, 2])
ax[1, 2].set_title("Monthly Charges vs Churn")
plt.tight_layout(); plt.savefig(f"{OUT}/01_eda.png", dpi=110); plt.close()

# ---------------------------------------------------------------- 4. PREPROCESS
X = pd.get_dummies(df.drop(columns="Churn"), drop_first=True).astype(float)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler().fit(X_train[num_cols])
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[num_cols] = scaler.transform(X_train[num_cols])
X_test_s[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------------- 5. MODELS
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # handle class imbalance
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"), True),
    "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=10,
                      class_weight="balanced", random_state=42, n_jobs=-1), False),
    "XGBoost": (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                scale_pos_weight=pos_weight, eval_metric="logloss", random_state=42), False),
}

rows, fitted, probs = [], {}, {}
for name, (m, scaled) in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
    m.fit(Xtr, y_train)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    cv = cross_val_score(m, Xtr, y_train, cv=5, scoring="roc_auc").mean()
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "ROC-AUC": roc_auc_score(y_test, p),
                 "CV ROC-AUC": cv})
    fitted[name], probs[name] = m, p

results = pd.DataFrame(rows).set_index("Model").round(3)
print("\n=== MODEL COMPARISON ===\n", results)
results.to_csv(f"{OUT}/model_comparison.csv")

best_name = results["ROC-AUC"].idxmax()
best = fitted[best_name]
print("\nBest model:", best_name)
best_pred = (probs[best_name] >= 0.5).astype(int)
print(classification_report(y_test, best_pred, target_names=["Stayed", "Churned"]))

# ---------------------------------------------------------------- 6. EVALUATION PLOTS
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--"); ax[0].legend(); ax[0].set_title("ROC Curves")
ax[0].set_xlabel("False Positive Rate"); ax[0].set_ylabel("True Positive Rate")
sns.heatmap(confusion_matrix(y_test, best_pred), annot=True, fmt="d", cmap="Blues", ax=ax[1],
            xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
ax[1].set_title(f"Confusion Matrix - {best_name}")
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual")
plt.tight_layout(); plt.savefig(f"{OUT}/02_evaluation.png", dpi=110); plt.close()

# ---------------------------------------------------------------- 7. FEATURE IMPORTANCE
rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
plt.figure(figsize=(9, 6))
imp[::-1].plot.barh(color="#4c9be8")
plt.title("Top 12 Factors Driving Churn (Random Forest)")
plt.tight_layout(); plt.savefig(f"{OUT}/03_feature_importance.png", dpi=110); plt.close()
print("\nTop churn drivers:\n", imp.round(3))

# ---------------------------------------------------------------- 8. SAVE MODEL + RISK LIST
joblib.dump({"model": best, "columns": list(X.columns), "scaler": scaler,
             "num_cols": num_cols, "scaled": models[best_name][1]}, f"{OUT}/churn_model.pkl")

risk = X_test.copy()
risk["Churn_Probability"] = probs[best_name]
risk["Actual_Churn"] = y_test.values
risk.sort_values("Churn_Probability", ascending=False).head(50)[
    ["tenure", "MonthlyCharges", "Churn_Probability", "Actual_Churn"]
].round(3).to_csv(f"{OUT}/top50_high_risk_customers.csv")
print("\nSaved all outputs to ./outputs/")
