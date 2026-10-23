import os
import runpy

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Always work relative to this file, no matter where streamlit is launched from
BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

st.set_page_config(page_title="ChurnGuard - Churn Predictor", page_icon="📉", layout="wide")

RED, ORANGE, GREEN, BLUE = "#e8654c", "#f0a030", "#3cb371", "#4c9be8"


# ------------------------------------------------------------------ LOADERS
@st.cache_resource
def load_bundle():
    if not os.path.exists("outputs/churn_model.pkl"):
        with st.spinner("Training model for the first time (about 1 minute)..."):
            runpy.run_path(os.path.join(BASE, "churn_project.py"))
    return joblib.load("outputs/churn_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("telco.csv")


bundle = load_bundle()
model, columns = bundle["model"], bundle["columns"]
scaler, num_cols, scaled = bundle["scaler"], bundle["num_cols"], bundle["scaled"]
data = load_data()


# ------------------------------------------------------------------ PREDICTION HELPERS
def build_raw(v: dict) -> dict:
    r = dict(v)
    r["TotalCharges"] = r["MonthlyCharges"] * r["tenure"]
    return r


def predict_many(raws: list) -> list:
    """Raw form dicts -> churn probabilities (same one-hot layout as training)."""
    rows = []
    for raw in raws:
        row = {c: 0.0 for c in columns}
        for k, val in raw.items():
            if k in num_cols or k == "SeniorCitizen":
                row[k] = float(val)
            else:
                key = f"{k}_{val}"
                if key in row:
                    row[key] = 1.0
        rows.append(row)
    X = pd.DataFrame(rows)[columns]
    if scaled:
        X[num_cols] = scaler.transform(X[num_cols])
    return model.predict_proba(X)[:, 1].tolist()


# ------------------------------------------------------------------ INPUT DEFINITIONS
YN = ["Yes", "No"]
NET = ["No", "Yes", "No internet service"]
CHOICES = {
    "gender": ["Female", "Male"],
    "Partner": YN, "Dependents": YN, "PhoneService": YN,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": NET, "OnlineBackup": NET, "DeviceProtection": NET,
    "TechSupport": NET, "StreamingTV": NET, "StreamingMovies": NET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YN,
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}
DEFAULT = dict(gender="Female", SeniorCitizen=0, Partner="Yes", Dependents="No", tenure=12,
               PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
               OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
               StreamingTV="No", StreamingMovies="No", Contract="Month-to-month",
               PaperlessBilling="Yes", PaymentMethod="Electronic check", MonthlyCharges=70.0)
HIGH = dict(DEFAULT, tenure=2, MonthlyCharges=95.0, Contract="Month-to-month",
            InternetService="Fiber optic", PaymentMethod="Electronic check",
            OnlineSecurity="No", TechSupport="No")
LOW = dict(DEFAULT, tenure=60, MonthlyCharges=42.0, Contract="Two year", InternetService="DSL",
           PaymentMethod="Credit card (automatic)", OnlineSecurity="Yes", TechSupport="Yes",
           Partner="Yes", Dependents="Yes")

# initialise widget state once
for k, val in DEFAULT.items():
    st.session_state.setdefault(f"in_{k}", val)


def apply_preset(preset: dict):
    for k, val in preset.items():
        st.session_state[f"in_{k}"] = val


# ------------------------------------------------------------------ SIDEBAR
with st.sidebar:
    st.title("📉 ChurnGuard")
    st.caption("Change any input - the prediction updates instantly.")

    st.markdown("**Quick scenarios**")
    b1, b2, b3 = st.columns(3)
    b1.button("🔴 Risky", on_click=apply_preset, args=(HIGH,), width="stretch")
    b2.button("🟢 Loyal", on_click=apply_preset, args=(LOW,), width="stretch")
    b3.button("↺ Reset", on_click=apply_preset, args=(DEFAULT,), width="stretch")

    with st.expander("👤 Customer profile", expanded=True):
        st.selectbox("Gender", CHOICES["gender"], key="in_gender")
        st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No",
                     key="in_SeniorCitizen")
        st.selectbox("Partner", CHOICES["Partner"], key="in_Partner")
        st.selectbox("Dependents", CHOICES["Dependents"], key="in_Dependents")
        st.slider("Tenure (months)", 0, 72, key="in_tenure")

    with st.expander("📡 Services", expanded=False):
        st.selectbox("Phone service", CHOICES["PhoneService"], key="in_PhoneService")
        st.selectbox("Multiple lines", CHOICES["MultipleLines"], key="in_MultipleLines")
        st.selectbox("Internet service", CHOICES["InternetService"], key="in_InternetService")
        st.selectbox("Online security", CHOICES["OnlineSecurity"], key="in_OnlineSecurity")
        st.selectbox("Online backup", CHOICES["OnlineBackup"], key="in_OnlineBackup")
        st.selectbox("Device protection", CHOICES["DeviceProtection"], key="in_DeviceProtection")
        st.selectbox("Tech support", CHOICES["TechSupport"], key="in_TechSupport")
        st.selectbox("Streaming TV", CHOICES["StreamingTV"], key="in_StreamingTV")
        st.selectbox("Streaming movies", CHOICES["StreamingMovies"], key="in_StreamingMovies")

    with st.expander("💳 Contract & billing", expanded=True):
        st.selectbox("Contract", CHOICES["Contract"], key="in_Contract")
        st.selectbox("Paperless billing", CHOICES["PaperlessBilling"], key="in_PaperlessBilling")
        st.selectbox("Payment method", CHOICES["PaymentMethod"], key="in_PaymentMethod")
        st.slider("Monthly charges ($)", 18.0, 120.0, step=0.5, key="in_MonthlyCharges")

    st.divider()
    threshold = st.slider("🚨 Alert threshold", 0.20, 0.80, 0.50, 0.05,
                          help="Customers at or above this churn probability are flagged for action.")

current = {k: st.session_state[f"in_{k}"] for k in DEFAULT}
raw = build_raw(current)
prob = predict_many([raw])[0]

# ------------------------------------------------------------------ MAIN
st.title("Customer Churn Prediction")
tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📊 Data Insights", "🏆 Model Performance"])

# ==================================================================== TAB 1
with tab1:
    flagged = prob >= threshold
    level = "HIGH" if prob >= max(threshold, 0.6) else "MEDIUM" if prob >= 0.35 else "LOW"
    level_color = {"HIGH": RED, "MEDIUM": ORANGE, "LOW": GREEN}[level]

    # ---- row 1: gauge | KPIs + actions
    g_col, k_col = st.columns([1, 1.2])
    with g_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": "Churn probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": level_color},
                   "steps": [{"range": [0, 35], "color": "rgba(60,179,113,0.25)"},
                             {"range": [35, 60], "color": "rgba(240,160,48,0.25)"},
                             {"range": [60, 100], "color": "rgba(232,101,76,0.25)"}],
                   "threshold": {"line": {"color": "black", "width": 4},
                                 "thickness": 0.8, "value": threshold * 100}}))
        fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(fig, width="stretch")

    with k_col:
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk level", level)
        m2.metric("Flagged?", "🚨 Yes" if flagged else "✅ No")
        monthly = current["MonthlyCharges"]
        m3.metric("Revenue at risk / yr", f"${monthly * 12 * prob:,.0f}",
                  help="Monthly charges x 12 x churn probability")

        tips = []
        if current["Contract"] == "Month-to-month":
            tips.append("Offer a discount to move to a 1- or 2-year contract.")
        if current["OnlineSecurity"] == "No" and current["InternetService"] != "No":
            tips.append("Offer Online Security as a free trial or bundle.")
        if current["TechSupport"] == "No" and current["InternetService"] != "No":
            tips.append("Offer Tech Support - customers with it churn less.")
        if current["PaymentMethod"] == "Electronic check":
            tips.append("Incentivise automatic payment (card / bank transfer).")
        if current["InternetService"] == "Fiber optic" and monthly > 80:
            tips.append("Review the fiber plan price - high-bill fiber is a churn hotspot.")
        if current["tenure"] < 12:
            tips.append("New customer: run an onboarding / first-year loyalty campaign.")
        st.markdown("**💡 Suggested retention actions**")
        for t in tips or ["Customer looks stable. Keep service quality high."]:
            st.markdown(f"- {t}")

    st.divider()

    # ---- row 2: why this prediction | what-if on contract
    w_col, c_col = st.columns(2)

    # model-agnostic factor impact: replace each factor by a typical value, see how risk moves
    baseline = {c: data[c].mode()[0] for c in CHOICES}
    baseline.update(SeniorCitizen=0, tenure=int(data["tenure"].median()),
                    MonthlyCharges=float(data["MonthlyCharges"].median()))
    labels = {"gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Partner",
              "Dependents": "Dependents", "tenure": "Tenure", "PhoneService": "Phone service",
              "MultipleLines": "Multiple lines", "InternetService": "Internet service",
              "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
              "DeviceProtection": "Device protection", "TechSupport": "Tech support",
              "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
              "Contract": "Contract", "PaperlessBilling": "Paperless billing",
              "PaymentMethod": "Payment method", "MonthlyCharges": "Monthly charges"}
    variants = []
    for feat in labels:
        alt = dict(current)
        alt[feat] = baseline[feat]
        variants.append(build_raw(alt))
    alt_probs = predict_many(variants)
    impact = pd.DataFrame({
        "Factor": [f"{labels[f]}: {current[f] if f != 'SeniorCitizen' else ('Yes' if current[f] else 'No')}"
                   for f in labels],
        "Impact": [(prob - p) * 100 for p in alt_probs]})
    impact = impact[impact["Impact"].abs() > 0.05]
    impact = impact.reindex(impact["Impact"].abs().sort_values(ascending=False).index).head(8)
    impact = impact.iloc[::-1]

    with w_col:
        st.subheader("Why this prediction?")
        if impact.empty:
            st.info("This customer is very close to the average profile.")
        else:
            fig = go.Figure(go.Bar(
                x=impact["Impact"], y=impact["Factor"], orientation="h",
                marker_color=[RED if x > 0 else GREEN for x in impact["Impact"]],
                text=[f"{x:+.1f}" for x in impact["Impact"]], textposition="outside"))
            fig.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=30),
                              xaxis_title="Change in churn probability (percentage points)")
            st.plotly_chart(fig, width="stretch")
            st.caption("🔴 pushes churn risk up · 🟢 pushes it down, compared with a typical customer.")

    with c_col:
        st.subheader("What if the contract changes?")
        opts = []
        for c in CHOICES["Contract"]:
            alt = dict(current, Contract=c)
            opts.append(predict_many([build_raw(alt)])[0] * 100)
        fig = go.Figure(go.Bar(
            x=CHOICES["Contract"], y=opts,
            marker_color=[BLUE if c == current["Contract"] else "#b8c4d0" for c in CHOICES["Contract"]],
            text=[f"{v:.1f}%" for v in opts], textposition="outside"))
        fig.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10),
                          yaxis=dict(title="Churn probability (%)", range=[0, max(opts) * 1.25 + 5]))
        st.plotly_chart(fig, width="stretch")
        best = min(range(3), key=lambda i: opts[i])
        if CHOICES["Contract"][best] != current["Contract"]:
            st.success(f"Moving to a **{CHOICES['Contract'][best]}** contract would cut risk from "
                       f"{prob * 100:.1f}% to {opts[best]:.1f}%.")

    st.divider()

    # ---- row 3: risk curves
    st.subheader("How risk changes with tenure and price")
    t_col, p_col = st.columns(2)
    with t_col:
        tenures = list(range(0, 73, 2))
        fig = go.Figure()
        for c, col in zip(CHOICES["Contract"], [RED, ORANGE, GREEN]):
            ps = predict_many([build_raw(dict(current, Contract=c, tenure=t)) for t in tenures])
            fig.add_trace(go.Scatter(x=tenures, y=[p * 100 for p in ps], name=c,
                                     mode="lines", line=dict(color=col, width=3)))
        fig.add_trace(go.Scatter(x=[current["tenure"]], y=[prob * 100], name="This customer",
                                 mode="markers", marker=dict(size=14, color="black", symbol="star")))
        fig.update_layout(height=340, margin=dict(t=30, b=10), title="Risk vs tenure",
                          xaxis_title="Tenure (months)", yaxis_title="Churn probability (%)",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with p_col:
        prices = [18 + i * 3 for i in range(35)]
        fig = go.Figure()
        for c, col in zip(CHOICES["Contract"], [RED, ORANGE, GREEN]):
            ps = predict_many([build_raw(dict(current, Contract=c, MonthlyCharges=float(m))) for m in prices])
            fig.add_trace(go.Scatter(x=prices, y=[p * 100 for p in ps], name=c,
                                     mode="lines", line=dict(color=col, width=3)))
        fig.add_trace(go.Scatter(x=[current["MonthlyCharges"]], y=[prob * 100], name="This customer",
                                 mode="markers", marker=dict(size=14, color="black", symbol="star")))
        fig.update_layout(height=340, margin=dict(t=30, b=10), title="Risk vs monthly charges",
                          xaxis_title="Monthly charges ($)", yaxis_title="Churn probability (%)",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

# ==================================================================== TAB 2
with tab2:
    df = data.copy()
    df["Churned"] = (df["Churn"] == "Yes").astype(int)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(df):,}")
    k2.metric("Churn rate", f"{df['Churned'].mean() * 100:.1f}%")
    k3.metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():.2f}")
    k4.metric("Avg tenure", f"{df['tenure'].mean():.0f} months")

    col = st.selectbox("Churn rate by...", ["Contract", "InternetService", "PaymentMethod",
                                            "TechSupport", "OnlineSecurity", "PaperlessBilling",
                                            "SeniorCitizen", "Partner", "Dependents"])
    g = (df.groupby(col)["Churned"].mean() * 100).reset_index()
    g[col] = g[col].astype(str)
    fig = px.bar(g, x=col, y="Churned", text=g["Churned"].round(1).astype(str) + "%",
                 labels={"Churned": "Churn %"}, color_discrete_sequence=[RED])
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    fig = px.histogram(df, x="tenure", color="Churn", nbins=36, barmode="overlay",
                       color_discrete_map={"Yes": RED, "No": BLUE}, title="Tenure distribution")
    a.plotly_chart(fig, width="stretch")
    fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                 color_discrete_map={"Yes": RED, "No": BLUE}, title="Monthly charges")
    b.plotly_chart(fig, width="stretch")
    st.image("outputs/03_feature_importance.png", caption="Top factors driving churn (Random Forest)")

# ==================================================================== TAB 3
with tab3:
    res = pd.read_csv("outputs/model_comparison.csv", index_col=0)
    st.dataframe(res, width="stretch")
    metric = st.radio("Compare models by", list(res.columns), horizontal=True, index=4)
    fig = px.bar(res.reset_index(), x="Model", y=metric, text=res[metric].round(3).values,
                 color="Model", color_discrete_sequence=[BLUE, GREEN, ORANGE])
    fig.update_layout(height=360, showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    st.image("outputs/02_evaluation.png", caption="ROC curves and confusion matrix")
    st.info("Metric to watch: **Recall** - the share of customers who really churned that the model "
            "catches. Classes are imbalanced (~26% churn), so accuracy alone is misleading.")