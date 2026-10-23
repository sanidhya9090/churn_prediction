# Customer Churn Prediction 

Streamlit web app that predicts telecom customer churn (Logistic Regression / Random Forest / XGBoost, ROC-AUC ~0.84).

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Retrain the model
    python churn_project.py

## Deploy free on Streamlit Community Cloud
1. Push this folder to a public GitHub repo (app.py, churn_project.py, telco.csv, requirements.txt, outputs/).
2. Go to share.streamlit.io, sign in with GitHub, click "Create app".
3. Pick the repo, branch main, main file `app.py`, then Deploy.
4. You get a public link like https://yourname-churn.streamlit.app

## Files
- app.py - web app (Predict / Data Insights / Model Performance tabs)
- churn_project.py - training pipeline
- telco.csv - dataset
- outputs/ - saved model, charts, metrics
