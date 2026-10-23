# Customer Churn Prediction System
`Overview`

Customer churn is when customers stop using a company’s service. Winning a new customer costs far more than keeping an existing one, so telecom companies want to spot customers who are likely to leave before they do. This project is a machine learning system that predicts whether a telecom customer will churn. It is deployed as a live web app where a user can enter customer details and get an instant risk assessment.

`Objectives`
Analyze customer data to find the factors that drive churn.
Build and compare classification models that predict churn.
Handle the class imbalance in the data (about 26.5% of customers churn).
Deploy the best model as an interactive web application.
Suggest retention actions based on each customer’s risk profile.
Dataset.

# The IBM Telco Customer Churn dataset has 7,043 customers and 21 features:

`Demographics:` gender, senior citizen, partner, dependents
`Services:` phone, internet, online security, tech support, streaming
`Account and billing:` tenure, contract type, payment method, monthly and total charges
`Target:` Churn (Yes/No)
`Methodology`
`Data cleaning:` Fixed the missing values in TotalCharges (new customers with zero tenure) and removed the customer ID.
`Exploratory data analysis:` Looked at churn by contract, internet service, payment method and tenure.
`Preprocessing:` One-hot encoded the categorical features, scaled the numeric features, and made an 80/20 stratified train/test split.
`Model training:` Trained Logistic Regression, Random Forest and XGBoost, using class weighting to handle the imbalance.
Evaluation: Compared the models on accuracy, precision, recall, F1 and ROC-AUC, and checked them with 5-fold cross-validation.
`Deployment:` Built a Streamlit web app with prediction, insights and model performance tabs.
# Results
Model	Accuracy	Recall	F1	ROC-AUC
Logistic Regression	73.8%	78.3%	0.614	0.842
Random Forest	76.3%	71.9%	0.617	0.841
XGBoost	75.2%	78.6%	0.627	0.839

Logistic Regression had the highest ROC-AUC and was selected as the final model. The model correctly identifies about 78% of customers who actually churn. Recall matters most here, because missing a customer who is about to leave costs more than a false alarm.

`Key Findings`
Customers with short tenure are the most likely to churn.
Month-to-month contracts have much higher churn than one- or two-year contracts.
Fiber optic internet users and customers paying by electronic check churn more.
Customers without online security or tech support churn more.
Web Application Features
Predict: Enter customer details to get a churn probability, a risk level (Low/Medium/High) and personalized retention suggestions.
Data Insights: Interactive charts of churn by customer segment.
Model Performance: Model comparison table, ROC curve and confusion matrix.
Business Impact

The company can contact high-risk customers early, offer long-term contract discounts, promote add-on services like security and tech support, and encourage automatic payments, all of which reduce churn.

# Technologies Used

`Python`, `Pandas`, `NumPy`, `Scikit-learn`, `XGBoost`, `Matplotlib`, `Seaborn`, `Streamlit`, `Joblib`, `GitHub`, `Streamlit Community Cloud`

# Future Scope
Hyperparameter tuning and model explainability with SHAP
Retraining on live customer data
Customer lifetime value estimates alongside churn risk
Integration with a CRM system for automatic alerts


