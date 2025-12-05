🛠️ Predictive Maintenance for Industrial Machines
Team 4 — Raj & Vaibhav | EM 538 Final Project | Spring 2025
📌 Overview

This project builds an end-to-end predictive maintenance system that uses industrial sensor data to predict machine failures before they occur.
We combine:

Machine Learning (XGBoost, Random Forest, Logistic Regression)

Feature Engineering

Explainability (SHAP)

AI-Assisted Maintenance Reports (Groq LLM)

Deployment-ready design (Streamlit)


Our final model achieves 99% accuracy and 82% precision & recall for the failure class, making it highly reliable for real-world applications.

🧩 Problem Statement

Industrial machines generate a large amount of sensor data, but failures are rare and often unpredictable.
Traditional maintenance relies on:

Reactive maintenance — fix after failure

Preventive maintenance — fixed schedules, often inefficient

Predictive maintenance uses sensor data + ML to detect early signs of failure, reducing downtime, improving safety, and saving cost.

Goal:
Build a binary classification model to predict:

0 → Machine is normal

1 → Machine is likely to fail

📚 Dataset

We use the AI4I 2020 Predictive Maintenance Dataset, containing ~10,000 rows of sensor readings.

Features:

Air temperature

Process temperature

Rotational speed

Torque

Tool wear

Product type

Target:

Machine failure (0 or 1)

Key Challenge:

The dataset is imbalanced — failures are rare (~3–5%).
Thus, evaluation must focus on precision, recall, and F1-score, especially for the failure class.

🔧 Preprocessing Pipeline

Removed unnecessary ID columns

One-hot encoded categorical Type

Scaled numerical features for logistic regression

Stratified train-test split

Feature Engineering:

Temp_delta = ProcessTemp – AirTemp → thermal stress

Power_est = Torque × RotationalSpeed → mechanical load

These engineered features significantly improved model performance.

🤖 Machine Learning Models

We trained and compared three ML algorithms:

Model	Notes
Logistic Regression	Linear baseline model
Random Forest	Nonlinear ensemble, strong performance but missed failures
XGBoost	Best-performing model, handles imbalance & nonlinearities
🏆 Final Model: XGBoost

Balanced performance:

Accuracy: 99%

Precision (Failure): 82%

Recall (Failure): 82%

F1-Score (Failure): 82%

XGBoost was chosen because predictive maintenance requires both catching failures and minimizing false alarms.

📊 Model Evaluation

Since accuracy alone is misleading for imbalanced data, we evaluated:

Precision (failure class)

Recall (failure class)

F1-score

Confusion Matrix

ROC-AUC

Error analysis (false positives, false negatives)

🔍 Confusion Matrix Insights (XGBoost)

False Positives: 12

False Negatives: 12

Balanced errors → reliable model for real-world deployment.

🧠 Explainability & AI Angle
⭐ SHAP Explainability

We used SHAP (SHapley Additive exPlanations) to understand:

Global explanations

Which features MOST influence the model?

Torque

Rotational speed

Tool wear

Temperature delta

Power estimate

These match real mechanical behavior → increases trust.

Local explanations

For each prediction, SHAP shows how each feature pushed the outcome toward failure or normal.

⭐ AI-Assisted Maintenance Reports (Groq LLM)

We integrated Groq’s LLM to generate:

Natural-language explanations for model predictions

Error reasoning for false positives & negatives

Technician-friendly maintenance reports

Example output:

“Torque and temperature delta show an abnormal rise, indicating early mechanical stress. Recommended action: inspect spindle alignment.”

This bridges ML with real-world usability.

🚀 Deployment Architecture

A Streamlit-based interactive app is designed to:

Accept new sensor readings

Run XGBoost model prediction

Show SHAP explanation

Generate Groq AI maintenance reports

Can be integrated with IoT pipelines for real-time monitoring.