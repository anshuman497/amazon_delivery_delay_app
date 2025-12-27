📦 Amazon Delivery Delay Prediction App

A production-ready Machine Learning web application that predicts whether an Amazon delivery will be On-Time or Delayed based on agent, traffic, weather, and order details.

🔗 Live App:
👉 https://amazondeliverydelayapp-rhnklkgc44j233y6zappc2f.streamlit.app/

🚀 Project Overview

This project demonstrates an end-to-end data analytics + machine learning pipeline, starting from data cleaning and model training to real-time prediction using a deployed Streamlit web application.

The goal is to simulate a real-world logistics delay prediction system similar to what is used in e-commerce and supply-chain companies.

🧠 Tech Stack

Python 3.11

Pandas, NumPy – data handling

Scikit-learn – preprocessing pipeline

XGBoost – classification model

Streamlit – web application & deployment

GitHub + Streamlit Cloud – CI/CD & hosting

🗂 Project Structure
amazon_delivery_delay_app/
│
├── app/
│   └── app.py                 # Streamlit application
│
├── models/
│   ├── PREPROCESSOR.pkl        # Saved preprocessing pipeline
│   └── XGBMODEL.json           # Trained XGBoost model (production)
│
├── data/
│   ├── amazon_delivery.csv
│   └── cleaned_amazon_delivery.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_delay_prediction_model.ipynb
│   └── 03_inference_and_testing.ipynb
│
├── requirements.txt
├── runtime.txt
├── README.md
└── LICENSE

📊 Features

📈 Real-time delivery delay prediction

🧹 Automated feature preprocessing (One-Hot Encoding + Scaling)

⚡ High-performance XGBoost classifier

🌐 Fully deployed cloud web application

🧪 Notebook-based training & experimentation

🔁 Production-ready inference pipeline

🖥 How the App Works

User enters order & delivery details

Input is passed through a saved preprocessing pipeline

Preprocessed data is fed into an XGBoost model

Model predicts delay probability

Result shown as:

✅ On-Time Delivery

🚨 High Delay Risk

🧪 Model Details

Algorithm: XGBoost Classifier

Problem Type: Binary Classification

Target: Delivery Delay (Yes / No)

Evaluation: Accuracy & Probability-based decision threshold

🧠 Learning Outcomes

Built an end-to-end ML system (training → inference → deployment)

Understood production model packaging & versioning

Learned Streamlit deployment best practices

Implemented clean repository structure for recruiters

📌 Why This Project Matters

This project reflects industry-relevant skills used in:

E-commerce analytics

Supply-chain optimization

Business intelligence & ML-driven decision systems

It is designed to be resume-ready for Data Analyst / Data Science internships.

👤 Author

Anshuman Mishra
Aspiring Data Analyst / Data Scientist
🔗 GitHub: https://github.com/anshuman497

🔗 LinkedIn: https://www.linkedin.com/in/contactanshuman/

📜 License

This project is licensed under the MIT License.
