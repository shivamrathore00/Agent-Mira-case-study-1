# Real Estate Price Prediction & API Deployment

A complete end-to-end Machine Learning pipeline to predict house prices, featuring multi-core processing and a RESTful API.

## 📌 Project Objective

The goal of this project is to develop a robust machine learning model that predicts property sale prices based on features like location, size, and condition, and deploy it as a production-ready API.

## 🛠️ Key Features

* 
**Data Pipeline**: Automated preprocessing including One-Hot Encoding for categorical features (Location, Type, Condition) and Median Imputation for missing values.


* 
**Feature Engineering**: Derived "Property Age" from Year Built and Date Sold to capture market trends.


* 
**High Performance**: Implemented **Random Forest Regressor** with **multi-core processing** (`n_jobs=-1`) to optimize training speed.


* 
**REST API**: Built a Flask API that serves model predictions in real-time via JSON payloads.



## 📂 Dataset Overview

The model uses real estate data with the following target and features:

* 
**Target**: Price 


* 
**Features**: Property ID, Location, Size (sq ft), Bedrooms, Bathrooms, Year Built, Condition, Type, and Date Sold.



## 🚀 How to Run

1. **Install Dependencies**:
```bash
pip install pandas scikit-learn flask joblib openpyxl

```


2. **Train the Model**:
```bash
python model_train.py

```


3. **Launch the API**:
```bash
python app.py

```


4. **Test a Prediction**:
Use the following PowerShell command:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -Headers @{"Content-Type"="application/json"} -Body '{"Location": "Downtown", "Size": 1500, "Bedrooms": 3, "Bathrooms": 2, "Condition": "Good", "Type": "Single Family", "Year Built": 2010}'

```



## 📊 Results

* **Model**: Random Forest Regressor
* **Multi-core Support**: Enabled
* 
**API Endpoint**: `/predict` (POST) 



---

**Would you like me to help you draft the final "Presentation" slides content based on these results?**
# Real Estate Price Prediction & API Deployment

A complete end-to-end Machine Learning pipeline to predict house prices, featuring multi-core processing and a RESTful API.

## 📌 Project Objective

The goal of this project is to develop a robust machine learning model that predicts property sale prices based on features like location, size, and condition, and deploy it as a production-ready API.

## 🛠️ Key Features

* 
**Data Pipeline**: Automated preprocessing including One-Hot Encoding for categorical features (Location, Type, Condition) and Median Imputation for missing values.


* 
**Feature Engineering**: Derived "Property Age" from Year Built and Date Sold to capture market trends.


* 
**High Performance**: Implemented **Random Forest Regressor** with **multi-core processing** (`n_jobs=-1`) to optimize training speed.


* 
**REST API**: Built a Flask API that serves model predictions in real-time via JSON payloads.



## 📂 Dataset Overview

The model uses real estate data with the following target and features:

* 
**Target**: Price 


* 
**Features**: Property ID, Location, Size (sq ft), Bedrooms, Bathrooms, Year Built, Condition, Type, and Date Sold.



## 🚀 How to Run

1. **Install Dependencies**:
```bash
pip install pandas scikit-learn flask joblib openpyxl

```


2. **Train the Model**:
```bash
python model_train.py

```


3. **Launch the API**:
```bash
python app.py

```


4. **Test a Prediction**:
Use the following PowerShell command:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -Headers @{"Content-Type"="application/json"} -Body '{"Location": "Downtown", "Size": 1500, "Bedrooms": 3, "Bathrooms": 2, "Condition": "Good", "Type": "Single Family", "Year Built": 2010}'

```



## 📊 Results

* **Model**: Random Forest Regressor
* **Multi-core Support**: Enabled
* 
**API Endpoint**: `/predict` (POST) 



---

**Would you like me to help you draft the final "Presentation" slides content based on these results?**