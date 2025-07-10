# 🧠 Pathrise Placement Prediction Project

## 📋 Overview

**Pathrise** is an online program that helps job seekers secure top tech jobs through personalized 1-on-1 mentorship and expert training. Every two weeks, a new cohort of fellows joins the program. On average, fellows who continue after their free trial are placed in approximately **4 months**, but outcomes vary significantly.

This project aims to **predict**:

1. **Whether a fellow will be placed at a company** (Classification)  
2. **How long it will take** for the placement to happen (Regression)

Accurate predictions can help Pathrise improve resource allocation and provide more tailored support to its fellows.

---

## 🎯 Project Objectives

- Perform **Exploratory Data Analysis (EDA)** and **Feature Engineering**
- Build and evaluate multiple **Classification** models to predict placement status
- Build and evaluate multiple **Regression** models to estimate placement duration

---

## 🔍 Data Analysis & Preprocessing

Data preprocessing steps (handled in a separate script) include:

- Handling missing values  
- Encoding categorical features  
- Normalizing numerical features  
- Feature selection/engineering  

---

## ⚙️ Stage 1: Classification – Predicting Placement Status

This stage uses supervised learning to predict whether a fellow will eventually be placed.

### ✅ Models Evaluated:
- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree Classifier  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  
- XGBoost Classifier

> **Goal:** Identify the most accurate model for binary classification (Placed vs. Not Placed)

---

## ⏳ Stage 2: Regression – Predicting Job Search Duration

This stage predicts how many days it will take for a placed fellow to land a job.

### ✅ Models Evaluated:
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- ElasticNet  
- Random Forest Regressor  
- XGBoost Regressor

> **Goal:** Find the best-performing regression model based on metrics such as **MAE**

---

## 📊 Tools & Libraries

- Python (Pandas, NumPy)  
- Scikit-learn  
- Matplotlib, Seaborn  
- XGBoost  
- Jupyter Notebooks  

---

## 🚀 Getting Started

1. **Clone the repository**  
   ```bash
  
   git clone https://github.com/Mahdi-Alijani-Lati/Pathrise-ML-Pipeline.git
   cd Pathrise-ML-Pipeline

   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the scripts or notebooks**  
   - EDA: `jupyter notebook Pathrise_Preprocessing.ipynb`  
   - Classification models : `jupyter notebook Pathrise_Classification.ipynb`  
   - Regression models: `jupyter notebook Pathrise_Regression.ipynb`

---

## 📁 Project Structure

```
.
├── data/
│   └── Data_Pathrise.xlsx
├── notebooks/
│   ├── 1_Pathrise_Preprocessing.ipynb
│   ├── 2_Pathrise_Classification.ipynb
│   └── 3_Pathrise_Regression.ipynb
├── src/
│   └── (Python modules will be added here soon)
├── models/
│   ├── Pathrise_Classification.pkl
│   ├── Pathrise_model_columns.pkl
│   ├── Pathrise_scaler.pkl
│   └── program_duration_days_Estimater_regression.pkl
└── README.md
```

---

## ✅ Results Summary

| Task           | Best Model               | Metric       |
|----------------|--------------------------|--------------|
| Classification | Support Vector Machine   | F1 = 0.77    |
| Regression     | XGBoost Regressor        | MAE = 78     |


---

## 📌 Notes

- The classification and regression tasks are handled separately.  
- All preprocessing is done beforehand to allow flexible model experimentation.  
- Project is modular and ready for deployment or further development.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

- Mahdi Alijani Lati 
- Mahdi.Alijani.Lati@gmail.com  
- GitHub: [Mahdi-Alijani-Lati](https://github.com/Mahdi-Alijani-Lati)
