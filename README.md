# 👁️ Eye Health Classification

A machine learning project that predicts whether a patient has an **eye disease** using clinical and metabolic health data.

## 📌 Project Overview

The project includes:

- Exploratory Data Analysis
- Clinical feature engineering
- Data scaling
- Neural network classification
- Multiple machine learning model comparisons
- Performance evaluation

## 📊 Dataset

The dataset contains **20,000 patient records**.

Main features include:

- Age
- Glucose
- Cholesterol
- Obesity percentage
- Blood pressure
- Heart rate
- Diabetic retinopathy status

Target:

```text
has_eye_disease
```

Class distribution:

```text
No Eye Disease : 10,130
Eye Disease    : 9,870
```

## 🧹 Feature Engineering

The notebook creates:

- Pulse Pressure
- Mean Arterial Pressure
- Age × Glucose
- Age × Systolic BP
- Metabolic Risk Score

## 🤖 Models

The project tests:

- Neural Network
- Bernoulli Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- KNN
- AdaBoost
- Multinomial Naive Bayes

Best classical-model accuracy:

```text
≈ 75.2%
```

## 📈 Key Finding

Performance stays around **75% accuracy**, suggesting clinical tabular data alone may not be enough for highly accurate eye-disease detection.

Future versions could include **retinal fundus images or OCT scans**.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## 🎯 Skills Demonstrated

- Exploratory Data Analysis
- Feature Engineering
- Neural Networks
- Classification
- Model Comparison
- Healthcare Data Analysis

---

Built with Python, TensorFlow, and Scikit-learn. 👁️🧠
