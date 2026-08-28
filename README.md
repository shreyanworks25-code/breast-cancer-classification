# 🔬 Breast Cancer Classification using Machine Learning

A machine learning classification project that predicts the class of breast cancer samples using multiple classification algorithms and evaluates their performance using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

## 🚀 Live Demo

**Streamlit App:**  
https://breast-cancer-classification-frzm8rqkqdzgpmjnavwzj9.streamlit.app/

---

## 📌 Project Objective

The objective of this project is to build and evaluate machine learning classification models for predicting the class of breast cancer samples.

Multiple models were trained and compared to determine the best-performing model based on several evaluation metrics.

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin dataset** available through Scikit-learn.

The dataset contains numerical measurements derived from breast cancer cell nuclei.

The model uses **30 numerical features**, including:

- Mean measurements
- Measurement error values
- Worst-case measurements

These features are used as inputs to the classification models.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 🤖 Machine Learning Models

The following classification models were trained and evaluated:

1. Logistic Regression
2. Balanced Logistic Regression
3. Decision Tree Classifier

The balanced version of Logistic Regression was implemented using:

```python
class_weight="balanced"
```

to handle class imbalance.

---

## 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

---

## 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 95.61% | 94.59% | 98.59% | 96.55% | 99.77% |
| **Balanced Logistic Regression** | **97.37%** | **97.22%** | **98.59%** | **97.90%** | **99.77%** |
| Decision Tree | 94.74% | 95.77% | 95.77% | 95.77% | 94.40% |

---

## 🏆 Best Model

**Balanced Logistic Regression** was selected as the final model.

It achieved:

- **Accuracy:** 97.37%
- **Precision:** 97.22%
- **Recall:** 98.59%
- **F1 Score:** 97.90%
- **ROC-AUC:** 99.77%

The model provided the best overall balance of the evaluation metrics among the models tested.

The trained model was saved using Joblib as:

```text
best_classification_model.pkl
```

---

## 🌐 Streamlit Application

A Streamlit web application was developed to provide an interactive interface for the trained model.

The application allows users to:

1. Enter the required feature values.
2. Submit the values to the trained model.
3. Generate a classification prediction.
4. View the predicted class probabilities.

### Run the application locally

Clone the repository:

```bash
git clone https://github.com/shreyanworks25-code/breast-cancer-classification.git
```

Move into the project directory:

```bash
cd breast-cancer-classification
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
python -m streamlit run app1.py
```

---

## 📁 Project Structure

```text
breast-cancer-classification/
│
├── app1.py
├── best_classification_model.pkl
├── AI_ML_Task4_Classification_Models.ipynb
├── requirements.txt
└── README.md
```

### File Description

| File | Description |
|---|---|
| `app1.py` | Streamlit web application |
| `best_classification_model.pkl` | Saved trained model |
| `AI_ML_Task4_Classification_Models.ipynb` | Complete ML experimentation and evaluation |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## ⚠️ Disclaimer

This project is an educational machine learning demonstration. It is **not a medical diagnostic system** and predictions from the application should not be used for medical decisions.

---

## 👩‍💻 Project

**Breast Cancer Classification using Machine Learning**

Built using Python, Scikit-learn and Streamlit.