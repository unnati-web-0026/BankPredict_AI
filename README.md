# 🏦 BankPredict AI — Bank Marketing Prediction System

A production-style Streamlit web app that predicts whether a bank customer will subscribe to a **term deposit**, based on their demographic, financial, and marketing-campaign data. Built on the **UCI Bank Marketing Dataset** and powered by a **Random Forest classifier**.

---

## 📌 Overview

Banks run large outbound marketing campaigns to sell term deposits, but most calls don't convert. This project uses machine learning to predict, in real time, how likely a given customer is to subscribe — helping target the right customers and improve campaign ROI.

The app walks through the full ML lifecycle: EDA → feature engineering → model training/comparison → evaluation → deployment as an interactive Streamlit dashboard.

---

## ✨ Features

- **🏠 Dashboard** — high-level overview and key stats
- **📋 Dataset Analytics** — dataset summary, distributions, class balance
- **📈 Data Visualization** — interactive Plotly charts (age, job, balance, correlation, campaign effectiveness, etc.)
- **🔮 Customer Prediction** — enter a customer's details and get a real-time subscription prediction with probability score
- **📊 Analytics** — model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and feature importances
- **🧠 Model Information** — model architecture and preprocessing pipeline details
- **ℹ️ About Project** — project documentation and tech stack

---

## 🗂️ Repository Structure

```
├── bank_marketing_prediction.py   # Data prep, EDA, model training & export (Colab-style script)
├── bk_3.py                        # Streamlit dashboard app (rename to app.py to run)
├── model.pkl                      # Trained Random Forest model (generated)
├── scaler.pkl                     # StandardScaler used on features (generated)
├── features.pkl                   # Training feature column list (generated)
├── cat_cols.pkl                   # Categorical columns used for encoding (generated)
├── requirements.txt               # Python dependencies
└── README.md
```

> **Note:** `model.pkl`, `scaler.pkl`, `features.pkl`, and `cat_cols.pkl` are produced by running `bank_marketing_prediction.py` on the dataset. If these files are missing, the app will still run using the **Dataset Analytics / Visualization** pages with synthetic sample data, but live prediction requires the trained model files.

---

## 📊 Dataset

The [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) contains ~45,000 records of direct marketing phone calls made by a Portuguese banking institution.

| Column | Description |
|---|---|
| `age` | Age of the customer |
| `job` | Occupation |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Has credit in default? |
| `balance` | Average yearly account balance |
| `housing` | Has a housing loan? |
| `loan` | Has a personal loan? |
| `contact` | Contact communication type |
| `day` / `month` | Last contact day/month |
| `duration` | Last contact duration (seconds) |
| `campaign` | Number of contacts in current campaign |
| `pdays` | Days since last contact in a previous campaign |
| `previous` | Number of contacts before this campaign |
| `poutcome` | Outcome of the previous campaign |
| **`deposit`** | **Target** — did the customer subscribe? (yes/no) |

---

## 🧠 Model

- **Algorithm:** Random Forest Classifier (100 estimators, `random_state=42`)
- **Also compared:** Logistic Regression, Decision Tree
- **Preprocessing:**
  - IQR-based outlier capping on `balance` and `duration`
  - Binary mapping for `default`, `housing`, `loan`, `deposit` (yes/no → 1/0)
  - `pdays == -1` converted to a `was_contacted_before` flag
  - One-hot encoding for `job`, `marital`, `education`, `contact`, `month`, `poutcome`
  - Feature scaling via `StandardScaler`
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

---

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** — web app framework
- **Scikit-learn** — model training & preprocessing
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn / Plotly** — visualization
- **Joblib** — model serialization

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the model files (if not already present)
Run the training script on the dataset to produce `model.pkl`, `scaler.pkl`, `features.pkl`, and `cat_cols.pkl`:
```bash
python bank_marketing_prediction.py
```

### 5. Run the app
The Streamlit app file is `bk_3.py`. You can either rename it to `app.py` or run it directly:
```bash
streamlit run bk_3.py
```

Then open your browser at:
```
http://localhost:8501
```

---

## 🔮 How Prediction Works

1. User fills in customer details (age, job, balance, contact history, etc.) via the sidebar form.
2. Inputs are one-hot encoded and aligned to match the model's training feature set.
3. Features are scaled using the saved `StandardScaler`.
4. The Random Forest model returns a prediction (**Subscribe / Not Subscribe**) with a probability score.

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~87% |
| Precision | ~85% |
| Recall | ~83% |
| F1 Score | ~84% |
| ROC AUC | ~0.92 |

**Top predictive features:** `duration`, `balance`, `age`, `pdays`, `campaign`, `previous`, `poutcome`

---

## 📄 License

This project is intended for educational purposes. The dataset is publicly available from the UCI Machine Learning Repository.

---

## 🙋 Acknowledgements

- Dataset: [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Built as a final-year data science project demonstrating the end-to-end ML lifecycle from EDA to deployment.
