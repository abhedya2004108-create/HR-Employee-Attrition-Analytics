# # HR Employee Attrition Prediction

An end-to-end machine learning project that predicts whether an IBM employee is likely to leave the company.

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost + SMOTE (imbalanced-learn) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Preprocessing | scikit-learn ColumnTransformer |

---

## Project Structure

```
HR Employee Attrition Analytics/
│
├── data/
│   └── HR-Employee-Attrition.csv       # Source dataset
│
├── backend/
│   ├── train_model.py                  # Training script
│   └── app.py                          # FastAPI REST API
│
├── frontend/
│   └── streamlit_app.py                # Streamlit UI
│
├── models/                             # Auto-created by training
│   ├── attrition_model.pkl
│   ├── metadata.json
│   ├── confusion_matrix.png
│   └── feature_importances.png
│
└── requirements.txt
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python backend/train_model.py
```
This will output cross-validation scores, test ROC-AUC, and save the model + plots to `models/`.

### 3. Start the FastAPI backend
```bash
uvicorn backend.app:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

### 4. Start the Streamlit frontend (new terminal)
```bash
streamlit run frontend/streamlit_app.py
```
Opens at: http://localhost:8501

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/metadata` | Feature names, options & ranges |
| POST | `/predict` | Single employee prediction |
| POST | `/predict/batch` | Batch prediction (up to 500 rows) |
| GET | `/health` | Model load status |

---

## Dateset Source :
https://www.kaggle.com/datasets/rishikeshkonapure/hr-analytics-prediction

## Frontend Pages

| Page | Description |
|---|---|
| 🏠 Dashboard | KPI cards + department/role/overtime charts |
| 🔍 Single Prediction | Interactive form → gauge chart + risk factors |
| 📂 Batch Prediction | Upload CSV → bulk predictions + download results |
| 📊 EDA & Insights | Distributions, correlations, satisfaction analysis |
| 🤖 Model Performance | Confusion matrix, feature importances, CV scores |

---

## Model Details

- **Algorithm**: XGBoost Classifier
- **Imbalance handling**: SMOTE oversampling on training split only
- **Cross-validation**: Stratified 5-Fold, scored by ROC-AUC
- **Preprocessing**: StandardScaler (numeric) + OneHotEncoder (categorical)
- **Features**: 30 input features (4 constant/ID columns dropped)
HR-Employee-Attrition-Analytics
