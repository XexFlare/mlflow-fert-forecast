# mlflow-fert-forecast

An end-to-end machine learning project for fertilizer demand forecasting and farmer behavior analysis, built using MLflow for experiment tracking, reproducibility, and model management.

---

## 🌍 Business Context

This project is inspired by real-world agricultural operations in Malawi, where fertilizer distribution plays a critical role in food security and farmer productivity.

Accurate forecasting helps:

- Optimize fertilizer supply chains
- Reduce stockouts and overstocking
- Improve planning for seasonal demand
- Support data-driven agricultural programs

---

## 🎯 Project Objectives

- Forecast fertilizer demand based on historical farmer data
- Identify patterns in farmer purchasing behavior
- Track and compare multiple model experiments using MLflow
- Build a reproducible and production-ready ML pipeline

---

## 🧠 Models Used

This project experiments with multiple classical machine learning models:

- Random Forest
- XGBoost
- Logistic Regression (baseline)

Models are evaluated using:

- Accuracy
- Precision / Recall
- F1 Score
- ROC-AUC

---

## ⚙️ Tech Stack

- Python 3.8+
- MLflow (experiment tracking & model registry)
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Matplotlib / Seaborn

---

## 📁 Project Structure


mlflow-fert-forecast/
├── data/
│ ├── raw/
│ └── processed/
├── models/
├── notebooks/
├── src/
│ ├── data_processing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ └── inference.py
├── mlruns/
├── results/
├── MLproject
├── conda.yaml
├── requirements.txt
└── README.md


---

## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/XexFlare/mlflow-fert-forecast.git
cd mlflow-fert-forecast


### 2. Install dependencies
pip install -r requirements.txt

Or with conda:

conda env create -f conda.yaml
conda activate mlflow-fert-forecast


### 3. Run the pipeline
mlflow run .


### 4. Launch MLflow UI
mlflow ui


Open in browser:

http://localhost:5000
📊 What MLflow Tracks


## Each run logs:

Parameters (e.g., max_depth, n_estimators)
Metrics (accuracy, F1 score, etc.)
Model artifacts
Experiment history
📈 Example Use Case

## Predict future fertilizer demand or identify farmers at risk of reduced purchasing, enabling:

Targeted interventions
Improved logistics planning
Better inventory management
🔮 Future Improvements
Add time-series forecasting (Prophet / LSTM)
Integrate real weather data
Add API for real-time predictions
Combine with LLMs for explainability
Deploy via Docker / cloud


# 🤝 Contributing

This project is part of a broader exploration into AI-driven agriculture systems. Contributions and ideas are welcome.


# 📜 License

This project is licensed under the MIT License.
