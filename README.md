
# Financial Fraud Detection with PySpark

Academic project focused on detecting fraudulent financial transactions using Apache Spark and Machine Learning.

## Project Overview

The objective is to identify fraudulent transactions in a highly imbalanced dataset containing over 6.3 million simulated financial operations.

The project includes exploratory data analysis, preprocessing, feature selection, model training, hyperparameter optimization and evaluation.

## Technologies

- Python
- Apache Spark / PySpark
- Spark MLlib
- Pandas and NumPy
- Matplotlib
- Jupyter Notebook

## Machine Learning Models

Three classification algorithms were compared:

1. Logistic Regression
2. Random Forest
3. Gradient Boosted Trees (GBT)

The best-performing model was Gradient Boosted Trees.

## Results

| Metric | GBT |
|---|---|
| AUC-PR | 0.7547 |
| AUC-ROC | 0.9923 |
| Precision | 87.44% |
| Recall | 69.49% |
| F1-score | 0.7744 |

The selected classification threshold was 0.30.

## Dataset

The project uses a processed version of the PaySim simulated financial transactions dataset, containing 6,362,620 transactions.

Dataset reference:
https://www.kaggle.com/datasets/ealaxi/paysim1

## Project Files

- `Fraude_Spark_P1.ipynb`: complete analysis and model training.
- `fraude_spark_p1.py`: Python script version.
- `memoria_p1_aage.pdf`: academic report.
- `cleaned.csv`: dataset used for training.
- `transactions_features.csv`: additional engineered features.
- `requirements.txt`: Python dependencies.

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open the Jupyter notebook and run all cells. The notebook reads `cleaned.csv` from the project directory by default.

The reported metrics correspond to the original experimental execution. They have not been independently reproduced for this repository.

## Limitations

The dataset is simulated and the train/validation/test split is random. Results may not generalize to future real-world transactions.

Post-transaction balances are excluded from the operational model to reduce data leakage.
