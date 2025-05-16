# last change: 2025-05-16 22:08
"""
Airflow DAG zum Training von ML-Modellen für Aktienrenditen.
- Kombiniert CSVs aus verschiedenen Aktien
- Erstellt Features & Zielwerte
- Trainiert LightGBM-Regressoren pro Aktie
- Loggt Modelle & Metriken in MLflow
"""

from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
import mlflow
import mlflow.sklearn

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Konfiguration
# -----------------------------

DATA_DIR = "/home/holu/airflow/data/stock_data"
MODEL_DIR = "/home/holu/airflow/data/models"
COMBINED_FILE = os.path.join(DATA_DIR, "combined_stock_data.csv")
EXPERIMENT_NAME = "stock_return_prediction"

# -----------------------------
# 1️⃣ CSV-Dateien kombinieren
# -----------------------------

def combine_stock_csvs():
    """Kombiniert Einzel-Aktien-CSV-Dateien zu einer gemeinsamen Datei."""
    all_dfs = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv") and file != "combined_stock_data.csv":
            filepath = os.path.join(DATA_DIR, file)
            try:
                df = pd.read_csv(filepath)

                # Suche nach einer Zeitspalte
                datetime_col = next((col for col in df.columns if "date" in col.lower()), None)
                if not datetime_col:
                    raise ValueError(f"Keine Zeitspalte in {file} gefunden.")

                df["Datetime"] = pd.to_datetime(df[datetime_col])
                df["stock"] = file.replace(".csv", "")
                all_dfs.append(df)

            except Exception as e:
                print(f"❌ Fehler bei Datei {file}: {e}")

    if not all_dfs:
        raise ValueError("⚠️ Keine gültigen Stock-Daten gefunden!")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(COMBINED_FILE, index=False)
    print(f"✅ combined_stock_data.csv mit {len(combined)} Zeilen erfolgreich erstellt.")

# -----------------------------
# 2️⃣ Daten vorbereiten
# -----------------------------

def prepare_data():
    """Lädt und verarbeitet kombinierte Daten: Feature Engineering & Bereinigung."""
    df = pd.read_csv(COMBINED_FILE)
    print("📊 Spaltenübersicht:", df.columns.tolist())

    datetime_column = next((col for col in df.columns if "date" in col.lower()), None)
    if not datetime_column:
        raise ValueError("❌ Keine geeignete Zeitspalte gefunden.")

    try:
        df["Datetime"] = pd.to_datetime(df[datetime_column], errors="coerce", utc=True)
    except Exception as e:
        raise ValueError(f"❌ Fehler beim Konvertieren der Zeitspalte: {e}")

    df = df.dropna(subset=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values(by=["stock", "Datetime"])

    # Renditeberechnung
    df["prev_close"] = df.groupby("stock")["Close"].shift(1)
    df["return"] = (df["Close"] - df["prev_close"]) / df["prev_close"]
    df = df.dropna(subset=["return"])

    # Zeitfeatures
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month

    df = df.drop(columns=["prev_close"])
    df.to_csv(COMBINED_FILE, index=False)  # Überschreibt Datei mit Feature-Set

    return df

# -----------------------------
# 3️⃣ Modell trainieren & loggen
# -----------------------------

def train_and_log_model():
    """Trainiert je Aktie ein Modell und loggt es inklusive Metriken & Feature-Info in MLflow."""
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = prepare_data()
    feature_columns = ["Open", "High", "Low", "Volume", "hour", "dayofweek", "month"]
    target_column = "return"

    tickers = df["stock"].unique()

    for ticker in tickers:
        ticker_df = df[df["stock"] == ticker].copy()
        X = ticker_df[feature_columns]
        y = ticker_df[target_column]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LGBMRegressor(n_estimators=100, learning_rate=0.05)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        if r2 < 0.0:
            print(f"⚠️ Achtung: Modellleistung zu niedrig (R2={r2:.2f}) – wird trotzdem geloggt.")

        with mlflow.start_run(run_name=f"{ticker}_{datetime.now().isoformat()}") as run:
            mlflow.log_params({"ticker": ticker, "model_type": "LGBM", "features": feature_columns})
            mlflow.log_metrics({"r2": r2, "rmse": rmse})
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Speichere Feature-Info
            info = pd.DataFrame({"feature_columns": [feature_columns]})
            info_path = os.path.join(MODEL_DIR, "info.json")
            info.to_json(info_path)
            mlflow.log_artifact(info_path, artifact_path="info")

            # Modell registrieren & in Produktion setzen
            result = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name="stock_price_model"
            )

            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="stock_price_model",
                version=result.version,
                stage="Production",
                archive_existing_versions=True
            )

            print(f"✅ Modell für {ticker} erfolgreich mit R2={r2:.2f} geloggt und registriert.")

# -----------------------------
# ⚙️ Airflow DAG Definition
# -----------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 4, 20),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "stock_model_trainer_mlflow",
    default_args=default_args,
    description="Trainiert ML-Modelle für Aktienrenditen und loggt sie in MLflow",
    schedule_interval="10 * * * *",  # stündlich um xx:10
    catchup=False,
)

# -----------------------------
# Tasks
# -----------------------------

combine_csvs_task = PythonOperator(
    task_id="combine_stock_csvs_task",
    python_callable=combine_stock_csvs,
    dag=dag,
)

train_and_log_model_task = PythonOperator(
    task_id="train_and_log_model_task",
    python_callable=train_and_log_model,
    dag=dag,
)

# Task-Abhängigkeiten
combine_csvs_task >> train_and_log_model_task
