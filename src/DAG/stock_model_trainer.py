"""
Airflow DAG zum Training eines gemeinsamen ML-Modells fuer Aktienrenditen.

Die CSV-Dateien einzelner Aktien werden kombiniert. Das Modell lernt anschliessend
ueber alle Aktien hinweg und nutzt die Aktie selbst als One-Hot-Feature.
"""

from datetime import datetime, timedelta
import logging
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from lightgbm import LGBMRegressor
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from feature_engineering import MARKET_SYMBOLS, build_feature_frame, get_numeric_feature_columns


DATA_DIR = os.getenv("STOCK_DATA_DIR", "/opt/airflow/data/stock_data")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/data/models")
COMBINED_FILE = os.path.join(DATA_DIR, "combined_stock_data.csv")
PREPARED_FILE = os.path.join(DATA_DIR, "combined_stock_data_prepared.csv")
EXPERIMENT_NAME = "stock_return_prediction"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
MODEL_NAME = "stock_price_model"

REQUIRED_COLUMNS = ["Datetime", "stock", "Open", "High", "Low", "Close", "Volume"]


def combine_stock_csvs():
    """Kombiniert Einzel-Aktien-CSV-Dateien zu einer gemeinsamen Datei."""
    all_dfs = []

    for file_name in os.listdir(DATA_DIR):
        if not file_name.endswith(".csv") or file_name in {"combined_stock_data.csv", "combined_stock_data_prepared.csv"}:
            continue

        file_path = os.path.join(DATA_DIR, file_name)
        try:
            df = pd.read_csv(file_path)
            if "Datetime" not in df.columns:
                raise ValueError(f"Keine Datetime-Spalte in {file_name} gefunden.")

            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
            df["stock"] = file_name.removesuffix(".csv")
            all_dfs.append(df)
        except Exception as exc:
            logging.warning("Fehler bei Datei %s: %s", file_name, exc)

    if not all_dfs:
        raise ValueError("Keine gueltigen Stock-Daten gefunden.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(COMBINED_FILE, index=False)
    logging.info("combined_stock_data.csv mit %s Zeilen erstellt.", len(combined))


def prepare_data():
    """Laedt kombinierte Daten, erstellt technische Features und codiert die Aktie."""
    df = pd.read_csv(COMBINED_FILE)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Fehlende Spalten in {COMBINED_FILE}: {missing_columns}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = build_feature_frame(df, include_target=True)
    df = df[~df["stock"].isin(MARKET_SYMBOLS)]
    feature_columns = get_numeric_feature_columns(df)
    df[feature_columns] = df[feature_columns].fillna(0)
    df = df.dropna(subset=["target_return"])

    df = pd.get_dummies(
        df,
        columns=["stock"],
        prefix="stock",
        dtype=int,
    )
    df.to_csv(PREPARED_FILE, index=False)

    return df


def get_or_create_experiment(experiment_name: str):
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(experiment_name)


def train_and_log_model():
    """Trainiert ein gemeinsames Modell ueber alle Aktien und loggt es in MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_id=experiment_id)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = prepare_data()
    stock_feature_columns = sorted(column for column in df.columns if column.startswith("stock_"))
    numeric_feature_columns = [
        column for column in get_numeric_feature_columns(df)
        if not column.startswith("stock_")
    ]
    feature_columns = numeric_feature_columns + stock_feature_columns

    if len(df) < 10:
        raise ValueError("Zu wenige Trainingsdaten fuer ein gemeinsames Modell.")
    if not stock_feature_columns:
        raise ValueError("Keine Aktien-Feature-Spalten gefunden.")

    X = df[feature_columns]
    y = df["target_return"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LGBMRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    if r2 < 0.0:
        logging.warning("Modellleistung niedrig: R2=%.2f. Modell wird trotzdem geloggt.", r2)

    with mlflow.start_run(run_name=f"all_stocks_{datetime.now().isoformat()}") as run:
        mlflow.log_params({
            "model_type": "LGBM",
            "training_mode": "all_stocks",
            "num_features": len(feature_columns),
            "num_rows": len(df),
        })
        mlflow.log_metrics({"r2": r2, "rmse": rmse})

        info = pd.DataFrame({"feature_columns": [feature_columns]})
        info_path = os.path.join(MODEL_DIR, "info.json")
        info.to_json(info_path)
        mlflow.log_artifact(info_path, artifact_path="info")

        mlflow.sklearn.log_model(model, artifact_path="model")

        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=MODEL_NAME,
        )

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )

        logging.info("Gemeinsames Modell mit R2=%.2f und RMSE=%.6f registriert.", r2, rmse)


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
    description="Trainiert ein gemeinsames Modell fuer Aktienrenditen und loggt es in MLflow",
    schedule_interval="10 * * * *",
    catchup=False,
)

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

combine_csvs_task >> train_and_log_model_task
