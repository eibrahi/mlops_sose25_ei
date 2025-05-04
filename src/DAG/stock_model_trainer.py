# -*- coding: utf-8 -*- v.2.0.4
"""
Airflow DAG für stündliches Neutrainieren des Modells auf Basis aktueller Aktienkursdaten.

Features:
- Läuft stündlich, aber 10 Minuten zeitversetzt (nach CSV-Update)
- Nutzt LightGBM für Vorhersage-Modellierung
- Speichert nur Modelle, deren R²-Score über 0.5 liegt
- Ablage der Modelle inklusive Metadaten (info.json)
"""

# ====== Importierte Pakete ======
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pytz
import os
import pickle
import pandas as pd
import numpy as np
import json
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ====== Konfigurationswerte ======
DATA_DIR = "/home/holu/airflow/data/stock_data"
MODEL_DIR = "/home/holu/airflow/data/models"
BERLIN_TZ = pytz.timezone('Europe/Berlin')

# ====== Airflow DAG Standardargumente ======
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 4, 20, tzinfo=BERLIN_TZ),
}

# ====== DAG-Definition ======
dag = DAG(
    'stock_model_trainer',
    default_args=default_args,
    description='Trainiert stündlich ein neues Aktienkurs-Modell basierend auf aktuellen CSV-Daten',
    schedule_interval='10 * * * *',
    catchup=False,
    tags=['stock', 'finance', 'model'],
)

# ====== Feature Engineering ======
def create_features(df):
    df = df.copy()
    df['return'] = df['Close'].pct_change(fill_method=None)

    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)

    for window in [5, 10, 20]:
        df[f'close_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'close_std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'return_mean_{window}'] = df['return'].rolling(window=window).mean()
        df[f'return_std_{window}'] = df['return'].rolling(window=window).std()

    df['volume_lag_1'] = df['Volume'].shift(1)
    df['volume_mean_5'] = df['Volume'].rolling(window=5).mean()
    df['volume_std_5'] = df['Volume'].rolling(window=5).std()

    return df

# ====== Datenvorbereitung ======
def prepare_data(stock_data_folder):
    all_stocks_data = []
    files = os.listdir(stock_data_folder)
    csv_files = [f for f in files if f.endswith('.csv') and not f.startswith('combined_')]

    for file in csv_files:
        file_path = os.path.join(stock_data_folder, file)
        df = pd.read_csv(file_path)

        # Einheitliche Datumsspalte erstellen und als Index setzen
        if 'Datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['Datetime'], errors='coerce', utc=True)
            df['Date'] = df['Date'].dt.tz_convert(BERLIN_TZ)
            df.set_index('Date', inplace=True)
            df.drop('Datetime', axis=1, errors='ignore', inplace=True)
        elif 'Date' in df.columns and df.index.name != 'Date':
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            df.set_index('Date', inplace=True)

        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = create_features(df)
        stock_name = file.split('_')[0].upper()
        df['stock'] = stock_name
        all_stocks_data.append(df)

    combined_df = pd.concat(all_stocks_data)

    if not combined_df.empty:
        combined_df = combined_df[~combined_df.index.isna()]
        combined_df.sort_index(inplace=True)
        combined_df.dropna(inplace=True)

        combined_csv_path = os.path.join(stock_data_folder, "combined_stock_data.csv")
        combined_df.to_csv(combined_csv_path)
        print(f"✅ combined_stock_data.csv erfolgreich gespeichert unter {combined_csv_path}")
    else:
        print("⚠️ Keine gültigen Daten gefunden. Keine combined_stock_data.csv erstellt.")

    return combined_df

# ====== Training und Speichern ======
def train_and_save_model(**kwargs):
    print("🔄 Starte Datenvorbereitung...")
    df = prepare_data(DATA_DIR)

    if df.empty:
        raise ValueError("❌ Keine Daten zum Trainieren verfügbar.")

    print(f"📊 Gesamtdaten: {len(df)} Zeilen, {df['stock'].nunique()} Aktien")

    df = df.dropna()
    if len(df) == 0:
        raise ValueError("❌ Nach dem Entfernen von NaN-Werten sind keine Daten mehr vorhanden.")

    feature_columns = [
        col for col in df.columns
        if col not in ['return', 'stock', 'Close', 'Volume', 'Price', 'Open', 'High', 'Low']
    ]
    print(f"🧮 Verwende {len(feature_columns)} Features für das Training")

    X = df[feature_columns]
    y = df['return']
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) < 20:
        raise ValueError(f"❌ Zu wenig gültige Daten zum Trainieren ({len(X)} Zeilen).")

    n_splits = min(5, len(X) // 100)
    if n_splits < 2:
        n_splits = 2

    print(f"🔄 Starte Cross-Validation mit {n_splits} Splits")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        random_state=42
    )

    scores = []
    best_iterations = []

    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"🔄 Cross-Validation Fold {i+1}/{n_splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(stopping_rounds=50)]
        )

        best_iterations.append(model.best_iteration_)
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        scores.append(score)
        print(f"📈 Fold {i+1} R² Score: {score:.4f} mit {model.best_iteration_} Iterationen")

    mean_r2 = np.mean(scores)
    median_best_iteration = int(np.median(best_iterations))
    print(f"📊 Durchschnittlicher R² Score: {mean_r2:.4f}")

    if mean_r2 <= 0.5:
        raise ValueError(f"❌ Modellqualität zu schlecht (R²={mean_r2:.4f}). Training abgebrochen.")

    print(f"🔄 Trainiere finales Modell mit {median_best_iteration} Iterationen")
    final_model = LGBMRegressor(
        n_estimators=median_best_iteration,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        random_state=42
    )
    final_model.fit(X, y)

    now = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d_%H")
    model_folder = os.path.join(MODEL_DIR, now)
    os.makedirs(model_folder, exist_ok=True)

    model_path = os.path.join(model_folder, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_columns': feature_columns
        }, f)

    info = {
        "model_version": now,
        "trained_on_stocks": list(df['stock'].unique()),
        "r2_score_mean": float(mean_r2),
        "r2_scores": [float(s) for s in scores],
        "n_training_samples": len(X),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "best_iteration": median_best_iteration
    }

    info_path = os.path.join(model_folder, "info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

    print(f"✅ Modell und Metadaten erfolgreich gespeichert unter {model_folder}")
    return model_folder

# ====== Airflow Task ======
train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_and_save_model,
    dag=dag,
)

# ====== Lokaler Test ======
if __name__ == "__main__":
    dag.test()