"""
FastAPI-Service fuer Aktienkursprognosen mit einem MLflow-Modell.

Der Service laedt das gemeinsame Modell aus der MLflow Registry und erzeugt
bei Vorhersagen dieselben Feature-Spalten wie im Training.
"""

from datetime import datetime
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pandas as pd

from feature_engineering import MARKET_SYMBOLS, build_feature_frame


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "stock_price_model"
DATA_FILE = os.getenv("COMBINED_STOCK_DATA_FILE", "/opt/airflow/data/stock_data/combined_stock_data.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")

MODEL = None
FEATURE_COLUMNS = []
COMBINED_DF = None
LAST_DATA_FILE_TIMESTAMP = None


def load_model_from_registry():
    """Laedt das Production-Modell und die gespeicherte Feature-Liste."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    client = MlflowClient()
    model_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not model_versions:
        raise RuntimeError(f"Kein Production-Modell fuer {MODEL_NAME} gefunden.")

    info_path = client.download_artifacts(model_versions[0].run_id, "info/info.json")
    info_df = pd.read_json(info_path)
    feature_columns = info_df["feature_columns"].iloc[0]

    return model, feature_columns


def load_combined_data():
    """Laedt die kombinierte CSV-Datei."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} nicht gefunden.")

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        raise ValueError("CSV konnte nicht geladen werden oder ist leer.")
    if "stock" not in df.columns:
        raise ValueError("CSV enthaelt keine stock-Spalte.")

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
        df = df.sort_values(["stock", "Datetime"])

    return df


def check_reload_combined_data():
    """Laedt die CSV-Datei neu, falls sie geaendert wurde."""
    global COMBINED_DF, LAST_DATA_FILE_TIMESTAMP

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} nicht gefunden.")

    current_timestamp = os.path.getmtime(DATA_FILE)
    if LAST_DATA_FILE_TIMESTAMP is None or current_timestamp > LAST_DATA_FILE_TIMESTAMP:
        COMBINED_DF = load_combined_data()
        LAST_DATA_FILE_TIMESTAMP = current_timestamp
        print(f"Neue combined_stock_data.csv geladen: {datetime.now().isoformat()}")


def build_prediction_features(feature_frame, stock):
    """Erstellt eine Feature-Zeile mit den im Training gespeicherten Spalten."""
    if feature_frame.empty:
        raise ValueError("Keine Feature-Zeilen erzeugt.")

    stock_features = feature_frame[feature_frame["stock"] == stock].sort_values("Datetime")
    if stock_features.empty:
        raise ValueError(f"Keine Feature-Zeile fuer {stock} erzeugt.")

    latest_row = stock_features.iloc[-1:].copy()

    for column in FEATURE_COLUMNS:
        if column.startswith("stock_"):
            latest_row[column] = 1 if column == f"stock_{stock}" else 0
        elif column not in latest_row.columns:
            latest_row[column] = 0

    latest_row = latest_row[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], 0).fillna(0)
    return latest_row


@app.get("/health")
async def health():
    checks = []

    checks.append("Modell geladen" if MODEL is not None else "Modell nicht geladen")
    checks.append(f"Feature-Spalten gefunden: {len(FEATURE_COLUMNS)}" if FEATURE_COLUMNS else "Feature-Spalten fehlen")
    checks.append(
        f"Combined DataFrame geladen mit {len(COMBINED_DF)} Zeilen"
        if COMBINED_DF is not None and not COMBINED_DF.empty
        else "Combined DataFrame fehlt oder leer"
    )

    status = MODEL is not None and bool(FEATURE_COLUMNS) and COMBINED_DF is not None and not COMBINED_DF.empty

    return JSONResponse(
        content={
            "status": "healthy" if status else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
        },
        status_code=200 if status else 503,
    )


@app.get("/api/v1/predictions")
async def get_predictions():
    """Gibt pro Aktie die letzte bekannte Close-Preis- und Return-Prognose zurueck."""
    try:
        check_reload_combined_data()
        df = COMBINED_DF

        if df is None or df.empty:
            raise ValueError("Kein DataFrame geladen oder CSV leer.")

        predictions = {}
        feature_frame = build_feature_frame(df, include_target=False)

        for stock in df["stock"].dropna().unique():
            if stock in MARKET_SYMBOLS:
                continue

            stock_df = df[df["stock"] == stock].copy()
            latest_features = build_prediction_features(feature_frame, stock)

            predicted_return = MODEL.predict(latest_features)[0]
            last_close = stock_df["Close"].iloc[-1]
            predicted_price = last_close * (1 + predicted_return)
            last_timestamp = stock_df["Datetime"].iloc[-1] if "Datetime" in stock_df.columns else stock_df.index[-1]

            predictions[stock] = {
                "last_close": f"${last_close:.2f}",
                "last_timestamp": str(last_timestamp),
                "predicted_return": f"{predicted_return:.2%}",
                "predicted_price": f"${predicted_price:.2f}",
            }

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Vorhersage: {exc}") from exc


@app.on_event("startup")
async def startup_event():
    global MODEL, FEATURE_COLUMNS, COMBINED_DF, LAST_DATA_FILE_TIMESTAMP

    try:
        MODEL, FEATURE_COLUMNS = load_model_from_registry()
        COMBINED_DF = load_combined_data()
        LAST_DATA_FILE_TIMESTAMP = os.path.getmtime(DATA_FILE)
        print("Modell und Daten erfolgreich geladen.")
    except Exception as exc:
        print(f"Fehler beim Laden beim Startup: {exc}")
