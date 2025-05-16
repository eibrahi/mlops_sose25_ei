# fastapi_model_server.py
"""
FastAPI-Service zur Bereitstellung von Aktienkursprognosen mithilfe eines MLflow-Modells.
- Lädt Modell & Feature-Metadaten beim Start
- Automatischer Reload der CSV-Datei bei Änderungen
- Healthcheck-Endpoint
- Vorhersage-Endpunkt für aktuelle Kursprognosen pro Aktie
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pandas as pd
import os

# ------------------------------------------------
# 🌍 FastAPI Setup
# ------------------------------------------------

app = FastAPI()

# CORS für alle Domains freigeben (z. B. für Frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# 📁 Konfiguration & globale Variablen
# ------------------------------------------------

MODEL_DIR = "/home/holu/airflow/data/models"
DATA_FILE = "/home/holu/airflow/data/stock_data/combined_stock_data.csv"
MLFLOW_TRACKING_URI = "http://localhost:5001"

MODEL = None
FEATURE_COLUMNS = []
COMBINED_DF = None
LAST_DATA_FILE_TIMESTAMP = None

# ------------------------------------------------
# 📦 Modell & Daten laden
# ------------------------------------------------

def load_model_from_registry():
    """
    Lädt das Modell, das in MLflow unter 'Production' registriert ist,
    inklusive der Feature-Metadaten aus info.json.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = "models:/stock_price_model/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    client = MlflowClient()
    reg_model = client.get_latest_versions("stock_price_model", stages=["Production"])[0]
    run_id = reg_model.run_id

    try:
        info_path = client.download_artifacts(run_id, "info/info.json")
        info_df = pd.read_json(info_path)
        feature_columns = info_df["feature_columns"].iloc[0]
    except Exception as e:
        raise RuntimeError(f"Fehler beim Laden von info.json: {e}")

    return model, feature_columns


def load_combined_data():
    """Lädt die kombinierte CSV-Datei und prüft auf Fehler oder Leere."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} nicht gefunden.")
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError("❌ CSV konnte nicht geladen werden oder ist leer.")
    return df


def check_reload_combined_data():
    """Lädt die CSV-Datei neu, falls sie zwischenzeitlich geändert wurde."""
    global COMBINED_DF, LAST_DATA_FILE_TIMESTAMP
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} nicht gefunden.")

    current_timestamp = os.path.getmtime(DATA_FILE)
    if LAST_DATA_FILE_TIMESTAMP is None or current_timestamp > LAST_DATA_FILE_TIMESTAMP:
        COMBINED_DF = load_combined_data()
        LAST_DATA_FILE_TIMESTAMP = current_timestamp
        print(f"🔄 Neue combined_stock_data.csv geladen (Stand: {datetime.now()})")

# ------------------------------------------------
# 🔍 Healthcheck Endpoint
# ------------------------------------------------

@app.get("/health")
async def health():
    """Gibt Informationen über Modellstatus, Feature-Set und Datensatzzustand zurück."""
    checks = []

    if MODEL is not None:
        checks.append("✅ Modell geladen")
    else:
        checks.append("❌ Modell nicht geladen")

    if FEATURE_COLUMNS:
        checks.append(f"✅ Feature-Spalten gefunden: {len(FEATURE_COLUMNS)}")
    else:
        checks.append("❌ Feature-Spalten fehlen")

    if COMBINED_DF is not None and not COMBINED_DF.empty:
        checks.append(f"✅ Combined DataFrame geladen mit {len(COMBINED_DF)} Zeilen")
    else:
        checks.append("❌ Combined DataFrame fehlt oder leer")

    status = all("✅" in c for c in checks)

    return JSONResponse(
        content={
            "status": "healthy" if status else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "checks": checks
        },
        status_code=200 if status else 503
    )

# ------------------------------------------------
# 📈 Vorhersage-Endpunkt
# ------------------------------------------------

@app.get("/api/v1/predictions")
async def get_predictions():
    """
    Gibt für jede Aktie die letzte bekannte Close-Preis, Return-Prognose und Preis-Prognose zurück.
    Nutzt dabei die aktuellsten Features und ein MLflow-Modell.
    """
    try:
        check_reload_combined_data()
        df = COMBINED_DF

        if df is None or df.empty:
            raise ValueError("❌ Kein DataFrame geladen oder CSV leer.")

        predictions = {}

        for stock in df['stock'].unique():
            stock_df = df[df['stock'] == stock].copy()
            latest_features = stock_df[FEATURE_COLUMNS].iloc[-1:]

            predicted_return = MODEL.predict(latest_features)[0]
            last_close = stock_df['Close'].iloc[-1]
            predicted_price = last_close * (1 + predicted_return)
            last_timestamp = stock_df.index[-1]

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Vorhersage: {e}")

# ------------------------------------------------
# ⚙️ Initialisierung bei Serverstart
# ------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Lädt Modell und Daten beim Start des FastAPI-Servers in den Speicher.
    """
    global MODEL, FEATURE_COLUMNS, COMBINED_DF, LAST_DATA_FILE_TIMESTAMP
    try:
        MODEL, FEATURE_COLUMNS = load_model_from_registry()
        COMBINED_DF = load_combined_data()
        LAST_DATA_FILE_TIMESTAMP = os.path.getmtime(DATA_FILE)
        print("✅ Modell & Daten erfolgreich geladen.")
    except Exception as e:
        print(f"❌ Fehler beim Laden beim Startup: {e}")
