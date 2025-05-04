# fastapi_model_server.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import os
from datetime import datetime
from glob import glob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "/home/holu/airflow/data/models"
DATA_FILE = "/home/holu/airflow/data/stock_data/combined_stock_data.csv"

# In-Memory-Cache
MODEL = None
FEATURE_COLUMNS = []
COMBINED_DF = None
LAST_DATA_FILE_TIMESTAMP = None

def load_latest_model():
    """Lade das neueste Modell basierend auf Ordnernamen."""
    model_folders = sorted(glob(os.path.join(MODEL_DIR, "*")), reverse=True)
    if not model_folders:
        raise FileNotFoundError("Kein Modellordner gefunden.")

    latest_model_folder = model_folders[0]
    model_path = os.path.join(latest_model_folder, "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelldatei nicht gefunden in {latest_model_folder}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_columns']

def load_combined_data():
    """Lade die kombinierte Stock-Daten-Datei."""
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    return df

def check_reload_combined_data():
    """Prüfe, ob combined_stock_data.csv aktualisiert wurde und lade ggf. neu."""
    global COMBINED_DF, LAST_DATA_FILE_TIMESTAMP
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} nicht gefunden.")

    current_timestamp = os.path.getmtime(DATA_FILE)
    if LAST_DATA_FILE_TIMESTAMP is None or current_timestamp > LAST_DATA_FILE_TIMESTAMP:
        # Datei wurde geändert ➔ neu laden
        COMBINED_DF = load_combined_data()
        LAST_DATA_FILE_TIMESTAMP = current_timestamp
        print(f"🔄 Neue combined_stock_data.csv geladen (Stand: {datetime.now()})")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/v1/predictions")
async def get_predictions():
    try:
        # Prüfen, ob neue CSV geladen werden muss
        check_reload_combined_data()

        df = COMBINED_DF

        latest_data = {}
        for stock in df['stock'].unique():
            stock_df = df[df['stock'] == stock].copy()
            latest_features = stock_df[FEATURE_COLUMNS].iloc[-1:].copy()

            predicted_return = MODEL.predict(latest_features)[0]
            last_close = stock_df['Close'].iloc[-1]
            predicted_price = last_close * (1 + predicted_return)
            last_timestamp = stock_df.index[-1]

            latest_data[stock] = {
                'last_close': f"${last_close:.2f}",
                'last_timestamp': str(last_timestamp),
                'predicted_return': f"{predicted_return:.2%}",
                'predicted_price': f"${predicted_price:.2f}",
            }

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "predictions": latest_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    global MODEL, FEATURE_COLUMNS, COMBINED_DF, LAST_DATA_FILE_TIMESTAMP
    try:
        MODEL, FEATURE_COLUMNS = load_latest_model()
        COMBINED_DF = load_combined_data()
        LAST_DATA_FILE_TIMESTAMP = os.path.getmtime(DATA_FILE)
        print("✅ Modell und kombinierte Daten erfolgreich beim Start geladen.")
    except Exception as e:
        print(f"❌ Fehler beim Laden der Initialdaten: {e}")