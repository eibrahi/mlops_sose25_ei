# -*- coding: utf-8 T-O-R-V-5.2.4 -*-

"""
Airflow DAG zur stündlichen Erfassung von Aktienkursen über die TwelveData API.
Daten werden für mehrere Aktien gesammelt, im CSV-Format gespeichert und ggf. fortgeschrieben.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
import pytz
import os
import logging
import requests
import random

# -----------------------------
# Konfiguration
# -----------------------------

STOCKS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NFLX"]  # Zu trackende Aktien
DATA_DIR = "/home/holu/airflow/data/stock_data"  # Speicherpfad für CSV-Dateien
BERLIN_TZ = pytz.timezone('Europe/Berlin')
HISTORY_DAYS = 30  # Maximale Anzahl an Tagen, die im Free-Plan abrufbar sind

# Rotierende User-Agents zur Minimierung von Rate Limits oder Blockaden
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, wie Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, wie Gecko) Chrome/133.0.0.0 Safari/537.36"
]

# API Key aus Umgebungsvariable laden
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
if not TWELVEDATA_API_KEY:
    raise RuntimeError("TWELVEDATA_API_KEY nicht gesetzt!")

# Sicherstellen, dass Verzeichnis existiert
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Default Args & DAG Definition
# -----------------------------

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 20, tzinfo=BERLIN_TZ),
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}

dag = DAG(
    'stock_data_collector_twelvedata',
    default_args=default_args,
    description='Stündliche Kursdaten über TwelveData API',
    schedule_interval='0 * * * *',  # Jede volle Stunde
    catchup=False,
    max_active_runs=1,
    tags=['stock', 'twelvedata', 'intraday'],
)

# -----------------------------
# Hilfsfunktion: Letzten Zeitstempel aus CSV lesen
# -----------------------------

def get_last_timestamp(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
    try:
        df = pd.read_csv(file_path)
        ts = pd.to_datetime(df['Datetime'], utc=True)
        return ts.max()
    except Exception as e:
        logging.error(f"Fehler beim Lesen von {file_path}: {e}")
        return None

# -----------------------------
# Hauptfunktion: Datenabruf & Speicherung
# -----------------------------

def get_stock_data_twelvedata(stock_symbol, **kwargs):
    logging.info(f"[{stock_symbol}] Abruf über TwelveData API")

    session = requests.Session()
    session.headers.update({'User-Agent': random.choice(USER_AGENTS)})

    file_path = os.path.join(DATA_DIR, f"{stock_symbol}.csv")
    last_ts = get_last_timestamp(file_path)

    now_utc = datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
    start_utc = (now_utc - timedelta(days=HISTORY_DAYS)) if not last_ts else (last_ts - timedelta(hours=1))

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": stock_symbol,
        "interval": "1h",
        "outputsize": 5000,
        "start_date": start_utc.strftime('%Y-%m-%d %H:%M:%S'),
        "end_date": now_utc.strftime('%Y-%m-%d %H:%M:%S'),
        "apikey": TWELVEDATA_API_KEY,
        "timezone": "UTC"
    }

    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "values" not in data:
            logging.warning(f"[{stock_symbol}] Keine Daten empfangen: {data.get('message', 'unbekannt')}")
            return

        df = pd.DataFrame(data["values"])
        df['Datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(BERLIN_TZ)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.sort_values('Datetime', inplace=True)

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            df.to_csv(file_path, index=False)
            logging.info(f"[{stock_symbol}] Datei neu geschrieben mit {len(df)} Einträgen.")
        else:
            existing = pd.read_csv(file_path)
            existing['Datetime'] = pd.to_datetime(existing['Datetime'], utc=True).dt.tz_convert(BERLIN_TZ)
            last_existing = existing['Datetime'].max()
            new_rows = df[df['Datetime'] > last_existing]

            if not new_rows.empty:
                new_rows.to_csv(file_path, mode='a', header=False, index=False)
                logging.info(f"[{stock_symbol}] {len(new_rows)} neue Zeilen angehängt.")
            else:
                logging.info(f"[{stock_symbol}] Keine neuen Zeilen.")
    except Exception as e:
        logging.error(f"[{stock_symbol}] API-Fehler: {e}")

# -----------------------------
# Tasks generieren (seriell)
# -----------------------------

previous = None
for stock in STOCKS:
    task = PythonOperator(
        task_id=f"get_{stock}_data",
        python_callable=get_stock_data_twelvedata,
        op_kwargs={'stock_symbol': stock},
        dag=dag
    )
    if previous:
        previous >> task  # Seriell: Jeder Task startet nach dem vorherigen
    previous = task

# -----------------------------
# Lokaler Test-Trigger
# -----------------------------

if __name__ == '__main__':
    dag.test()
    