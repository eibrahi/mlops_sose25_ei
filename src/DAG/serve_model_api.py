# DAG: serve_model_api – startet regelmäßig den FastAPI-Server für Modellvorhersagen über MLflow

from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
import psutil
import subprocess
import os

# -----------------------------
# 🔧 Konfiguration
# -----------------------------

DATA_DIR = os.getenv("STOCK_DATA_DIR", "/opt/airflow/data/stock_data")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/data/models")
FASTAPI_SCRIPT = os.getenv("FASTAPI_SCRIPT", "/opt/airflow/api_server/fastapi_model_server.py")  # Pfad zum API-Skript

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'serve_model_api',
    default_args=default_args,
    description='Starte FastAPI Server für Modellvorhersagen',
    schedule_interval='15 * * * *',  # Alle Stunde um Minute 15
    catchup=False,
    tags=['fastapi', 'model', 'prediction'],
)

# -----------------------------
# 🚀 Serverstart-Funktion
# -----------------------------

def start_server():
    """
    Beendet laufende FastAPI-Prozesse (uvicorn), falls vorhanden,
    und startet den Server neu im gewünschten Verzeichnis.
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'uvicorn' in proc.info.get('cmdline', []):
                print(f"🛑 Stoppe existierenden Server-Prozess PID {proc.pid}")
                proc.terminate()
        except Exception as e:
            print(f"⚠️ Fehler beim Stoppen: {e}")

    print("🚀 Starte neuen FastAPI-Server...")

    # Starte FastAPI-Server mit Hot-Reload für schnelles Debugging
    subprocess.Popen(
        ["uvicorn", "fastapi_model_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=os.path.dirname(FASTAPI_SCRIPT),
    )

# -----------------------------
# 🧱 Task-Definition
# -----------------------------

start_fastapi_task = PythonOperator(
    task_id='start_fastapi_server',
    python_callable=start_server,
    dag=dag,
)
