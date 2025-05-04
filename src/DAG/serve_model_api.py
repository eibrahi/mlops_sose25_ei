from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import psutil  # NEU oben importieren
import subprocess
import os

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
    schedule_interval='@hourly',
    catchup=False,
    tags=['fastapi', 'model', 'prediction'],
)

FASTAPI_SCRIPT = "/home/holu/airflow/data/fastapi_model_server.py"

def start_server():
    """Starte FastAPI-Server auf Python-Weise."""
    # Suche nach existierenden uvicorn Prozessen und beende sie
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'uvicorn' in proc.info['cmdline']:
                print(f"🛑 Stoppe existierenden Server-Prozess PID {proc.pid}")
                proc.terminate()
        except Exception as e:
            print(f"Fehler beim Stoppen eines Prozesses: {e}")

    # ➔ Starte neuen FastAPI-Server-Prozess
    print("🚀 Starte neuen FastAPI-Server...")
    subprocess.Popen(
        ["uvicorn", "fastapi_model_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=os.path.dirname(FASTAPI_SCRIPT),
    )


start_fastapi_task = PythonOperator(
    task_id='start_fastapi_server',
    python_callable=start_server,
    dag=dag,
)
