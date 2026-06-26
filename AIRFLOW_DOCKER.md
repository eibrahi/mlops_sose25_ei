# Airflow mit Docker Desktop starten

## 1. TwelveData API Key setzen

Lege im Projektordner eine lokale `.env` Datei an:

```env
TWELVEDATA_API_KEY=dein_api_key_hier
```

Die Datei wird durch `.gitignore` ignoriert.

## 2. Container bauen und starten

```bash
docker compose build
docker compose up -d
```

## 3. Oberflaechen

- Airflow: http://localhost:8080
- MLflow: http://localhost:5001
- FastAPI, nachdem der `serve_model_api` DAG erfolgreich lief: http://localhost:8000

Airflow Login:

```text
Username: admin
Password: admin
```

## 4. DAGs verwenden

In Airflow sind diese DAGs verfuegbar:

- `stock_data_collector_twelvedata`
- `stock_model_trainer_mlflow`
- `serve_model_api`

Die DAGs sind initial pausiert. Aktiviere sie in der Airflow UI oder starte sie manuell.
Die sinnvolle Reihenfolge ist:

1. `stock_data_collector_twelvedata`
2. `stock_model_trainer_mlflow`
3. `serve_model_api`

## 5. Status und Logs

```bash
docker compose ps
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-webserver
```

## 6. Stoppen

```bash
docker compose down
```

Die lokalen Laufzeitdaten bleiben in `airflow-data/`, `airflow-logs/` und `mlflow-data/`.
