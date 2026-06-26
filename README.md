# MLOps Stock Return Prediction

Dieses Repository enthaelt eine lokale MLOps-Pipeline zur stundenweisen Erfassung von Aktienkursen, zum Training eines gemeinsamen Machine-Learning-Modells und zur Bereitstellung von Vorhersagen ueber eine FastAPI-Schnittstelle. Das Projekt wurde fuer den MLOps-Kurs im Sommersemester 2025 erstellt.

Die Pipeline laeuft lokal mit Docker Compose und verbindet:

- Apache Airflow fuer Orchestrierung und Scheduling
- TwelveData als Datenquelle fuer intraday OHLCV-Aktiendaten
- PostgreSQL als Airflow-Metadatenbank
- MLflow fuer Experiment Tracking und Model Registry
- LightGBM als Regressionsmodell
- FastAPI und Uvicorn fuer den Prediction-Service

Die Installations- und Startanleitung steht in [AIRFLOW_DOCKER.md](AIRFLOW_DOCKER.md). Diese README beschreibt, was im Repository enthalten ist, wie die Pipeline fachlich funktioniert und wie die Komponenten zusammenspielen.

## Ziel des Projekts

Das System prognostiziert fuer mehrere Aktien den naechsten erwarteten Return. Aus diesem Return wird in der API ein erwarteter naechster Preis berechnet:

```text
predicted_price = last_close * (1 + predicted_return)
```

Das Modell wird nicht pro Aktie einzeln trainiert, sondern als gemeinsames Modell ueber alle Aktien. Die jeweilige Aktie wird als One-Hot-Feature mitgegeben. Dadurch kann das Modell gemeinsame Muster ueber mehrere Titel hinweg lernen und trotzdem aktienspezifische Unterschiede abbilden.

## Aktuell verarbeitete Symbole

Die Datenpipeline erfasst aktuell diese Aktien:

```text
AAPL, NVDA, MSFT, AMZN, GOOGL, TSLA, META, NFLX
```

Zusaetzlich werden optionale Marktreihen geladen:

```text
SPY, QQQ, VIX
```

`SPY` und `QQQ` dienen als Marktindex-Proxies. `VIX` dient als Stimmungs- bzw. Volatilitaetsindikator. Diese Marktreihen werden nicht als normale Aktien vorhergesagt. Wenn eine optionale Marktreihe von TwelveData nicht geliefert wird, bricht die Pipeline nicht ab; die betroffenen Marktfeatures werden dann mit neutralen Werten gefuellt.

## Repository-Struktur

```text
.
|-- README.md
|-- AIRFLOW_DOCKER.md
|-- docker-compose.yml
|-- Architekturschaubild.drawio
|-- project-canvas/
|   `-- ml-canvas_ei_1.0.pdf
|-- docker/
|   `-- airflow/
|       |-- Dockerfile
|       `-- requirements-airflow.txt
`-- src/
    |-- dag/
    |   |-- stock_data_ingest.py
    |   |-- stock_model_trainer.py
    |   `-- serve_model_api.py
    |-- common/
    |   `-- feature_engineering.py
    `-- api_server/
        `-- fastapi_model_server.py
```

### Wichtige Dateien

`src/dag/stock_data_ingest.py`  
Airflow-DAG fuer den stundenweisen Abruf von OHLCV-Daten ueber die TwelveData API. Pro Symbol wird eine CSV-Datei geschrieben oder erweitert.

`src/dag/stock_model_trainer.py`  
Airflow-DAG fuer Datenkombination, Feature Engineering, Modelltraining, MLflow-Logging und Registrierung des Production-Modells.

`src/common/feature_engineering.py`  
Zentrale Feature-Engineering-Logik. Diese Datei wird sowohl vom Training als auch vom FastAPI-Server verwendet, damit Training und Prediction exakt dieselben Features erzeugen.

`src/api_server/fastapi_model_server.py`  
FastAPI-App, die das aktuelle Production-Modell aus MLflow laedt, aktuelle Daten einliest, Features erzeugt und Vorhersagen ueber HTTP bereitstellt.

`src/dag/serve_model_api.py`  
Airflow-DAG, der den FastAPI-Server ueber Uvicorn startet bzw. neu startet.

`docker-compose.yml`  
Definiert Airflow-Webserver, Airflow-Scheduler, PostgreSQL und MLflow. Ausserdem werden die lokalen Quellcode-Ordner in die Container gemountet.

`docker/airflow/requirements-airflow.txt`  
Python-Abhaengigkeiten fuer Airflow, MLflow, LightGBM, FastAPI und Uvicorn.

## Architektur

Der Ablauf ist:

```text
TwelveData API
   |
   v
stock_data_collector_twelvedata
   |
   v
CSV-Dateien in airflow-data/stock_data
   |
   v
stock_model_trainer_mlflow
   |
   v
Feature Engineering + LightGBM Training
   |
   v
MLflow Experiment + Model Registry
   |
   v
serve_model_api
   |
   v
FastAPI Prediction Endpoint
```

Die drei Airflow-DAGs bilden die Hauptpipeline:

1. `stock_data_collector_twelvedata`
2. `stock_model_trainer_mlflow`
3. `serve_model_api`

## Datenpipeline

### 1. Datenerfassung

Der DAG `stock_data_collector_twelvedata` laeuft stuendlich zur vollen Stunde. Er ruft fuer jedes konfigurierte Symbol Daten von TwelveData ab:

- Zeitintervall: `1h`
- Historie beim ersten Abruf: bis zu 30 Tage
- Ausgabeformat: CSV
- Zeitspalte: `Datetime`
- Kurs-/Volumenspalten: `Open`, `High`, `Low`, `Close`, `Volume`

Die Dateien werden unter folgendem Pfad abgelegt:

```text
airflow-data/stock_data/
```

Beispiele:

```text
airflow-data/stock_data/AAPL.csv
airflow-data/stock_data/NVDA.csv
airflow-data/stock_data/SPY.csv
```

Wenn eine Datei bereits existiert, werden nur neue Zeilen angehaengt.

### 2. Datenkombination

Der Trainer-DAG kombiniert alle einzelnen Symbol-CSV-Dateien zu:

```text
airflow-data/stock_data/combined_stock_data.csv
```

Dabei bekommt jede Zeile eine zusaetzliche Spalte:

```text
stock
```

So weiss das Modell spaeter, zu welcher Aktie die jeweilige Zeile gehoert.

### 3. Feature Engineering

Das Feature Engineering liegt zentral in:

```text
src/common/feature_engineering.py
```

## Feature Engineering im Detail

Die Pipeline erzeugt technische Indikatoren aus den OHLCV-Daten.

### Basisdaten

Direkt genutzt werden:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

### Lag-Features

Fuer den Schlusskurs werden Rueckblicke erzeugt:

```text
close_lag_1, close_lag_2, close_lag_3, close_lag_5,
close_lag_10, close_lag_20, close_lag_50
```

### Returns

Prozentuale Kursveraenderungen:

```text
return_1, return_5, return_10, return_20
```

### Rolling Means

Gleitende Mittelwerte des Schlusskurses:

```text
rolling_mean_5, rolling_mean_10, rolling_mean_20,
rolling_mean_50, rolling_mean_100, rolling_mean_200
```

### Rolling Standard Deviation

Gleitende Standardabweichungen als Volatilitaetsmass:

```text
rolling_std_5, rolling_std_10, rolling_std_20,
rolling_std_50, rolling_std_100, rolling_std_200
```

### EMA

Exponential Moving Averages:

```text
ema_12, ema_26, ema_50
```

### RSI

Relative Strength Index:

```text
rsi_7, rsi_14, rsi_21
```

### MACD

MACD-Indikator inklusive Signal und Histogramm:

```text
macd, macd_signal, macd_histogram
```

### Bollinger Bands

Bollinger-Band-Features:

```text
bollinger_middle, bollinger_upper, bollinger_lower, bollinger_bandwidth
```

### ATR

Average True Range:

```text
atr_14
```

### Distanz zu Hochs und Tiefs

Distanz zum 20- und 50-Perioden-Hoch bzw. -Tief:

```text
distance_high_20, distance_low_20,
distance_high_50, distance_low_50
```

### Z-Score

Standardisierter Abstand zum 20er Rolling Mean:

```text
zscore_20
```

### Momentum

Absolute Kursveraenderung ueber mehrere Perioden:

```text
momentum_5, momentum_10, momentum_20
```

### Zyklische Kalenderfeatures

Wochentag und Monat werden nicht als einfache Zahlen kodiert, sondern zyklisch ueber Sinus und Kosinus:

```text
dayofweek_sin, dayofweek_cos,
month_sin, month_cos
```

Dadurch versteht das Modell besser, dass z. B. Sonntag/Montag oder Dezember/Januar zyklisch nahe beieinander liegen.

### Markt- und Sentimentfeatures

Wenn Marktdaten vorhanden sind, werden Features aus Index-Proxies und VIX erzeugt:

```text
market_return_1, market_return_5,
vix_return_1, vix_return_5,
vix_close
```

Wenn keine passenden Marktdaten vorhanden sind, werden diese Features mit `0.0` gefuellt. So bleibt die Feature-Struktur stabil.

### Target

Das Modell lernt den naechsten Return:

```text
target_return = Close.pct_change().shift(-1)
```

Wichtig: Dadurch wird nicht der Return derselben Zeile vorhergesagt. Das vermeidet Leakage, weil aktuelle OHLCV-Daten als Features verwendet werden duerfen, aber das Ziel die naechste Periode ist.

## Modelltraining

Der DAG `stock_model_trainer_mlflow` laeuft stuendlich um Minute 10.

Er macht:

1. Einzelne CSV-Dateien kombinieren
2. Technische Features erzeugen
3. Optionale Markt-/VIX-Reihen aus dem eigentlichen Trainingsziel entfernen
4. Aktien per One-Hot-Encoding kodieren
5. Feature-Liste speichern
6. LightGBM-Regressor trainieren
7. Metriken berechnen
8. Modell und Artefakte in MLflow loggen
9. Modell in der MLflow Model Registry als `Production` registrieren

Das Modell:

```text
LGBMRegressor(n_estimators=100, learning_rate=0.05)
```

Metriken:

- `r2`
- `rmse`

Das Modell wird unter diesem Namen registriert:

```text
stock_price_model
```

Die Feature-Liste wird als MLflow-Artefakt gespeichert:

```text
info/info.json
```

Diese Feature-Liste ist wichtig, weil die API spaeter exakt dieselben Spalten in exakt derselben Reihenfolge erzeugen muss.

## FastAPI Prediction Service

Der FastAPI-Server steht in:

```text
src/api_server/fastapi_model_server.py
```

Beim Start macht der Server:

1. MLflow Tracking URI setzen
2. Aktuelles `Production`-Modell aus MLflow laden
3. `info/info.json` mit der Feature-Liste laden
4. `combined_stock_data.csv` laden
5. API-Endpunkte bereitstellen

### Healthcheck

```text
GET http://localhost:8000/health
```

Beispielantwort:

```json
{
  "status": "healthy",
  "checks": [
    "Modell geladen",
    "Feature-Spalten gefunden: 67",
    "Combined DataFrame geladen mit 1526 Zeilen"
  ]
}
```

### Predictions

```text
GET http://localhost:8000/api/v1/predictions
```

Die API gibt pro Aktie zurueck:

- letzter bekannter Close-Preis
- Zeitstempel der letzten Datenzeile
- vorhergesagter Return
- daraus berechneter Preis

Beispielstruktur:

```json
{
  "status": "success",
  "predictions": {
    "AAPL": {
      "last_close": "$275.05",
      "last_timestamp": "2026-06-25 19:30:00+00:00",
      "predicted_return": "0.14%",
      "predicted_price": "$275.44"
    }
  }
}
```

## API-Deployment ueber Airflow

Der FastAPI-Server wird ueber den Airflow-DAG `serve_model_api` gestartet.

Die Datei:

```text
src/dag/serve_model_api.py
```

macht:

1. Laufende `uvicorn`-Prozesse suchen
2. Existierenden FastAPI-Server beenden
3. Neuen Server starten:

```text
uvicorn fastapi_model_server:app --host 0.0.0.0 --port 8000 --reload
```

Der DAG laeuft stuendlich um Minute 15. Dadurch kann nach einem neuen Training der API-Server neu gestartet werden und das aktuelle Production-Modell aus MLflow laden.

## Docker-Deployment

Das lokale Deployment erfolgt ueber Docker Compose. Die genaue Schritt-fuer-Schritt-Installation steht in:

```text
AIRFLOW_DOCKER.md
```

Kurzueberblick:

```bash
docker compose build
docker compose up -d
```

Die Compose-Datei startet:

- `postgres`
- `mlflow`
- `airflow-webserver`
- `airflow-scheduler`
- einmalig `airflow-init`

Die lokalen Ordner werden in die Container gemountet:

```yaml
./src/dag:/opt/airflow/dags
./src/api_server:/opt/airflow/api_server
./src/common:/opt/airflow/common
./airflow-data:/opt/airflow/data
./airflow-logs:/opt/airflow/logs
./mlflow-data:/mlflow
```

Wichtige URLs:

```text
Airflow: http://localhost:8080
MLflow:  http://localhost:5001
FastAPI: http://localhost:8000
```

Airflow Login:

```text
Username: admin
Password: admin
```

## Voraussetzungen

Du benoetigst:

- Docker Desktop
- Docker Compose
- Git
- TwelveData API Key

Der API-Key wird in einer lokalen `.env` Datei hinterlegt:

```env
TWELVEDATA_API_KEY=dein_api_key_hier
```

Die Datei `.env` wird nicht versioniert.

## Nutzung

Nach dem Start der Container sollten die DAGs in dieser Reihenfolge ausgefuehrt werden:

1. `stock_data_collector_twelvedata`
2. `stock_model_trainer_mlflow`
3. `serve_model_api`

Danach pruefen:

```text
http://localhost:8000/health
```

Wenn der Status `healthy` ist, koennen Vorhersagen abgerufen werden:

```text
http://localhost:8000/api/v1/predictions
```

## Laufzeitdaten

Diese Ordner entstehen lokal beim Betrieb:

```text
airflow-data/
airflow-logs/
mlflow-data/
```

Sie enthalten:

- geladene CSV-Daten
- vorbereitete Trainingsdaten
- Airflow-Logs
- MLflow-Experimente
- MLflow-Modellartefakte

Diese Laufzeitdaten sind nicht Teil des Quellcodes.

## Weitere Artefakte

`Architekturschaubild.drawio`  
Architekturdiagramm der Pipeline.

`project-canvas/ml-canvas_ei_1.0.pdf`  
Projekt-/ML-Canvas mit konzeptioneller Beschreibung.

## Wichtige Hinweise

Dieses Projekt ist eine lokale MLOps-Demo und keine Finanzberatung. Die Vorhersagen sind technische Modelloutputs auf Basis historischer Intraday-Daten. Sie sollten nicht als Grundlage fuer reale Handelsentscheidungen verwendet werden.

Bei Aenderungen an `docker-compose.yml`, z. B. neuen Mounts oder Umgebungsvariablen, sollten die Airflow-Container neu erstellt werden:

```bash
docker compose up -d --force-recreate airflow-scheduler airflow-webserver
```

Bei reinen Codeaenderungen in `src/dag`, `src/common` oder `src/api_server` reicht in vielen Faellen ein Neustart der betroffenen DAGs bzw. des API-DAGs.
