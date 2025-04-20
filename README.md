# mlops_sose25_ei

Prüfungsleistung für den MLOPS Kurs Sommersemester 25 beim Prof. David Klotz

## Hetzner Cloud

### Loggin over Bash Terminal

```bash
ssh -p 2222 holu@91.99.13.45
```

### Aktiviere die virtuelle Umgebung

```bash
source airflow_venv/bin/activate
```

### Starte Airflow

```bash
# Webserver starten
airflow webserver --port 8080 
```

### Starte Scheduler

```bash
# In neuem Terminal: Scheduler starten
airflow scheduler
```

### Starten über das Script

```bash
# In neuem Terminal: Scheduler starten
./start_airflow.sh
```

### Airflow Trubbleshooting

#### Finde alle laufenden Prozesse

```bash
# Alle Airflow-Prozesse finden:
ps aux | grep airflow
```

#### Terminiere alle laufenden Prozesse

```bash
# Alle Airflow-Prozesse stoppen:
killall -9 airflow
```

#### Restarte den Webserver

```bash
# Webserver neu starten:
airflow webserver -D
```

#### Restarte den Scheduler

```bash
# Scheduler neu starten:
airflow scheduler -D
```
