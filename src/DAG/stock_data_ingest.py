
'''
Das ist die für MLFlow und mit TOR
'''

# -*- coding: utf-8 T-O-R-V-5.0.9-*-

"""

Airflow DAG zur seriellen Erfassung von Aktienkursdaten während US-Handelszeiten

über das Tor-Netzwerk mit IP-Rotation und dynamischem User-Agent.

"""

from datetime import datetime, timedelta

from airflow import DAG

from airflow.operators.python import PythonOperator

import yfinance as yf

import pandas as pd

import pandas_market_calendars as mcal

import pytz

import os

import logging

import requests

from stem import Signal

from stem.control import Controller

import random

import time


# Konfiguration

STOCKS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NFLX"]

DATA_DIR = "/home/holu/airflow/data/stock_data"

BERLIN_TZ = pytz.timezone('Europe/Berlin')

US_EASTERN_TZ = pytz.timezone('US/Eastern')

HISTORY_DAYS = 729

USER_AGENTS = [

    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",

    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",

    "Mozilla/5.0 (X11; Linux i686; rv:135.0) Gecko/20100101 Firefox/135.0",

    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:135.0) Gecko/20100101 Firefox/135.0"

]


os.makedirs(DATA_DIR, exist_ok=True)


default_args = {

    'owner': 'airflow',

    'depends_on_past': False,

    'email_on_failure': False,

    'email_on_retry': False,

    'retries': 3,

    'retry_delay': timedelta(minutes=5),

    'execution_timeout': timedelta(minutes=10),

    'start_date': datetime(2025, 4, 20, tzinfo=BERLIN_TZ),

}


dag = DAG(

    'stock_data_collector_serial',

    default_args=default_args,

    description='Serielle Sammlung von Aktienkursdaten über Tor mit IP-Rotation und dynamischem User-Agent',

    schedule_interval='0 * * * *',

    catchup=False,

    tags=['stock', 'finance', 'tor'],

)


def is_market_open():

    now = datetime.now(US_EASTERN_TZ)

    if now.weekday() >= 5:

        logging.info("Wochenende – US-Markt geschlossen.")

        return False

    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

    market_close = now.replace(hour=16, minute=0, microsecond=0)

    if market_open <= now <= market_close:

        nyse = mcal.get_calendar('NYSE')

        schedule = nyse.schedule(start_date=now.date(), end_date=now.date())

        return not schedule.empty

    return False


def get_last_timestamp(file_path):

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:

        return None

    try:

        df = pd.read_csv(file_path)

        if df.empty:

            return None

        return pd.to_datetime(df.iloc[-1]['Datetime'], utc=True)

    except Exception as e:

        logging.error(f"Fehler beim Lesen von {file_path}: {e}")

        return None


def rotate_tor_ip():

    try:

        with Controller.from_port(port=9051) as controller:

            controller.authenticate()

            controller.signal(Signal.NEWNYM)

            logging.info("Neue Tor-IP wurde angefordert.")

    except Exception as e:

        logging.warning(f"Tor-Rotation fehlgeschlagen: {e}")


def get_stock_data(stock_symbol, **kwargs):

    logging.info(f"Verarbeite Aktie: {stock_symbol}")


    proxies = {

        'http': 'socks5h://127.0.0.1:9050',

        'https': 'socks5h://127.0.0.1:9050',

    }

    headers = {

        'User-Agent': random.choice(USER_AGENTS)

    }


    session = requests.Session()

    session.proxies.update(proxies)

    session.headers.update(headers)


    rotate_tor_ip()

    time.sleep(5)


    try:

        ip = session.get('https://icanhazip.com', timeout=10).text.strip()

        logging.info(f"IP-Wechsel bestätigt: {ip}")

    except Exception as e:

        logging.warning(f"IP konnte nicht überprüft werden: {e}")

        return


    if not is_market_open():

        logging.info("US-Markt geschlossen. Abbruch.")

        return


    file_path = os.path.join(DATA_DIR, f"{stock_symbol}.csv")

    last_timestamp = get_last_timestamp(file_path)

    now = BERLIN_TZ.localize(datetime.now())


    if not last_timestamp or pd.isna(last_timestamp):

        start_date = now - timedelta(days=HISTORY_DAYS)

    else:

        start_date = (last_timestamp - timedelta(hours=1)).astimezone(BERLIN_TZ)


    end_date = now


    try:

        stock_data = yf.download(

            stock_symbol,

            start=start_date,

            end=end_date,

            interval="1h",

            progress=False,

            proxy="socks5h://127.0.0.1:9050",

        )

    except Exception as e:

        logging.error(f"Fehler beim Laden von {stock_symbol}: {e}")

        return


    if stock_data.empty:

        logging.info(f"{stock_symbol}: Keine neuen Daten.")

        return


    stock_data = stock_data.reset_index()

    if 'Datetime' not in stock_data.columns:

        stock_data.rename(columns={'Date': 'Datetime'}, inplace=True)


    try:

        stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'], utc=True)

        stock_data['Datetime'] = stock_data['Datetime'].dt.tz_convert(BERLIN_TZ)

    except Exception as e:

        logging.error(f"Fehler bei Zeitumwandlung: {e}")

        return


    columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']

    stock_data = stock_data[columns]


    try:

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:

            stock_data.to_csv(file_path, index=False)

            logging.info(f"{stock_symbol}: Neue Datei mit {len(stock_data)} Einträgen.")

        else:

            existing_data = pd.read_csv(file_path)

            existing_data['Datetime'] = pd.to_datetime(existing_data['Datetime'], utc=True)

            last_datetime = existing_data['Datetime'].max()

            new_data = stock_data[stock_data['Datetime'] > last_datetime]

            if not new_data.empty:

                new_data.to_csv(file_path, mode='a', header=False, index=False)

                logging.info(f"{stock_symbol}: {len(new_data)} neue Zeilen gespeichert.")

            else:

                logging.info(f"{stock_symbol}: Keine neuen Zeilen.")

    except Exception as e:

        logging.error(f"Fehler beim Speichern von {stock_symbol}: {e}")


# Serielle Task-Verknüpfung

previous_task = None

for stock in STOCKS:

    task = PythonOperator(

        task_id=f"get_{stock}_data",

        python_callable=get_stock_data,

        op_kwargs={'stock_symbol': stock},

        dag=dag,

    )

    if previous_task:

        previous_task >> task

    previous_task = task


if __name__ == "__main__":

    dag.test()

'''
Das ist die ursprüngliche die funktioniert ohne ML Flow

'''

# # -*- coding: utf-8 -*- v.3.
# """
# Airflow DAG zur stündlichen Erfassung von Aktienkursdaten während US-Handelszeiten
# """

# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# import yfinance as yf
# import pandas as pd
# import pandas_market_calendars as mcal
# import pytz
# import os
# import logging

# # Konfigurationsparameter
# STOCKS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NFLX"]
# DATA_DIR = "/home/holu/airflow/data/stock_data"
# BERLIN_TZ = pytz.timezone('Europe/Berlin')
# US_EASTERN_TZ = pytz.timezone('US/Eastern')
# HISTORY_DAYS = 730  # Historische Daten der letzten 730 Tage (ca. 2 Jahre)

# # Sicherstellen, dass das Verzeichnis existiert
# os.makedirs(DATA_DIR, exist_ok=True)

# # Standard-Argumente für den DAG
# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 3,
#     'retry_delay': timedelta(minutes=5),
#     'execution_timeout': timedelta(minutes=10),
#     'start_date': datetime(2025, 4, 20, tzinfo=BERLIN_TZ),
# }

# # DAG-Definition
# dag = DAG(
#     'stock_data_collector',
#     default_args=default_args,
#     description='Sammelt stündlich Aktienkursdaten während US-Handelszeiten',
#     schedule_interval='0 * * * *',  # Stündlich
#     catchup=False,
#     tags=['stock', 'finance', 'data'],
# )

# def is_market_open():
#     """
#     Überprüft, ob der US-Aktienmarkt aktuell geöffnet ist.
#     """
#     now = datetime.now(US_EASTERN_TZ)
    
#     # Prüfe, ob heute ein Wochentag ist (Montag=0, Sonntag=6)
#     if now.weekday() >= 5:  # Samstag oder Sonntag
#         logging.info("Heute ist Wochenende. Der US-Aktienmarkt ist geschlossen.")
#         return False
    
#     # Überprüfe, ob die aktuelle Zeit während der Handelszeiten liegt (9:30-16:00 ET)
#     market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
#     market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
#     if market_open <= now <= market_close:
#         # Verwende pandas_market_calendars, um zu prüfen, ob heute ein Handelstag ist
#         nyse = mcal.get_calendar('NYSE')
#         schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
        
#         if not schedule.empty:
#             logging.info("Der US-Aktienmarkt ist geöffnet.")
#             return True
    
#     logging.info("Der US-Aktienmarkt ist derzeit geschlossen.")
#     return False

# def get_last_timestamp(file_path):
#     """
#     Liest den letzten Zeitstempel aus einer CSV-Datei.
#     Gibt None zurück, wenn die Datei nicht existiert oder leer ist.
#     """
#     if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
#         return None
    
#     try:
#         df = pd.read_csv(file_path)
#         if df.empty:
#             return None
        
#         # Konvertiere Zeitstempel zu Datetime-Objekt mit explizitem utc=True Parameter
#         last_timestamp = pd.to_datetime(df.iloc[-1]['Datetime'], utc=True)
#         return last_timestamp
#     except Exception as e:
#         logging.error(f"Fehler beim Lesen des letzten Zeitstempels aus {file_path}: {e}")
#         return None

# def get_stock_data(stock_symbol, **kwargs):
#     """
#     Lädt Aktienkursdaten für ein bestimmtes Symbol und speichert sie im CSV-Format.
#     """
#     logging.info(f"Verarbeite Aktie: {stock_symbol}")
    
#     # Überprüfe zuerst, ob der Markt geöffnet ist
#     if not is_market_open():
#         logging.info(f"Überspringe Datenerfassung für {stock_symbol}, da der Markt geschlossen ist.")
#         return
    
#     file_path = os.path.join(DATA_DIR, f"{stock_symbol}.csv")
#     last_timestamp = get_last_timestamp(file_path)
    
#     try:
#         if last_timestamp is None:
#             # Erster Durchlauf oder leere Datei: Lade historische Daten
#             logging.info(f"Lade historische Daten der letzten {HISTORY_DAYS} Tage für {stock_symbol}")
#             end_date = datetime.now(BERLIN_TZ)
#             start_date = end_date - timedelta(days=HISTORY_DAYS)
            
#             # Verwende yfinance, um Daten zu laden (1h Intervall)
#             stock_data = yf.download(
#                 stock_symbol,
#                 start=start_date,
#                 end=end_date,
#                 interval="1h",
#                 progress=False
#             )
#         else:
#             # Lade nur neue Daten seit dem letzten Zeitstempel
#             # Füge einen kleinen Puffer hinzu, um sicherzustellen, dass wir keine Daten verpassen
#             start_date = last_timestamp - timedelta(hours=1)
#             end_date = datetime.now(BERLIN_TZ)
            
#             logging.info(f"Lade neue Daten für {stock_symbol} seit {start_date}")
#             stock_data = yf.download(
#                 stock_symbol,
#                 start=start_date,
#                 end=end_date,
#                 interval="1h",
#                 progress=False
#             )
#     except Exception as e:
#         logging.error(f"Fehler beim Abrufen von Daten für {stock_symbol}: {e}")
#         raise
    
#     if stock_data.empty:
#         logging.info(f"Keine neuen Daten für {stock_symbol} verfügbar.")
#         return
    
#     # Bereite Daten auf und konvertiere zu Berliner Zeit
#     stock_data = stock_data.reset_index()
#     stock_data = stock_data.rename(columns={'Datetime': 'Datetime'} 
#                                    if 'Datetime' in stock_data.columns else {'Date': 'Datetime'})
    
#     # Stelle sicher, dass die Zeitzone korrekt ist (konvertiere zu Berliner Zeit)
#     try:
#         # Konvertiere mit explizitem utc=True Parameter, um die Future-Warnung zu vermeiden
#         if not pd.api.types.is_datetime64_dtype(stock_data['Datetime']):
#             stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'], utc=True)
        
#         if stock_data['Datetime'].dt.tz is None:
#             # Wenn keine Zeitzone gesetzt ist, nehmen wir an, dass es UTC ist
#             stock_data['Datetime'] = stock_data['Datetime'].dt.tz_localize('UTC')
        
#         # Konvertiere zu Berliner Zeit
#         stock_data['Datetime'] = stock_data['Datetime'].dt.tz_convert(BERLIN_TZ)
#     except Exception as e:
#         logging.error(f"Fehler bei der Zeitstempelkonvertierung für {stock_symbol}: {e}")
#         raise
    
#     # Behalte nur die benötigten Spalten
#     columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
#     stock_data = stock_data[columns]
    
#     # Speichere in CSV
#     try:
#         if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
#             # Neue Datei erstellen
#             stock_data.to_csv(file_path, index=False)
#             logging.info(f"Neue CSV-Datei für {stock_symbol} erstellt mit {len(stock_data)} Einträgen.")
#         else:
#             # Vorhandene Daten lesen
#             existing_data = pd.read_csv(file_path)
            
#             # KORREKTUR: Zeitstempel zu Datetime konvertieren mit explizitem utc=True
#             existing_data['Datetime'] = pd.to_datetime(existing_data['Datetime'], utc=True)
            
#             # Sicherstellen, dass alle Datetime-Werte tatsächlich Timestamps sind
#             if not pd.api.types.is_datetime64_dtype(existing_data['Datetime']):
#                 logging.warning(f"Datetime-Spalte in bestehender Datei für {stock_symbol} hat gemischte Typen. Konvertiere...")
#                 existing_data['Datetime'] = pd.to_datetime(existing_data['Datetime'], utc=True)
            
#             if existing_data['Datetime'].isna().any():
#                 logging.warning(f"NaN-Werte in Datetime-Spalte gefunden für {stock_symbol}. Entferne diese...")
#                 existing_data = existing_data.dropna(subset=['Datetime'])
            
#             # Überprüfe, ob die Datetime-Spalte leer ist
#             if existing_data.empty:
#                 logging.warning(f"Nach Bereinigung ist die Datei für {stock_symbol} leer. Erstelle neue...")
#                 stock_data.to_csv(file_path, index=False)
#                 logging.info(f"Neue CSV-Datei für {stock_symbol} erstellt mit {len(stock_data)} Einträgen.")
#                 return
                
#             # Vermeide Duplikate durch Vergleich mit dem letzten Zeitstempel
#             last_datetime = existing_data['Datetime'].max()
#             new_data = stock_data[stock_data['Datetime'] > last_datetime]
            
#             if not new_data.empty:
#                 # Füge neue Daten an
#                 new_data.to_csv(file_path, mode='a', header=False, index=False)
#                 logging.info(f"{len(new_data)} neue Einträge zu {stock_symbol} hinzugefügt.")
#             else:
#                 logging.info(f"Keine neuen Daten für {stock_symbol} hinzuzufügen.")
#     except Exception as e:
#         logging.error(f"Fehler beim Speichern der Daten für {stock_symbol}: {e}")
#         raise

# # Erstelle PythonOperator für jede Aktie
# for stock in STOCKS:
#     task = PythonOperator(
#         task_id=f'get_{stock}_data',
#         python_callable=get_stock_data,
#         op_kwargs={'stock_symbol': stock},
#         dag=dag,
#     )

# if __name__ == "__main__":
#     # Für lokale Tests
#     dag.test()