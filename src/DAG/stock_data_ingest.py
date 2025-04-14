from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import yfinance as yf
import pandas as pd
import pytz
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

ticker_list = [
    "AAPL", "NVDA", "MSFT", "AMZN", 
    "GOOGL", "TSLA", "META", "NFLX"
]

def is_us_market_open():
    """Prüft ob der US-Markt aktuell geöffnet ist"""
    ny_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(ny_tz)
    
    if current_time.weekday() > 4:
        return False
    
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= current_time <= market_close

def convert_to_berlin_time(df):
    """Konvertiert den DataFrame-Index in Berliner Zeit"""
    berlin_tz = pytz.timezone('Europe/Berlin')
    
    # Stelle sicher, dass der Index ein DatetimeIndex ist und in UTC ist
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    
    # Konvertiere zu Berliner Zeit
    return df.tz_convert(berlin_tz)

def get_current_data(symbol):
    """Aktuelle Tagesdaten mit Stunden-Intervall holen"""
    try:
        ticker = yf.Ticker(symbol)
        current_data = ticker.history(period="1d", interval="60m")
        
        if not current_data.empty:
            current_data = current_data[['Close', 'High', 'Low', 'Open', 'Volume']]
            current_data = convert_to_berlin_time(current_data)
            return current_data
        else:
            print(f"Keine aktuellen Daten für {symbol} verfügbar")
            return None
    except Exception as e:
        print(f"Fehler bei history-Abfrage für {symbol}: {e}")
        return None

def process_stock_data():
    if not is_us_market_open():
        print("US-Markt ist aktuell geschlossen. Keine Ausführung.")
        return
    
    output_folder = "/home/holu/airflow/data/stock_data"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for ticker in ticker_list:
        filename = os.path.join(output_folder, f"{ticker.lower()}_stock_hourly.csv")
        
        try:
            if not os.path.exists(filename):
                print(f"Erstelle neue Datei für {ticker}")
                hist_data = yf.download(ticker, period="730d", interval="60m")
                if hist_data.empty:
                    print(f"Keine historischen Daten für {ticker} gefunden")
                    continue
                    
                hist_data = hist_data[['Close', 'High', 'Low', 'Open', 'Volume']]
                hist_data = convert_to_berlin_time(hist_data)
                # Speichere ohne Zeitzoneninformation
                hist_data.index = hist_data.index.strftime('%Y-%m-%d %H:%M:%S')
                hist_data.to_csv(filename)
                print(f"Historische Daten für {ticker} gespeichert")
                
                current_data = get_current_data(ticker)
                if current_data is not None and not current_data.empty:
                    last_hist_time = pd.to_datetime(hist_data.index[-1])
                    current_data.index = current_data.index.strftime('%Y-%m-%d %H:%M:%S')
                    new_data = current_data[pd.to_datetime(current_data.index) > last_hist_time]
                    if not new_data.empty:
                        new_data.to_csv(filename, mode='a', header=False)
                        print(f"Aktuelle Daten für {ticker} angehängt")
            
            else:
                # Lade existierende Daten
                try:
                    # Versuche zuerst, die Datei ohne Header zu lesen
                    existing_df = pd.read_csv(filename, index_col=0, header=None)
                except:
                    # Falls das fehlschlägt, lies mit Header
                    existing_df = pd.read_csv(filename, index_col=0)
                
                # Entferne mögliche nicht-datetime Zeilen
                existing_df.index = pd.to_datetime(existing_df.index, format='mixed', errors='coerce')
                existing_df = existing_df.dropna(subset=[existing_df.columns[0]])  # Entferne Zeilen mit NaT im Index
                
                if existing_df.empty:
                    print(f"Keine gültigen Daten in der Datei für {ticker} gefunden")
                    continue
                
                last_entry_time = existing_df.index[-1]
                
                current_data = get_current_data(ticker)
                
                if current_data is not None and not current_data.empty:
                    # Konvertiere zu String-Format für konsistente Speicherung
                    current_data.index = current_data.index.strftime('%Y-%m-%d %H:%M:%S')
                    new_data = current_data[pd.to_datetime(current_data.index) > last_entry_time]
                    
                    if not new_data.empty:
                        for idx, row in new_data.iterrows():
                            if (abs(float(row['Close']) - float(existing_df.iloc[-1]['Close'])) > 0.0001 or
                                abs(float(row['Volume']) - float(existing_df.iloc[-1]['Volume'])) > 0):
                                new_data.to_csv(filename, mode='a', header=False)
                                print(f"Neue Daten für {ticker} hinzugefügt - Zeitpunkt: {idx}")
                                print(f"Neue Werte - Close: {row['Close']}, Volume: {row['Volume']}")
                                break
                            else:
                                print(f"Keine signifikanten Änderungen für {ticker}")
                    else:
                        print(f"Keine neueren Daten für {ticker} verfügbar")
                        
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {ticker}: {e}")
            continue


# DAG Definition
with DAG(
    'stock_data_ingest_us_market',
    default_args=default_args,
    description='Lädt Aktiendaten während der US-Handelszeiten',
    schedule_interval='30 14-21 * * 1-5',  # Entspricht 9:30-16:00 ET in deutscher Zeit
    catchup=False
) as dag:

    stock_data_task = PythonOperator(
        task_id='process_stock_data',
        python_callable=process_stock_data,
    )
