import os
import yfinance as yf   # type: ignore

# Liste der Ticker-Symbole
ticker_list = [
    "AAPL", 
    "NVDA", 
    "MSFT", 
    "AMZN", 
    "GOOGL", 
    "TSLA", 
    "META", 
    "NFLX",
    ]

# Zeitraum und Intervall definieren
period = "730d"  # Letzte 2 Jahre
interval = "60m"  # Stündliche Daten

# Zielordner für CSV-Dateien
output_folder = "stock_data"
os.makedirs(output_folder, exist_ok=True)  # Ordner erstellen, falls nicht vorhanden

# Daten abrufen und speichern
for ticker in ticker_list:
    df = yf.download(ticker, period=period, interval=interval)
    filename = os.path.join(output_folder, f"{ticker.lower()}_stock_hourly.csv")
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    df.to_csv(filename, index=True)
    print(f"Daten gespeichert als {filename}")