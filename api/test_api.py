import requests
import json

# API-Endpunkt
url = "http://localhost:5000/api/v1/predictions"

try:
    # GET-Anfrage senden
    response = requests.get(url)
    
    # Prüfen ob Request erfolgreich war
    response.raise_for_status()
    
    # JSON-Antwort in Python-Objekt umwandeln
    data = response.json()
    
    # Ausgabe der Ergebnisse
    print("Status Code:", response.status_code)
    print("\nAntwort:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    # # Optional: Spezifische Werte aus der Antwort extrahieren
    # if data['status'] == 'success':
    #     predictions = data['predictions']
    #     print("\nAnzahl der Vorhersagen:", len(predictions))
        
    #     # Beispiel: Erste Aktie ausgeben
    #     first_stock = next(iter(predictions))
    #     print(f"\nDetails für {first_stock}:")
    #     print(json.dumps(predictions[first_stock], indent=2))

except requests.exceptions.ConnectionError:
    print("Fehler: Konnte keine Verbindung zum Server herstellen. Läuft der Server?")
except requests.exceptions.RequestException as e:
    print(f"Fehler bei der API-Anfrage: {e}")
except json.JSONDecodeError as e:
    print(f"Fehler beim Parsen der JSON-Antwort: {e}")
