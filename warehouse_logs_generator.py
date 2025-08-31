import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- KONFIGURATION ---
NUM_ROWS = 500  # Reduzierte Anzahl für ein kleineres Lager
START_DATE = datetime(2025, 8, 4, 7, 30, 0)  # Angepasste Startzeit (Schichtbeginn)
FILENAME = "werkstattlager_logs_changed_pop.csv"

# --- STAMMDATEN FÜR WERKSTATTLAGER (10 ARTIKEL) ---
# Simuliert das Pareto-Prinzip für typische Werkstattartikel
# A-Artikel (Verbrauchsmaterial, 2 SKUs, 70% der Bewegungen)
fast_movers = ["SCHRAUBE-M8x40", "KABELBINDER-200mm", "FILTER-LUFT-A45", "MOTOR-KEILRIEMEN-XPA", "BREMSFLUESSIGKEIT-DOT4"]
# B-Artikel (Regelmäßige Ersatzteile, 3 SKUs, 25% der Bewegungen)
medium_movers = ["KUGELLAGER-6204-2RS", "SICHERUNG-10A"]
# C-Artikel (Selten benötigte Teile, 5 SKUs, 5% der Bewegungen)
slow_movers = ["SENSOR-DRUCK-P20", "DICHTUNG-GUMMI-S12", "RELAIS-12V-KFZ"]

ALL_SKUS = fast_movers + medium_movers + slow_movers
SKU_PROBABILITIES = ([0.70 / len(fast_movers)] * len(fast_movers) +
                     [0.25 / len(medium_movers)] * len(medium_movers) +
                     [0.05 / len(slow_movers)] * len(slow_movers))

# Kleinere Anzahl an Lagerplätzen und nur 3 Werker
LOCATIONS = [f"REGAL-{rack}-{fach}" for rack in ["A", "B"] for fach in range(1, 11)]
USERS = ["MEIER", "HUBER", "SCHMIDT"]

# Angepasste Transaktionstypen und Wahrscheinlichkeiten
# MATERIALENTNAHME ist die häufigste Aktion in einer Werkstatt
TRANSACTION_TYPES = ["MATERIALENTNAHME", "WARENEINGANG", "BESTANDSKORREKTUR"]
TRANSACTION_PROBABILITIES = [0.85, 0.14, 0.01]


# --- GENERIERUNGSLOGIK ---
def generate_workshop_logs():
    """Generiert eine Liste von Log-Einträgen für ein Werkstattlager."""
    log_data = []
    current_time = START_DATE

    # Angepasste Auftrags-IDs
    work_order_id = 4000
    purchase_order_id = 850

    for i in range(NUM_ROWS):
        # 1. Zeitstempel generieren (kurze Abstände, da oft viel los ist)
        time_delta = timedelta(seconds=random.randint(10, 600))
        current_time += time_delta

        # Simuliert Arbeitszeiten (z.B. 7:30 bis 16:30 Uhr)
        if current_time.hour >= 16 and current_time.minute > 30:
            current_time += timedelta(hours=15)
            current_time = current_time.replace(hour=7, minute=30)

        # 2. Transaktionstyp auswählen
        transaction_type = np.random.choice(TRANSACTION_TYPES, p=TRANSACTION_PROBABILITIES)

        # 3. SKU basierend auf Pareto-Prinzip auswählen
        sku = np.random.choice(ALL_SKUS, p=SKU_PROBABILITIES)

        # 4. Menge und Auftrags-ID an Werkstatt anpassen
        if transaction_type == "MATERIALENTNAHME":
            quantity = -random.randint(1, 10)  # Typischerweise kleine Mengen für einen Auftrag
            order_id = f"AUFTRAG-{work_order_id}"
            if i % random.randint(2, 5) == 0:  # Oft mehrere Entnahmen pro Auftrag
                work_order_id += 1
        elif transaction_type == "WARENEINGANG":
            quantity = random.randint(10, 50)  # Kleinere Liefermengen als in einem Großlager
            order_id = f"LIEF-{purchase_order_id}"
            if i % random.randint(20, 40) == 0:
                purchase_order_id += 1
        else:  # BESTANDSKORREKTUR
            quantity = random.choice([-1, 1])
            order_id = "INVENTUR"

        # 5. Restliche Daten zufällig auswählen
        location = random.choice(LOCATIONS)
        user = random.choice(USERS)

        # Eintrag zum Log hinzufügen
        log_data.append({
            "Timestamp": current_time.strftime('%Y-%m-%d %H:%M:%S'),
            "TransactionType": transaction_type,
            "SKU": sku,
            "Quantity": quantity,
            "Location": location,
            "UserID": user,
            "OrderID": order_id
        })

    return log_data


# --- Hauptprogramm ---
if __name__ == "__main__":
    print("Generiere Logs für Werkstattlager...")
    logs = generate_workshop_logs()

    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(FILENAME, index=False)

    print(f"Fertig! {NUM_ROWS} Log-Einträge wurden in '{FILENAME}' gespeichert.")
    print("Hier ist eine Vorschau der ersten 5 Einträge:")
    print(df_logs.head())