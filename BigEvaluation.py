import pandas as pd


def analyze_popularity(filename="werkstattlager_logs.csv"):
    """
    Liest eine CSV-Datei mit Lager-Logs ein und bestimmt die Popularität
    der Artikel basierend auf der Häufigkeit der Materialentnahmen.

    Args:
        filename (str): Der Dateipfad zur CSV-Datei.

    Returns:
        pandas.Series: Eine nach Popularität sortierte Liste der Artikel (SKUs)
                       und der Anzahl ihrer Entnahmen.
                       Gibt None zurück, wenn die Datei nicht gefunden wird.
    """
    try:
        # 1. CSV-Datei einlesen
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{filename}' wurde nicht gefunden.")
        print("Bitte führe zuerst das Skript zum Generieren der Logs aus.")
        return None

    # 2. Nur relevante Transaktionen filtern (tatsächliche Nachfrage)
    demand_df = df[df['TransactionType'] == 'MATERIALENTNAHME']

    if demand_df.empty:
        print("Warnung: Keine 'MATERIALENTNAHME'-Transaktionen in der Datei gefunden.")
        return pd.Series(dtype=int)

    # 3. Popularität bestimmen: Häufigkeit jedes Artikels zählen
    # value_counts() zählt die Vorkommen und sortiert automatisch absteigend
    popularity = demand_df['SKU'].value_counts()

    return popularity


# --- Beispiel für die Anwendung ---
if __name__ == "__main__":
    # Die Funktion mit dem Standard-Dateinamen aufrufen
    article_popularity = analyze_popularity()

    if article_popularity is not None and not article_popularity.empty:
        total_articles = article_popularity.sum()
        popularity_percentage = (article_popularity / total_articles) * 100
        print("\n--- Prozentuale Popularitätsverteilung der Artikel ---")
        print(popularity_percentage)

    # Ergebnis ausgeben, wenn die Analyse erfolgreich war
    if article_popularity is not None:
        print("--- Artikel-Popularität (Anzahl der Entnahmen) ---")
        print(article_popularity)