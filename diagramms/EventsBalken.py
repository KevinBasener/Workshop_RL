import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Die Daten für die Events
data = {
    'Event-Art': [
        'Erstiparty / Party', 'Erstiparty / Party', 'Erstiparty / Party',
        'Pubcrawl', 'Pubcrawl', 'Pubcrawl', 'Pubcrawl',
        'Lasertag', 'Lasertag', 'Lasertag',
        'Karaoke', 'Karaoke', 'Karaoke', 'Karaoke',
        'Bowling & Pubsport', 'Bowling & Pubsport', 'Bowling & Pubsport', 'Bowling & Pubsport'
    ],
    'Ticketanzahl': [
        70, 600, 904,
        168, 500, 124, 542,
        43, 55, 29,
        142, 58, 25, 76,
        39, 185, 35, 107
    ]
}
df = pd.DataFrame(data)

# --- Erstellung des Diagramms ---

# 1. Leere Zeichenfläche mit einer bestimmten Größe erstellen
plt.figure(figsize=(11, 7))

# 2. Balkendiagramm für den Durchschnitt zeichnen
# Seaborn berechnet automatisch den Durchschnitt für die Höhe der Balken.
sns.barplot(x='Event-Art', y='Ticketanzahl', data=df, color='red', alpha=0.7, errorbar=None)

# 3. Streudiagramm für die einzelnen Datenpunkte darüberlegen
# sns.stripplot(x='Event-Art', y='Ticketanzahl', data=df, color='black', jitter=True)

# 4. Titel und Achsenbeschriftungen hinzufügen
plt.title('Ticketverkäufe: Durchschnitt und einzelne Events', fontsize=16)
plt.ylabel('Anzahl verkaufter Tickets')
plt.xlabel('Event-Kategorie')

# Stellt sicher, dass die Beschriftungen nicht überlappen
plt.tight_layout()

# 5. Das fertige Diagramm als Bilddatei speichern
# Die Datei 'event_vergleich.png' wird in deinem Projektordner erstellt.
plt.savefig('event_vergleich.png')

print("Diagramm wurde als 'event_vergleich.png' gespeichert.")