import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Daten in ein passendes Format bringen (Pandas DataFrame)
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

# Den Durchschnitt (Mittelwert) für jede Kategorie berechnen
# Ein Boxplot zeigt den Median, aber wir können den Mittelwert hinzufügen.
mittelwerte = df.groupby('Event-Art')['Ticketanzahl'].mean().reset_index()

# 2. Diagramm erstellen
plt.style.use('seaborn-v0_8-whitegrid') # Setzt einen schönen Stil
fig, ax = plt.subplots(figsize=(12, 7)) # Erstellt die Zeichenfläche und Achsen

# Boxplot erstellen, um die Verteilung darzustellen
sns.boxplot(x='Event-Art', y='Ticketanzahl', data=df, ax=ax,
            showfliers=False, # Ausreißer werden durch den Strip Plot dargestellt
            boxprops=dict(alpha=0.6)) # Boxen leicht transparent machen

# Strip Plot darüberlegen, um alle einzelnen Datenpunkte anzuzeigen
sns.stripplot(x='Event-Art', y='Ticketanzahl', data=df, ax=ax,
              jitter=True, # Punkte leicht streuen, um Überlappung zu vermeiden
              color='black', alpha=0.7, size=7)

# Mittelwerte als rote Rauten hinzufügen
sns.pointplot(x='Event-Art', y='Ticketanzahl', data=mittelwerte, ax=ax,
              markers='D', color='red', linestyles='')


# 3. Titel und Beschriftungen hinzufügen für bessere Lesbarkeit
ax.set_title('Ticketverkäufe nach Event-Art', fontsize=16, weight='bold')
ax.set_xlabel('Event-Art', fontsize=12)
ax.set_ylabel('Anzahl verkaufter Tickets', fontsize=12)
plt.xticks(rotation=15, ha='right') # Achsenbeschriftung leicht drehen

# Diagramm anzeigen
plt.tight_layout() # Stellt sicher, dass alles gut sichtbar ist
plt.show()