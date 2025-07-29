import numpy as np
import matplotlib.pyplot as plt


def generate_cost_heatmap(width=20, height=10):
    """
    Generiert und speichert eine Heatmap der Manhattan-Distanzkosten
    für ein gegebenes Lagerlayout, ohne abgeschnittene Ränder.
    """
    # 1. Definition des Lagerlayouts und des I/O-Punktes
    io_point = (width // 2, 0)

    # 2. Erstellung eines Kosten-Gitters
    cost_grid = np.zeros((height, width))

    # 3. Berechnung der Manhattan-Distanz für jeden Punkt
    for y in range(height):
        for x in range(width):
            distance = abs(x - io_point[0]) + abs(y - io_point[1])
            cost_grid[y, x] = distance

    # 4. Visualisierung des Gitters als Heatmap
    # Etwas größere Figur für bessere Lesbarkeit
    fig, ax = plt.subplots(figsize=(width * 0.6, height * 0.6))

    cax = ax.imshow(cost_grid, cmap='plasma', origin='lower')

    # 5. Beschriftung der Zellen und des I/O-Punktes
    for y in range(height):
        for x in range(width):
            color = 'white' if cost_grid[y, x] < cost_grid.max() / 4 else 'black'
            text = 'I/O' if (x, y) == io_point else f'{int(cost_grid[y, x])}'
            weight = 'bold' if (x, y) == io_point else 'normal'
            ax.text(x, y, text, ha='center', va='center', color=color, fontsize=8, weight=weight)

    # 6. Hinzufügen einer Farblegende und Titel
    fig.colorbar(cax, ax=ax, label='Kosten (Manhattan-Distanz)')
    ax.set_title('Kostenverteilung der Wegstrecke zum I/O-Punkt', fontsize=14, pad=20)
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_xticklabels(np.arange(width))
    ax.set_yticklabels(np.arange(height))
    ax.set_xlabel('X-Koordinate')
    ax.set_ylabel('Y-Koordinate')

    # 7. Speichern der Grafik
    plt.tight_layout(pad=1.5)  # Sorgt für eine gute Anordnung

    # --- KORREKTUR HIER ---
    # Fügen Sie `bbox_inches='tight'` hinzu, um das Abschneiden zu verhindern.
    plt.savefig('lager_kosten_heatmap.png', dpi=300, bbox_inches='tight')
    # --- ENDE DER KORREKTUR ---

    print("Heatmap wurde als 'lager_kosten_heatmap.png' gespeichert.")
    plt.show()


# --- Hauptskript ausführen ---
if __name__ == '__main__':
    generate_cost_heatmap(width=9, height=5)