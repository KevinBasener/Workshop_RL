import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_professional_flowchart():
    """
    Erstellt ein professionelles, von links nach rechts verlaufendes UML-Aktivitätsdiagramm.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # --- Stile definieren ---
    node_style = dict(boxstyle="round,pad=0.5", fc="#e3f2fd", ec="#1e88e5", lw=2, zorder=5)
    decision_style = dict(boxstyle="round,pad=0.5", fc="#fff3e0", ec="#fb8c00", lw=2, zorder=5)

    # --- Knotenpositionen (x, y) ---
    positions = {
        "start": (5, 50),
        "check_queue": (25, 50),
        "check_task_type": (50, 50),
        "einlagern_action": (75, 75),
        "einlagern_reward": (100, 75),
        "auslagern_action": (75, 25),
        "auslagern_reward": (100, 25),
        "end": (115, 50)
    }

    # --- Knoten zeichnen ---
    ax.plot(positions["start"][0], positions["start"][1], 'o', markersize=15, color='black', zorder=5)
    ax.plot(positions["end"][0], positions["end"][1], 'o', markersize=15, mfc='white', mec='black', mew=2, zorder=5)
    ax.plot(positions["end"][0], positions["end"][1], 'o', markersize=8, color='black', zorder=5)

    ax.text(positions["check_queue"][0], positions["check_queue"][1], "Noch Aufträge in\nder Warteschlange?",
            ha="center", va="center", bbox=decision_style, fontsize=11)
    ax.text(positions["check_task_type"][0], positions["check_task_type"][1], "Einlagern oder\nAuslagern?", ha="center",
            va="center", bbox=decision_style, fontsize=11)

    ax.text(positions["einlagern_action"][0], positions["einlagern_action"][1], "Agent lagert Artikel ein", ha="center",
            va="center", bbox=node_style, fontsize=11)
    ax.text(positions["einlagern_reward"][0], positions["einlagern_reward"][1], "Platzierungsbelohnung\nwird vergeben",
            ha="center", va="center", bbox=node_style, fontsize=11)

    ax.text(positions["auslagern_action"][0], positions["auslagern_action"][1], "System lagert automatisch aus",
            ha="center", va="center", bbox=node_style, fontsize=11)
    ax.text(positions["auslagern_reward"][0], positions["auslagern_reward"][1], "Auslagerungsbelohnung",
            ha="center", va="center", bbox=node_style, fontsize=11)

    # --- Pfeile zeichnen ---
    arrow_props = dict(facecolor='black', arrowstyle="->", lw=1.5, zorder=1)

    def connect(pos_a, pos_b, text="", rad=0, offset_a=(0, 0), offset_b=(0, 0), text_offset=(0, 2)):
        """Hilfsfunktion zum Zeichnen von Pfeilen."""
        x_a, y_a = pos_a[0] + offset_a[0], pos_a[1] + offset_a[1]
        x_b, y_b = pos_b[0] + offset_b[0], pos_b[1] + offset_b[1]
        arrow = patches.FancyArrowPatch((x_a, y_a), (x_b, y_b), **arrow_props, connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arrow)
        if text:
            ax.text((x_a + x_b) / 2 + text_offset[0], (y_a + y_b) / 2 + text_offset[1], text,
                    ha="center", va="center", fontsize=10, backgroundcolor='white', zorder=2)

    # Hauptfluss
    connect(positions["start"], positions["check_queue"], offset_a=(2, 0), offset_b=(-10, 0))
    connect(positions["check_queue"], positions["check_task_type"], "Ja", offset_a=(10, 0), offset_b=(-8, 0))
    connect(positions["check_queue"], positions["end"], "Nein", offset_a=(10, -3), offset_b=(-2, -3), rad=-0.2)

    # Verzweigung
    connect(positions["check_task_type"], positions["einlagern_action"], "Einlagern", offset_a=(8, 10),
            offset_b=(-10, 0), rad=0.2)
    connect(positions["check_task_type"], positions["auslagern_action"], "Auslagern", offset_a=(8, -10),
            offset_b=(-12, 0), rad=-0.2)

    # Einlagerungspfad
    connect(positions["einlagern_action"], positions["einlagern_reward"], offset_a=(10, 0), offset_b=(-10, 0))

    # Auslagerungspfad
    connect(positions["auslagern_action"], positions["auslagern_reward"], offset_a=(12, 0), offset_b=(-10, 0))

    # Rückschleifen - ANGEPASST
    p1 = positions["einlagern_reward"]
    p2 = positions["check_queue"]
    # Zeichnet die Linie der Schleife ohne Pfeilspitze
    loop_path1 = patches.PathPatch(
        patches.Path([(p1[0] + 10, p1[1]), (p1[0] + 10, p1[1] + 20), (p2[0] - 10, p1[1] + 20), (p2[0] - 10, p2[1] + 5)],
                     [patches.Path.MOVETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.LINETO]),
        facecolor='none', lw=1.5)
    ax.add_patch(loop_path1)
    # Fügt eine separate Pfeilspitze am Ende der Linie hinzu
    ax.add_patch(patches.FancyArrowPatch((p2[0] - 10, p1[1] + 20), (p2[0] - 10, p2[1] + 5), **arrow_props))

    p3 = positions["auslagern_reward"]
    # Zeichnet die Linie der zweiten Schleife
    loop_path2 = patches.PathPatch(
        patches.Path([(p3[0] + 10, p3[1]), (p3[0] + 10, p3[1] - 20), (p2[0] - 10, p3[1] - 20), (p2[0] - 10, p2[1] - 5)],
                     [patches.Path.MOVETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.LINETO]),
        facecolor='none', lw=1.5)
    ax.add_patch(loop_path2)
    # Fügt eine separate Pfeilspitze am Ende der Linie hinzu
    ax.add_patch(patches.FancyArrowPatch((p2[0] - 10, p3[1] - 20), (p2[0] - 10, p2[1] - 5), **arrow_props))

    # Titel und Speichern
    plt.title("Aktivitätsdiagramm: Ein- und Auslagerungsprozess", fontsize=18, pad=20)
    output_filename = "prozess_diagramm_links_rechts.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Diagramm erfolgreich als '{output_filename}' gespeichert.")
    plt.close()


if __name__ == '__main__':
    create_professional_flowchart()
