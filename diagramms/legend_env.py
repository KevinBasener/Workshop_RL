import json

import pygame
import numpy as np

# --- Konfiguration ---
CELL_SIZE = 64  # Größe der einzelnen Bilder in Pixeln
GRID_DIM = 4  # Dimension des Gitter-Beispielbildes (4x4)
OUTPUT_PATH = "."  # Speicherort (aktuelles Verzeichnis)

# --- Farben ---
COLORS = {
    "popularity_a": (0, 150, 0),  # Grün für Schnelldreher
    "popularity_b": (255, 191, 0),  # Gelb für Mitteldreher
    "popularity_c": (200, 0, 0),  # Rot für Langsamdreher
    "rack": (100, 100, 100),  # Dunkelgrau für Regale
    "aisle": (255, 255, 255),  # Weiß für laufbare Felder
    "io_point": (255, 165, 0),  # Orange für I/O-Punkt
    "border": (0, 0, 0),  # Schwarz für Ränder
}


def generate_images():
    """
    Initialisiert pygame und generiert alle benötigten Bilder für die Legende.
    """
    pygame.init()
    font = pygame.font.SysFont(None, 48)
    print("Starte die Generierung der Legendenbilder...")

    # 1. Schnelldreher A (Grün mit "A")
    surface_a = pygame.Surface((CELL_SIZE, CELL_SIZE))
    surface_a.fill(COLORS["popularity_a"])
    pygame.draw.rect(surface_a, COLORS["border"], surface_a.get_rect(), 1)
    text_a = font.render("A", True, COLORS["border"])
    text_rect_a = text_a.get_rect(center=surface_a.get_rect().center)
    surface_a.blit(text_a, text_rect_a)
    pygame.image.save(surface_a, f"{OUTPUT_PATH}/schnelldreher_a.png")
    print("... schnelldreher_a.png gespeichert.")

    # 2. Schnelldreher B (Gelb mit "B")
    surface_b = pygame.Surface((CELL_SIZE, CELL_SIZE))
    surface_b.fill(COLORS["popularity_b"])
    pygame.draw.rect(surface_b, COLORS["border"], surface_b.get_rect(), 1)
    text_b = font.render("B", True, COLORS["border"])
    text_rect_b = text_b.get_rect(center=surface_b.get_rect().center)
    surface_b.blit(text_b, text_rect_b)
    pygame.image.save(surface_b, f"{OUTPUT_PATH}/schnelldreher_b.png")
    print("... schnelldreher_b.png gespeichert.")

    # 3. Schnelldreher C (Rot mit "C")
    surface_c = pygame.Surface((CELL_SIZE, CELL_SIZE))
    surface_c.fill(COLORS["popularity_c"])
    pygame.draw.rect(surface_c, COLORS["border"], surface_c.get_rect(), 1)
    text_c = font.render("C", True, COLORS["border"])
    text_rect_c = text_c.get_rect(center=surface_c.get_rect().center)
    surface_c.blit(text_c, text_rect_c)
    pygame.image.save(surface_c, f"{OUTPUT_PATH}/schnelldreher_c.png")
    print("... schnelldreher_c.png gespeichert.")

    # 4. Einzelner Regalkasten (Grau)
    surface_rack = pygame.Surface((CELL_SIZE, CELL_SIZE))
    surface_rack.fill(COLORS["rack"])
    pygame.draw.rect(surface_rack, COLORS["border"], surface_rack.get_rect(), 1)
    pygame.image.save(surface_rack, f"{OUTPUT_PATH}/regalkasten.png")
    print("... regalkasten.png gespeichert.")

    # 5. Laufbares Feld (Weißes Gitter)
    surface_aisle = pygame.Surface((CELL_SIZE, CELL_SIZE))
    surface_aisle.fill(COLORS["aisle"])
    pygame.draw.rect(surface_aisle, COLORS["border"], surface_aisle.get_rect(), 1)
    pygame.image.save(surface_aisle, f"{OUTPUT_PATH}/laufbares_feld.png")
    print("... laufbares_feld.png gespeichert.")

    # 6. I/O-Punkt (Orange mit "I/O")
    surface_io = pygame.Surface((CELL_SIZE, CELL_SIZE))
    surface_io.fill(COLORS["io_point"])
    pygame.draw.rect(surface_io, COLORS["border"], surface_io.get_rect(), 1)
    text_io = font.render("I/O", True, COLORS["border"])
    text_rect_io = text_io.get_rect(center=surface_io.get_rect().center)
    surface_io.blit(text_io, text_rect_io)
    pygame.image.save(surface_io, f"{OUTPUT_PATH}/io_punkt.png")
    print("... io_punkt.png gespeichert.")

    # 7. Agent (Blauer Kreis)
    surface_agent = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    surface_agent.fill((0, 0, 0, 0))  # Transparenter Hintergrund
    pygame.draw.circle(surface_agent, (0, 0, 255), (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 3)
    pygame.draw.circle(surface_agent, COLORS["border"], (CELL_SIZE // 2, CELL_SIZE // 2), CELL_SIZE // 3, 1)
    pygame.image.save(surface_agent, f"{OUTPUT_PATH}/agent.png")
    print("... agent.png gespeichert.")

    pygame.quit()
    print("\nAlle Bilder wurden erfolgreich generiert.")


def generate_layout_image(layout_data, filename, font):
    """
    Generiert ein Bild eines einzelnen Lagerlayouts basierend auf den übergebenen Daten.
    """
    rows, cols = len(layout_data), len(layout_data[0])
    surface = pygame.Surface((cols * CELL_SIZE, rows * CELL_SIZE))

    for row_idx, row in enumerate(layout_data):
        for col_idx, cell_char in enumerate(row):
            rect = pygame.Rect(col_idx * CELL_SIZE, row_idx * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # Zellfarben und Buchstaben basierend auf dem Layout-Zeichen zuweisen
            color_map = {
                '0': COLORS["aisle"],
                '1': COLORS["rack"],
                '2': COLORS["popularity_a"],
                '3': COLORS["popularity_b"],
                '4': COLORS["popularity_c"],
                'I': COLORS["io_point"],
            }
            text_map = {'2': 'A', '3': 'B', '4': 'C', 'I': 'I/O'}

            color = color_map.get(cell_char, COLORS["aisle"])
            text_char = text_map.get(cell_char, '')

            surface.fill(color, rect)
            pygame.draw.rect(surface, COLORS["border"], rect, 1)

            # Buchstaben in die Zelle zeichnen
            if text_char:
                text_surf = font.render(text_char, True, COLORS["border"])
                text_rect = text_surf.get_rect(center=rect.center)
                surface.blit(text_surf, text_rect)

    pygame.image.save(surface, f"{OUTPUT_PATH}/{filename}")
    print(f"Das Bild '{filename}' wurde erfolgreich generiert.")

if __name__ == "__main__":
    # 1. Generiere die Legendenbilder (optional, kann auskommentiert werden)
    generate_images()

    # 2. Lade die Layouts aus der JSON-Datei und generiere die Bilder
    print("\nStarte die Generierung der Layout-Bilder aus JSON...")
    pygame.init()
    # Kleinere Schriftart für "I/O", damit es gut in die Zelle passt
    io_font = pygame.font.SysFont(None, 36)

    try:
        with open('dqn_middle.json', 'r') as f:
            all_layouts = json.load(f)

        for layout_name, layout_data in all_layouts.items():
            filename = f"{layout_name}.png"
            generate_layout_image(layout_data, filename, io_font)

    except FileNotFoundError:
        print("FEHLER: Die Datei 'layouts.json' wurde nicht gefunden. Bitte erstellen Sie sie.")

    pygame.quit()
    print("\nAlle Layout-Bilder wurden erfolgreich generiert.")