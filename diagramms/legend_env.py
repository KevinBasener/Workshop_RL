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

def generate_layout_image():
    """
    Generiert ein Bild des Lagerlayouts basierend auf der Vorgabe.
    """
    pygame.init()
    layout = [
        "0000000",  # Obere 0-Reihe
        "0110110",  # Reihe 1
        "0110110",  # Reihe 2
        "0110110",  # Reihe 3
        "0110110",  # Reihe 4
        "0110110",  # Reihe 5
        "0110110",  # Reihe 6
        "0110110",  # Reihe 7
        "0000000"   # Untere 0-Reihe
    ]

    rows, cols = len(layout), len(layout[0])
    surface = pygame.Surface((cols * CELL_SIZE, rows * CELL_SIZE))

    for row_idx, row in enumerate(layout):
        for col_idx, cell in enumerate(row):
            rect = pygame.Rect(col_idx * CELL_SIZE, row_idx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if cell == "1":
                color = COLORS["rack"]
            else:
                color = COLORS["aisle"]
            surface.fill(color, rect)
            pygame.draw.rect(surface, COLORS["border"], rect, 1)

    pygame.image.save(surface, f"{OUTPUT_PATH}/lagerlayout.png")
    pygame.quit()
    print("Das Bild 'lagerlayout.png' wurde erfolgreich generiert.")

if __name__ == "__main__":
    generate_images()
    generate_layout_image()