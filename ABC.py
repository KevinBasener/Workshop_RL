import pygame
import numpy as np
import random
import sys
import pandas as pd

# <<< NEU: Globale Kapazität pro Lagerplatz >>>
LOCATION_CAPACITY = 50


class CostTracker:
    """Berechnet und speichert die operativen Kosten (Wegstrecken)."""

    def __init__(self, travel_times):
        self.travel_times = travel_times
        self.total_cost = 0

    def record_putaway_costs(self, plan):
        """Addiert die Kosten für einen Einlagerungsplan."""
        # Annahme: Jeder Schritt im Plan, der einen Lagerplatz anfährt, ist eine separate Bewegung.
        for step in plan:
            loc = step['location']
            self.total_cost += self.travel_times.get(loc, 0)

    def record_picking_costs(self, locations_visited):
        """Addiert die Kosten für die besuchten Entnahmeorte."""
        # Annahme: Jeder einzigartige besuchte Ort erfordert eine Bewegung.
        for loc in set(
                locations_visited):  # set() um doppelte Wege zu vermeiden, falls von einem Ort mehrfach gepickt wird
            self.total_cost += self.travel_times.get(loc, 0)

    def get_total_cost(self):
        return self.total_cost


class WarehouseVisualizer:
    def __init__(self, layout_matrix, io_point, zones, item_classes, cell_size=60):
        pygame.init()
        self.layout = layout_matrix
        self.height, self.width = layout_matrix.shape
        self.io_point = io_point
        self.zones = zones
        self.item_classes = item_classes
        self.cell_size = cell_size
        self.window_size = (self.width * self.cell_size, self.height * self.cell_size)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("ABC-Simulation mit Kostenfunktion")
        self.COLORS = {
            "background": (255, 255, 255), "grid": (220, 220, 220),
            "aisle": (255, 255, 255), "rack": (211, 211, 211),
            "io_point": (255, 195, 0),
            "zone_A": (0, 255, 0, 40), "zone_B": (255, 255, 0, 40), "zone_C": (255, 0, 0, 40),
            "item_A": (0, 180, 0), "item_B": (230, 190, 0), "item_C": (200, 0, 0),
            "text": (0, 0, 0), "text_light": (100, 100, 100)
        }
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

    # <<< GEÄNDERT: Nimmt jetzt auch die Gesamtkosten entgegen >>>
    def draw(self, occupied_locations, last_action="", total_cost=0):
        self.screen.fill(self.COLORS["background"])
        self._draw_grid()
        self._draw_zones()
        self._draw_io_point()
        self._draw_items(occupied_locations)

        # Anzeige für letzte Aktion und Kosten
        action_text_surf = self.font.render(last_action, True, self.COLORS["text"])
        cost_text_surf = self.font.render(f"Gesamtkosten: {total_cost}", True, self.COLORS["text"])

        self.screen.blit(action_text_surf, (10, 10))
        self.screen.blit(cost_text_surf, (10, self.window_size[1] - 30))  # Unten links
        pygame.display.flip()

    def _draw_grid(self):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                color = self.COLORS["rack"] if self.layout[y, x] == 1 else self.COLORS["aisle"]
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLORS["grid"], rect, 1)

    def _draw_zones(self):
        for zone_key, locations in self.zones.items():
            zone_color = self.COLORS[f"zone_{zone_key}"]
            zone_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            zone_surface.fill(zone_color)
            for y, x in locations:
                self.screen.blit(zone_surface, (x * self.cell_size, y * self.cell_size))

    def _draw_io_point(self):
        y, x = self.io_point
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLORS["io_point"], rect)
        io_text_surf = self.font.render("I/O", True, self.COLORS["text"])
        io_text_rect = io_text_surf.get_rect(center=rect.center)
        self.screen.blit(io_text_surf, io_text_rect)

    def _draw_items(self, occupied_locations):
        for loc, data in occupied_locations.items():
            y, x = loc
            item_id, quantity = data['sku'], data['quantity']
            item_class = self.item_classes.get(item_id, "C")
            item_color = self.COLORS[f"item_{item_class}"]
            fill_ratio = quantity / LOCATION_CAPACITY
            base_rect = pygame.Rect(x * self.cell_size + 5, y * self.cell_size + 5, self.cell_size - 10,
                                    self.cell_size - 10)
            fill_height = (self.cell_size - 10) * fill_ratio
            fill_rect = pygame.Rect(base_rect.left, base_rect.bottom - fill_height, base_rect.width, fill_height)
            pygame.draw.rect(self.screen, item_color, fill_rect)
            display_id = str(item_id).replace("SCHRAUBE-", "S-").replace("KABELBINDER-", "K-").replace("KUGELLAGER-",
                                                                                                       "L-")
            sku_surf = self.font_small.render(display_id, True, self.COLORS["background"])
            qty_surf = self.font_small.render(f"{quantity}/{LOCATION_CAPACITY}", True, self.COLORS["background"])
            self.screen.blit(sku_surf, sku_surf.get_rect(center=(base_rect.centerx, base_rect.centery - 8)))
            self.screen.blit(qty_surf, qty_surf.get_rect(center=(base_rect.centerx, base_rect.centery + 8)))

    def wait_for_quit(self):
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
        pygame.quit()
        sys.exit()

class ABCAgent:
    def __init__(self, layout_matrix, io_point, item_catalog):
        self.io_point = io_point
        self.item_catalog = item_catalog
        self.storage_locations = self._get_storage_locations(layout_matrix)
        self.travel_times = {loc: self._manhattan_distance(loc, self.io_point) for loc in self.storage_locations}
        self.zones = self._define_zones(self.storage_locations)
        self.item_classes = self._classify_items()

    def _get_storage_locations(self, layout):
        locs = np.where(layout == 1)
        return list(zip(locs[0], locs[1]))

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _define_zones(self, locations):
        sorted_locs = sorted(locations, key=lambda loc: self.travel_times[loc])
        num_locs = len(sorted_locs)
        a_count = int(num_locs * 0.2)
        b_count = int(num_locs * 0.3)
        return {"A": sorted_locs[:a_count], "B": sorted_locs[a_count: a_count + b_count],
                "C": sorted_locs[a_count + b_count:]}

    def _classify_items(self):
        sorted_items = sorted(self.item_catalog.keys(), key=lambda k: self.item_catalog[k]['popularity'], reverse=True)
        num_items, a_count, b_count = len(sorted_items), int(len(sorted_items) * 0.2), int(len(sorted_items) * 0.3)
        item_classes = {}
        for i, item_id in enumerate(sorted_items):
            if i < a_count:
                item_classes[item_id] = "A"
            elif i < a_count + b_count:
                item_classes[item_id] = "B"
            else:
                item_classes[item_id] = "C"
        return item_classes

    # <<< GEÄNDERT: Komplett neue Einlagerungslogik >>>
    def find_storage_for_batch(self, item_id, quantity_to_store, occupied_locations):
        """Findet Lagerplätze für eine ganze Charge und gibt einen Einlagerungsplan zurück."""
        storage_plan = []

        # 1. Priorität: Bestehende Plätze mit demselben Artikel auffüllen
        possible_locations = [loc for loc, data in occupied_locations.items() if
                              data['sku'] == item_id and data['quantity'] < LOCATION_CAPACITY]
        possible_locations.sort(key=lambda loc: self.travel_times[loc])  # Nächstgelegene zuerst

        for loc in possible_locations:
            if quantity_to_store == 0: break
            free_space = LOCATION_CAPACITY - occupied_locations[loc]['quantity']
            add_qty = min(quantity_to_store, free_space)
            storage_plan.append({'location': loc, 'add_quantity': add_qty})
            quantity_to_store -= add_qty

        if quantity_to_store == 0:
            return storage_plan

        # 2. Priorität: Neue, leere Plätze in bevorzugten Zonen suchen
        item_class = self.item_classes.get(item_id, "C")
        preferred_zones = []
        if item_class == "A":
            preferred_zones = ["A", "B", "C"]
        elif item_class == "B":
            preferred_zones = ["B", "C", "A"]
        else:
            preferred_zones = ["C", "B", "A"]

        for zone_key in preferred_zones:
            if quantity_to_store == 0: break
            available_in_zone = [loc for loc in self.zones[zone_key] if loc not in occupied_locations]
            available_in_zone.sort(key=lambda loc: self.travel_times[loc])  # Nächstgelegene zuerst

            for loc in available_in_zone:
                if quantity_to_store == 0: break
                add_qty = min(quantity_to_store, LOCATION_CAPACITY)
                storage_plan.append({'location': loc, 'new_sku': item_id, 'add_quantity': add_qty})
                quantity_to_store -= add_qty

        if quantity_to_store > 0:
            print(f"WARNUNG: Kein Platz für {quantity_to_store} Stk. von {item_id} gefunden!")

        return storage_plan


def prepare_data_from_logs(filename="werkstattlager_logs.csv"):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        return None, None
    demand_df = df[df['TransactionType'] == 'MATERIALENTNAHME']
    popularity_counts = demand_df['SKU'].value_counts()
    item_catalog = {sku: {'popularity': count} for sku, count in popularity_counts.items()}
    all_skus = df['SKU'].unique()
    for sku in all_skus:
        if sku not in item_catalog:
            item_catalog[sku] = {'popularity': 0}
    transactions = df.sort_values(by="Timestamp").to_dict('records')
    return item_catalog, transactions


if __name__ == '__main__':
    layout = np.zeros((9, 7), dtype=int)
    layout[1:8, 1:3] = 1
    layout[1:8, 4:6] = 1
    io_point = (4, 3)

    item_catalog, transactions = prepare_data_from_logs()
    if not transactions:
        sys.exit()

    abc_agent = ABCAgent(layout_matrix=layout, io_point=io_point, item_catalog=item_catalog)
    visualizer = WarehouseVisualizer(layout_matrix=layout, io_point=io_point, zones=abc_agent.zones,
                                     item_classes=abc_agent.item_classes)

    # <<< NEU: CostTracker initialisieren >>>
    cost_tracker = CostTracker(abc_agent.travel_times)

    occupied = {}

    visualizer.draw(occupied, "Simulation startet...", cost_tracker.get_total_cost())
    pygame.time.wait(1500)

    for trans in transactions:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

        item_id, quantity = trans['SKU'], trans['Quantity']
        action_info = ""

        if trans['TransactionType'] == 'WARENEINGANG':
            action_info = f"Einlagerung: {quantity}x {item_id}"
            plan = abc_agent.find_storage_for_batch(item_id, quantity, occupied)

            # <<< NEU: Kosten für Einlagerung erfassen >>>
            cost_tracker.record_putaway_costs(plan)

            for step in plan:
                loc, add = step['location'], step['add_quantity']
                if loc in occupied:
                    occupied[loc]['quantity'] += add
                else:
                    occupied[loc] = {'sku': item_id, 'quantity': add}

        elif trans['TransactionType'] == 'MATERIALENTNAHME':
            quantity_to_pick = abs(quantity)
            action_info = f"Entnahme: {quantity_to_pick}x {item_id}"

            locations_of_item = [loc for loc, data in occupied.items() if data['sku'] == item_id]
            locations_of_item.sort(key=lambda l: abc_agent.travel_times[l])

            # <<< NEU: Besuchte Orte für Kostenberechnung sammeln >>>
            locations_visited_for_this_pick = []

            if not locations_of_item:
                action_info = f"FEHLER: {item_id} nicht auf Lager!"
            else:
                for loc in locations_of_item:
                    if quantity_to_pick == 0: break

                    locations_visited_for_this_pick.append(loc)
                    available_qty = occupied[loc]['quantity']
                    pick_qty = min(quantity_to_pick, available_qty)

                    occupied[loc]['quantity'] -= pick_qty
                    quantity_to_pick -= pick_qty

                    if occupied[loc]['quantity'] == 0:
                        del occupied[loc]

            # <<< NEU: Kosten für Entnahme erfassen >>>
            cost_tracker.record_picking_costs(locations_visited_for_this_pick)

        # <<< GEÄNDERT: Gesamtkosten an Visualizer übergeben >>>
        visualizer.draw(occupied, action_info, cost_tracker.get_total_cost())
        pygame.time.wait(100)  # Beschleunigt für schnellere Analyse

    # <<< NEU: Finale Kostenausgabe >>>
    final_cost = cost_tracker.get_total_cost()
    print("\n" + "=" * 30)
    print(f"Simulation abgeschlossen!")
    print(f"FINALE GESAMTKOSTEN: {final_cost}")
    print("=" * 30)

    visualizer.draw(occupied, f"ENDE! Finale Kosten: {final_cost}", final_cost)
    visualizer.wait_for_quit()