import os
import json  # Hinzugefügt
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
import pygame
import random

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import pandas as pd  # Am Anfang der Datei hinzufügen

LOCATION_CAPACITY = 1000

def load_and_prepare_logs_for_rl(log_path="werkstattlager_logs.csv"):
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Fehler: Log-Datei '{log_path}' nicht gefunden.")
        return None, None

    # Popularität und Item-Katalog erstellen
    demand_df = df[df['TransactionType'] == 'MATERIALENTNAHME']
    popularity_counts = demand_df['SKU'].value_counts()

    item_catalog = {}
    sku_to_id_map = {sku: i for i, sku in enumerate(df['SKU'].unique())}
    id_to_sku_map = {i: sku for sku, i in sku_to_id_map.items()}

    for i, sku in id_to_sku_map.items():
        item_catalog[i] = {'id': i, 'name': sku, 'popularity': popularity_counts.get(sku, 0)}

    # Task-Liste mit Mengen erstellen
    task_queue = []
    for _, row in df.sort_values(by="Timestamp").iterrows():
        task_type = "PUT" if row['TransactionType'] == 'WARENEINGANG' else "PICK"
        item_id = sku_to_id_map[row['SKU']]
        quantity = abs(row['Quantity'])
        # Füge eine Aufgabe pro Transaktion hinzu, nicht pro Stück
        task_queue.append({'type': task_type, 'item_id': item_id, 'quantity': quantity})

    return item_catalog, task_queue

class DynamicWorkshop(gym.Env):
    """
    A more realistic workshop environment that handles both put-away and
    picking tasks with dynamic item popularity, configured via a JSON file.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    ACTION_MAP = {0: "MOVE UP", 1: "MOVE DOWN", 2: "MOVE LEFT", 3: "MOVE RIGHT", 4: "PUT"}

    def __init__(self, config_path, log_path=None):
        super(DynamicWorkshop, self).__init__()

        with open(config_path, 'r') as f:
            self.config = json.load(f)
        env_config, sim_config = self.config['environment'], self.config['simulation']
        self.rewards_config = self.config['rewards']
        self.height, self.width = env_config['height'], env_config['width']
        self.io_point = tuple(env_config['io_point'])
        self.layout_matrix = np.array(env_config['layout_matrix'], dtype=int)
        self.metadata['render_fps'] = sim_config['render_fps']

        if log_path:
            loaded_catalog, self.master_task_queue = load_and_prepare_logs_for_rl(log_path)
            if loaded_catalog is None: raise ValueError("Fehler beim Laden der Log-Datei.")
            self.item_catalog = {item['id']: item for item in loaded_catalog.values()}
        else:
            item_config = self.config['items']
            self.item_catalog = {item['id']: item for item in item_config['item_catalog']}
            self.master_task_queue = None

        print(f"[MASTER QUEUE]: {self.master_task_queue}")

        self.num_item_types = len(self.item_catalog)
        self.rack_locations = self._get_rack_locations()
        self.num_racks = len(self.rack_locations)

        # <<< GEÄNDERT: Agenten-Inventar ist jetzt ein Dictionary >>>
        self.agent_inventory = None  # Format: {'sku': id, 'quantity': int}
        # <<< GEÄNDERT: rack_contents speichert jetzt auch die Menge >>>
        self.rack_contents = {}  # Format: {loc: {'sku': id, 'quantity': int}}

        self.travel_times = {loc: self._calculate_travel_time_manhattan(self.io_point, loc) for loc in
                             self.rack_locations}
        self.max_travel_time = max(self.travel_times.values()) if self.travel_times else 1
        self.action_space = spaces.Discrete(5)

        # <<< GEÄNDERT: Beobachtungsraum erweitert um Mengeninformationen >>>
        # agent_pos(2), agent_inv(2), task(3), racks(num_racks * 2)
        obs_size = 2 + 2 + 3 + self.num_racks * 2
        self.observation_space = spaces.Box(
            low=-1, high=max(self.width, self.height, self.num_item_types, LOCATION_CAPACITY),
            shape=(obs_size,), dtype=np.float32
        )
        self.cell_size, self.window, self.clock, self.font = 80, None, None, None
        self.window_size = (self.width * self.cell_size, self.height * self.cell_size)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.io_point
        self.agent_inventory = None
        self.rack_contents = {}
        self.accumulated_reward = 0

        if self.master_task_queue is not None:
            self.task_queue = self.master_task_queue.copy()
        else: self._initialize_task_queue()

        self.current_task = self._get_next_task_from_queue()
        return self._get_obs(), {}

    def step(self, action):
        reward = self.rewards_config['step_penalty']
        terminated = False

        if action < 4:
            reward += self._move_agent(action)
        elif action == 4:
            reward += self._handle_put_action()

        reward += self.accumulated_reward
        self.accumulated_reward = 0

        if self.current_task is None:
            self.current_task = self._get_next_task_from_queue()
            if self.current_task is None:
                reward += self.rewards_config['completion_bonus']
                terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def _handle_put_action(self):
        print(f"\n[PUT ACTION] Agent an Position {self.agent_pos} versucht, abzulegen.")

        # Nur ausführen, wenn der Agent eine PUT-Aufgabe hat und etwas trägt
        if not (self.current_task and self.current_task["type"] == "PUT" and self.agent_inventory):
            print(f"  ❌ FEHLER: Keine gültige PUT-Aufgabe oder leeres Inventar.")
            return self.rewards_config['failed_put_penalty']

        # Finde ein benachbartes Regal
        adjacent_rack = None
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            check_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            if check_pos in self.rack_locations:
                adjacent_rack = check_pos
                break

        if not adjacent_rack:
            print(f"  ❌ FEHLER: Kein benachbartes Regal gefunden.")
            return self.rewards_config['failed_put_penalty']

        print(f"  ➡️ Benachbartes Regal gefunden bei: {adjacent_rack}")
        item_in_hand_sku = self.agent_inventory['sku']
        item_in_hand_qty = self.agent_inventory['quantity']
        print(f"  ➡️ Agent trägt: {item_in_hand_qty}x ID:{item_in_hand_sku}")

        # Prüfen, ob das Regal leer ist oder denselben Artikel enthält
        is_empty = adjacent_rack not in self.rack_contents
        is_compatible = not is_empty and self.rack_contents[adjacent_rack]['sku'] == item_in_hand_sku

        if is_empty or is_compatible:
            if is_empty:
                print(f"  ➡️ Regal {adjacent_rack} ist leer. Einlagerung möglich.")
            else:
                print(f"  ➡️ Regal {adjacent_rack} enthält bereits ID:{item_in_hand_sku}. Auffüllen möglich.")

            # Berechne verfügbaren Platz
            current_qty = self.rack_contents.get(adjacent_rack, {'quantity': 0})['quantity']
            free_space = LOCATION_CAPACITY - current_qty
            print(f"  ➡️ Verfügbarer Platz im Regal: ({free_space} von {LOCATION_CAPACITY})")

            if free_space > 0:
                put_qty = min(item_in_hand_qty, free_space)
                print(f"  ➡️ Lege {put_qty} Stück ab.")

                if is_empty:
                    self.rack_contents[adjacent_rack] = {'sku': item_in_hand_sku, 'quantity': 0}

                self.rack_contents[adjacent_rack]['quantity'] += put_qty
                self.agent_inventory['quantity'] -= put_qty

                print(f"  ➡️ Neuer Bestand im Regal: {self.rack_contents[adjacent_rack]['quantity']}")
                print(f"  ➡️ Rest im Inventar: {self.agent_inventory['quantity']}")

                # Wenn der Agent alles abgelegt hat, ist die Aufgabe fertig
                if self.agent_inventory['quantity'] == 0:
                    self.agent_inventory = None
                    self.current_task = None
                    travel_time = self.travel_times[adjacent_rack]
                    placement_score = 1 - (travel_time / self.max_travel_time)
                    reward = placement_score * self.rewards_config['put_score_multiplier']
                    print(f"  ✅ AUFGABE ERLEDIGT! Belohnung für Platzierung: {reward:.2f}")
                    return reward

                reward = self.rewards_config.get('partial_put_reward', 5)
                print(f"  ✅ TEIL-ABLAGE ERFOLGREICH! Belohnung: {reward}")
                return reward

        # Fall: Das Regal ist mit einem anderen Artikel belegt
        else:
            mismatch_sku = self.rack_contents[adjacent_rack]['sku']
            print(f"  ❌ FEHLER: Regal {adjacent_rack} ist mit falschem Artikel (ID:{mismatch_sku}) belegt.")
            return self.rewards_config['failed_put_penalty']

        # Fall: Das kompatible Regal war bereits voll
        print(f"  ❌ FEHLER: Regal {adjacent_rack} ist bereits voll.")
        return self.rewards_config['failed_put_penalty']

    def _get_next_task_from_queue(self):
        while self.task_queue:
            next_task = self.task_queue.pop(0)
            print(f"[TASK]: {next_task}")

            if next_task['type'] == 'PUT':
                self.agent_pos = self.io_point
                self.agent_inventory = {'sku': next_task['item_id'], 'quantity': next_task['quantity']}
                return next_task

            elif next_task['type'] == 'PICK':
                # PICK-Aufgaben werden sofort von der Umgebung gelöst
                item_to_pick = next_task['item_id']
                quantity_to_pick = next_task['quantity']

                # Finde alle Orte mit dem gesuchten Artikel
                locations_of_item = [loc for loc, data in self.rack_contents.items() if data['sku'] == item_to_pick]
                locations_of_item.sort(key=lambda loc: self.travel_times[loc])  # Nächstgelegene zuerst

                pick_cost = 0

                if not locations_of_item:
                    self.accumulated_reward += self.rewards_config.get('stockout_penalty', -100)
                else:
                    for loc in locations_of_item:
                        if quantity_to_pick == 0: break

                        pick_cost += self.travel_times[loc]  # Kosten für den Weg zum Regal

                        available_qty = self.rack_contents[loc]['quantity']
                        pick_qty = min(quantity_to_pick, available_qty)

                        self.rack_contents[loc]['quantity'] -= pick_qty
                        quantity_to_pick -= pick_qty

                        if self.rack_contents[loc]['quantity'] == 0:
                            del self.rack_contents[loc]

                    # Belohnung basiert auf den (hoffentlich niedrigen) Kosten
                    # max_cost wäre quantity * max_travel_time
                    max_possible_cost = next_task['quantity'] * self.max_travel_time
                    pick_score = 1 - (pick_cost / max_possible_cost) if max_possible_cost > 0 else 1
                    self.accumulated_reward += pick_score * self.rewards_config['pick_score_multiplier']

        # Da PICK sofort passiert, wird kein Task zurückgegeben, die Schleife sucht den nächsten
        return None

    def _move_agent(self, action):
        move_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = move_deltas[action]
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        print(f"    [MOVE] Attempting to move from {self.agent_pos} to {new_pos}")
        if (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height and
                self.layout_matrix[new_pos[1], new_pos[0]] == 0):
            self.agent_pos = new_pos
            print(f"    [MOVE] Success. New position: {self.agent_pos}")
            return 0
        else:
            print(f"    [MOVE] Failed. Wall or out of bounds.")
            # Belohnung aus Konfiguration verwenden
            return self.rewards_config['invalid_move_penalty']

    # --- Die restlichen Methoden bleiben unverändert ---
    # (z.B. _get_obs, _get_rack_locations, _update_popularity, etc.)
    def _get_obs(self):
        # Agent Position
        obs = [self.agent_pos[0], self.agent_pos[1]]
        # Agent Inventar
        if self.agent_inventory:
            obs.extend([self.agent_inventory['sku'], self.agent_inventory['quantity']])
        else:
            obs.extend([-1, 0])
        # Aktueller Task
        if self.current_task:
            task_type = 1 if self.current_task['type'] == 'PUT' else 2
            obs.extend([task_type, self.current_task['item_id'], self.current_task.get('quantity', 0)])
        else:
            obs.extend([-1, -1, 0])
        # Regal-Inhalte
        rack_obs = [-1, 0] * self.num_racks # [sku, qty, sku, qty, ...]
        loc_to_idx = {loc: i for i, loc in enumerate(self.rack_locations)}
        for rack_pos, data in self.rack_contents.items():
            if rack_pos in loc_to_idx:
                idx = loc_to_idx[rack_pos]
                rack_obs[idx * 2] = data['sku']
                rack_obs[idx * 2 + 1] = data['quantity']
        obs.extend(rack_obs)
        return np.array(obs, dtype=np.float32)

    def _get_rack_locations(self):
        return [(x, y) for y in range(self.height) for x in range(self.width) if self.layout_matrix[y, x] == 1]

    def _update_popularity(self, item_id):
        self.popularity_counts[item_id] += 1

    def _calculate_travel_time_manhattan(self, start, end):
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def _initialize_task_queue(self):
        self.task_queue = []
        available_for_picking = list(self.rack_contents.keys())
        random.shuffle(available_for_picking)
        for _ in range(self.num_initial_tasks):
            can_pick = bool(available_for_picking)
            can_put = True
            task_choice = random.random()
            if task_choice < 0.5 and can_pick:
                rack_pos = available_for_picking.pop()
                item_id = self.rack_contents[rack_pos]
                # Wichtig: JSON speichert Listen, wir brauchen Tupel. Hier aber wieder Liste für JSON-Kompatibilität.
                task = {"type": "PICK", "item_id": item_id, "origin": list(rack_pos)}
                self.task_queue.append(task)
            elif can_put:
                item_id = random.randint(0, self.num_item_types - 1)
                task = {"type": "PUT", "item_id": item_id}
                self.task_queue.append(task)
            else:
                break

    def plot_popularity_history(self, filepath=None):
        """
        Generates and saves a plot of item popularity changes over the episode.
        """
        if not self.popularity_history:
            print("⚠️ Popularity history is empty. Nothing to plot.")
            return

        history_array = np.array(self.popularity_history)

        fig, ax = plt.subplots(figsize=(12, 8))

        for i in range(self.num_item_types):
            ax.plot(history_array[:, i], label=f'Item {i}')

        ax.set_title('Item Popularity Over Episode', fontsize=16)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Popularity Count', fontsize=12)
        ax.legend(title='Items')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        fig.tight_layout()

        if filepath:
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                plt.savefig(filepath)
                print(f"✅ Popularity graph saved to {filepath}")
            except Exception as e:
                print(f"❌ Error saving popularity graph: {e}")
        else:
            plt.show()

        plt.close(fig)

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Dynamic Workshop Warehouse")
        if self.clock is None: self.clock = pygame.time.Clock()
        if self.font is None: self.font = pygame.font.Font(None, 24)
        if not hasattr(self, 'font_small'): self.font_small = pygame.font.Font(None, 18)

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        colors = {"aisle": (200, 200, 200), "rack": (100, 100, 100), "io_point": (255, 223, 0),
                  "agent": (65, 105, 225), "text": (0, 0, 0)}

        # 1. Gitter und leere Regale zeichnen
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                color = colors["aisle"] if self.layout_matrix[y, x] == 0 else colors["rack"]
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (220, 220, 220), rect, 1)

        # 2. I/O-Punkt zeichnen
        io_rect_obj = pygame.Rect(self.io_point[0] * self.cell_size, self.io_point[1] * self.cell_size, self.cell_size,
                                  self.cell_size)
        pygame.draw.rect(canvas, colors["io_point"], io_rect_obj)
        io_text = self.font.render("I/O", True, (0, 0, 0))
        canvas.blit(io_text, io_text.get_rect(center=io_rect_obj.center))

        # 3. <<< GEÄNDERT: Füllstand der belegten Regale zeichnen >>>
        for rack_pos, data in self.rack_contents.items():
            rect = pygame.Rect(rack_pos[0] * self.cell_size, rack_pos[1] * self.cell_size, self.cell_size,
                               self.cell_size)

            item_id = data['sku']
            quantity = data['quantity']
            popularity = self.item_catalog[item_id]['popularity']

            # Farbe basierend auf Popularität
            if popularity > 20:
                color = (0, 150, 0)  # Grün für A-Artikel
            elif popularity > 5:
                color = (255, 191, 0)  # Gelb für B-Artikel
            else:
                color = (200, 0, 0)  # Rot für C-Artikel

            # Zeichne Füllstands-Balken
            base_rect = rect.inflate(-10, -10)
            fill_ratio = quantity / LOCATION_CAPACITY
            fill_height = base_rect.height * fill_ratio
            fill_rect = pygame.Rect(base_rect.left, base_rect.bottom - fill_height, base_rect.width, fill_height)
            pygame.draw.rect(canvas, color, fill_rect)

            # Zeichne Text für ID und Menge
            id_surf = self.font_small.render(f"ID:{item_id}", True, (255, 255, 255))
            qty_surf = self.font_small.render(f"{quantity}", True, (255, 255, 255))
            canvas.blit(id_surf, id_surf.get_rect(center=(rect.centerx, rect.centery - 8)))
            canvas.blit(qty_surf, qty_surf.get_rect(center=(rect.centerx, rect.centery + 8)))

        # 4. Agenten zeichnen
        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size, self.cell_size,
                                 self.cell_size)
        pygame.draw.circle(canvas, colors["agent"], agent_rect.center, self.cell_size // 2 - 5)

        # <<< GEÄNDERT: Inventar des Agenten anzeigen >>>
        if self.agent_inventory:
            # Weißer Kreis, um anzuzeigen, dass der Agent etwas trägt
            pygame.draw.circle(canvas, (255, 255, 255), agent_rect.center, self.cell_size // 4)
            # Text, der anzeigt, was getragen wird
            inv_text = f"{self.agent_inventory['quantity']}x ID:{self.agent_inventory['sku']}"
            inv_surf = self.font_small.render(inv_text, True, colors['text'])
            canvas.blit(inv_surf, inv_surf.get_rect(midbottom=agent_rect.topleft))

        # 5. Overlay-Text für die aktuelle Aufgabe zeichnen
        if self.current_task:
            # <<< GEÄNDERT: Task-Anzeige mit Menge >>>
            task_str = f"Task: {self.current_task['type']} {self.current_task['quantity']}x item {self.current_task['item_id']}"
            text = self.font.render(task_str, True, (200, 0, 0))
            canvas.blit(text, (10, 10))

        # Anzeige aktualisieren
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def save_episode_render(self, filepath):
        """Renders the current state of the environment and saves it to a file."""
        # --- Ensure PyGame is initialized ---
        if self.window is None:
            pygame.init()
            # Run in headless mode for saving without a window popping up
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size, pygame.HIDDEN)
        if self.font is None: self.font = pygame.font.Font(None, 18)

        # Create the canvas and draw all elements (copied from render method)
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        colors = {"aisle": (200, 200, 200), "rack": (100, 100, 100), "io_point": (255, 223, 0),
                  "agent": (65, 105, 225)}

        # Draw grid, racks, and IO point
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                color = colors["aisle"] if self.layout_matrix[y, x] == 0 else colors["rack"]
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (220, 220, 220), rect, 1)
        pygame.draw.rect(canvas, colors["io_point"],
                         pygame.Rect(self.io_point[0] * self.cell_size, self.io_point[1] * self.cell_size,
                                     self.cell_size, self.cell_size))

        # Draw items in racks
        for rack_pos, item_id in self.rack_contents.items():
            popularity = self.popularity_counts[item_id]
            color = (255, 0, 0)  # Red for low pop
            if popularity >= 80:
                color = (0, 255, 0)  # Green for high pop
            elif 20 <= popularity < 80:
                color = (255, 255, 0)  # Yellow for mid pop
            rect = pygame.Rect(rack_pos[0] * self.cell_size, rack_pos[1] * self.cell_size, self.cell_size,
                               self.cell_size)
            pygame.draw.rect(canvas, color, rect.inflate(-8, -8), border_radius=5)
            text = self.font.render(f"{item_id}", True, (0, 0, 0))  # Black text for yellow bg
            canvas.blit(text, text.get_rect(center=rect.center))

        # Draw agent and inventory
        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size, self.cell_size,
                                 self.cell_size)
        pygame.draw.circle(canvas, colors["agent"], agent_rect.center, self.cell_size // 2 - 5)
        if self.agent_inventory is not None:
            pygame.draw.circle(canvas, (255, 255, 255), agent_rect.center, self.cell_size // 4)

        # --- Save the canvas to a file ---
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            pygame.image.save(canvas, filepath)
            print(f"✅ Episode render saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving render: {e}")

    def close(self):
        if self.window is not None: pygame.display.quit(); pygame.quit()

# -------------EVAL---------------
# (Der Evaluations- und Spiel-Code bleibt größtenteils gleich,
# außer bei der Instanziierung der Umgebung)

# --- Control Flags ---
TRAIN_MODELS = False
EVALUATE_MODEL = True
PLAY_GAME = False

base_log_dir = "dqn_workshop_runs"


def get_next_run_number(base_log_dir):
    if not os.path.exists(base_log_dir): return 1
    existing_runs = [d for d in os.listdir(base_log_dir) if d.startswith("run_")]
    if not existing_runs: return 1
    max_run = 0
    for run in existing_runs:
        try:
            num = int(run.split('_')[1])
            if num > max_run: max_run = num
        except (ValueError, IndexError):
            continue
    return max_run + 1


if __name__ == '__main__':

    # Pfad zur Konfigurationsdatei
    CONFIG_FILE = "two_racks.json"
    LOG_FILE = "werkstattlager_logs.csv"

    if TRAIN_MODELS:
        run_number = get_next_run_number(base_log_dir)
        log_dir = os.path.join(base_log_dir, f"run_{run_number}")
        model_path = os.path.join(log_dir, "dqn_workshop_agent.zip")
        os.makedirs(log_dir, exist_ok=True)
        print(f"--- Starting new training run #{run_number} ---")

        # <<< KORRIGIERT: log_path wird jetzt übergeben >>>
        env_dqn = Monitor(DynamicWorkshop(config_path=CONFIG_FILE, log_path=LOG_FILE), filename=log_dir)

        train_env_dqn = DummyVecEnv([lambda: env_dqn])
        policy_kwargs = dict(net_arch=[128, 64, 32])

        model_dqn = DQN("MlpPolicy", train_env_dqn, verbose=1, buffer_size=50000,
                        learning_starts=1000, batch_size=32, exploration_fraction=0.9,
                        exploration_final_eps=0.001, tensorboard_log=log_dir, policy_kwargs=policy_kwargs)

        print("--- Starting DQN Training... ---")
        model_dqn.learn(total_timesteps=1000000)  # Ggf. Timesteps für Tests reduzieren
        model_dqn.save(model_path)
        print(f"--- DQN Training finished. Model saved to {model_path} ---")

    else:
        if EVALUATE_MODEL:
            run_to_load = 56  # Passe die Nummer des zu ladenden Modells an
            model_path = os.path.join(base_log_dir, f"run_{run_to_load}", "dqn_workshop_agent.zip")

            if not os.path.exists(model_path):
                print(f"Error: Model not found at {model_path}")
            else:
                print(f"--- STARTING EVALUATION OF MODEL: {model_path} ---")

                # <<< KORRIGIERT: log_path wird jetzt übergeben >>>
                eval_env = DynamicWorkshop(config_path=CONFIG_FILE, log_path=LOG_FILE)
                model = DQN.load(model_path, env=eval_env)

                eval_episodes = 5
                episode_rewards = []
                max_steps_per_episode = 1000  # Erhöht, da eine Episode jetzt alle Log-Einträge abarbeitet

                for episode in range(eval_episodes):
                    obs, _ = eval_env.reset()
                    done = False
                    current_episode_reward = 0
                    current_steps = 0

                    while not done:
                        current_steps += 1
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        done = terminated or truncated
                        current_episode_reward += reward
                        eval_env.render()

                        if current_steps >= max_steps_per_episode:
                            print(
                                f"  [WARNUNG] Episode {episode + 1} bei {current_steps} Schritten abgebrochen (Limit erreicht).")
                            done = True

                    # (Der Rest der Evaluationsschleife bleibt gleich)
                    output_dir = "dqn_dynamic_layout_eval"
                    graph_path = os.path.join(output_dir, f"episode_{episode + 1}_popularity.png")
                    # eval_env.plot_popularity_history(filepath=graph_path) # Diese Funktion existiert in der neuen Version nicht mehr
                    image_path = os.path.join(output_dir, f"episode_{episode + 1}_final_state.png")
                    # eval_env.save_episode_render(image_path) # Diese Funktion existiert in der neuen Version nicht mehr
                    print(f"Episode {episode + 1}/{eval_episodes} | Reward: {current_episode_reward:.2f}")
                    episode_rewards.append(current_episode_reward)

                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                print("\n--- EVALUATION COMPLETE ---")
                print(f"Average reward over {eval_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
                eval_env.close()

        elif PLAY_GAME:
            run_to_load = 56  # Passe die Nummer des zu ladenden Modells an
            model_path = os.path.join(base_log_dir, f"run_{run_to_load}", "dqn_workshop_agent.zip")
            print(f"--- STARTING GAME MODE WITH MODEL: {model_path} ---")

            # <<< KORRIGIERT: log_path wird jetzt übergeben >>>
            game_env = DynamicWorkshop(config_path=CONFIG_FILE, log_path=LOG_FILE)
            model = DQN.load(model_path, env=game_env)

            # (Der Rest des Spiel-Codes bleibt gleich)
            running = True
            while running:
                obs, _ = game_env.reset()
                done = False
                total_reward = 0
                print("\n" + "=" * 50 + "\n🚀 NEW EPISODE STARTED 🚀")
                print("Press [SPACE] to advance one step. Close the window to quit.")
                print("=" * 50)

                while not done:
                    game_env.render()
                    action_taken = False
                    while not action_taken:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                done, running, action_taken = True, False, True
                            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                                action_taken = True
                    if not running: continue

                    action, _ = model.predict(obs, deterministic=True)
                    action_int = action.item()
                    action_str = game_env.ACTION_MAP.get(action_int, f"UNKNOWN({action_int})")
                    print(f"\nModel chose action: {action_int} [{action_str}]")
                    obs, reward, terminated, truncated, info = game_env.step(action_int)
                    done = terminated or truncated
                    total_reward += reward
                    print(f"  -> Step Reward: {reward:.2f} | Total Reward: {total_reward:.2f}")

                    if done: print(f"\n--- ✅ EPISODE FINISHED --- Final Reward: {total_reward:.2f}")
            game_env.close()