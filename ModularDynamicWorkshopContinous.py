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


class DynamicWorkshop(gym.Env):
    """
    A more realistic workshop environment that handles both put-away and
    picking tasks with dynamic item popularity, configured via a JSON file.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    ACTION_MAP = {0: "MOVE UP", 1: "MOVE DOWN", 2: "MOVE LEFT", 3: "MOVE RIGHT", 4: "PUT"}

    def __init__(self, config_path):
        super(DynamicWorkshop, self).__init__()
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # --- Konfiguration laden ---
        env_config = self.config['environment']
        sim_config = self.config['simulation']
        item_config = self.config['items']
        self.rewards_config = self.config['rewards']

        self.height = env_config['height']
        self.width = env_config['width']
        self.io_point = tuple(env_config['io_point'])
        self.layout_matrix = np.array(env_config['layout_matrix'], dtype=int)

        # NEU: Episoden-Terminierung über Aufgabenanzahl
        self.max_episode_steps = sim_config.get('max_episode_steps', 1000)  # Standardwert 1000

        # --- Artikelkatalog ---
        self.item_catalog = {item['id']: item for item in item_config['item_catalog']}
        self.num_item_types = len(self.item_catalog)
        self.max_popularity = 100.0

        # --- Umgebungsvariablen ---
        self.rack_locations = self._get_rack_locations()
        self.num_racks = len(self.rack_locations)
        self.travel_times = {loc: self._calculate_travel_time_manhattan(self.io_point, loc) for loc in
                             self.rack_locations}
        self.max_travel_time = max(self.travel_times.values()) if self.travel_times else 1

        # --- Agenten- und Episodenstatus ---
        self.agent_pos = None
        self.agent_inventory = None
        self.rack_contents = {}
        self.current_task = None
        self.accumulated_reward = 0
        self.popularity_counts = None
        self.tasks_completed = 0  # NEU: Zähler für abgeschlossene Aufgaben

        # --- RL-Definitionen ---
        self.action_space = spaces.Discrete(5)
        obs_size = 2 + 1 + 2 + self.num_racks
        self.observation_space = spaces.Box(
            low=-1, high=max(self.width, self.height, self.num_item_types), shape=(obs_size,), dtype=np.float32
        )

        # (Visualisierungs-Variablen bleiben gleich)
        self.cell_size, self.window, self.clock, self.font = 80, None, None, None
        self.window_size = (self.width * self.cell_size, self.height * self.cell_size)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.io_point
        self.agent_inventory = None
        self.rack_contents = {}
        self.popularity_counts = np.array([item['popularity'] for item in self.item_catalog.values()], dtype=float)
        self.accumulated_reward = 0
        self.current_step = 0

        # Initiales Füllen des Lagers
        items_to_place = list(self.item_catalog.keys())
        random.shuffle(items_to_place)
        initial_fill_count = int(self.num_racks * 0.5) # Fülle das Lager zu 50%
        for i, rack_pos in enumerate(self.rack_locations):
            if i < initial_fill_count:
                self.rack_contents[rack_pos] = items_to_place[i % self.num_item_types]

        # Erste Aufgabe generieren
        self.current_task = self._generate_new_task()
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        reward = self.rewards_config['step_penalty']
        terminated = False
        truncated = False
        task_was_completed_this_step = False

        # --- 1. Prüfen, welche Art von Aufgabe aktiv ist ---
        if self.current_task:
            task_type = self.current_task['type']

            if task_type == 'PICK':
                # Führe die PICK-Aufgabe sofort und automatisch aus.
                # Die Aktion des Agenten (z.B. eine Bewegung) wird in diesem Schritt ignoriert.
                self._handle_automatic_pick(self.current_task)
                task_was_completed_this_step = True
                self.current_task = None  # Aufgabe als erledigt markieren

            elif task_type == 'PUT':
                # Nur bei einer PUT-Aufgabe ist die Aktion des Agenten relevant.
                if self.agent_inventory is not None:
                    if action == 4:  # PUT-Aktion
                        put_reward = self._handle_put_action()
                        reward += put_reward
                        # Prüfen, ob die Ablage erfolgreich war
                        if put_reward > self.rewards_config['failed_put_penalty']:
                            task_was_completed_this_step = True
                            # current_task wird in _handle_put_action auf None gesetzt
                    else:  # Bewegungsaktion
                        reward += self._move_agent(action)
                else:
                    # Sollte nicht vorkommen, aber zur Sicherheit bewegt sich der Agent
                    reward += self._move_agent(action)

        else:
            # Falls es aus irgendeinem Grund keine Aufgabe gibt, wird eine neue generiert
            task_was_completed_this_step = True

        # --- 2. Belohnungen sammeln und Status aktualisieren ---
        reward += self.accumulated_reward
        self.accumulated_reward = 0

        if task_was_completed_this_step:
            # Eine neue Aufgabe wird nur generiert, wenn die alte abgeschlossen ist
            if self.current_task is None:
                self.current_task = self._generate_new_task()

        # --- 3. Prüfen, ob die Episode beendet ist ---
        if self.current_step >= self.max_episode_steps:
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _generate_new_task(self):
        """Generiert eine einzelne neue Aufgabe (PUT oder PICK)."""
        # Bedingungen prüfen: Kann kommissioniert werden? Kann eingelagert werden?
        can_pick = bool(self.rack_contents)
        can_put = len(self.rack_contents) < self.num_racks

        # Entscheidung, welcher Task-Typ generiert wird
        if can_pick and (not can_put or random.random() < 0.5):  # Bevorzuge PICK, wenn PUT nicht geht
            rack_pos = random.choice(list(self.rack_contents.keys()))
            item_id = self.rack_contents[rack_pos]
            return {"type": "PICK", "item_id": item_id, "origin": list(rack_pos)}
        elif can_put:
            item_id = random.choice(list(self.item_catalog.keys()))
            # Agent wird für die neue PUT-Aufgabe zum I/O-Punkt teleportiert und erhält den Artikel
            self.agent_pos = self.io_point
            self.agent_inventory = item_id
            return {"type": "PUT", "item_id": item_id}

        return None  # Kein Task möglich (Lager voll und leer zugleich, unwahrscheinlich)

    def _handle_put_action(self):
        if self.current_task and self.current_task["type"] == "PUT" and self.agent_inventory is not None:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
                if check_pos in self.rack_locations and check_pos not in self.rack_contents:
                    popularity = self.item_catalog[self.agent_inventory]['popularity']
                    travel_time = self.travel_times[check_pos]
                    norm_pop = popularity / self.max_popularity
                    norm_dist = travel_time / self.max_travel_time
                    placement_score = (norm_pop * (1 - norm_dist)) + ((1 - norm_pop) * norm_dist)
                    self.rack_contents[check_pos] = self.agent_inventory
                    self._update_popularity(self.agent_inventory)
                    self.agent_inventory = None
                    self.current_task = None
                    # Belohnungsmultiplikator aus Konfiguration verwenden
                    return placement_score * self.rewards_config['put_score_multiplier']
        # Belohnung aus Konfiguration verwenden
        return self.rewards_config['failed_put_penalty']

    def _handle_automatic_pick(self, pick_task):
        """Führt eine spezifische Auslagerungsaufgabe automatisch aus."""
        task_origin = tuple(pick_task['origin'])

        # Überprüfen, ob der richtige Artikel am richtigen Ort ist
        if task_origin in self.rack_contents and self.rack_contents[task_origin] == pick_task['item_id']:
            item_id = self.rack_contents.pop(task_origin)

            travel_time = self.travel_times[task_origin]
            retrieval_score = self.max_travel_time - travel_time
            instant_reward = retrieval_score * self.rewards_config['pick_score_multiplier']
            self.accumulated_reward += instant_reward

            print(f"🤖 Automatic PICK: Item {item_id} from {task_origin}, Reward: {instant_reward:.2f}")
        else:
            # Dieser Fall kann eintreten, wenn eine Aufgabe generiert wurde, der Artikel aber nicht mehr da ist.
            # Dies sollte selten passieren, führt aber nicht zu einem Fehler.
            print(f"🤖 Automatic PICK SKIPPED: Item {pick_task['item_id']} not found at {task_origin}.")

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
        obs = [self.agent_pos[0], self.agent_pos[1]]
        obs.append(self.agent_inventory if self.agent_inventory is not None else -1)
        task_type, task_item = -1, -1
        if self.current_task:
            task_type = 1
            task_item = self.current_task["item_id"]
        obs.extend([task_type, task_item])
        rack_obs = [-1] * self.num_racks
        loc_to_idx = {loc: i for i, loc in enumerate(self.rack_locations)}
        for rack_pos, item_id in self.rack_contents.items():
            if rack_pos in loc_to_idx: rack_obs[loc_to_idx[rack_pos]] = item_id
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
        if self.font is None: self.font = pygame.font.Font(None, 24)  # Increased font size slightly for clarity

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        colors = {"aisle": (200, 200, 200), "rack": (100, 100, 100), "io_point": (255, 223, 0),
                  "agent": (65, 105, 225)}

        # 1. Draw the basic grid layout (aisles and empty rack placeholders)
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                color = colors["aisle"] if self.layout_matrix[y, x] == 0 else colors["rack"]
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (220, 220, 220), rect, 1)

        # 2. Draw the I/O Point
        pygame.draw.rect(canvas, colors["io_point"],
                         pygame.Rect(self.io_point[0] * self.cell_size, self.io_point[1] * self.cell_size,
                                     self.cell_size, self.cell_size))
        io_text = self.font.render("I/O", True, (0, 0, 0))
        io_rect = io_text.get_rect(center=(self.io_point[0] * self.cell_size + self.cell_size // 2,
                                           self.io_point[1] * self.cell_size + self.cell_size // 2))
        canvas.blit(io_text, io_rect)

        # 3. Draw items in occupied racks OR Manhattan distance in empty racks
        for rack_pos in self.rack_locations:
            rect = pygame.Rect(rack_pos[0] * self.cell_size, rack_pos[1] * self.cell_size, self.cell_size,
                               self.cell_size)

            # Check if the rack is occupied
            if rack_pos in self.rack_contents:
                item_id = self.rack_contents[rack_pos]
                popularity = self.popularity_counts[item_id]

                # Determine color based on popularity
                if popularity >= 80:
                    color = (0, 150, 0)  # Dark Green
                elif 20 <= popularity < 80:
                    color = (255, 191, 0)  # Amber
                else:
                    color = (200, 0, 0)  # Dark Red

                pygame.draw.rect(canvas, color, rect.inflate(-8, -8), border_radius=5)

                # Render the item ID
                text_surf = self.font.render(f"ID:{item_id}", True, (255, 255, 255))
                canvas.blit(text_surf, text_surf.get_rect(center=rect.center))

            # If the rack is empty, draw the Manhattan distance
            else:
                travel_time = self.travel_times.get(rack_pos, '?')  # Get distance, default to '?'
                text_surf = self.font.render(str(travel_time), True, (255, 255, 255))  # White text
                text_rect = text_surf.get_rect(center=rect.center)
                canvas.blit(text_surf, text_rect)

        # 4. Draw the agent and its inventory status
        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size, self.cell_size,
                                 self.cell_size)
        pygame.draw.circle(canvas, colors["agent"], agent_rect.center, self.cell_size // 2 - 5)
        if self.agent_inventory is not None:
            # Draw a white circle inside the agent to show it's carrying an item
            pygame.draw.circle(canvas, (255, 255, 255), agent_rect.center, self.cell_size // 4)

        # 5. Draw overlay text (task info, etc.)
        if self.current_task:
            task_str = f"Task: {self.current_task['type']} item {self.current_task['item_id']}"
            if self.current_task['type'] == "PICK": task_str += f" from {self.current_task['origin']}"
            text = self.font.render(task_str, True, (200, 0, 0))
            canvas.blit(text, (10, 10))

        # Update the display
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
TRAIN_MODELS = True
EVALUATE_MODEL = False
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

    if TRAIN_MODELS:
        run_number = get_next_run_number(base_log_dir)
        log_dir = os.path.join(base_log_dir, f"run_{run_number}")
        model_path = os.path.join(log_dir, "dqn_workshop_agent.zip")
        os.makedirs(log_dir, exist_ok=True)
        print(f"--- Starting new training run #{run_number} ---")

        # *** MODIFIZIERT: Übergabe des Konfigurationspfads ***
        env_dqn = Monitor(DynamicWorkshop(config_path=CONFIG_FILE), filename=log_dir)
        train_env_dqn = DummyVecEnv([lambda: env_dqn])
        policy_kwargs = dict(net_arch=[128, 64, 32])

        model_dqn = DQN("MlpPolicy", train_env_dqn, verbose=1, buffer_size=50000,
                        learning_starts=1000, batch_size=32, exploration_fraction=0.9,
                        exploration_final_eps=0.01, tensorboard_log=log_dir, policy_kwargs=policy_kwargs)

        print("--- Starting DQN Training... ---")
        model_dqn.learn(total_timesteps=1000000)
        model_dqn.save(model_path)
        print(f"--- DQN Training finished. Model saved to {model_path} ---")

    else:
        if EVALUATE_MODEL:
            run_to_load = 24
            model_path = os.path.join(base_log_dir, f"run_{run_to_load}", "dqn_workshop_agent.zip")

            if not os.path.exists(model_path):
                print(f"Error: Model not found at {model_path}")
            else:
                print(f"--- STARTING EVALUATION OF MODEL: {model_path} ---")
                eval_env = DynamicWorkshop(config_path=CONFIG_FILE)
                model = DQN.load(model_path, env=eval_env)
                eval_episodes = 20
                episode_rewards = []

                for episode in range(eval_episodes):
                    obs, _ = eval_env.reset()
                    done = False
                    current_episode_reward = 0

                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        done = terminated or truncated
                        current_episode_reward += reward
                        eval_env.render()

                    # *** NEW: Generate and save the popularity plot after the episode ends ***
                    output_dir = "dqn_dynamic_layout"
                    graph_path = os.path.join(output_dir, f"episode_{episode + 1}_popularity.png")
                    eval_env.plot_popularity_history(filepath=graph_path)

                    image_path = os.path.join(output_dir, f"episode_{episode + 1}_final_state.png")
                    eval_env.save_episode_render(image_path)

                    print(f"Episode {episode + 1}/{eval_episodes} | Reward: {current_episode_reward:.2f}")
                    episode_rewards.append(current_episode_reward)

                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                print("\n--- EVALUATION COMPLETE ---")
                print(f"Average reward over {eval_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

                eval_env.close()

        elif PLAY_GAME:
            run_to_load = 30
            model_path = os.path.join(base_log_dir, f"run_{run_to_load}", "dqn_workshop_agent.zip")
            print(f"--- STARTING GAME MODE WITH MODEL: {model_path} ---")
            # *** MODIFIZIERT: Übergabe des Konfigurationspfads ***
            game_env = DynamicWorkshop(config_path=CONFIG_FILE)
            model = DQN.load(model_path, env=game_env)
            # (restlicher Spiel-Code bleibt gleich)
            running = True
            while running:
                obs, _ = game_env.reset()
                done = False
                total_reward = 0
                print("\n" + "=" * 50)
                print("🚀 NEW EPISODE STARTED 🚀")
                print("Press [SPACE] to advance one step.")
                print("Press [R] to reset the episode.")
                print("Close the window to quit.")
                print("=" * 50)

                while not done:
                    game_env.render()
                    action_taken = False
                    while not action_taken:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                done = True
                                running = False
                                action_taken = True
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_r:
                                    done = True
                                    action_taken = True
                                if event.key == pygame.K_SPACE:
                                    action_taken = True
                    if not running or done:
                        continue
                    action, _ = model.predict(obs, deterministic=True)
                    action_int = action.item()
                    action_str = game_env.ACTION_MAP.get(action_int, f"UNKNOWN({action_int})")
                    print(f"\nModel chose action: {action_int} [{action_str}]")
                    obs, reward, terminated, truncated, info = game_env.step(action_int)
                    done = terminated or truncated
                    total_reward += reward
                    print(f"  -> Reward for this step: {reward:.2f}")
                    print(f"  -> Total Episode Reward: {total_reward:.2f}")
                    if done:
                        output_dir = "dqn_dynamic_layout"
                        graph_path = os.path.join(output_dir, f"episode_game_popularity.png")
                        game_env.plot_popularity_history(filepath=graph_path)
                        print(f"\n--- ✅ EPISODE FINISHED --- Final Reward: {total_reward:.2f}")
            game_env.close()