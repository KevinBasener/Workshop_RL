import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import pygame
import random

# --- Set this to True to see detailed logs of every step ---
DEBUG_MODE = True


def debug_print(message):
    """Helper function to print only if debug mode is on."""
    if DEBUG_MODE:
        print(message)


class WorkshopEnv(gym.Env):
    """
    --- MODIFIED: Simplified environment with high-level actions ---
    An environment simulating a workshop warehouse. The agent uses high-level
    actions (e.g., "go get item") and a pathfinder handles the movement.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, width=10, height=5, item_catalog=None, num_tasks_per_episode=5, render_mode="human"):
        super(WorkshopEnv, self).__init__()
        debug_print("--- [ENV INIT] ---")

        # --- Layout, Items, Agent (Mostly Unchanged) ---
        self.width, self.height = width, height
        self.staging_area_pos = (width // 2, 0)
        # Layout: 1 for racks (obstacles), 0 for aisles (walkable)
        self.layout_matrix = np.ones((height, width), dtype=int)
        self.layout_matrix[:, width // 2] = 0
        self.layout_matrix[self.staging_area_pos[1], self.staging_area_pos[0]] = 0
        self.max_popularity = 100.0
        self.rack_locations = self._get_rack_locations()
        self.num_racks = len(self.rack_locations)

        self.agent_pos = None
        self.agent_inventory = None
        self.num_items = self.num_racks
        self.item_catalog = self._generate_item_catalog() if item_catalog is None else item_catalog

        self.num_tasks_per_episode = num_tasks_per_episode
        self.staging_area_queue = []
        self.rack_contents = {}

        # --- MODIFIED: RL Definitions ---
        # Action space is now high-level: 0: Go to Staging & Pick, 1: Go to Rack & Put
        self.action_space = spaces.Discrete(2)

        # Observation space is now smaller and fixed:
        # [agent_x, agent_y, inventory_item_id, next_task_id, closest_rack_x, closest_rack_y]
        obs_size = 6
        self.observation_space = spaces.Box(
            low=-1, high=max(width, height, self.num_items), shape=(obs_size,), dtype=np.float32
        )

        # --- Rendering (Unchanged) ---
        self.render_mode = render_mode
        self.cell_size = 50
        self.window_size = (self.width * self.cell_size, self.height * self.cell_size)
        self.window = None
        self.clock = None

        debug_print("--- [ENV INIT] Simplified Environment created successfully. ---\n")
        self.reset()

    # --- HELPER METHODS (Mostly Unchanged) ---
    def _get_rack_locations(self):
        return [(x, y) for y in range(self.height) for x in range(self.width) if self.layout_matrix[y, x] == 1]

    def _calculate_travel_time_manhattan(self, start, end):
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def _generate_item_catalog(self):
        catalog = {}
        for i in range(self.num_items):
            rand_val = np.random.rand()
            if rand_val < 0.2: popularity = np.random.randint(80, self.max_popularity + 1)
            elif rand_val < 0.5: popularity = np.random.randint(20, 80)
            else: popularity = np.random.randint(1, 20)
            catalog[i] = {'id': i, 'popularity': popularity}
        return catalog

    # --- NEW: Helper to find the closest empty storage location ---
    def _find_closest_empty_rack(self):
        empty_racks = [rack for rack in self.rack_locations if rack not in self.rack_contents]
        if not empty_racks:
            return None

        closest_rack = min(
            empty_racks,
            key=lambda rack: self._calculate_travel_time_manhattan(self.agent_pos, rack)
        )
        return closest_rack

    # --- MODIFIED: Create the new, smaller observation vector ---
    def _get_obs(self):
        # Agent position
        obs = [float(self.agent_pos[0]), float(self.agent_pos[1])]
        # Agent inventory
        obs.append(float(self.agent_inventory) if self.agent_inventory is not None else -1.0)
        # Next item in queue
        obs.append(float(self.staging_area_queue[0]) if self.staging_area_queue else -1.0)

        # Position of the closest empty rack
        closest_rack = self._find_closest_empty_rack()
        if closest_rack:
            obs.extend([float(closest_rack[0]), float(closest_rack[1])])
        else:
            obs.extend([-1.0, -1.0])  # No empty racks available

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        debug_print("\n--- [RESET] ---")
        self.agent_pos = self.staging_area_pos
        self.agent_inventory = None
        self.rack_contents = {}
        # Generate a list of unique items for the episode's tasks
        items = list(range(self.num_items))
        random.shuffle(items)
        self.staging_area_queue = items[:self.num_tasks_per_episode]

        debug_print(f"  [RESET] New tasks in staging area: {self.staging_area_queue}")
        return self._get_obs(), {}

    # --- MODIFIED: Core step logic now uses pathfinding for high-level actions ---
    def step(self, action):
        debug_print(f"\n--- [STEP] ---")
        debug_print(f"Agent at {self.agent_pos} (carrying: {self.agent_inventory}) chose high-level action: {action}")

        reward = 0
        terminated = False

        # --- MODIFIED: Pathfinding is no longer needed ---

        # Action 0: Go to Staging Area & Pick
        if action == 0:
            if self.agent_inventory is not None:
                reward -= 50  # Heavy penalty for trying to pick when already carrying something
            else:
                # Calculate distance for penalty, then teleport
                distance = self._calculate_travel_time_manhattan(self.agent_pos, self.staging_area_pos)
                reward -= distance  # Time penalty based on distance
                self.agent_pos = self.staging_area_pos  # Teleport agent to destination

                if self.staging_area_queue:
                    self.agent_inventory = self.staging_area_queue.pop(0)
                    reward += 20  # Reward for successful pick
                    debug_print(
                        f"  [STEP-PICK] Success! Agent picked item {self.agent_inventory}. Distance: {distance}")
                else:
                    reward -= 20  # Penalty for going to staging when it's empty

        # Action 1: Go to Closest Empty Rack & Put
        elif action == 1:
            if self.agent_inventory is None:
                reward -= 50  # Heavy penalty for trying to put when empty-handed
            else:
                target_rack = self._find_closest_empty_rack()
                debug_print(f"[CLOSET_RACK]: {target_rack}")
                if target_rack is None:
                    reward -= 50  # Heavy penalty, no place to put the item
                else:
                    # Find the closest walkable spot next to the rack
                    access_spots = []
                    rx, ry = target_rack
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        spot = (rx + dx, ry + dy)
                        if (0 <= spot[0] < self.width and 0 <= spot[1] < self.height and
                                self.layout_matrix[spot[1], spot[0]] == 0):
                            access_spots.append(spot)

                    if not access_spots:
                        reward -= 50
                    else:
                        destination = min(access_spots,
                                          key=lambda spot: self._calculate_travel_time_manhattan(self.agent_pos, spot))

                        self.agent_pos = destination  # Teleport agent to the access spot

                        self.rack_contents[target_rack] = self.agent_inventory
                        self.agent_inventory = None
                        reward += 30  # Reward for successful placement
                        debug_print(f"  [STEP-PUT] Success! Placed item at {target_rack}.")

        # --- Termination Logic (Unchanged) ---
        if not self.staging_area_queue and self.agent_inventory is None:
            debug_print("  [STEP-TERMINAL] All tasks complete. Terminating episode.")
            reward += 500  # Large completion bonus
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        """Renders the environment using PyGame with ABC labels on items."""
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        # Initialize PyGame window, clock, and font if they don't exist
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Workshop Warehouse Simulation")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))  # White background

        # --- Draw all elements ---
        colors = {
            "aisle": (200, 200, 200),
            "rack": (100, 100, 100),
            "staging": (255, 255, 0),
            "agent": (0, 0, 255),
            "item_A": (255, 0, 0),
            "item_B": (0, 255, 0),
            "item_C": (255, 165, 0),
        }

        # Draw aisles and racks
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if self.layout_matrix[y, x] == 0:
                    pygame.draw.rect(canvas, colors["aisle"], rect)
                else:
                    pygame.draw.rect(canvas, colors["rack"], rect)

        # Draw items in racks
        for rack_pos, item_id in self.rack_contents.items():
            pop = self.item_catalog[item_id]['popularity']
            item_class = 'A' if pop >= 80 else 'B' if pop >= 20 else 'C'
            color = colors[f"item_{item_class}"]
            rect = pygame.Rect(rack_pos[0] * self.cell_size, rack_pos[1] * self.cell_size, self.cell_size,
                               self.cell_size)
            pygame.draw.rect(canvas, color, rect, border_radius=5)

            # --- ADDED: Render the class letter on top ---
            text_surf = pygame.font.SysFont("Arial", size=36).render(item_class, True, (255, 255, 255))  # White text
            text_rect = text_surf.get_rect(center=rect.center)
            canvas.blit(text_surf, text_rect)

        # Draw staging area
        rect = pygame.Rect(self.staging_area_pos[0] * self.cell_size, self.staging_area_pos[1] * self.cell_size,
                           self.cell_size, self.cell_size)
        pygame.draw.rect(canvas, colors["staging"], rect)
        if self.staging_area_queue:
            # Render the next item ID in the staging area
            text = pygame.font.SysFont("Arial", size=36).render(f"S:{self.staging_area_queue[0]}", True, (0, 0, 0))  # Black text
            canvas.blit(text, text.get_rect(center=rect.center))

        # Draw agent
        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size, self.cell_size,
                                 self.cell_size)
        pygame.draw.circle(canvas, colors["agent"], agent_rect.center, self.cell_size // 2)

        # Draw item carried by agent
        if self.agent_inventory is not None:
            pop = self.item_catalog[self.agent_inventory]['popularity']
            item_class = 'A' if pop >= 80 else 'B' if pop >= 20 else 'C'
            color = colors[f"item_{item_class}"]
            pygame.draw.circle(canvas, color, agent_rect.center, self.cell_size // 4)

            # --- ADDED: Render the class letter on the carried item ---
            text_surf = pygame.font.SysFont("Arial", size=36).render(item_class, True, (0, 0, 0))  # Black text for contrast
            text_rect = text_surf.get_rect(center=agent_rect.center)
            canvas.blit(text_surf, text_rect)

        if self.render_mode == "human":
            # The screen window shows what is drawn on the canvas
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Properly close the PyGame window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def run_game_mode(model_path):
    """
    Starts an interactive session where the user can step through the
    agent's decisions by pressing Enter.
    """
    print("\n--- Starting Interactive Game Mode ---")
    print("Press ENTER to advance one step.")
    print("Close the PyGame window to quit.")

    env = WorkshopEnv(width=3, height=5, num_tasks_per_episode=10)

    try:
        model = DQN.load(model_path, env=env)
        print(f"Trained DQN model loaded from '{model_path}'.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'. Please train a model first.")
        return

    obs, info = env.reset()
    env.render()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Wait for the user to press a key
            if event.type == pygame.KEYDOWN:
                # Check if the key is ENTER
                if event.key == pygame.K_RETURN:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    env.render()

                    print(f"Action: {action}, Reward: {reward:.2f}")

                    if terminated or truncated:
                        print("\n--- Episode Finished! Press ENTER to start a new one. ---")
                        obs, info = env.reset()
                        env.render()

    env.close()

# ==============================================================================
# --- Phase 2: Hauptskript zum Trainieren und Evaluieren ---
# ==============================================================================

if __name__ == '__main__':

    TRAIN_MODELS = False
    EVALUATE_MODEL = False

    # --- DQN Training ---
    dqn_log_dir = "dqn_workshop_logs"
    dqn_model_path = os.path.join(dqn_log_dir, "dqn_workshop_agent.zip")
    os.makedirs(dqn_log_dir, exist_ok=True)

    if TRAIN_MODELS:
        print("\n--- Modus: TRAINING DQN ---")
        env_dqn = Monitor(WorkshopEnv(width=3, height=5, num_tasks_per_episode=5), filename=dqn_log_dir)
        train_env_dqn = DummyVecEnv([lambda: env_dqn])
        policy_kwargs = dict(net_arch=[128, 64, 32])

        model_dqn = DQN(
            "MlpPolicy",
            train_env_dqn,
            verbose=1,
            buffer_size=50000,  # How many past experiences to store
            learning_starts=1000,  # How many steps to take before learning starts
            batch_size=32,
            exploration_fraction=0.9,  # The fraction of training to explore
            exploration_final_eps=0.05,  # Final exploration probability
            tensorboard_log=dqn_log_dir,
            policy_kwargs=policy_kwargs
        )
        print("--- Starte DQN Training... ---")
        model_dqn.learn(total_timesteps=1000000)
        model_dqn.save(dqn_model_path)
        print(f"--- DQN Training abgeschlossen. Modell gespeichert unter {dqn_model_path} ---")

    # --- Evaluationsteil ---
    print("\n--- Modus: EVALUATION ---")

    if EVALUATE_MODEL:
        print("\n--- Mode: EVALUATION ---")
        # Lade beide trainierten Modelle
        try:
            dqn_model = DQN.load(dqn_model_path)
            print(f"Trainiertes DQN Modell von '{dqn_model_path}' geladen.")
        except FileNotFoundError:
            print(f"Fehler: Eine der Modell-Dateien wurde nicht gefunden. Bitte beide Modelle trainieren.")
            exit()

        num_eval_episodes = 50

        print("\n--- Starte systematische Evaluation ---")

        # Evaluiere DQN
        dqn_rewards = []
        for i in range(num_eval_episodes):
            eval_env = WorkshopEnv(width=3, height=5, num_tasks_per_episode=5)
            obs, info = eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = dqn_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
                eval_env.render()
            dqn_rewards.append(total_reward)
            eval_env.close()

        mean_reward_dqn = np.mean(dqn_rewards)
        std_reward_dqn = np.std(dqn_rewards)

        print("\n--- ERGEBNIS DER EVALUATION ---")
        print(
            f"DQN Durchschnittliche Belohnung über {num_eval_episodes} Episoden: {mean_reward_dqn:.2f} +/- {std_reward_dqn:.2f}")

        # --- Interactive Game Mode ---
    else:
        run_game_mode(dqn_model_path)