from time import sleep

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
    An environment simulating a workshop warehouse with a moving agent.
    This version uses an event-driven reward structure similar to the TUM warehouse.
    """

    def __init__(self, width=10, height=5, item_catalog=None, num_tasks_per_episode=5, render_mode="human"):
        super(WorkshopEnv, self).__init__()
        debug_print("--- [ENV INIT] ---")

        # --- Layout, Items, Agent ---
        self.width, self.height = width, height
        self.staging_area_pos = (width // 2, 0)
        self.layout_matrix = np.ones((height, width), dtype=int)
        self.layout_matrix[:, width // 2] = 0
        self.layout_matrix[self.staging_area_pos[1], self.staging_area_pos[0]] = 0
        self.rack_locations = self._get_rack_locations()
        self.num_racks = len(self.rack_locations)

        self.agent_pos = None
        self.agent_inventory = None
        self.num_items = self.num_racks
        self.font = None
        self.max_popularity = 100
        self.item_catalog = self._generate_item_catalog() if item_catalog is None else item_catalog

        self.num_tasks_per_episode = num_tasks_per_episode
        self.staging_area_queue = []
        self.rack_contents = {}

        # --- RL Definitions ---
        self.action_space = spaces.Discrete(6)
        obs_size = 2 + 1 + 1 + self.num_racks
        self.observation_space = spaces.Box(
            low=-1, high=max(width, height, self.num_items), shape=(obs_size,), dtype=np.float32
        )
        debug_print("--- [ENV INIT] Environment created successfully. ---\n")
        self.reset()

        self.render_mode = render_mode
        self.cell_size = 50
        self.window_size = (self.width * self.cell_size, self.height * self.cell_size)
        self.window = None
        self.clock = None
        self.metadata = {"render_fps": 30}

        debug_print("--- [ENV INIT] Environment created successfully. ---\n")
        self.reset()

    def _get_rack_locations(self):
        return [(x, y) for y in range(self.height) for x in range(self.width) if self.layout_matrix[y, x] == 1]

    def _calculate_travel_time_manhattan(self, start, end):
        """Calculates travel time using Manhattan distance."""
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def _generate_item_catalog(self):
        catalog = {}
        for i in range(self.num_items):
            rand_val = np.random.rand()
            if rand_val < 0.2:
                popularity = np.random.randint(80, self.max_popularity + 1)
            elif rand_val < 0.5:
                popularity = np.random.randint(20, 80)
            else:
                popularity = np.random.randint(1, 20)
            catalog[i] = {'id': i, 'popularity': popularity}
        return catalog

    def _get_obs(self):
        obs = [self.agent_pos[0], self.agent_pos[1]]
        obs.append(self.agent_inventory if self.agent_inventory is not None else -1)
        next_item = self.staging_area_queue[0] if self.staging_area_queue else -1
        obs.append(next_item)
        rack_obs = [-1] * self.num_racks
        for i, rack_pos in enumerate(self.rack_locations):
            if rack_pos in self.rack_contents:
                rack_obs[i] = self.rack_contents[rack_pos]
        obs.extend(rack_obs)
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        debug_print("\n--- [RESET] ---")
        self.agent_pos = self.staging_area_pos
        self.agent_inventory = None
        self.rack_contents = {}
        self.staging_area_queue = [random.randint(0, self.num_items - 1) for _ in range(self.num_tasks_per_episode)]
        debug_print(f"  [RESET] New tasks in staging area: {self.staging_area_queue}")
        return self._get_obs(), {}

    def step(self, action):
        debug_print(f"\n--- [STEP] ---")
        debug_print(f"Agent at {self.agent_pos} (carrying: {self.agent_inventory}) chose action: {action}")
        reward = 0  # Default penalty for time
        terminated = False

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in [0, {self.action_space.n - 1}])")

        if action < 4:  # Move
            move_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = move_deltas[action]
            new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            if (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height and
                    self.layout_matrix[new_pos[1], new_pos[0]] == 0):
                self.agent_pos = new_pos
                reward -= 1
            else:
                reward -= 10  # Penalty for bumping into a wall

        elif action == 4:  # Pick
            if self.agent_pos == self.staging_area_pos and self.agent_inventory is None and self.staging_area_queue:
                self.agent_inventory = self.staging_area_queue.pop(0)
                reward += 10  # Success reward for making progress
            else:
                reward -= 20  # Penalty for invalid action

        elif action == 5:  # Put
            if self.agent_inventory is not None:
                target_rack = None
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    check_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
                    if check_pos in self.rack_locations and check_pos not in self.rack_contents:
                        target_rack = check_pos
                        break
                if target_rack:
                    item_id_placed = self.agent_inventory
                    item_popularity = self.item_catalog[item_id_placed]['popularity']
                    distance = self._calculate_travel_time_manhattan(target_rack, self.staging_area_pos)

                    # Reward is higher for popular items and closer placements.
                    # Add 1 to distance to prevent division by zero and moderate rewards for the closest spots.
                    placement_reward = (item_popularity / (distance + 1)) * 10  # Scaled reward
                    reward += placement_reward

                    self.rack_contents[target_rack] = self.agent_inventory
                    self.agent_inventory = None
                    debug_print(f"  [STEP-PUT] Success! Placed item at {target_rack}.")
                else:
                    reward -= 20  # Penalty for invalid action
            else:
                reward -= 20  # Penalty for invalid action

            if self.agent_inventory is None:
                self.agent_pos = self.staging_area_pos

        # --- Termination Logic ---
        if not self.staging_area_queue and self.agent_inventory is None:
            debug_print("  [STEP-TERMINAL] All tasks complete. Terminating episode.")
            reward += 500  # Large completion bonus
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def render_to_file(self, folder_path="rendered_episodes"):
        """
        Renders the environment with a professional aesthetic and saves it to a file.
        """
        # --- Create the folder if it doesn't exist ---
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # --- Initialization (Headless Mode) ---
        # Initialize PyGame without a visible window for saving files
        if self.window is None:
            pygame.init()
            pygame.display.init()
            # Using pygame.HIDDEN prevents a window from appearing
            self.window = pygame.display.set_mode(self.window_size, pygame.HIDDEN)
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("sans-serif", 24, bold=True)
            self.font_small = pygame.font.SysFont("sans-serif", 16)

        # --- Professional Color Palette (Copied from render method) ---
        COLORS = {
            "background": (248, 249, 250),
            "grid": (222, 226, 230),
            "rack": (208, 212, 216),
            "rack_outline": (173, 181, 189),
            "staging": (255, 236, 179),
            "staging_outline": (253, 126, 20),
            "agent": (0, 123, 255),
            "agent_outline": (0, 86, 179),
            "item_A": (0, 255, 0),
            "item_B": (255, 255, 0),
            "item_C": (255, 0, 0),
            "text_dark": (33, 37, 41),
            "text_light": (248, 249, 250),
        }

        canvas = pygame.Surface(self.window_size)
        canvas.fill(COLORS["background"])

        # --- Draw Grid and Layout ---
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if self.layout_matrix[y, x] == 1:
                    pygame.draw.rect(canvas, COLORS["rack"], rect)
                    pygame.draw.rect(canvas, COLORS["rack_outline"], rect, 2)
                pygame.draw.rect(canvas, COLORS["grid"], rect, 1)

        # --- Draw Items in Racks ---
        for rack_pos, item_id in self.rack_contents.items():
            pop = self.item_catalog[item_id]['popularity']
            item_class = 'A' if pop >= 80 else 'B' if pop >= 20 else 'C'
            color = COLORS[f"item_{item_class}"]
            item_rect = pygame.Rect(rack_pos[0] * self.cell_size + 5, rack_pos[1] * self.cell_size + 5,
                                    self.cell_size - 10, self.cell_size - 10)
            pygame.draw.rect(canvas, color, item_rect, border_radius=5)
            text_surf = self.font.render(item_class, True, COLORS["text_light"])
            text_rect = text_surf.get_rect(center=item_rect.center)
            canvas.blit(text_surf, text_rect)

        # --- Draw Staging Area ---
        staging_rect = pygame.Rect(self.staging_area_pos[0] * self.cell_size, self.staging_area_pos[1] * self.cell_size,
                                   self.cell_size, self.cell_size)
        pygame.draw.rect(canvas, COLORS["staging"], staging_rect)
        pygame.draw.rect(canvas, COLORS["staging_outline"], staging_rect, 3)
        if self.staging_area_queue:
            text_surf = self.font_small.render(f"Next: {self.staging_area_queue[0]}", True, COLORS["text_dark"])
            text_rect = text_surf.get_rect(center=staging_rect.center)
            canvas.blit(text_surf, text_rect)

        # --- Draw Agent ---
        agent_center = (int(self.agent_pos[0] * self.cell_size + self.cell_size / 2),
                        int(self.agent_pos[1] * self.cell_size + self.cell_size / 2))
        agent_radius = self.cell_size // 2 - 8
        pygame.draw.circle(canvas, COLORS["agent_outline"], agent_center, agent_radius)
        pygame.draw.circle(canvas, COLORS["agent"], agent_center, agent_radius - 2)

        # Draw item carried by agent
        if self.agent_inventory is not None:
            pop = self.item_catalog[self.agent_inventory]['popularity']
            item_class = 'A' if pop >= 80 else 'B' if pop >= 20 else 'C'
            item_color = COLORS[f"item_{item_class}"]
            pygame.draw.circle(canvas, item_color, agent_center, self.cell_size // 4)
            pygame.draw.circle(canvas, COLORS["background"], agent_center, self.cell_size // 4, 2)

        # --- Save the canvas to a file ---
        # Use a unique name for each step to avoid overwriting
        step_filename = os.path.join(folder_path, f"step_{pygame.time.get_ticks()}.png")
        pygame.image.save(canvas, step_filename)
        print(f"Saved step to {step_filename}")

    def render(self):
        """Renders the environment with a clean, professional aesthetic."""
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        # --- Initialization ---
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Warehouse Simulation")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("sans-serif", 24, bold=True)
            self.font_small = pygame.font.SysFont("sans-serif", 16)

        # --- Professional Color Palette ---
        COLORS = {
            "background": (248, 249, 250),
            "grid": (222, 226, 230),
            "rack": (208, 212, 216),
            "rack_outline": (173, 181, 189),
            "staging": (255, 236, 179),
            "staging_outline": (253, 126, 20),
            "agent": (0, 123, 255),
            "agent_outline": (0, 86, 179),
            "item_A": (0, 255, 0),
            "item_B": (255, 255, 0),
            "item_C": (255, 0, 0),
            "text_dark": (33, 37, 41),
            "text_light": (248, 249, 250),
        }

        canvas = pygame.Surface(self.window_size)
        canvas.fill(COLORS["background"])

        # --- Draw Grid and Layout ---
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                # Draw rack
                if self.layout_matrix[y, x] == 1:
                    pygame.draw.rect(canvas, COLORS["rack"], rect)
                    pygame.draw.rect(canvas, COLORS["rack_outline"], rect, 2)
                # Draw grid lines for the entire area
                pygame.draw.rect(canvas, COLORS["grid"], rect, 1)

        # --- Draw Items in Racks ---
        for rack_pos, item_id in self.rack_contents.items():
            pop = self.item_catalog[item_id]['popularity']
            item_class = 'A' if pop >= 80 else 'B' if pop >= 20 else 'C'
            color = COLORS[f"item_{item_class}"]
            item_rect = pygame.Rect(rack_pos[0] * self.cell_size + 5, rack_pos[1] * self.cell_size + 5,
                                    self.cell_size - 10, self.cell_size - 10)
            pygame.draw.rect(canvas, color, item_rect, border_radius=5)

            text_surf = self.font.render(item_class, True, COLORS["text_light"])
            text_rect = text_surf.get_rect(center=item_rect.center)
            canvas.blit(text_surf, text_rect)

        # --- Draw Staging Area ---
        staging_rect = pygame.Rect(self.staging_area_pos[0] * self.cell_size, self.staging_area_pos[1] * self.cell_size,
                                   self.cell_size, self.cell_size)
        pygame.draw.rect(canvas, COLORS["staging"], staging_rect)
        pygame.draw.rect(canvas, COLORS["staging_outline"], staging_rect, 3)
        if self.staging_area_queue:
            text_surf = self.font_small.render(f"Next: {self.staging_area_queue[0]}", True, COLORS["text_dark"])
            text_rect = text_surf.get_rect(center=staging_rect.center)
            canvas.blit(text_surf, text_rect)

        # --- Draw Agent ---
        agent_center = (int(self.agent_pos[0] * self.cell_size + self.cell_size / 2),
                        int(self.agent_pos[1] * self.cell_size + self.cell_size / 2))
        agent_radius = self.cell_size // 2 - 8
        pygame.draw.circle(canvas, COLORS["agent_outline"], agent_center, agent_radius)
        pygame.draw.circle(canvas, COLORS["agent"], agent_center, agent_radius - 2)

        # Draw item carried by agent
        if self.agent_inventory is not None:
            pop = self.item_catalog[self.agent_inventory]['popularity']
            item_class = 'A' if pop >= 80 else 'B' if pop >= 20 else 'C'
            item_color = COLORS[f"item_{item_class}"]
            pygame.draw.circle(canvas, item_color, agent_center, self.cell_size // 4)
            pygame.draw.circle(canvas, COLORS["background"], agent_center, self.cell_size // 4, 2)

        # --- Update Display ---
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

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
    EVALUATE_MODEL = True

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
        model_dqn.learn(total_timesteps=5000000)
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
            eval_env.render_to_file(folder_path=f"evaluation_steps/dqn_episode")
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