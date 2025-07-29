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
    An environment simulating an open-plan workshop warehouse where every
    cell is a potential storage location. The agent can move freely.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, width=20, height=10, item_catalog=None, num_tasks_per_episode=5):
        super(WorkshopEnv, self).__init__()
        debug_print("--- [ENV INIT] ---")
        debug_print(f"Creating OPEN workshop with size {width}x{height}.")

        # --- 1. Physisches Layout ---
        self.width, self.height = width, height
        self.staging_area_pos = (width // 2, 0)

        # --- NEW: Entire grid is walkable ---
        self.layout_matrix = np.zeros((height, width), dtype=int)

        # --- NEW: All cells (except staging) are storage locations ---
        self.rack_locations = self._get_rack_locations()
        self.num_racks = len(self.rack_locations)

        # Agent and Item properties
        self.agent_pos = None
        self.agent_inventory = None
        self.num_items = self.num_racks
        self.max_popularity = 100
        self.item_catalog = self._generate_item_catalog() if item_catalog is None else item_catalog

        self.num_tasks_per_episode = num_tasks_per_episode
        self.staging_area_queue = []
        self.rack_contents = {}  # Stores {(x, y): item_id}

        # --- RL Definitions ---
        self.action_space = spaces.Discrete(6)  # 4 move, 1 pick, 1 put
        obs_size = 2 + 1 + 1 + self.num_racks
        self.observation_space = spaces.Box(
            low=-1, high=max(width, height, self.num_items), shape=(obs_size,), dtype=np.float32
        )

        # --- PyGame Attributes ---
        self.cell_size = 60
        self.window_size = (self.width * self.cell_size, self.height * self.cell_size)
        self.window = None
        self.clock = None
        self.font = None

        debug_print("--- [ENV INIT] Environment created successfully. ---\n")
        self.reset()

    def _get_rack_locations(self):
        """All cells are storage locations, except the staging area."""
        locations = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) != self.staging_area_pos:
                    locations.append((x, y))
        return locations

    def _calculate_travel_time_manhattan(self, start, end):
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
        # Create a mapping from location coordinate to its index in the list
        loc_to_idx = {loc: i for i, loc in enumerate(self.rack_locations)}
        for rack_pos, item_id in self.rack_contents.items():
            if rack_pos in loc_to_idx:
                rack_obs[loc_to_idx[rack_pos]] = item_id

        obs.extend(rack_obs)
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        debug_print("\n--- [RESET] ---")
        debug_print("Episode starting. Resetting environment.")
        self.agent_pos = self.staging_area_pos
        self.agent_inventory = None
        self.rack_contents = {}
        num_tasks = min(self.num_tasks_per_episode, self.num_racks)
        self.staging_area_queue = [random.randint(0, self.num_items - 1) for _ in range(num_tasks)]
        debug_print(f"  [RESET] New tasks in staging area: {self.staging_area_queue}")
        return self._get_obs(), {}

    def step(self, action):
        debug_print(f"\n--- [STEP] ---")
        debug_print(f"Agent at {self.agent_pos} (carrying: {self.agent_inventory}) chose action: {action}")
        reward = -1
        terminated = False

        if action < 4:  # Move
            move_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = move_deltas[action]
            new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            if (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height):
                self.agent_pos = new_pos
                debug_print(f"  [STEP-MOVE] Agent moved to {self.agent_pos}")
            else:
                debug_print(f"  [STEP-MOVE] Move to {new_pos} is invalid (out of bounds).")
                reward -= 2

        elif action == 4:  # Pick
            debug_print("  [STEP-PICK] Agent attempts to PICK.")
            if self.agent_pos == self.staging_area_pos and self.agent_inventory is None and self.staging_area_queue:
                self.agent_inventory = self.staging_area_queue.pop(0)
                reward += 20
                debug_print(
                    f"  [STEP-PICK] Success! Agent picked up {self.agent_inventory}. Queue: {self.staging_area_queue}")
            else:
                debug_print("  [STEP-PICK] Failed. Conditions not met.")
                reward -= 10

        elif action == 5:  # Put
            debug_print("  [STEP-PUT] Agent attempts to PUT.")
            if self.agent_inventory is not None:
                target_rack = self.agent_pos
                if target_rack in self.rack_locations and target_rack not in self.rack_contents:
                    reward += 40
                    self.rack_contents[target_rack] = self.agent_inventory
                    self.agent_inventory = None
                    debug_print(f"  [STEP-PUT] Success! Placed item at {target_rack}.")
                else:
                    debug_print(f"  [STEP-PUT] Failed. Current location {target_rack} is invalid or occupied.")
                    reward -= 10
            else:
                debug_print("  [STEP-PUT] Failed. Agent inventory is empty.")
                reward -= 5

        if not self.staging_area_queue and self.agent_inventory is None:
            debug_print("  [STEP-TERMINAL] All tasks complete. Terminating episode.")
            reward += 1000
            terminated = True

        debug_print(f"  [STEP-REWARD] Reward for this step: {reward}")
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Open Workshop Simulation")
        if self.clock is None: self.clock = pygame.time.Clock()
        # Increased font size for better readability
        if self.font is None: self.font = pygame.font.Font(None, 36)

        canvas = pygame.Surface(self.window_size)
        canvas.fill((200, 200, 200))  # Gray background
        colors = {"staging": (255, 223, 0), "agent": (65, 105, 225),
                  "item_A": (220, 20, 60), "item_B": (50, 205, 50),
                  "item_C": (255, 140, 0)}

        # Draw items in racks
        for rack_pos, item_id in self.rack_contents.items():
            pop = self.item_catalog[item_id]['popularity']
            item_class = 'A' if pop >= 80 else 'B' if pop >= 20 else 'C'
            color = colors[f"item_{item_class}"]
            rect = pygame.Rect(rack_pos[0] * self.cell_size, rack_pos[1] * self.cell_size, self.cell_size,
                               self.cell_size)
            pygame.draw.rect(canvas, color, rect.inflate(-8, -8), border_radius=5)

            # --- NEW: Render the class letter on top ---
            text_surf = self.font.render(item_class, True, (255, 255, 255))  # White text
            text_rect = text_surf.get_rect(center=rect.center)
            canvas.blit(text_surf, text_rect)
            # --- END OF CHANGE ---

        # Draw grid lines
        for x in range(self.width + 1):
            pygame.draw.line(canvas, (180, 180, 180), (x * self.cell_size, 0),
                             (x * self.cell_size, self.height * self.cell_size))
        for y in range(self.height + 1):
            pygame.draw.line(canvas, (180, 180, 180), (0, y * self.cell_size),
                             (self.width * self.cell_size, y * self.cell_size))

        # Draw staging area
        rect = pygame.Rect(self.staging_area_pos[0] * self.cell_size, self.staging_area_pos[1] * self.cell_size,
                           self.cell_size, self.cell_size)
        pygame.draw.rect(canvas, colors["staging"], rect)
        if self.staging_area_queue:
            text = self.font.render(f"S:{self.staging_area_queue[0]}", True, (0, 0, 0))
            canvas.blit(text, text.get_rect(center=rect.center))

        # Draw agent
        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size, self.cell_size,
                                 self.cell_size)
        pygame.draw.circle(canvas, colors["agent"], agent_rect.center, self.cell_size // 2 - 5)

        if self.agent_inventory is not None:
            pop = self.item_catalog[self.agent_inventory]['popularity']
            item_class = 'A' if pop >= 80 else 'B' if pop >= 20 else 'C'
            color = colors[f"item_{item_class}"]
            pygame.draw.circle(canvas, color, agent_rect.center, self.cell_size // 4)

            # --- NEW: Render the class letter on the carried item ---
            text_surf = self.font.render(item_class, True, (0, 0, 0))  # Black text for contrast
            text_rect = text_surf.get_rect(center=agent_rect.center)
            canvas.blit(text_surf, text_rect)
            # --- END OF CHANGE ---

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
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

    env = WorkshopEnv(width=3, height=3, num_tasks_per_episode=10)

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
        env_dqn = Monitor(WorkshopEnv(width=3, height=3, num_tasks_per_episode=5), filename=dqn_log_dir)
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
        model_dqn.learn(total_timesteps=300000)
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
            eval_env = WorkshopEnv(width=3, height=3, num_tasks_per_episode=5)
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