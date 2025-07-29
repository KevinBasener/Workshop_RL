import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Set a flag to easily turn debugging on or off
DEBUG_MODE = True


def debug_print(message):
    """Helper function to print only if debug mode is on."""
    if DEBUG_MODE:
        print(message)


class WarehouseEnv(gym.Env):
    """
    An environment with extensive debugging to trace the agent's actions
    and the environment's state changes.
    """

    def __init__(self, width=20, height=10, item_catalog=None):
        super(WarehouseEnv, self).__init__()
        debug_print("--- [ENV INIT] ---")
        debug_print(f"Creating warehouse with size {width}x{height}.")

        # --- 1. Physisches Layout ---
        self.width = width
        self.height = height
        self.io_point = (self.width // 2, 0)
        self.layout_matrix = np.ones((self.height, self.width), dtype=int)
        self.layout_matrix[:, self.width // 2] = 0
        self.layout_matrix[self.io_point[1], self.io_point[0]] = 0
        self.storage_locations = self._get_storage_locations()
        self.num_locations = len(self.storage_locations)
        self.travel_times = {loc: self._calculate_travel_time_manhattan(self.io_point, loc) for loc in self.storage_locations}

        # --- 2. Artikel und ihre Eigenschaften ---
        self.num_items = self.num_locations
        self.max_popularity = 100
        if item_catalog is None:
            self.item_catalog = self._generate_item_catalog()
        else:
            self.item_catalog = item_catalog

        # --- RL-Spezifische Definitionen ---
        self.action_space = spaces.Discrete(self.num_locations)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0, high=1, shape=(1 + self.num_locations,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.num_locations,), dtype=np.int8)
        })

        # --- Interner Zustand ---
        self.location_contents = {}
        self.item_to_be_stored = None

        self.render_dir = "ppo_action_mask_episode_layouts"
        os.makedirs(self.render_dir, exist_ok=True)

        debug_print("--- [ENV INIT] Environment created successfully. ---\n")
        self.reset()

    def _get_storage_locations(self):
        locs = []
        for y in range(self.height):
            for x in range(self.width):
                if self.layout_matrix[y, x] == 1:
                    locs.append((x, y))
        return locs

    def _calculate_travel_time_manhattan(self, start, end):
        """Berechnet die Wegstrecke mit Manhattan-Distanz."""
        return abs(start[0] - end[0]) + abs(start[1] - end[1]) + 1

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

    def _get_new_item_to_be_stored(self):
        placed_items = set(self.location_contents.values())
        available_items = [i for i in range(self.num_items) if i not in placed_items]
        if not available_items: return None
        return np.random.choice(available_items)

    def _get_obs(self):
        """Generiert die Beobachtung für den Agenten."""
        debug_print("  [GET_OBS] Generating new observation...")
        if self.item_to_be_stored is None:
            debug_print("  [GET_OBS] Episode finished, returning empty observation and mask.")
            obs_vec = np.zeros(self.observation_space["observation"].shape, dtype=np.float32)
            mask = np.zeros(self.num_locations, dtype=np.int8)
            return {"observation": obs_vec, "action_mask": mask}

        popularity = self.item_catalog[self.item_to_be_stored]['popularity']
        normalized_popularity = popularity / self.max_popularity
        occupancy = np.zeros(self.num_locations, dtype=np.float32)
        action_mask = np.ones(self.num_locations, dtype=np.int8)

        for i, loc in enumerate(self.storage_locations):
            if loc in self.location_contents:
                occupancy[i] = 1.0
                action_mask[i] = 0

        obs_vector = np.concatenate([[normalized_popularity], occupancy]).astype(np.float32)

        debug_print(f"  [GET_OBS] Item to store: ID {self.item_to_be_stored} (Pop: {popularity})")
        debug_print(f"  [GET_OBS] Valid actions (mask): {np.sum(action_mask)} of {self.num_locations}")
        return {"observation": obs_vector, "action_mask": action_mask}

    def render_to_file(self, episode_num):
        """Creates and saves a PNG image of the current warehouse layout
           with a white background and black grid."""
        fig, ax = plt.subplots(figsize=(self.width / 2, self.height / 2))
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)

        # Define colors
        colors = {"io_point": "gold"}
        item_colors = {'A': (220, 20, 60), 'B': (50, 205, 50), 'C': (255, 140, 0)}

        # --- NEW: Draw a white background and black grid lines ---
        # 1. Draw a single white rectangle for the background
        ax.add_patch(plt.Rectangle((-0.5, -0.5), self.width, self.height, facecolor='white'))

        # 2. Draw vertical and horizontal grid lines in black
        for x in range(self.width + 1):
            ax.axvline(x - 0.5, color='black', linewidth=0.5)
        for y in range(self.height + 1):
            ax.axhline(y - 0.5, color='black', linewidth=0.5)
        # --- END OF NEW LOGIC ---

        # Draw placed items on top of the grid
        for loc, item_id in self.location_contents.items():
            x, y = loc
            popularity = self.item_catalog[item_id]['popularity']
            item_class = 'A' if popularity >= 80 else 'B' if popularity >= 20 else 'C'
            item_rect = plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                      facecolor=tuple(c / 255 for c in item_colors[item_class]),
                                      edgecolor='black')
            ax.add_patch(item_rect)
            ax.text(x, y, item_class, ha='center', va='center', color='white', weight='bold', fontsize=8)

        # Highlight the I/O point
        io_rect = plt.Rectangle((self.io_point[0] - 0.5, self.io_point[1] - 0.5), 1, 1, facecolor=colors["io_point"],
                                edgecolor='black')
        ax.add_patch(io_rect)
        ax.text(self.io_point[0], self.io_point[1], 'I/O', ha='center', va='center', color='black', weight='bold')

        ax.set_title(f'Final Layout - Episode {episode_num}')
        ax.set_xticks([]);
        ax.set_yticks([])  # Cleaner look without axis numbers

        filepath = os.path.join(self.render_dir, f'layout_episode_{episode_num}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        debug_print(f"  [RENDER] Saved final layout image to {filepath}")

    def render(self, mode='human'):
        grid_representation = np.full((self.height, self.width), '   ', dtype='<U3')
        for y in range(self.height):
            for x in range(self.width):
                if self.layout_matrix[y, x] == 0: grid_representation[y, x] = ' . '
        for loc, item_id in self.location_contents.items():
            x, y = loc
            popularity = self.item_catalog[item_id]['popularity']
            if popularity >= 80:
                item_class = ' A '
            elif popularity >= 20:
                item_class = ' B '
            else:
                item_class = ' C '
            grid_representation[y, x] = item_class
        grid_representation[self.io_point[1], self.io_point[0]] = 'I/O'
        print("\n" + "=" * (self.width * 3))
        print("Aktueller Lagerzustand:")
        for row in np.flipud(grid_representation): print("".join(row))
        print("=" * (self.width * 3))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        debug_print("\n--- [RESET] ---")
        debug_print("Episode starts. Resetting environment.")
        self.location_contents = {}
        self.item_to_be_stored = self._get_new_item_to_be_stored()
        return self._get_obs(), {}

    def step(self, action):
        """Führt eine Aktion aus. Geht davon aus, dass die Aktion gültig ist."""
        debug_print(f"\n--- [STEP] ---")
        debug_print(f"Agent chose action: {action}")

        chosen_location = self.storage_locations[action]

        # This check should ideally not be triggered if action masking works
        if chosen_location in self.location_contents:
            debug_print(
                f"  [STEP-ERROR] Agent chose an occupied location: {chosen_location}. This shouldn't happen with action masking.")
            reward = -1000.0  # High penalty as a fallback
            terminated = True
            return self._get_obs(), reward, terminated, False, {}

        # 1. Ermittle die Kosten nur für die AKTUELLE Aktion
        item_id_to_place = self.item_to_be_stored
        item_popularity = self.item_catalog[item_id_to_place]['popularity']
        travel_time = self.travel_times[chosen_location]
        action_cost = item_popularity * travel_time
        debug_print(f"  [STEP-COST] Cost for this action (Pop {item_popularity} * Time {travel_time}) = {action_cost}")

        # 2. Die Belohnung ist der negative Wert dieser Kosten
        reward = -action_cost
        debug_print(f"  [STEP-REWARD] Reward = {-action_cost}")

        # Platziere den Artikel
        self.location_contents[chosen_location] = item_id_to_place
        debug_print(f"  [STEP-STATE] Placed item {item_id_to_place} at {chosen_location}.")

        # Prüfen, ob das Lager voll ist
        if len(self.location_contents) == self.num_locations:
            debug_print("  [STEP-TERMINAL] Warehouse is full. Terminating episode.")
            terminated = True
        else:
            self.item_to_be_stored = self._get_new_item_to_be_stored()
            terminated = self.item_to_be_stored is None
            if terminated:
                debug_print("  [STEP-TERMINAL] All items placed. Terminating episode.")

        return self._get_obs(), reward, terminated, False, {}


def evaluate_policy(policy_name, env_fn, policy_func=None, model=None, num_episodes=50):
    """
    Evaluates a given policy and correctly renders the final state of each episode.
    This version works with a raw, non-vectorized environment for clarity.
    """
    episode_costs = []
    print(f"\n--- Evaluating Strategy: {policy_name} ---")

    for i in range(num_episodes):
        # Create a fresh, raw environment for each episode
        eval_env = env_fn()
        obs, info = eval_env.reset()
        done = False

        while not done:
            if model:
                # model.predict works on raw observations too
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = policy_func(eval_env)

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

        # --- CORRECT: Render the final layout AFTER the episode is done ---
        # We call this on the eval_env, which is now in its final, filled state.
        eval_env.render_to_file(i)

        # Calculate final cost from the filled warehouse
        final_pfs_cost = 0
        for loc, item_id in eval_env.location_contents.items():
            final_pfs_cost += eval_env.item_catalog[item_id]['popularity'] * eval_env.travel_times[loc]

        episode_costs.append(final_pfs_cost)
        print(f"Episode {i + 1} finished. Final Cost: {final_pfs_cost:.2f}. Image saved.")

    mean_cost = np.mean(episode_costs)
    std_cost = np.std(episode_costs)
    print(f"Result for {policy_name}: Average Cost = {mean_cost:.2f} +/- {std_cost:.2f}")
    return mean_cost, std_cost


if __name__ == '__main__':
    log_dir = "ppo_action_mask_logs"
    model_path = os.path.join(log_dir, "ppo_action_mask_agent.zip")
    os.makedirs(log_dir, exist_ok=True)

    TRAIN_MODEL = False

    if TRAIN_MODEL:
        print("--- Mode: TRAINING ---")
        # For action masking, the Monitor wrapper should wrap the base environment
        train_env_fn = lambda: Monitor(WarehouseEnv(width=9, height=5), log_dir)
        train_env = DummyVecEnv([train_env_fn])

        # Use "MultiInputPolicy" for Dict observation spaces
        model = PPO("MultiInputPolicy", train_env, verbose=1, n_steps=2048, n_epochs=10, tensorboard_log=log_dir)

        print("--- Starting Training... ---")
        model.learn(total_timesteps=500000)
        model.save(model_path)
        print(f"--- Training Finished. Model saved to {model_path} ---")

    print("\n\n" + "=" * 50)
    print(" Step 1: Performance Comparison in Base Scenario")
    print("=" * 50)

    try:
        ppo_model = PPO.load(model_path)
        print(f"Trained model loaded from '{model_path}'.")
    except FileNotFoundError:
        print(f"Error: Model file not found. Please train first.")
        exit()

    # Evaluate all three strategies
    ppo_cost, _ = evaluate_policy("PPO", lambda: WarehouseEnv(width=9, height=5), model=ppo_model)

    print("\n\n" + "=" * 50)
    print(" Final Results Summary")
    print("=" * 50)
    print(f"  - PPO: \t\t\t{ppo_cost:.2f}")