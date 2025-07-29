import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Set this to True to see detailed logs of every step ---
DEBUG_MODE = True


def debug_print(message):
    """Helper function to print only if debug mode is on."""
    if DEBUG_MODE:
        print(message)


class WarehouseEnv(gym.Env):
    """
    A simulation environment for the SLAP representing a completely open
    warehouse where every cell is a potential storage spot.
    """

    def __init__(self, width=20, height=10, item_catalog=None):
        super(WarehouseEnv, self).__init__()
        debug_print("--- [ENV INIT] ---")
        debug_print(f"Creating OPEN warehouse with size {width}x{height}.")

        # --- 1. Physisches Layout ---
        self.width = width
        self.height = height
        self.io_point = (self.width // 2, 0)

        # --- NEW: All locations are potential storage spots ---
        self.storage_locations = self._get_storage_locations()
        self.num_locations = len(self.storage_locations)

        # --- NEW: Using Manhattan Distance for travel time ---
        self.travel_times = {loc: self._calculate_travel_time_manhattan(self.io_point, loc) for loc in
                             self.storage_locations}

        # --- 2. Artikel und ihre Eigenschaften ---
        self.num_items = self.num_locations
        self.max_popularity = 100
        self.item_catalog = self._generate_item_catalog() if item_catalog is None else item_catalog

        # --- RL-Spezifische Definitionen ---
        self.action_space = spaces.Discrete(self.num_locations)
        obs_shape = (1 + self.num_locations,)
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

        # --- Interner Zustand der Simulation ---
        self.location_contents = {}
        self.item_to_be_stored = None

        self.render_dir = "episode_layouts"
        os.makedirs(self.render_dir, exist_ok=True)

        self.reset()

    def _get_storage_locations(self):
        """Returns every cell in the grid except the I/O point."""
        locs = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) != self.io_point:
                    locs.append((x, y))
        return locs

    def _calculate_travel_time_manhattan(self, start, end):
        """Calculates travel distance using Manhattan distance."""
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

    def _get_new_item_to_be_stored(self):
        placed_items = set(self.location_contents.values())
        available_items = [i for i in range(self.num_items) if i not in placed_items]
        return np.random.choice(available_items) if available_items else None

    def _get_obs(self):
        if self.item_to_be_stored is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        popularity = self.item_catalog[self.item_to_be_stored]['popularity']
        normalized_popularity = popularity / self.max_popularity
        occupancy = np.zeros(self.num_locations, dtype=np.float32)

        # Create a mapping from location coordinate to its index in the list
        loc_to_idx = {loc: i for i, loc in enumerate(self.storage_locations)}
        for loc in self.location_contents:
            if loc in loc_to_idx:
                occupancy[loc_to_idx[loc]] = 1.0

        return np.concatenate([[normalized_popularity], occupancy]).astype(np.float32)

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        debug_print("\n--- [RESET] ---")

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