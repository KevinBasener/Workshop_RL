import os
import gymnasium as gym
import numpy as np
import pandas as pd
import heapq
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

import pygame
# Import the pathfinding library
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from ABC import ABCAgent, WarehouseVisualizer


class WarehouseEnv(gym.Env):
    """
    A generalized Gymnasium environment for warehouse optimization. It accepts any
    layout via a grid and uses the 'pathfinding' library for A* distance calculations.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, csv_path: str, layout_grid: np.ndarray, io_point: Tuple[int, int],
                 render_mode: Optional[str] = None):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self._prepare_sku_mappings()

        # --- Layout and Pathfinding Setup ---
        self.grid_matrix = layout_grid.T  # Transpose grid to treat (x, y) intuitively
        self.io_point = io_point

        # --- Dynamic Environment Configuration from Grid ---
        self._map_storable_locations()  # Creates mappings from grid

        self.action_space = gym.spaces.Discrete(self.n_rows)
        self.observation_space = gym.spaces.Dict({
            "racks": gym.spaces.Box(low=0, high=self.max_sku_id, shape=(self.n_locations,), dtype=np.int32),
            "task_sku": gym.spaces.Discrete(self.max_sku_id + 1)
        })

        # Internal state
        self.racks_state = None
        self.racks_quantity_state = None
        self.current_step_index = 0
        self.render_mode = render_mode

    def _prepare_sku_mappings(self):
        unique_skus = sorted(self.df['SKU'].unique())
        self.sku_to_id = {sku: i + 1 for i, sku in enumerate(unique_skus)}
        self.id_to_sku = {i + 1: sku for i, sku in enumerate(unique_skus)}
        self.max_sku_id = len(unique_skus)

    def _map_storable_locations(self):
        """
        Scans the grid to find all storable locations (value=1) and creates
        helper mappings for quick lookups during simulation.
        """
        storable_xs, storable_ys = np.where(self.grid_matrix == 1)

        self.storage_rows = sorted(list(np.unique(storable_ys)))
        self.n_rows = len(self.storage_rows)  # This is now the basis for the action space
        self.action_to_y_map = {action: y for action, y in enumerate(self.storage_rows)}

        self.storable_coords: List[Tuple[int, int]] = []
        self.coord_to_idx_map: Dict[Tuple[int, int], int] = {}
        self.row_to_indices_map: Dict[int, List[int]] = {y: [] for y in self.storage_rows}

        for idx, (x, y) in enumerate(zip(storable_xs, storable_ys)):
            coord = (x, y)
            self.storable_coords.append(coord)
            self.coord_to_idx_map[coord] = idx
            if y in self.row_to_indices_map:
                self.row_to_indices_map[y].append(idx)

        self.n_locations = len(self.storable_coords)

    def _calculate_distance(self, item_coord_idx: int) -> float:
        """
        Calculates the direct Manhattan distance from the I/O point to an item's coordinates,
        ignoring any obstacles.
        """
        item_coords = self.storable_coords[item_coord_idx]
        item_x, item_y = item_coords
        io_x, io_y = self.io_point

        return float(abs(item_x - io_x) + abs(item_y - io_y))

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> (Dict, Dict):
        super().reset(seed=seed)
        self.racks_state = np.zeros(self.n_locations, dtype=np.int32)
        self.racks_quantity_state = np.zeros(self.n_locations, dtype=np.int32)
        self.current_step_index = 0
        return self._get_observation(), self._get_info()

    def step(self, action: int) -> (Dict, float, bool, bool, Dict):
        total_reward = 0.0
        while self.current_step_index < len(self.df):
            task = self.df.loc[self.current_step_index]
            if task['TransactionType'] == 'MATERIALENTNAHME':
                sku_to_pick_id = self.sku_to_id[task['SKU']]
                quantity_to_pick = abs(task['Quantity'])
                locations_indices = np.where(self.racks_state == sku_to_pick_id)[0]
                if len(locations_indices) > 0:
                    locations_indices = sorted(locations_indices, key=lambda idx: self._calculate_distance(idx))
                    visited_locations_for_this_pick = set()
                    for loc_idx in locations_indices:
                        if quantity_to_pick == 0: break

                        if loc_idx not in visited_locations_for_this_pick:
                            distance = self._calculate_distance(loc_idx)
                            total_reward -= distance
                            visited_locations_for_this_pick.add(loc_idx)

                        available_qty = self.racks_quantity_state[loc_idx]
                        pick_qty = min(quantity_to_pick, available_qty)

                        self.racks_quantity_state[loc_idx] -= pick_qty
                        quantity_to_pick -= pick_qty

                        if self.racks_quantity_state[loc_idx] <= 0:
                            self.racks_state[loc_idx] = 0
                self.current_step_index += 1
            else:
                break
        if self.current_step_index >= len(self.df):
            return self._get_observation(), total_reward, True, False, self._get_info()

        current_task = self.df.loc[self.current_step_index]
        sku_to_place_id = self.sku_to_id[current_task['SKU']]
        quantity_to_place = current_task['Quantity']

        placed = False
        for row_offset in range(self.n_rows):
            # The action from the agent is an index from 0 to n_rows-1
            current_action_index = (action + row_offset) % self.n_rows

            # Use the map to get the physical y-coordinate for this action
            y_coord_to_check = self.action_to_y_map[current_action_index]

            # Get all storable location indices for that physical row
            loc_indices_in_row = self.row_to_indices_map.get(y_coord_to_check, [])

            existing_stack_idx = next((idx for idx in loc_indices_in_row if self.racks_state[idx] == sku_to_place_id),
                                      -1)
            if existing_stack_idx != -1:
                self.racks_quantity_state[existing_stack_idx] += quantity_to_place
                placed = True
                break

            empty_slot_idx = next((idx for idx in loc_indices_in_row if self.racks_state[idx] == 0), -1)
            if empty_slot_idx != -1:
                self.racks_state[empty_slot_idx] = sku_to_place_id
                self.racks_quantity_state[empty_slot_idx] = quantity_to_place
                placed = True
                break

        if not placed:
            info = self._get_info();
            info['error'] = 'Warehouse is full.'
            return self._get_observation(), total_reward, True, False, info

        self.current_step_index += 1
        terminated = self.current_step_index >= len(self.df)
        if self.render_mode == "human": self.render()
        return self._get_observation(), total_reward, terminated, False, self._get_info()

    def _get_observation(self) -> Dict[str, Any]:
        task_sku_id = 0
        search_idx = self.current_step_index
        while search_idx < len(self.df):
            if self.df.loc[search_idx, 'TransactionType'] == 'WARENEINGANG':
                task_sku_id = self.sku_to_id[self.df.loc[search_idx, 'SKU']];
                break
            search_idx += 1
        return {"racks": self.racks_state.copy(), "task_sku": task_sku_id}

    def _get_info(self) -> Dict[str, Any]:
        return {"current_step": self.current_step_index, "total_steps": len(self.df)}

    def render(self):
        """
        Renders the current state of the warehouse, including SKU IDs and quantities,
        correctly mapping items to their storage locations on the grid.
        """
        # Adjust the header width based on the grid size for a clean look
        header_width = self.grid_matrix.shape[0] * 8
        print("\n" + "=" * header_width)
        print(f"Step: {self.current_step_index}/{len(self.df)} | I/O Point: {self.io_point}")

        # Use a wider cell for better formatting of "SKU:QTY"
        cell_width = 7
        # Create a grid for rendering based on the original layout (transposed back)
        grid_render = self.grid_matrix.T.astype(f'<U{cell_width}')

        # Set default characters for aisles and empty racks
        grid_render[self.grid_matrix.T == 0] = '·'.center(cell_width)
        grid_render[self.grid_matrix.T == 1] = '███'.center(cell_width)

        # Populate the grid with items, showing "SKU:QTY"
        for idx, sku_id in enumerate(self.racks_state):
            if sku_id > 0:
                # Get the correct (x, y) coordinate for this item's storage index
                x, y = self.storable_coords[idx]
                qty = self.racks_quantity_state[idx]

                # Format the string as "SKU:QTY"
                item_str = f"{sku_id}:{qty}"

                # Place the formatted string at the correct [y, x] position on the render grid
                grid_render[y, x] = item_str.center(cell_width)

        # Place the I/O point on the grid
        try:
            io_y, io_x = self.io_point[1], self.io_point[0]
            grid_render[io_y, io_x] = 'I/O'.center(cell_width)
        except IndexError:
            # This handles cases where the I/O point is outside the drawn grid
            pass

        print("Warehouse State:")
        # Print the grid row by row
        for row in grid_render:
            print(" ".join(row))
        print("=" * header_width)

    def close(self):
        pass

def prepare_data_from_logs(filename="werkstattlager_logs.csv"):
    df = pd.read_csv(filename);
    demand_df = df[df['TransactionType'] == 'MATERIALENTNAHME'];
    popularity_counts = demand_df['SKU'].value_counts();
    item_catalog = {sku: {'popularity': count} for sku, count in popularity_counts.items()};
    [item_catalog.update({sku: {'popularity': 0}}) for sku in df['SKU'].unique() if sku not in item_catalog];
    print(item_catalog)
    return item_catalog, df.sort_values(by="Timestamp").to_dict('records')


def evaluate_abc_agent(csv_path: str, layout_grid: np.ndarray, io_point: tuple, item_catalog_preset: dict = None, render: bool = False):
    """
    Evaluates the ABC agent and calculates the final picking cost.
    Optionally visualizes the process.
    """
    title = "ABC Agent (Visual)" if render else "ABC Agent (Heuristic)"
    print(f"\n--- Evaluating {title} ---")

    item_catalog, transactions = prepare_data_from_logs(csv_path)

    if not transactions:
        print("Could not load transaction data for ABC agent.")
        return 0

    # ABC agent uses (y, x) coordinates, so we pass the layout and IO point in that format.
    abc_agent = ABCAgent(layout_matrix=layout_grid, io_point=(io_point[1], io_point[0]), item_catalog=item_catalog if item_catalog_preset is None else item_catalog_preset)

    visualizer = None
    if render:
        visualizer = WarehouseVisualizer(layout_matrix=layout_grid, io_point=(io_point[1], io_point[0]),
                                         zones=abc_agent.zones, item_classes=abc_agent.item_classes)

    occupied = {}
    total_picking_cost = 0

    for trans in transactions:
        item_id, quantity = trans['SKU'], trans['Quantity']

        if trans['TransactionType'] == 'WARENEINGANG':
            # Use the correct method to get the storage plan
            plan = abc_agent.find_storage_for_batch(item_id, quantity, occupied)

            # PUTAWAY COSTS ARE NOT RECORDED FOR A FAIR BENCHMARK
            for step in plan:
                loc, add_qty = step['location'], step['add_quantity']
                if 'new_sku' in step:
                    occupied[loc] = {'sku': item_id, 'quantity': add_qty}
                elif loc in occupied:
                    occupied[loc]['quantity'] += add_qty

        elif trans['TransactionType'] == 'MATERIALENTNAHME':
            quantity_to_pick = abs(quantity)
            locations_of_item = [loc for loc, data in occupied.items() if data['sku'] == item_id]
            locations_of_item.sort(key=lambda l: abc_agent.travel_times[l])

            locations_visited_for_this_pick = set()

            if locations_of_item:
                for loc in locations_of_item:
                    if quantity_to_pick == 0: break
                    locations_visited_for_this_pick.add(loc)

                    available_qty = occupied[loc]['quantity']
                    pick_qty = min(quantity_to_pick, available_qty)

                    occupied[loc]['quantity'] -= pick_qty
                    quantity_to_pick -= pick_qty

                    if occupied[loc]['quantity'] == 0:
                        del occupied[loc]

            # PICKING COSTS ARE RECORDED
            for loc in locations_visited_for_this_pick:
                total_picking_cost += abc_agent.travel_times.get(loc, 0)

        if render and visualizer:
            visualizer.draw(occupied, f"Action: {trans['TransactionType']}", total_picking_cost)
            pygame.time.wait(50)

    if render and visualizer:
        visualizer.draw(occupied, f"ENDE! Finale Kosten: {total_picking_cost}", total_picking_cost)
        pygame.time.wait(3000)
        pygame.quit()

    print(f"Final Picking Cost: {total_picking_cost:.2f}")
    return total_picking_cost

def evaluate_model(model_path: str, csv_path: str, layout_grid: np.ndarray, io_point: tuple, num_episodes: int = 10):
    print(f"\n--- Evaluating PPO Agent: {os.path.basename(model_path)} ---")
    eval_env = WarehouseEnv(csv_path=csv_path, layout_grid=layout_grid, io_point=io_point, render_mode="human")
    model = PPO.load(model_path, env=eval_env)
    total_rewards = [];
    [total_rewards.append(run_episode(eval_env, model)) for _ in range(num_episodes)];
    eval_env.close()
    avg_cost = -np.mean(total_rewards);
    std_cost = np.std(total_rewards);
    print(f"Average Picking Cost: {avg_cost:.2f} +/- {std_cost:.2f}");
    return avg_cost


def evaluate_random_agent(csv_path: str, layout_grid: np.ndarray, io_point: tuple, num_episodes: int = 10):
    print("\n--- Evaluating Random Agent (Baseline) ---")
    eval_env = WarehouseEnv(csv_path=csv_path, layout_grid=layout_grid, io_point=io_point)
    total_rewards = [];
    [total_rewards.append(run_episode(eval_env, None)) for _ in range(num_episodes)];
    eval_env.close()
    avg_cost = -np.mean(total_rewards);
    std_cost = np.std(total_rewards);
    print(f"Average Picking Cost: {avg_cost:.2f} +/- {std_cost:.2f}");
    return avg_cost


def run_episode(env, model=None):
    obs, _ = env.reset();
    terminated, episode_reward = False, 0
    while not terminated:
        action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action);
        episode_reward += reward
    return episode_reward


def train_agent(csv_path, layout_grid, io_point, timesteps=200_000):
    """Handles the complete training process for the PPO agent."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOGS_DIR, MODEL_DIR = "logs", "models"
    os.makedirs(LOGS_DIR, exist_ok=True);
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(WarehouseEnv(csv_path=csv_path, layout_grid=layout_grid, io_point=io_point, render_mode="human"))])
    episode_length = len(pd.read_csv(csv_path));
    buffer_size = 2048
    if buffer_size < episode_length: buffer_size = episode_length + (128 - episode_length % 128)
    model = PPO("MultiInputPolicy", env, verbose=0, n_steps=buffer_size, batch_size=64)
    model.set_logger(configure(os.path.join(LOGS_DIR, f"PPO_{timestamp}"), ["stdout", "tensorboard"]))
    print(f"\n🚀 Training PPO Agent for {timesteps} timesteps...");
    model.learn(total_timesteps=timesteps);
    print("✅ Training complete!")
    final_model_path = os.path.join(MODEL_DIR, f"final_model_{timestamp}.zip");
    model.save(final_model_path);
    print(f"📦 Final model saved to {final_model_path}")
    return final_model_path


def run_all_evaluations(model_path, csv_path, layout_grid, io_point):
    """Runs both statistical and visual evaluations for the agents."""
    print("\n" + "=" * 50 + "\nPERFORMANCE EVALUATION (STATISTICAL)\n" + "=" * 50)
    ppo_cost = evaluate_model(model_path, csv_path, layout_grid, io_point, num_episodes=20)

    preset_popularity = {
        'SCHRAUBE-M8x40': {'popularity': 149},
        'KABELBINDER-200mm': {'popularity': 147},
        'KUGELLAGER-6204-2RS': {'popularity': 42},
        'FILTER-LUFT-A45': {'popularity': 38},
        'SICHERUNG-10A': {'popularity': 32},
        'SENSOR-DRUCK-P20': {'popularity': 7},
        'RELAIS-12V-KFZ': {'popularity': 3},
        'BREMSFLUESSIGKEIT-DOT4': {'popularity': 3},
        'MOTOR-KEILRIEMEN-XPA': {'popularity': 1},
        'DICHTUNG-GUMMI-S12': {'popularity': 1}
    }

    abc_cost = evaluate_abc_agent(csv_path, layout_grid, io_point, preset_popularity, render=True)  # This needs to be created

    print("\n--- Summary ---")
    print(f"ABC Agent (Heuristic) Cost:   {abc_cost:.2f}")
    print(f"PPO Agent (Trained) Cost:     {ppo_cost:.2f}")

    print("\n" + "=" * 50 + "\nPERFORMANCE EVALUATION (VISUAL)\n" + "=" * 50)
    evaluate_model(model_path, csv_path, layout_grid, io_point, num_episodes=10)
    evaluate_abc_agent(csv_path, layout_grid, io_point)
    evaluate_random_agent(csv_path, layout_grid, io_point, num_episodes=10)


if __name__ == '__main__':
    layout = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0]
                       ])
    IO_POINT = (0, 4)
    CSV_FILE = 'werkstattlager_logs.csv'

    # Train the PPO Agent
    #trained_model_path = "models/final_model_20250830-183144"
    trained_model_path = train_agent(CSV_FILE, layout, IO_POINT, timesteps=1_000_000)

    # Run all evaluations
    run_all_evaluations(trained_model_path, CSV_FILE, layout, IO_POINT)
