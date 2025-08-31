import os
import gymnasium as gym
import numpy as np
import pandas as pd
import heapq
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

import pygame
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
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


def evaluate_abc_agent(csv_path: str, layout_grid: np.ndarray, io_point: tuple, item_catalog_preset: dict = None,
                       render: bool = False):
    """
    Evaluates the ABC agent, calculates the final picking cost, and returns the cost history.
    """
    title = "ABC Agent (Visual)" if render else "ABC Agent (Heuristic)"
    print(f"\n--- Evaluating {title} ---")

    item_catalog, transactions = prepare_data_from_logs(csv_path)

    if not transactions:
        print("Could not load transaction data for ABC agent.")
        return 0, []

    abc_agent = ABCAgent(layout_matrix=layout_grid, io_point=(io_point[1], io_point[0]),
                         item_catalog=item_catalog if item_catalog_preset is None else item_catalog_preset)

    visualizer = None
    if render:
        visualizer = WarehouseVisualizer(layout_matrix=layout_grid, io_point=(io_point[1], io_point[0]),
                                         zones=abc_agent.zones, item_classes=abc_agent.item_classes)

    occupied = {}
    total_picking_cost = 0
    cost_history = [0]  # Start with cost 0 at transaction 0

    for trans in transactions:
        item_id, quantity = trans['SKU'], trans['Quantity']

        if trans['TransactionType'] == 'WARENEINGANG':
            plan = abc_agent.find_storage_for_batch(item_id, quantity, occupied)
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
                    if occupied[loc]['quantity'] == 0: del occupied[loc]

            for loc in locations_visited_for_this_pick:
                total_picking_cost += abc_agent.travel_times.get(loc, 0)

        cost_history.append(total_picking_cost)  # Append cumulative cost after each transaction

        if render and visualizer:
            visualizer.draw(occupied, f"Action: {trans['TransactionType']}", total_picking_cost)
            pygame.time.wait(50)

    if render and visualizer:
        visualizer.draw(occupied, f"ENDE! Finale Kosten: {total_picking_cost}", total_picking_cost)
        pygame.time.wait(3000)
        pygame.quit()

    print(f"Final Picking Cost: {total_picking_cost:.2f}")
    return total_picking_cost, cost_history


def evaluate_model(model_path: str, csv_path: str, layout_grid: np.ndarray, io_point: tuple, num_episodes: int = 10):
    print(f"\n--- Evaluating PPO Agent: {os.path.basename(model_path)} ---")
    eval_env = WarehouseEnv(csv_path=csv_path, layout_grid=layout_grid, io_point=io_point)
    model = PPO.load(model_path, env=eval_env)

    total_rewards = []
    cost_history_for_plot = None

    for i in range(num_episodes):
        episode_reward, history = run_episode(eval_env, model)
        total_rewards.append(episode_reward)
        if i == 0:  # Store history from the first episode for plotting
            cost_history_for_plot = history

    eval_env.close()
    avg_cost = -np.mean(total_rewards)
    std_cost = np.std(total_rewards)
    print(f"Average Picking Cost: {avg_cost:.2f} +/- {std_cost:.2f}")
    return avg_cost, cost_history_for_plot


def evaluate_random_agent(csv_path: str, layout_grid: np.ndarray, io_point: tuple, num_episodes: int = 10):
    print("\n--- Evaluating Random Agent (Baseline) ---")
    eval_env = WarehouseEnv(csv_path=csv_path, layout_grid=layout_grid, io_point=io_point)

    total_rewards = []
    cost_history_for_plot = None

    for i in range(num_episodes):
        episode_reward, history = run_episode(eval_env, None)
        total_rewards.append(episode_reward)
        if i == 0:  # Store history from the first episode for plotting
            cost_history_for_plot = history

    eval_env.close()
    avg_cost = -np.mean(total_rewards)
    std_cost = np.std(total_rewards)
    print(f"Average Picking Cost: {avg_cost:.2f} +/- {std_cost:.2f}")
    return avg_cost, cost_history_for_plot


def run_episode(env, model=None):
    obs, _ = env.reset()
    terminated, episode_reward = False, 0

    # Stores tuples of (transaction_index, cumulative_cost)
    cost_milestones = [(0, 0)]

    while not terminated:
        action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        episode_reward += reward
        cost_milestones.append((info['current_step'], -episode_reward))

    # Create a list that maps 1-to-1 with transactions for consistent plotting
    total_transactions = len(env.df)
    cost_history = []
    milestone_idx = 0
    current_cost = 0
    for i in range(total_transactions + 1):
        if milestone_idx < len(cost_milestones) and i >= cost_milestones[milestone_idx][0]:
            current_cost = cost_milestones[milestone_idx][1]
            milestone_idx += 1
        cost_history.append(current_cost)

    return episode_reward, cost_history


def plot_comparison_graph(results: Dict[str, List[float]]):
    """
    Plots the cumulative picking cost of different agents over the course of all transactions.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))

    for agent_name, cost_history in results.items():
        # Ensure the list is not empty before plotting
        if cost_history:
            plt.plot(cost_history, label=agent_name, linewidth=2.5)

    plt.title('Agent Performance Comparison: Cumulative Picking Cost', fontsize=16)
    plt.xlabel('Transaction Number', fontsize=12)
    plt.ylabel('Cumulative Picking Cost', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

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
    # Evaluate the PPO (RL) Agent
    ppo_cost, ppo_history = evaluate_model(
        model_path, csv_path, layout_grid, io_point, num_episodes=10
    )

    # Evaluate the ABC Agent (set render=False for data collection)
    abc_cost, abc_history = evaluate_abc_agent(
        csv_path, layout_grid, io_point, render=False
    )

    # Evaluate the Random Agent
    random_cost, random_history = evaluate_random_agent(
        csv_path, layout_grid, io_point, num_episodes=10
    )

    print("\n" + "=" * 50 + "\n--- Summary of Average Costs ---\n" + "=" * 50)
    print(f"PPO Agent:      {ppo_cost:.2f}")
    print(f"ABC Agent:      {abc_cost:.2f}")
    print(f"Random Agent:   {random_cost:.2f}\n" + "=" * 50)

    if ppo_cost > 0:
        abc_change = ((abc_cost - ppo_cost) / ppo_cost) * 100
        random_change = ((random_cost - ppo_cost) / ppo_cost) * 100
        print("\n--- Performance vs. PPO Agent ---")
        print(f"ABC Agent cost is {abc_change:+.2f}% compared to PPO.")
        print(f"Random Agent cost is {random_change:+.2f}% compared to PPO.")
        print("(A positive percentage means higher/worse cost)")

    print("=" * 50)

    # Plot the results from a single, representative run of each agent
    plot_data = {
        'RL (PPO) Agent': ppo_history,
        'Heuristic (ABC) Agent': abc_history,
        'Random Agent': random_history,
    }
    plot_comparison_graph(plot_data)

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


def plot_popularity_comparison(original_csv_path: str, changed_csv_path: str, filename="popularity_comparison.png"):
    """
    Erstellt ein horizontales Balkendiagramm, das die Artikelpopularität
    (Anzahl der Entnahmen) zwischen zwei Datensätzen vergleicht.
    """
    # 1. Daten laden
    df_orig = pd.read_csv(original_csv_path)
    df_changed = pd.read_csv(changed_csv_path)

    # 2. Popularitäten für beide Szenarien berechnen
    pop_orig = df_orig[df_orig['TransactionType'] == 'MATERIALENTNAHME']['SKU'].value_counts().rename('Original')
    pop_changed = df_changed[df_changed['TransactionType'] == 'MATERIALENTNAHME']['SKU'].value_counts().rename(
        'Geändert')

    # 3. Daten für den Plot zusammenführen und sortieren
    df_comparison = pd.concat([pop_orig, pop_changed], axis=1).fillna(0)
    df_comparison = df_comparison.sort_values(by='Original', ascending=True)

    # 4. Diagramm erstellen
    ax = df_comparison.plot(kind='barh', figsize=(12, 10), width=0.8,
                            title="Vergleich der Artikelpopularität (Anzahl Entnahmen)")

    ax.set_xlabel("Anzahl der Entnahmen (höher ist populärer)")
    ax.set_ylabel("Artikel (SKU)")
    ax.legend(title="Szenario")
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Werte an die Balken schreiben für bessere Lesbarkeit
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fontsize=9, padding=3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n📊 Vergleichsdiagramm der Popularitäten gespeichert unter: {filename}")
    plt.close()

def plot_popularity_change_impact(results: dict, filename="popularity_impact.png"):
    """
    Creates a grouped bar chart showing the impact of changed popularity on agent performance.
    """
    labels = list(results.keys())
    costs = [res[0] for res in results.values()]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(labels, costs, color=['#d62728', '#9467bd', '#1f77b4'])

    ax.set_ylabel('Average Total Picking Cost on NEW Data')
    ax.set_title('Agent Adaptability to Changed Item Popularity')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha='center')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 5, f'{yval:.2f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved popularity impact plot to {filename}")
    plt.close()


def create_swapped_popularity_csv(original_csv_path: str, new_csv_path: str = "werkstattlager_logs_changed.csv"):
    """
    Creates a new CSV file by swapping the most and least popular SKUs from the original file.

    This simulates a drastic shift in product demand to test agent adaptability.
    """
    print(f"\n🔄 Creating a new dataset with swapped item popularities...")
    df = pd.read_csv(original_csv_path)

    # Calculate popularity based on picking transactions
    demand_df = df[df['TransactionType'] == 'MATERIALENTNAHME']
    popularity_counts = demand_df['SKU'].value_counts()

    if len(popularity_counts) < 2:
        print("Not enough unique SKUs to perform a swap. Aborting.")
        return original_csv_path

    # Identify the most and least popular items
    most_popular_sku = popularity_counts.index[0]
    least_popular_sku = popularity_counts.index[-1]
    print(f"Swapping most popular SKU '{most_popular_sku}' with least popular SKU '{least_popular_sku}'.")

    # Create a mapping for the swap
    swap_map = {
        most_popular_sku: least_popular_sku,
        least_popular_sku: most_popular_sku
    }

    # Apply the swap to the entire dataframe
    df['SKU'] = df['SKU'].replace(swap_map)

    # Save the new dataframe to a csv
    df.to_csv(new_csv_path, index=False)
    print(f"✅ New dataset saved to '{new_csv_path}'")
    return new_csv_path


def plot_adaptability_slopegraph(results: dict, filename="adaptability_slopegraph.png"):
    """
    Creates a Slopegraph to visualize the performance change of agents
    when item popularities are swapped.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define the two scenarios for the x-axis
    scenarios = ['Original Popularity', 'Changed Popularity']
    x_coords = [0, 1]
    colors = {'RL Agent': '#1f77b4', 'ABC Agent': '#ff7f0e', 'Random Agent': '#2ca02c'}

    # Plot lines and points for each agent
    for agent, data in results.items():
        original_cost = data.get('original', 0)
        changed_cost = data.get('changed', 0)

        # Plot the line connecting the two points
        ax.plot(x_coords, [original_cost, changed_cost],
                marker='o', markersize=8, label=agent,
                color=colors.get(agent, 'gray'), linewidth=2.5)

        # Add text labels for the costs
        ax.text(-0.05, original_cost, f'{original_cost:.0f}', ha='right', va='center', fontsize=10, weight='bold')
        ax.text(1.05, changed_cost, f'{changed_cost:.0f}', ha='left', va='center', fontsize=10, weight='bold')

    # --- Formatting the plot ---
    ax.set_title('Agent Adaptability to Popularity Change', fontsize=16, pad=20)
    ax.set_ylabel('Average Total Picking Cost (Lower is Better)', fontsize=12)
    ax.set_xticks(x_coords)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.legend(title='Agent Type', loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n📈 Saved adaptability slopegraph to {filename}")
    plt.close()


def run_adaptability_scenario(model_path: str, original_csv: str, changed_csv: str, layout_grid: np.ndarray, io_point: tuple):
    """
    Runs an evaluation scenario to test agent adaptability and plots the results as a slopegraph.
    """
    print("\n" + "=" * 50 + "\nADAPTABILITY SCENARIO\n" + "=" * 50)

    # 1. Create the new dataset with inverted popularities
    changed_csv = create_swapped_popularity_csv(original_csv)

    plot_popularity_comparison(original_csv, changed_csv)

    # 2. Get the "outdated" item catalog from the ORIGINAL data for the ABC agent
    outdated_item_catalog, _ = prepare_data_from_logs(original_csv)

    # --- 3. Evaluate agents on BOTH datasets to see the change ---
    print("\n--- Evaluating agents on ORIGINAL data (Baseline) ---")
    rl_cost_orig, _ = evaluate_model(model_path, original_csv, layout_grid, io_point, num_episodes=10)
    abc_cost_orig, _ = evaluate_abc_agent(original_csv, layout_grid, io_point)
    random_cost_orig, _ = evaluate_random_agent(original_csv, layout_grid, io_point, num_episodes=10)

    print("\n--- Evaluating agents on CHANGED data (Test) ---")
    rl_cost_changed, _ = evaluate_model(model_path, changed_csv, layout_grid, io_point, num_episodes=10)
    abc_cost_changed, _ = evaluate_abc_agent(changed_csv, layout_grid, io_point,
                                             item_catalog_preset=outdated_item_catalog)
    random_cost_changed, _ = evaluate_random_agent(changed_csv, layout_grid, io_point, num_episodes=10)

    # 4. Collate results in a structure suitable for the slopegraph
    results = {
        "RL Agent": {"original": rl_cost_orig, "changed": rl_cost_changed},
        "ABC Agent": {"original": abc_cost_orig, "changed": abc_cost_changed},
        "Random Agent": {"original": random_cost_orig, "changed": random_cost_changed}
    }

    # --- 5. Berechnen und ausgeben der prozentualen Veränderung für jeden Agenten (NEU) ---
    print("\n" + "=" * 50 + "\n--- Kostensteigerung durch Popularitätsänderung ---\n" + "=" * 50)

    if rl_cost_orig > 0:
        rl_change = ((rl_cost_changed - rl_cost_orig) / rl_cost_orig) * 100
        print(f"RL Agent Kostensteigerung:       {rl_change:+.2f}%")

    if abc_cost_orig > 0:
        abc_change = ((abc_cost_changed - abc_cost_orig) / abc_cost_orig) * 100
        print(f"ABC Agent Kostensteigerung:      {abc_change:+.2f}%")

    if random_cost_orig > 0:
        random_change = ((random_cost_changed - random_cost_orig) / random_cost_orig) * 100
        print(f"Zufälliger Agent Kostensteigerung: {random_change:+.2f}%")

    print("\n(Ein höherer Prozentsatz bedeutet eine geringere Anpassungsfähigkeit)")
    print("=" * 50)

    # Plot the results using the new slopegraph function
    plot_adaptability_slopegraph(results)

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
    CSV_FILE_ADAPTABILITY = "werkstattlager_logs_changed_pop.csv"

    # Train the PPO Agent
    trained_model_path = "models/final_model_20250830-191215"
    #trained_model_path = "models/final_model_20250831-160916"
    #trained_model_path = train_agent(CSV_FILE, layout, IO_POINT, timesteps=400_000)

    # Run all evaluations
    run_all_evaluations(trained_model_path, CSV_FILE, layout, IO_POINT)
    run_adaptability_scenario(trained_model_path, CSV_FILE, CSV_FILE_ADAPTABILITY, layout, IO_POINT)
