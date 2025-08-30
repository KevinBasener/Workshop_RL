import os

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class WarehouseEnv(gym.Env):
    """
    A custom Gymnasium environment for optimizing item placement in a warehouse.

    The agent's goal is to choose a row to place an incoming item (`WARENEINGANG`)
    to minimize the travel distance for future item picks (`MATERIALENTNAHME`).

    **Observation Space:** A dictionary with:
        - 'racks': A numpy array of shape (28,) representing the 28 storage
                   locations. Each element is an integer SKU_ID (0 for empty).
        - 'task_sku': An integer SKU_ID for the item to be placed.

    **Action Space:** A discrete space of size 7, where each action corresponds
                   to choosing a row (0-6) for placement.

    **Reward:**
        - A negative reward equal to the Manhattan distance for each pick.
        - 0 reward for placement actions.

    **Termination:**
        - The episode ends when all operations from the CSV file are processed.
        - The episode ends if an item needs to be placed but the warehouse is full.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, csv_path: str, render_mode: Optional[str] = None):
        """
        Initializes the Warehouse environment.

        Args:
            csv_path (str): The file path to the CSV containing warehouse operations.
            render_mode (Optional[str]): The rendering mode ('human' or None).
        """
        super().__init__()

        # Load and process the operations data
        self.df = pd.read_csv(csv_path)
        self._prepare_sku_mappings()

        # Environment constants
        self.n_rows = 7
        self.n_cols_per_rack = 2
        self.n_racks = 2
        self.io_point = (0,0)
        self.n_locations = self.n_rows * self.n_cols_per_rack * self.n_racks

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(self.n_rows)
        self.observation_space = gym.spaces.Dict({
            "racks": gym.spaces.Box(low=0, high=self.max_sku_id, shape=(self.n_locations,), dtype=np.int32),
            "task_sku": gym.spaces.Discrete(self.max_sku_id + 1)
        })

        # Pre-calculate coordinates for reward calculation
        self._initialize_location_coords()

        # Internal state
        self.racks_state = None
        self.racks_quantity_state = None
        self.current_step_index = 0

        self.render_mode = render_mode

    def _prepare_sku_mappings(self):
        """Creates integer mappings for SKU strings."""
        unique_skus = sorted(self.df['SKU'].unique())
        self.sku_to_id = {sku: i + 1 for i, sku in enumerate(unique_skus)}
        self.id_to_sku = {i + 1: sku for i, sku in enumerate(unique_skus)}
        self.max_sku_id = len(unique_skus)

    def _initialize_location_coords(self):
        """
        Assigns physical (x, y) coordinates to each storage location based on the
        new layout with surrounding aisles.
        """
        self.location_coords = {}
        # New Layout Definition:
        # Aisle: x=0
        # Rack A (front/back): x=1, x=2
        # Center Aisle: x=3
        # Rack B (front/back): x=4, x=5
        # Aisle: x=6
        for i in range(self.n_locations):
            rack_idx = i // (self.n_rows * self.n_cols_per_rack)
            loc_in_rack = i % (self.n_rows * self.n_cols_per_rack)
            row_idx = loc_in_rack // self.n_cols_per_rack
            col_idx = loc_in_rack % self.n_cols_per_rack

            y = row_idx + 1
            if rack_idx == 0:  # Rack A
                x = col_idx + 1
            else:  # Rack B
                x = col_idx + 4
            self.location_coords[i] = (x, y)

    def _calculate_distance(self, item_coords: tuple) -> float:
        """
        Calculates the Manhattan distance from the I/O point to an item,
        considering the aisle layout. The worker must travel to the correct
        aisle (x-travel) and then to the correct row (y-travel).
        """
        item_x, item_y = item_coords
        io_x, io_y = self.io_point

        # Y-distance is always the difference in rows
        dist_y = abs(item_y - io_y)

        # X-distance depends on the item's column and which aisle is closer to the I/O point
        if item_x == 1:  # Rack A, front column (accessible from left aisle x=0 or center x=3)
            dist_x = min(abs(0 - io_x), abs(3 - io_x))
        elif item_x == 2:  # Rack A, back column (accessible from center aisle x=3)
            dist_x = abs(3 - io_x)
        elif item_x == 4:  # Rack B, front column (accessible from center aisle x=3)
            dist_x = abs(3 - io_x)
        elif item_x == 5:  # Rack B, back column (accessible from center aisle x=3 or right aisle x=6)
            dist_x = min(abs(3 - io_x), abs(6 - io_x))
        else:  # Should not happen
            dist_x = 0

        return float(dist_x + dist_y)

    def _get_observation(self) -> Dict[str, Any]:
        """
        Constructs the observation, finding the next WARENEINGANG task for the agent
        to act on.
        """
        task_sku_id = 0
        # Search for the next WARENEINGANG from the current position
        search_idx = self.current_step_index
        while search_idx < len(self.df):
            task = self.df.loc[search_idx]
            if task['TransactionType'] == 'WARENEINGANG':
                task_sku_id = self.sku_to_id[task['SKU']]
                break
            search_idx += 1

        return {
            "racks": self.racks_state.copy(),
            "task_sku": task_sku_id
        }

    def _get_info(self) -> Dict[str, Any]:
        """Returns auxiliary info."""
        return {
            "current_step": self.current_step_index,
            "total_steps": len(self.df)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> (Dict, Dict):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        self.racks_state = np.zeros(self.n_locations, dtype=np.int32)
        self.racks_quantity_state = np.zeros(self.n_locations, dtype=np.int32)
        self.current_step_index = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action: int) -> (Dict, float, bool, bool, Dict):
        # 1. --- Process automatic PICK tasks ---
        total_reward = 0.0
        while self.current_step_index < len(self.df):
            task = self.df.loc[self.current_step_index]
            if task['TransactionType'] == 'MATERIALENTNAHME':
                sku_to_pick_id = self.sku_to_id[task['SKU']]
                quantity_to_pick = abs(task['Quantity'])
                locations = np.where(self.racks_state == sku_to_pick_id)[0]

                if len(locations) > 0:
                    loc_idx = locations[0]
                    coords = self.location_coords[loc_idx]
                    distance = self._calculate_distance(coords)
                    total_reward -= distance

                    # Update quantity and check if slot becomes empty
                    self.racks_quantity_state[loc_idx] -= quantity_to_pick
                    if self.racks_quantity_state[loc_idx] <= 0:
                        self.racks_state[loc_idx] = 0  # Mark as empty
                        self.racks_quantity_state[loc_idx] = 0

                self.current_step_index += 1
            else:
                break  # Stop at WARENEINGANG

        if self.current_step_index >= len(self.df):
            return self._get_observation(), total_reward, True, False, self._get_info()

        # 2. --- Execute agent's PLACEMENT action ---
        current_task = self.df.loc[self.current_step_index]
        sku_to_place_id = self.sku_to_id[current_task['SKU']]
        quantity_to_place = current_task['Quantity']

        placed = False
        # **UPDATED PLACEMENT LOGIC**
        # Priority 1: Find existing stack of the same SKU and add to it.
        # Priority 2: Find a new empty slot.
        for row_offset in range(self.n_rows):
            row_to_check = (action + row_offset) % self.n_rows

            # Search for existing stack in this row
            existing_stack_found = False
            for rack_idx in range(self.n_racks):
                for col_idx in range(self.n_cols_per_rack):
                    loc_idx = rack_idx * (
                                self.n_rows * self.n_cols_per_rack) + row_to_check * self.n_cols_per_rack + col_idx
                    if self.racks_state[loc_idx] == sku_to_place_id:
                        self.racks_quantity_state[loc_idx] += quantity_to_place
                        placed = True
                        existing_stack_found = True
                        break
                if existing_stack_found: break

            if placed: break

            # If no existing stack, search for an empty slot in this row
            empty_slot_found = False
            for rack_idx in range(self.n_racks):
                for col_idx in range(self.n_cols_per_rack):
                    loc_idx = rack_idx * (
                                self.n_rows * self.n_cols_per_rack) + row_to_check * self.n_cols_per_rack + col_idx
                    if self.racks_state[loc_idx] == 0:
                        self.racks_state[loc_idx] = sku_to_place_id
                        self.racks_quantity_state[loc_idx] = quantity_to_place
                        placed = True
                        empty_slot_found = True
                        break
                if empty_slot_found: break

            if placed: break

        if not placed:
            info = self._get_info()
            info['error'] = 'Warehouse is full. Could not place item.'
            return self._get_observation(), total_reward, True, False, info

        self.current_step_index += 1

        # 3. --- Prepare return values ---
        terminated = self.current_step_index >= len(self.df)
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human": self.render()
        return observation, total_reward, terminated, False, info

    def render(self):
        """Renders the current state of the warehouse, including quantities."""
        print("-" * 55)
        print(f"Step: {self.current_step_index}/{len(self.df)}")

        # Display next task
        if self.current_step_index < len(self.df):
            task = self.df.loc[self.current_step_index]
            if task['TransactionType'] == 'WARENEINGANG':
                sku_name = task['SKU']
                sku_id = self.sku_to_id[sku_name]
                qty = task['Quantity']
                print(f"Next Task: Place {qty} of '{sku_name}' (ID: {sku_id})")
        else:
            print("All tasks completed.")

        print("\n          Rack A           ||           Rack B")
        print("    (SKU: QTY)           ||      (SKU: QTY)")
        print("-" * 55)
        for r in range(self.n_rows):
            row_str = f"Row {r} | "
            for rack_idx in range(self.n_racks):
                for col_idx in range(self.n_cols_per_rack):
                    loc_idx = rack_idx * (self.n_rows * self.n_cols_per_rack) + r * self.n_cols_per_rack + col_idx
                    sku = self.racks_state[loc_idx]
                    qty = self.racks_quantity_state[loc_idx]

                    val_str = f"{sku:>2}:{qty:<4}" if sku else "__: 0   "
                    row_str += f"[{val_str}] "

                if rack_idx == 0:
                    row_str += " || "
            print(row_str)
        print("-" * 55)

    def close(self):
        pass


def evaluate_model(model_path: str, csv_path: str, num_episodes: int = 10, io_point: tuple = (0, 0)):
    """
    Evaluates a trained model's performance and calculates the average picking cost.

    Args:
        model_path (str): Path to the saved model file.
        csv_path (str): Path to the CSV file with warehouse operations.
        num_episodes (int): The number of episodes to run for evaluation.
        io_point (tuple): The (x, y) coordinates of the I/O point.
    """
    print(f"\n--- Evaluating Trained Model: {os.path.basename(model_path)} ---")

    # Create the evaluation environment (no rendering for speed)
    eval_env = WarehouseEnv(csv_path=csv_path, io_point=io_point)
    model = PPO.load(model_path, env=eval_env)

    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        terminated = False
        episode_reward = 0
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = eval_env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)

    eval_env.close()

    # Calculate costs (cost is the negative of the reward)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_cost = -avg_reward
    std_cost = std_reward

    print(f"Evaluation over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Average Picking Cost: {avg_cost:.2f} +/- {std_cost:.2f}")
    return avg_cost


def evaluate_random_agent(csv_path: str, num_episodes: int = 10, io_point: tuple = (0, 0)):
    """
    Evaluates a random agent as a baseline and calculates its average picking cost.
    """
    print("\n--- Evaluating Random Agent (Baseline) ---")

    eval_env = WarehouseEnv(csv_path=csv_path, io_point=io_point)
    total_rewards = []

    for _ in range(num_episodes):
        _, _ = eval_env.reset()
        terminated = False
        episode_reward = 0
        while not terminated:
            action = eval_env.action_space.sample()  # Choose a random action
            _, reward, terminated, _, _ = eval_env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)

    eval_env.close()

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_cost = -avg_reward
    std_cost = std_reward

    print(f"Evaluation over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Average Picking Cost: {avg_cost:.2f} +/- {std_cost:.2f}")
    return avg_cost


if __name__ == '__main__':
    # --- 1. Set up paths and a unique timestamp for this run ---
    from datetime import datetime
    from stable_baselines3.common.logger import configure

    # Generate a timestamp string like "20250830-145212"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    CSV_FILE = 'werkstattlager_logs.csv'
    LOGS_DIR = "logs"
    MODEL_DIR = "models"

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 2. Instantiate the Environment ---
    df = pd.read_csv(CSV_FILE)
    episode_length = len(df)
    print(f"Detected episode length: {episode_length} steps.")

    # Use a non-rendering environment for faster training
    env = DummyVecEnv([lambda: Monitor(WarehouseEnv(csv_path=CSV_FILE))])

    buffer_size = 2048
    if buffer_size < episode_length:
        buffer_size = episode_length + (128 - episode_length % 128)
        print(f"Adjusting n_steps to {buffer_size} to fit a full episode.")

    # --- 3. Set up the PPO model (without tensorboard_log initially) ---
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=buffer_size,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.05,
        gae_lambda=0.95
    )

    # --- 4. Manually set up a custom, timestamped logger ---
    run_log_path = os.path.join(LOGS_DIR, f"PPO_{timestamp}")
    new_logger = configure(run_log_path, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    print(f"--- Starting new training run ---")
    print(f"Logs will be saved to: {run_log_path}")

    # --- 5. Set up a callback to save timestamped checkpoints ---
    checkpoint_prefix = f"ppo_checkpoint_{timestamp}"
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=MODEL_DIR,
        name_prefix=checkpoint_prefix,
    )

    # --- 6. Train the agent ---
    TRAINING_TIMESTEPS = 200_000
    print(f"\n🚀 Starting training for {TRAINING_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=checkpoint_callback)
    print("✅ Training complete!")

    # --- 7. Save the final timestamped model ---
    final_model_path = os.path.join(MODEL_DIR, f"final_model_{timestamp}.zip")
    model.save(final_model_path)
    print(f"📦 Final model saved to {final_model_path}")

    # --- 5. Evaluate the trained model and a random baseline ---
    print("\n" + "=" * 50)
    print("PERFORMANCE EVALUATION")
    print("=" * 50)

    # Evaluate the model we just trained
    trained_model_cost = evaluate_model(model_path=final_model_path, csv_path=CSV_FILE, num_episodes=20)

    # Evaluate a random agent for comparison
    random_agent_cost = evaluate_random_agent(csv_path=CSV_FILE, num_episodes=20)

    print("\n--- Summary ---")
    print(f"Random Agent Average Cost:   {random_agent_cost:.2f}")
    print(f"Trained Agent Average Cost:  {trained_model_cost:.2f}")

    improvement = ((random_agent_cost - trained_model_cost) / random_agent_cost) * 100
    if improvement > 0:
        print(f"The trained agent is {improvement:.2f}% better than a random agent.")
    else:
        print(f"The trained agent did not perform better than a random agent.")
    print("=" * 50)