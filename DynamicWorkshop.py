import os

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
    picking tasks with dynamic item popularity.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    # DEBUG: Action mapping for clear print statements
    ACTION_MAP = {0: "MOVE UP", 1: "MOVE DOWN", 2: "MOVE LEFT", 3: "MOVE RIGHT", 4: "PUT"}
    STATIC_ITEM_CATALOG = {
        # --- High Popularity (80-100) ---
        0: {'id': 0, 'popularity': 98},
        1: {'id': 1, 'popularity': 85},
        # --- Medium Popularity (20-79) ---
        2: {'id': 2, 'popularity': 75},
        3: {'id': 3, 'popularity': 52},
        4: {'id': 4, 'popularity': 30},
        # --- Low Popularity (1-19) ---
        5: {'id': 5, 'popularity': 18},
        6: {'id': 6, 'popularity': 15},
        7: {'id': 7, 'popularity': 11},
        8: {'id': 8, 'popularity': 6},
        9: {'id': 9, 'popularity': 2},
    }

    def __init__(self):
        super(DynamicWorkshop, self).__init__()
        # --- Core Properties ---
        self.height = 9
        self.width = 7
        self.io_point = (0, 8)
        self.num_initial_tasks = 15
        self.task_queue = []
        self.layout_matrix = np.zeros((self.height, self.width), dtype=int)
        self.layout_matrix[:, 1:3] = 1
        self.layout_matrix[:, 4:6] = 1
        self.layout_matrix[0, :] = 0
        self.layout_matrix[8, :] = 0
        self.rack_locations = self._get_rack_locations()
        self.num_racks = len(self.rack_locations)
        self.max_popularity = 100.0
        self.num_item_types = len(self.STATIC_ITEM_CATALOG)
        self.popularity_counts = [item['popularity'] for item in self.STATIC_ITEM_CATALOG.values()]
        self.item_catalog = self.STATIC_ITEM_CATALOG

        # --- Agent and State ---
        self.agent_pos = None
        self.agent_inventory = None
        self.rack_contents = {}
        self.current_task = None
        self.accumulated_reward = 0

        # --- RL Definitions ---
        self.travel_times = {loc: self._calculate_travel_time_manhattan(self.io_point, loc) for loc in
                             self.rack_locations}
        self.max_travel_time = max(self.travel_times.values()) if self.travel_times else 1
        # MODIFIED: Action space is now 5 (4 moves + 1 put)
        self.action_space = spaces.Discrete(5)
        obs_size = 2 + 1 + 2 + self.num_racks
        self.observation_space = spaces.Box(
            low=-1, high=max(self.width, self.height, len(self.STATIC_ITEM_CATALOG)), shape=(obs_size,), dtype=np.float32
        )

        # --- Visualization ---
        self.render_mode = "human"
        self.cell_size, self.window, self.clock, self.font = 80, None, None, None
        self.window_size = (self.width * self.cell_size, self.height * self.cell_size)

        print("--- Final Put-Only DynamicWorkshop Environment Initialized ---")
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.io_point
        self.agent_inventory = None
        self.rack_contents = {}
        self.popularity_counts = [item['popularity'] for item in self.STATIC_ITEM_CATALOG.values()]
        self.accumulated_reward = 0
        items_to_place = list(range(self.num_item_types))
        random.shuffle(items_to_place)
        for i, rack_pos in enumerate(self.rack_locations):
            if i < len(items_to_place):
                self.rack_contents[rack_pos] = items_to_place[i]
        self._initialize_task_queue()
        print(f"Queue: {self.task_queue}")
        self.current_task = self._get_next_task_from_queue()
        return self._get_obs(), {}

    def step(self, action):
        reward = -1
        terminated = False

        # MODIFIED: Action routing updated for 5 actions
        if action < 4:  # Move
            reward += self._move_agent(action)
        elif action == 4:  # Put
            reward += self._handle_put_action()

        reward += self.accumulated_reward
        self.accumulated_reward = 0

        print(f"Queue: {self.task_queue}")

        if self.current_task is None:
            self.current_task = self._get_next_task_from_queue()
            if self.current_task is None:
                reward += 500
                terminated = True

        print(f"Inventory: {self.agent_inventory}")
        print(f"Content: {self.rack_contents}")

        return self._get_obs(), reward, terminated, False, {}

    def _get_next_task_from_queue(self):
        while self.task_queue:
            next_task = self.task_queue.pop(0)

            if next_task['type'] == 'PUT':
                self.agent_pos = self.io_point
                self.agent_inventory = next_task["item_id"]
                print(f"New task is PUT, {next_task['item_id']}")
                return next_task

            elif next_task['type'] == 'PICK':
                task_origin = next_task['origin']
                if task_origin in self.rack_contents:
                    self.rack_contents.pop(task_origin)
                    travel_time = self._calculate_travel_time_manhattan(self.io_point, task_origin)
                    retrieval_score = self.max_travel_time - travel_time
                    instant_reward = retrieval_score * 5
                    self.accumulated_reward += instant_reward
                    print(f"PICKED: {next_task['item_id']} from {task_origin}, reward: {instant_reward:.2f}")

        return None

    # REMOVED: _handle_pick_action is no longer necessary

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
                    return placement_score * 200
        return -10

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

    def _generate_item_catalog(self):
        catalog = {}
        for i in range(self.num_item_types):
            rand_val = np.random.rand()
            if rand_val < 0.2:
                popularity = np.random.randint(80, self.max_popularity + 1)
            elif rand_val < 0.5:
                popularity = np.random.randint(20, 80)
            else:
                popularity = np.random.randint(1, 20)
            catalog[i] = {'id': i, 'popularity': popularity}
        return catalog

    def _initialize_task_queue(self):
        """
        Generates a list of random tasks for the episode, ensuring no duplicate pick tasks.
        """
        self.task_queue = []

        # Get a list of locations that can be picked from at the start
        available_for_picking = list(self.rack_contents.keys())
        random.shuffle(available_for_picking)  # Randomize the order

        for _ in range(self.num_initial_tasks):
            # Decide on task type: 50/50 chance, but only if that type is possible
            can_pick = bool(available_for_picking)
            # For the initial queue, a PUT task is always considered possible
            can_put = True

            task_choice = random.random()

            # Generate a PICK task if possible and chosen, OR if putting is not an option (it always is here, but good practice)
            if task_choice < 0.5 and can_pick:
                # Pop the location to ensure it's used only once for a pick task
                rack_pos = available_for_picking.pop()
                item_id = self.rack_contents[rack_pos]
                task = {"type": "PICK", "item_id": item_id, "origin": rack_pos}
                self.task_queue.append(task)

            # Generate a PUT task
            elif can_put:
                item_id = random.randint(0, self.num_item_types - 1)
                task = {"type": "PUT", "item_id": item_id}
                self.task_queue.append(task)

            # If for some reason no task can be generated, stop.
            else:
                break

    def _generate_new_task(self):
        # 50/50 chance of a pick or put task
        if random.random() < 0.5:  # PICK task
            if not self.rack_contents: return None
            rack_pos = random.choice(list(self.rack_contents.keys()))
            item_id = self.rack_contents[rack_pos]
            return {"type": "PICK", "item_id": item_id, "origin": rack_pos}
        else:  # PUT task
            if len(self.rack_contents) >= self.num_racks: return None
            item_id = random.randint(0, self.num_item_types - 1)
            return {"type": "PUT", "item_id": item_id}

    def _move_agent(self, action):
        move_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT for PyGame
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
            return -10

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Dynamic Workshop Warehouse")
        if self.clock is None: self.clock = pygame.time.Clock()
        if self.font is None: self.font = pygame.font.Font(None, 18)

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        colors = {"aisle": (200, 200, 200), "rack": (100, 100, 100), "io_point": (255, 223, 0),
                  "agent": (65, 105, 225)}

        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                color = colors["aisle"] if self.layout_matrix[y, x] == 0 else colors["rack"]
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (220, 220, 220), rect, 1)

        pygame.draw.rect(canvas, colors["io_point"],
                         pygame.Rect(self.io_point[0] * self.cell_size, self.io_point[1] * self.cell_size,
                                     self.cell_size, self.cell_size))

        for rack_pos, item_id in self.rack_contents.items():
            # Get the popularity of the item
            popularity = self.popularity_counts[item_id]

            # Assign color based on popularity thresholds
            if popularity >= 80:
                color = (0, 255, 0)  # Green
            elif 20 <= popularity < 80:
                color = (255, 255, 0)  # Yellow
            else:
                color = (255, 0, 0)  # Red

            # Draw the rack rectangle
            rect = pygame.Rect(rack_pos[0] * self.cell_size, rack_pos[1] * self.cell_size, self.cell_size,
                               self.cell_size)
            pygame.draw.rect(canvas, color, rect.inflate(-8, -8), border_radius=5)

            # Render the item ID
            text = self.font.render(f"{item_id}", True, (255, 255, 255))
            canvas.blit(text, text.get_rect(center=rect.center))

        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size,
                                 self.cell_size, self.cell_size)
        pygame.draw.circle(canvas, colors["agent"], agent_rect.center, self.cell_size // 2 - 5)
        if self.agent_inventory is not None:
            pygame.draw.circle(canvas, (255, 255, 255), agent_rect.center, self.cell_size // 4)

        pop_normalized = self.popularity_counts / np.sum(self.popularity_counts)
        for i in range(self.num_item_types):
            text = self.font.render(f"Item {i}: {pop_normalized[i]:.1%}", True, (0, 0, 0))
            canvas.blit(text, (10, 10 + i * 15))
        if self.current_task:
            task_str = f"Task: {self.current_task['type']} item {self.current_task['item_id']}"
            if self.current_task['type'] == "PICK": task_str += f" from {self.current_task['origin']}"
            text = self.font.render(task_str, True, (200, 0, 0))
            canvas.blit(text, (self.width * self.cell_size - 300, 10))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None: pygame.display.quit(); pygame.quit()


# -------------EVAL---------------

# --- Control Flags ---
TRAIN_MODELS = True
EVALUATE_MODEL = False
PLAY_GAME = False

# --- Base directory for all logs and models ---
base_log_dir = "dqn_workshop_runs"


def get_next_run_number(base_log_dir):
    """Finds the next available run number in a log directory."""
    if not os.path.exists(base_log_dir):
        return 1

    existing_runs = [d for d in os.listdir(base_log_dir) if d.startswith("run_")]
    if not existing_runs:
        return 1

    max_run = 0
    for run in existing_runs:
        try:
            num = int(run.split('_')[1])
            if num > max_run:
                max_run = num
        except (ValueError, IndexError):
            continue

    return max_run + 1


if __name__ == '__main__':

    # --- Training ---
    if TRAIN_MODELS:
        # --- NEW: Automatically find the next run number and create a unique path ---
        run_number = get_next_run_number(base_log_dir)
        log_dir = os.path.join(base_log_dir, f"run_{run_number}")
        model_path = os.path.join(log_dir, "dqn_workshop_agent.zip")
        os.makedirs(log_dir, exist_ok=True)
        print(f"--- Starting new training run #{run_number} ---")

        env_dqn = Monitor(DynamicWorkshop(), filename=log_dir)
        train_env_dqn = DummyVecEnv([lambda: env_dqn])
        policy_kwargs = dict(net_arch=[128, 64, 32])

        model_dqn = DQN(
            "MlpPolicy",
            train_env_dqn,
            verbose=1,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            exploration_fraction=0.9,
            exploration_final_eps=0.63,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs
        )
        print("--- Starting DQN Training... ---")
        model_dqn.learn(total_timesteps=1000000)
        model_dqn.save(model_path)
        print(f"--- DQN Training finished. Model saved to {model_path} ---")

    # --- Evaluation or Game Mode ---
    else:
        if EVALUATE_MODEL:
            # --- Specify which trained model you want to load ---
            run_to_load = 20  # <--- CHANGE THIS NUMBER TO THE RUN YOU WANT TO EVALUATE

            model_path = os.path.join(base_log_dir, f"run_{run_to_load}", "dqn_workshop_agent.zip")

            if not os.path.exists(model_path):
                print(f"Error: Model not found at {model_path}")
                print("Please make sure you have trained a model for that run number.")

            elif EVALUATE_MODEL:
                print(f"--- STARTING EVALUATION OF MODEL: {model_path} ---")

                # 1. Create the evaluation environment
                # We use the same environment class the model was trained on
                eval_env = DynamicWorkshop()

                # 2. Load the trained model
                model = DQN.load(model_path, env=eval_env)

                # 3. Run the evaluation loop
                eval_episodes = 10
                episode_rewards = []

                for episode in range(eval_episodes):
                    obs, _ = eval_env.reset()
                    done = False
                    current_episode_reward = 0

                    while not done:
                        # Use the model to predict the best action (deterministic=True)
                        action, _ = model.predict(obs, deterministic=True)

                        # Take the action in the environment
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        done = terminated or truncated

                        # Add the reward to the total for this episode
                        current_episode_reward += reward

                        # Render the environment to watch the agent
                        eval_env.render()

                    print(f"Episode {episode + 1}/{eval_episodes} | Reward: {current_episode_reward:.2f}")
                    episode_rewards.append(current_episode_reward)

                # 4. Calculate and print summary statistics
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                print("\n--- EVALUATION COMPLETE ---")
                print(f"Average reward over {eval_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

                eval_env.close()
        elif PLAY_GAME:
            run_to_load = 21  # <--- CHANGE THIS NUMBER TO THE RUN YOU WANT TO EVALUATE

            model_path = os.path.join(base_log_dir, f"run_{run_to_load}", "dqn_workshop_agent.zip")

            print(f"--- STARTING GAME MODE WITH MODEL: {model_path} ---")
            game_env = DynamicWorkshop()
            model = DQN.load(model_path, env=game_env)

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

                    # --- Wait for user input to proceed ---
                    action_taken = False
                    while not action_taken:
                        for event in pygame.event.get():
                            # Handle window close
                            if event.type == pygame.QUIT:
                                done = True
                                running = False
                                action_taken = True
                            # Handle key presses
                            if event.type == pygame.KEYDOWN:
                                # Reset episode
                                if event.key == pygame.K_r:
                                    print("\n--- [R] key pressed. Resetting episode. ---")
                                    done = True
                                    action_taken = True
                                # Advance one step
                                if event.key == pygame.K_SPACE:
                                    action_taken = True

                    # If the loop was broken by quitting or resetting, skip the AI step
                    if not running or done:
                        continue

                    # --- AI takes one step ---
                    action, _ = model.predict(obs, deterministic=True)
                    action_int = action.item()  # Get integer from numpy array
                    action_str = game_env.ACTION_MAP.get(action_int, f"UNKNOWN({action_int})")
                    print(f"\nModel chose action: {action_int} [{action_str}]")

                    obs, reward, terminated, truncated, info = game_env.step(action_int)
                    done = terminated or truncated
                    total_reward += reward

                    print(f"  -> Reward for this step: {reward:.2f}")
                    print(f"  -> Total Episode Reward: {total_reward:.2f}")

                    if done:
                        print(f"\n--- ✅ EPISODE FINISHED --- Final Reward: {total_reward:.2f}")

            print("--- Exiting Game Mode. ---")
            game_env.close()