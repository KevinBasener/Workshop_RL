# Umgebung und Modellpfade definieren
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from stable_baselines3 import PPO
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from WarehouseEnv import WarehouseEnv


def random_policy(env):
    """Wählt einen zufälligen freien Platz."""
    # Nutzt die von der Umgebung bereitgestellte Action Mask
    action_mask = env._get_obs()['action_mask']
    valid_actions = np.where(action_mask == 1)[0]
    return np.random.choice(valid_actions)


def greedy_heuristic_policy(env):
    """Wählt den besten freien Platz für den aktuellen Artikel."""
    item_popularity = env.item_catalog[env.item_to_be_stored]['popularity']
    best_loc_idx = -1

    if item_popularity > 70:
        min_dist = float('inf')
        for i, loc in enumerate(env.storage_locations):
            if loc not in env.location_contents:
                dist = env.travel_times[loc]
                if dist < min_dist:
                    min_dist = dist
                    best_loc_idx = i
    if best_loc_idx == -1:
        best_loc_idx = random_policy(env)
    return best_loc_idx


def evaluate_policy(policy_name, env_fn, policy_func=None, model=None, num_episodes=50):
    """Evaluates a given policy (RL model or heuristic)."""
    eval_env = env_fn()
    episode_costs = []
    print(f"\n--- Evaluating Strategy: {policy_name} ---")

    for i in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = policy_func(eval_env)

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

        eval_env.render_to_file(i)
        final_pfs_cost = 0
        for loc, item_id in eval_env.location_contents.items():
            final_pfs_cost += eval_env.item_catalog[item_id]['popularity'] * eval_env.travel_times[loc]

        episode_costs.append(final_pfs_cost)

    mean_cost = np.mean(episode_costs)
    std_cost = np.std(episode_costs)
    print(f"Result for {policy_name}: Average Cost = {mean_cost:.2f} +/- {std_cost:.2f}")
    return mean_cost, std_cost


if __name__ == '__main__':
    log_dir = "ppo_planner_agent_logs"
    model_path = os.path.join(log_dir, "ppo_planner_agent.zip")
    os.makedirs(log_dir, exist_ok=True)

    TRAIN_MODEL = False

    if TRAIN_MODEL:
        print("--- Mode: TRAINING ---")
        train_env = DummyVecEnv([lambda: Monitor(WarehouseEnv(width=9, height=5), log_dir)])

        model = PPO("MlpPolicy", train_env, verbose=1, n_steps=2048, n_epochs=10, tensorboard_log=log_dir)

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
        print(f"Error: Model file not found at '{model_path}'. Please train a model first.")
        exit()

    ppo_cost, _ = evaluate_policy("PPO", lambda: WarehouseEnv(width=9, height=5), model=ppo_model)
    #random_cost, _ = evaluate_policy("Random Strategy", lambda: WarehouseEnv(width=10, height=5),
    #                                 policy_func=random_policy)
    #greedy_cost, _ = evaluate_policy("Greedy Heuristic", lambda: WarehouseEnv(width=10, height=5),
    #                                 policy_func=greedy_heuristic_policy)

    print("\n\n" + "=" * 50)
    print(" Step 2: Generalization Test (Demand Shift)")
    print("=" * 50)

    # base_catalog = WarehouseEnv(width=10, height=5).item_catalog
    # shifted_catalog = {k: v.copy() for k, v in base_catalog.items()}
    # for item_id in shifted_catalog:
    #     shifted_catalog[item_id]['popularity'] = 101 - shifted_catalog[item_id]['popularity']
    #
    # ppo_cost_gen, _ = evaluate_policy("PPO (Generalization)",
    #                                   lambda: WarehouseEnv(width=9, height=5, item_catalog=shifted_catalog),
    #                                   model=ppo_model)
    #random_cost_gen, _ = evaluate_policy("Random Strategy (Generalization)",
    #                                     lambda: WarehouseEnv(width=10, height=5, item_catalog=shifted_catalog),
    #                                     policy_func=random_policy)
    #greedy_cost_gen, _ = evaluate_policy("Greedy Heuristic (Generalization)",
    #                                     lambda: WarehouseEnv(width=10, height=5, item_catalog=shifted_catalog),
    #                                     policy_func=greedy_heuristic_policy)

    print("\n\n" + "=" * 50)
    print(" Final Results Summary")
    print("=" * 50)
    print(f"Base Scenario:")
    #print(f"  - Random: \t\t{random_cost:.2f}")
    #print(f"  - Greedy: \t\t{greedy_cost:.2f}")
    print(f"  - PPO: \t\t\t{ppo_cost:.2f}")
    #print("\nGeneralization Scenario (Demand Shift):")
    #print(f"  - Random: \t\t{random_cost_gen:.2f}")
    #print(f"  - Greedy: \t\t{greedy_cost_gen:.2f}")
    # print(f"  - PPO: \t\t\t{ppo_cost_gen:.2f}")
    # performance_drop = ((ppo_cost_gen - ppo_cost) / ppo_cost) * 100 if ppo_cost > 0 else 0
    #print(f"\nPPO Agent Performance Drop: {performance_drop:.2f}%")
