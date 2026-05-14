import os
import random
import time
import math
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict, deque
import numpy as np
import wandb
from tqdm import tqdm, trange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from main_env import CinchMainEnv
from agent import CinchTransformer
from obs_utils import build_transformer_obs
from time_it import timeit, enable_timer, time_start, time_end

# ================================
# Config
# ================================
WANDB_ENABLED = os.getenv("WANDB_API_KEY") is not None
EXPERIMENT_NAME = "transformer_small_batches"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODE_RUN_DEVICE = torch.device("cpu") # Run episodes on CPU to avoid GPU memory issues.

LR = 2.5e-4         # Learning rate
GAMMA = 0.99        # Discount factor for rewards
LAMBDA = 0.95       # GAE lambda
POLICY_CLIP = 0.2   # PPO policy clip parameter
# VALUE_CLIP = 0.2  # PPO value function clip parameter (added to prevent large value updates, similar to policy clipping)
TARGET_KL = 0.02    # Target KL divergence for early stopping in PPO epochs

# ================================
# PPO update
# ================================
PPO_EPOCHS = 4
TARGET_TRANSITIONS = 4800 # Each episode has 24 transitions (6 tricks x 4 players), so this corresponds to 200 episodes per PPO update.
MINIBATCH_SIZE = 600 # Each batch has 4800 transitions, so this corresponds to 4 minibatches per PPO epoch.
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5

MAX_UPDATES = 50_000

LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 50
EVALUATE_INTERVAL = 20

CHECKPOINT_DIR = f"checkpoints_{EXPERIMENT_NAME}"
LOG_DIR = f"runs_{EXPERIMENT_NAME}/cinch"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===============================
# Worker Initialization
# ===============================
def init_worker(model_state_dict):
    """
    Each worker gets:
    - its own environment
    - its own independent model copy

    This prevents wandb parameter tracking conflicts.

    Parameters:
    - model_state_dict: The state dict of the model to load for the worker.
    Returns:
    - worker_env: A new instance of the Cinch environment for the worker.
    - worker_model: A new instance of the CinchTransformer model with the state dict loaded.
    """
    torch.set_num_threads(1)

    worker_env = CinchMainEnv()

    worker_model = CinchTransformer(
        d_model=96,
        nhead=4,
        num_layers=3,
        dim_feedforward=256
    ).to(EPISODE_RUN_DEVICE)

    worker_model.load_state_dict(model_state_dict)
    worker_model.eval()

    return worker_env, worker_model


# ================================
# Run one episode
# ================================
@timeit
def run_episode(env, model):
    """
    Run one episode and collect transitions for PPO.

    Parameters:
    - env: The Cinch environment
    - model: The policy/value network
    
    Returns:
    - memory: A dictionary mapping player index to a list of transitions. Each transition is a dictionary containing:
        - "obs": The observation tensor for the state before taking the action.
        - "action": The action taken by the agent.
        - "log_prob": The log probability of the action under the policy at the time of action selection.
        - "value": The value estimate for the state before taking the action.
        - "reward": The reward received after taking the action (given at the end of the trick).
    - epsode_rewards: A list of total rewards for each player at the end of the episode.
    - bet_points: The bet points for the episode, which can be useful for monitoring trends in bet points over time.
    """
    obs = env.reset()
    done = False

    memory = defaultdict(list)

    # Track total reward per player for logging
    episode_rewards = [0.0] * 4

    players = []
    observations = []
    actions = []
    dists = []
    values = []
    while not done:
        time_start("run_episode_step")
        player = obs["player"]

        input_dict = build_transformer_obs(obs)
        cards = input_dict["cards"].to(EPISODE_RUN_DEVICE)
        global_features = input_dict["global"].to(EPISODE_RUN_DEVICE)

        time_start("model_forward")
        with torch.no_grad():
            logits, value = model(cards, global_features)
        time_end("model_forward")

        time_start("action_selection")
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        time_end("action_selection")

        players.append(player)
        observations.append(input_dict)
        actions.append(action)
        dists.append(dist)
        values.append(value)

        next_obs, rewards, done, _ = env.step(action.item())
        obs = next_obs
        time_end("run_episode_step")

        time_start("end_of_trick_processing")
        # The rewards are given after the end of each trick, so memory is updated after the end of the trick.
        if len(actions) == 4:
            for p, o, a, d, v in zip(players, observations, actions, dists, values):
                step_reward = rewards[p]
                episode_rewards[p] += step_reward

                memory[p].append({
                    "obs": o,
                    "action": a,
                    "log_prob": d.log_prob(a),
                    "value": v.squeeze(),
                    "reward": step_reward
                })

            players = []
            observations = []
            actions = []
            dists = []
            values = []
        time_end("end_of_trick_processing")

    return memory, episode_rewards, env.bet_points


# ================================
# Collect experience in parallel
# ================================
def collect_experience(model_state_dict, n):
    """
    This helper function is used to collect experience in parallel using multiple processes. Each process will run n episodes and return the collected memory, rewards, and bet points.
    
    Parameters:
    - model_state_dict: The state dictionary of the model.
    - n: The number of episodes to run in this worker process. 

    Returns:
    A tuple with 3 lists of length n (number of episodes). The list elements are:
    - memory: A dictionary mapping player index to a list of transitions collected from the episode. Each transition is a dictionary containing:
        - "obs": The observation tensor for the state before taking the action.
        - "action": The action taken by the agent.
        - "log_prob": The log probability of the action under the policy at the time of action selection.
        - "value": The value estimate for the state before taking the action.
        - "return": The return estimate for the state (calculated using GAE).
        - "advantage": The advantage estimate for the state (calculated using GAE).
        - "reward": The reward received after taking the action (given at the end of the trick).
    - rewards: A list of total rewards for each player at the end of the episode.
    - bet_points: A list of total bet points for each player at the end of the episode.
    """
    # This helper function allows the Pool to run run_episode
    env, model = init_worker(model_state_dict)

    memory_list = []
    rewards_list = []
    bet_points_list = []
    for _episode in trange(n, desc="Collecting experience", unit="episode"):
        memory, rewards, bet_points = run_episode(env, model)
        
        # Calculate GAE locally on the worker to keep the main process light
        for p in range(4):
            eps_adv, eps_returns = compute_gae(memory[p])
            for i, step in enumerate(memory[p]):
                step["return"] = eps_returns[i]
                step["advantage"] = eps_adv[i]
        memory_list.append(memory)
        rewards_list.append(rewards)
        bet_points_list.append(bet_points)
    return memory_list, rewards_list, bet_points_list


# ================================
# GAE
# ================================
def compute_gae(steps):
    """
    Compute Generalized Advantage Estimation (GAE) for a list of transitions.
    
    Parameters:
    - steps: A list of transitions for one player. Each transition is a dictionary containing:
        - "reward": The reward received after taking the action.
        - "value": The value estimate for the state before taking the action.
    
    Returns:
    - advantages: A list of advantage estimates for each transition.
    - returns: A list of return estimates for each transition.
    """
    rewards = [s["reward"] for s in steps]
    values = [s["value"].item() for s in steps] + [0]
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t + 1] - values[t]
        gae = delta + GAMMA * LAMBDA * gae
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values[:-1])]

    # print("Rewards:", rewards)
    # print("Advantages:", advantages)
    # print("Returns:", returns)

    return advantages, returns


def update(model, optimizer, memory):
    """
    Perform PPO update using the collected memory.
    
    Parameters:
    - model: The policy/value network to update.
    - optimizer: The optimizer for updating the model.
    - memory: A dictionary mapping player index to a list of transitions collected from episodes. Each transition is a dictionary containing:
        - "obs": The observation tensor for the state before taking the action.
        - "action": The action taken by the agent.
        - "log_prob": The log probability of the action under the policy at the time of action selection.
        - "value": The value estimate for the state before taking the action.
        - "return": The return estimate for the state (calculated using GAE).
        - "advantage": The advantage estimate for the state (calculated using GAE).
        - "reward": The reward received after taking the action (given at the end of the trick).

    Returns:
    - num_updates: The number of PPO updates performed (number of minibatches processed).
    - loss: The average total loss across all PPO epochs and minibatches.
    - policy_loss: The average policy loss across all PPO epochs and minibatches.
    - value_loss: The average value loss across all PPO epochs and minibatches.
    - entropy: The average policy entropy across all PPO epochs and minibatches.
    - kl: The average KL divergence between old and new policies across all PPO epochs and minibatches.
    - clip_fraction: The average fraction of actions that were clipped across all PPO epochs and minibatches.
    - explained_variance: The average explained variance of the value function across all PPO epochs and minibatches.
    """

    # ==========================================
    # Build full PPO batch
    # ==========================================
    obs_batch = []
    action_batch = []
    old_log_probs = []
    # old_values_batch = []
    returns_batch = []
    adv_batch = []

    for p in range(4):
        steps = memory[p]

        if len(steps) == 0:
            continue

        for step in steps:
            obs_batch.append(step["obs"])
            action_batch.append(step["action"])
            old_log_probs.append(step["log_prob"])
            # old_values_batch.append(step["value"])
            returns_batch.append(step["return"])
            adv_batch.append(step["advantage"])

    # ==========================================
    # Convert to tensors
    # ==========================================
    # obs_batch = torch.stack(obs_batch).to(DEVICE)
    cards_batch = torch.stack([o["cards"] for o in obs_batch]).to(DEVICE)
    global_batch = torch.stack([o["global"] for o in obs_batch]).to(DEVICE)

    action_batch = torch.stack(action_batch).to(DEVICE)

    old_log_probs = torch.stack(old_log_probs).detach().to(DEVICE)

    # old_values_batch = torch.stack(old_values_batch).detach().to(DEVICE)

    returns_batch = torch.tensor(
        returns_batch,
        dtype=torch.float32
    ).to(DEVICE)

    adv_batch = torch.tensor(
        adv_batch,
        dtype=torch.float32
    ).to(DEVICE)

    # ==========================================
    # Normalize advantages
    # ==========================================
    adv_batch = (
        adv_batch - adv_batch.mean()
    ) / (
        adv_batch.std() + 1e-8
    )

    dataset_size = len(obs_batch)

    # ==========================================
    # Track metrics
    # ==========================================
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clip_fraction = 0.0
    total_explained_variance = 0.0

    num_updates = 0

    # ==========================================
    # PPO epochs
    # ==========================================
    # Make a tqdm progress bar for PPO minibatches run.
    progress_bar = tqdm(
        total=PPO_EPOCHS * (dataset_size // MINIBATCH_SIZE),
        desc="PPO Update",
        unit="minibatch"
    )
    for epoch in range(PPO_EPOCHS):
        # Shuffle dataset each epoch
        indices = torch.randperm(dataset_size)

        stop_updating = False

        # ======================================
        # Minibatches
        # ======================================
        for start in range(
            0,
            dataset_size,
            MINIBATCH_SIZE
        ):

            end = start + MINIBATCH_SIZE

            mb_idx = indices[start:end]

            # mb_obs = obs_batch[mb_idx]
            mb_cards = cards_batch[mb_idx]

            mb_global = global_batch[mb_idx]

            mb_actions = action_batch[mb_idx]

            mb_old_log_probs = old_log_probs[mb_idx]

            # mb_old_values = old_values_batch[mb_idx]

            mb_returns = returns_batch[mb_idx]

            mb_adv = adv_batch[mb_idx]

            # ==================================
            # Forward pass
            # ==================================
            logits, values = model(
                mb_cards,
                mb_global
            )

            dist = torch.distributions.Categorical(
                logits=logits
            ) # If logits has shape (batch_size, 52), this creates a categorical distribution over 52 actions for each item in the batch.

            log_probs = dist.log_prob(mb_actions)

            entropy = dist.entropy().mean()

            # ==================================
            # PPO ratio
            # ==================================
            log_ratio = log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)

            # Approximate KL divergence for monitoring
            approx_kl = ((ratio - 1) - log_ratio).mean()

            if approx_kl.item() > 1.5 * TARGET_KL:
                stop_updating = True
                break

            clip_fraction = (
                (torch.abs(ratio - 1.0) > POLICY_CLIP)
                .float()
                .mean()
            )

            # ==================================
            # PPO clipped objective
            # ==================================
            surr1 = ratio * mb_adv

            surr2 = torch.clamp(
                ratio,
                1.0 - POLICY_CLIP,
                1.0 + POLICY_CLIP
            ) * mb_adv

            policy_loss = -torch.min(
                surr1,
                surr2
            ).mean()

            # ==================================
            # Value loss
            # ==================================
            # ==================================
            # Clipped value loss
            # ==================================
            # value_pred_clipped = (
            #     mb_old_values
            #     + torch.clamp(
            #         values - mb_old_values,
            #         -VALUE_CLIP,
            #         VALUE_CLIP
            #     )
            # )

            # value_loss_unclipped = (
            #     values - mb_returns
            # ).pow(2)

            # value_loss_clipped = (
            #     value_pred_clipped - mb_returns
            # ).pow(2)

            # value_loss = torch.max(
            #     value_loss_unclipped,
            #     value_loss_clipped
            # ).mean()

            value_loss = (
                mb_returns - values
            ).pow(2).mean()

            returns_var = torch.var(mb_returns)

            if returns_var > 1e-8:
                explained_variance = (
                    1.0
                    - torch.var(mb_returns - values)
                    / returns_var
                )
            else:
                explained_variance = torch.tensor(0.0)

            # ==================================
            # Total loss
            # ==================================
            ENTROPY_COEF = max(0.003, 0.02 * (0.9996 ** update_num))
            loss = (
                policy_loss
                + VALUE_COEF * value_loss
                - ENTROPY_COEF * entropy
            )

            # ==================================
            # Backprop
            # ==================================
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                0.5
            )

            optimizer.step()

            # ==================================
            # Metrics
            # ==================================
            total_loss += loss.item()

            total_policy_loss += policy_loss.item()

            total_value_loss += value_loss.item()

            total_entropy += entropy.item()

            total_kl += approx_kl.item()

            total_clip_fraction += clip_fraction.item()

            total_explained_variance += explained_variance.item()

            num_updates += 1

            progress_bar.update(1)

        if stop_updating:
            break

    # ==========================================
    # Average metrics across minibatches
    # ==========================================
    if num_updates > 0:
        total_loss /= num_updates
        total_policy_loss /= num_updates
        total_value_loss /= num_updates
        total_entropy /= num_updates
        total_kl /= num_updates
        total_clip_fraction /= num_updates
        total_explained_variance /= num_updates

    return (
        num_updates,
        total_loss,
        total_policy_loss,
        total_value_loss,
        total_entropy,
        total_kl,
        total_clip_fraction,
        total_explained_variance
    )


# ================================
# Checkpointing
# ================================
def save_checkpoint(model, optimizer, episode):
    """
    The checkpoint will be saved in the CHECKPOINT_DIR with a filename like "ckpt_{episode}.pt".

    Parameters:
    - model: The policy/value network to save.
    - optimizer: The optimizer to save.
    - episode: The current episode number (used for naming the checkpoint file).
    """
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{episode}.pt")

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode
    }, path)

    print(f"Saved checkpoint: {path}")


# ================================
# Evaluation against random agent
# ================================
def evaluate_vs_random(model, games=500):
    """
    Evaluate the trained model against a random agent.
    
    Parameters:
    - model: The trained policy/value network to evaluate.
    - games: The number of games to play against the random agent for evaluation.
    Returns:
    - A dictionary containing the average bet points for the trained model and a list of bet points for each game.
    """
    env = CinchMainEnv()

    total_bet_points = 0
    bet_points_list = []

    for _ in tqdm(range(games), desc="Evaluating against random agent"):

        obs = env.reset()

        done = False

        while not done:

            current_player = env.current_player

            legal = env.legal_actions()

            # Team 0 uses trained model
            if current_player % 2 == 0:

                input_dict = build_transformer_obs(obs)
                cards = input_dict["cards"].to(DEVICE)
                global_features = input_dict["global"].to(DEVICE)

                with torch.no_grad():
                    logits, _ = model(cards, global_features)

                dist = torch.distributions.Categorical(
                    logits=logits
                )

                action = dist.sample().item()

            else:
                # Random opponent
                action = random.choice(legal)

            obs, rewards, done, _ = env.step(action)

        team0_points = env.bet_points[0]
        team1_points = env.bet_points[1]

        total_bet_points += team0_points
        bet_points_list.append(team0_points)

    return {
        "avg_bet_points": total_bet_points / games,
        "bet_points_list": bet_points_list
    }


# ================================
# Training loop
# ================================
if __name__ == "__main__":
    # enable_timer()
    env = CinchMainEnv()

    input_dict = build_transformer_obs(env.reset())
    cards = input_dict["cards"].to(DEVICE)
    global_features = input_dict["global"].to(DEVICE)

    if WANDB_ENABLED:
        wandb.init(
            project="cinch-training",
            config={
                "lr": LR,
                "ppo_epochs": PPO_EPOCHS,
                "batch_size": MINIBATCH_SIZE
            },
            name=EXPERIMENT_NAME
        )

    model = CinchTransformer(
        d_model=96,
        nhead=4,
        num_layers=3,
        dim_feedforward=256
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    mp.set_start_method("spawn", force=True)

    if WANDB_ENABLED:
        wandb.watch(model, log="all", log_freq=100, log_graph=True)

    writer = SummaryWriter(LOG_DIR)

    reward_diff = deque(maxlen=1000)
    winning_reward = deque(maxlen=1000)
    losing_reward = deque(maxlen=1000)
    bet_points_1 = deque(maxlen=1000)
    bet_points_2 = deque(maxlen=1000)

    print(f"Card size: {cards.size()}")
    print(f"Global features size: {global_features.size()}")
    print(f"Device: {DEVICE}")
    print(f"Episode run device: {EPISODE_RUN_DEVICE}")

    # ==========================================
    # PPO batch settings
    # ==========================================
    update_num = 0
    episode = 0

    while update_num < MAX_UPDATES:
        # ==========================================
        # Multiprocessing pool
        # ==========================================
        batch_collection_start_time = time.time()
        target_episodes = TARGET_TRANSITIONS // 24

        # Each worker must do at least 20 episodes
        episodes_per_worker = 20
        required_workers = math.ceil(target_episodes / episodes_per_worker)
        num_workers = min(20, required_workers)

        # Redistribute episodes evenly
        base_episodes = target_episodes // num_workers
        remainder = target_episodes % num_workers

        worker_episode_counts = [
            base_episodes + (1 if i < remainder else 0)
            for i in range(num_workers)
        ]

        print(f"Using {num_workers} workers")
        print(f"Episodes per worker: {worker_episode_counts}")

        # Make workers by calling init_worker with the latest model parameters
        pool_memory = []
        pool_rewards = []
        pool_bet_points = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            state_dict = model.state_dict()
            # Each worker runs its assigned number of episodes and returns its collected memory, rewards, and bet points
            results = list(executor.map(
                collect_experience,
                [state_dict] * num_workers,
                worker_episode_counts
            ))

            for r in results:
                pool_memory.extend(r[0])
                pool_rewards.extend(r[1])
                pool_bet_points.extend(r[2])


        # ==========================================
        # Collect large PPO batch
        # ==========================================
        batch_memory = defaultdict(list)
        transitions_collected = 0
        for memory, rewards, bet_points in zip(pool_memory, pool_rewards, pool_bet_points):
            # --------------------------------------
            # Merge memory
            # --------------------------------------
            for p in range(4):                
                batch_memory[p].extend(memory[p])

            # --------------------------------------
            # Count transitions
            # --------------------------------------
            transitions_collected += 24 # Each episode has 24 transitions (6 tricks x 4 players)

            # --------------------------------------
            # Logging metrics
            # --------------------------------------
            team0 = rewards[0] + rewards[2]
            team1 = rewards[1] + rewards[3]

            episode_reward = team0 - team1

            reward_diff.append(episode_reward)
            winning_reward.append(max(team0, team1))
            losing_reward.append(min(team0, team1))
            bet_points_1.append(bet_points[0])
            bet_points_2.append(bet_points[1])

            episode += 1
        
        print(f"Collected {transitions_collected} transitions in {time.time() - batch_collection_start_time:.2f} seconds.")

        # ==========================================
        # PPO update
        # ==========================================
        (optimization_steps, loss, p_loss, v_loss, entropy, kl, clip_frac, explained_var) = update(
            model,
            optimizer,
            batch_memory
        )

        update_num += 1

        # ==========================================
        # Reward statistics
        # ==========================================
        avg_winning_reward = np.mean(winning_reward)
        avg_losing_reward = np.mean(losing_reward)
        avg_team_diff = np.mean(reward_diff)
        avg_bet_points_1 = np.mean(bet_points_1)
        avg_bet_points_2 = np.mean(bet_points_2)

        # ===========================================
        # Evaluation against random agent
        # ===========================================
        if update_num % EVALUATE_INTERVAL == 0:
            benchmark = evaluate_vs_random(
                model,
                games=500
            )

            print(
                f"[Benchmark] "
                f"Avg Bet Points: "
                f"{benchmark['avg_bet_points']:.3f} | "
            )

            writer.add_scalar(
                "Benchmark/avg_bet_points",
                benchmark["avg_bet_points"],
                update_num
            )

            if WANDB_ENABLED:
                wandb.log({
                    "benchmark/avg_bet_points":
                        benchmark["avg_bet_points"],
                    "update_step": update_num
                })

                benchmark_hist = wandb.Histogram(
                    np.array(benchmark["bet_points_list"])
                )

                wandb.log({
                    "benchmark/bet_points": benchmark_hist,
                    "update_step": update_num
                })

        # ==========================================
        # Logging
        # ==========================================
        if update_num % LOG_INTERVAL == 0:
            print(
                f"Update {update_num:6d} | "
                f"Episodes {episode:7d} | "
                f"Transitions {transitions_collected:5d} | "
                f"Loss {loss:.3f} | "
                f"P {p_loss:.3f} | "
                f"V {v_loss:.3f} | "
                f"Ent {entropy:.3f} | "
                f"AvgW {avg_winning_reward:.3f} | "
                f"AvgL {avg_losing_reward:.3f} | "
                f"AvgD {avg_team_diff:.3f} | "
                f"Bet1 {avg_bet_points_1:.3f} | "
                f"Bet2 {avg_bet_points_2:.3f}"
            )

            if WANDB_ENABLED:
                wandb.log({
                    "loss/total": loss,
                    "loss/policy": p_loss,
                    "loss/value": v_loss,
                    "entropy": entropy,
                    "ppo/optimization_steps": optimization_steps,
                    "ppo/kl_divergence": kl,
                    "ppo/clip_fraction": clip_frac,
                    "value/explained_variance": explained_var,
                    "update_step": update_num,
                })

            writer.add_scalar(
                "Loss/total",
                loss,
                update_num
            )

            writer.add_scalar(
                "Loss/policy",
                p_loss,
                update_num
            )

            writer.add_scalar(
                "Loss/value",
                v_loss,
                update_num
            )

            writer.add_scalar(
                "Loss/entropy",
                entropy,
                update_num
            )

            writer.add_scalar(
                "PPO/optimization_steps",
                optimization_steps,
                update_num
            )

            writer.add_scalar(
                "PPO/kl_divergence",
                kl,
                update_num
            )

            writer.add_scalar(
                "PPO/clip_fraction",
                clip_frac,
                update_num
            )

            writer.add_scalar(
                "Value/explained_variance",
                explained_var,
                update_num
            )

            writer.add_scalar(
                "Reward/avg_winning",
                avg_winning_reward,
                update_num
            )

            writer.add_scalar(
                "Reward/avg_losing",
                avg_losing_reward,
                update_num
            )

            writer.add_scalar(
                "Reward/avg_diff",
                avg_team_diff,
                update_num
            )

            writer.add_scalar(
                "Reward/avg_bet_points_1",
                avg_bet_points_1,
                update_num
            )

            writer.add_scalar(
                "Reward/avg_bet_points_2",
                avg_bet_points_2,
                update_num
            )

            writer.add_histogram(
                "Bet Points/Model 1",
                np.array(bet_points_1),
                update_num
            )

            writer.add_histogram(
                "Bet Points/Model 2",
                np.array(bet_points_2),
                update_num
            )

            if WANDB_ENABLED:
                # Make a histogram of bet points distribution in wandb
                bet_points_hist_1 = wandb.Histogram(
                    np.array(bet_points_1)
                )
                bet_points_hist_2 = wandb.Histogram(
                    np.array(bet_points_2)
                )
                wandb.log({
                    "Bet Points/Model 1": bet_points_hist_1,
                    "Bet Points/Model 2": bet_points_hist_2,
                    "update_step": update_num
                })

        # ==========================================
        # Checkpoints
        # ==========================================
        if update_num % CHECKPOINT_INTERVAL == 0:

            save_checkpoint(
                model,
                optimizer,
                update_num
            )

    writer.close()