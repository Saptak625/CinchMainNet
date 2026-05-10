import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from main_env import CinchMainEnv, SUITS, RANKS, card_to_index
from agent import CinchAgent
from obs_utils import obs_to_tensor

DEVICE = "cpu"
DEBUG = True
NUM_TESTS = 1_000
CHECKPOINT_PATH_1 = os.path.join("checkpoints", "ckpt_6300.pt")
CHECKPOINT_PATH_2 = os.path.join("checkpoints", "ckpt_6300.pt")

def load_model(path, env):
    input_size = len(obs_to_tensor(env.reset()))

    model = CinchAgent(input_size).to(DEVICE)

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])

    model.eval()
    return model

def select_action(model, obs, env):
    x = obs_to_tensor(obs).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(x)

    legal = env.legal_actions()

    # mask illegal
    mask = torch.full_like(logits, -1e9)
    mask[legal] = 0.0
    logits = logits + mask

    # Print the probabilities of the legal actions for debugging
    # if DEBUG:
    #     probs = torch.softmax(logits, dim=-1)
    #     legal_probs = {env._index_to_card(a): probs[a].item() for a in legal}
    #     print("Legal action probabilities:", {f"{rank}{suit}": prob for (suit, rank), prob in legal_probs.items()})

    return torch.argmax(logits).item()

def test_agents(model_1, model_2, env, model_1_is_human=False, model_2_is_human=False):
    obs = env.reset(no_reset=True)

    # if DEBUG:
    #     print("Trump Suit:", SUITS[obs["trump"]])
    #     print("Player 0's Hand:", [rank + suit for suit, rank in env.hands[0]])
    #     print("Player 1's Hand:", [rank + suit for suit, rank in env.hands[1]])
    #     print("Player 2's Hand:", [rank + suit for suit, rank in env.hands[2]])
    #     print("Player 3's Hand:", [rank + suit for suit, rank in env.hands[3]])

    # Play out a full round.
    count = 0
    while not env.done:
        legal = env.legal_actions()
        if env.current_player % 2 == 0:
            if model_1_is_human:
                print(f"\nPlayer {env.current_player} legal actions: {[''.join(env._index_to_card(a)[::-1]) for a in legal]}")
                print(f"Trump Suit: {SUITS[obs['trump']]}")
                action = input(f"Player {env.current_player}, enter your action RS: ").strip().upper()
                action = card_to_index((action[-1], action[:-1]))
                # Check what the agent would have done for debugging
                agent_action = select_action(model_1, obs, env)
                agent_card = env._index_to_card(agent_action)
                if DEBUG:
                    print(f"Agent would have played {agent_card[1]}{agent_card[0]}")     
            else:
                action = select_action(model_1, obs, env)
        else:
            if model_2_is_human:
                print(f"\nPlayer {env.current_player} legal actions: {[''.join(env._index_to_card(a)[::-1]) for a in legal]}")
                print(f"Trump Suit: {SUITS[obs['trump']]}")
                action = input(f"Player {env.current_player}, enter your action RS: ").strip().upper()
                action = card_to_index((action[-1], action[:-1]))
                # Check what the agent would have done for debugging
                agent_action = select_action(model_2, obs, env)
                agent_card = env._index_to_card(agent_action)
                if DEBUG:
                    print(f"Agent would have played {agent_card[1]}{agent_card[0]}")
            else:
                action = select_action(model_2, obs, env)
        card = env._index_to_card(action)
        if DEBUG:
            print(f"Player {env.current_player} played {card[1]}{card[0]}\n")
        obs, rewards, done, _ = env.step(action)
        count += 1
        # print("\nFinal Rewards:", rewards)
        if count == 4:
            if DEBUG:
                print('=' * 20, "Trick completed.", '=' * 20)
            count = 0
    return env.bet_points

if __name__ == "__main__":
    env = CinchMainEnv()

    model_1 = load_model(CHECKPOINT_PATH_1, env)
    model_2 = load_model(CHECKPOINT_PATH_2, env)

    bet_1, bet_2 = [], []
    total_bet_points = [0, 0]
    wins, losses, ties = 0, 0, 0
    for _ in range(NUM_TESTS):
        copy_env = env.copy()

        print("Count from deck:", env.count_from_deck)
        print("Count from dead wood:", env.count_from_dead_wood)
        print("Cards in widow:", env.count_in_widow)

        bet_points = test_agents(model_1, model_2, env, model_1_is_human=True, model_2_is_human=False)
        print("\nFinal Bet Points:", bet_points)

        print("\n" + "="*50 + "\n")
        if DEBUG:
            print("Player 0's Hand:", [rank + suit for suit, rank in env.starting_hands[0]])
            print("Player 1's Hand:", [rank + suit for suit, rank in env.starting_hands[1]])
            print("Player 2's Hand:", [rank + suit for suit, rank in env.starting_hands[2]])
            print("Player 3's Hand:", [rank + suit for suit, rank in env.starting_hands[3]])

        bet_points_agent = test_agents(model_1, model_2, copy_env, model_1_is_human=False, model_2_is_human=False)
        print("Agent Bet Points:", bet_points_agent)

        bet_1.append(bet_points[0])
        bet_2.append(bet_points[1])
        total_bet_points[0] += bet_points[0]
        total_bet_points[1] += bet_points[1]
        if bet_points[0] > bet_points[1]:
            wins += 1
        elif bet_points[0] < bet_points[1]:
            losses += 1
        else:
            ties += 1

        if DEBUG:
            input('Press Enter to continue to the next game...')
            print("\n" + "="*50 + "\n")
            print("\n" + "="*50 + "\n")
            print("\n" + "="*50 + "\n")
    
        env.reset()

    print("Total bet points for each team:", total_bet_points)
    print(f"Percentages - Model 1 wins: {wins/NUM_TESTS:.2%}, Model 2 wins: {losses/NUM_TESTS:.2%}, Ties: {ties/NUM_TESTS:.2%}")

    print("\nAverage bet points per game:")
    print(f"Model 1: {sum(bet_1)/len(bet_1):.2f}")
    print(f"Model 2: {sum(bet_2)/len(bet_2):.2f}")

    # Make a histogram of the bet points for each model
    plt.hist(bet_1, alpha=0.5, label=f"Model 1 ({CHECKPOINT_PATH_1})")
    plt.hist(bet_2, alpha=0.5, label=f"Model 2 ({CHECKPOINT_PATH_2})")
    plt.xlabel("Bet Points")
    plt.ylabel("Frequency")
    plt.title("Distribution of Bet Points for Each Model")
    plt.legend(loc=1)
    plt.grid()
    plt.show()