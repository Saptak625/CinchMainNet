import torch
import numpy as np

from main_env import SUITS, RANKS
from time_it import timeit

@timeit
def build_transformer_obs(obs):
    """
    Converts the raw observation from the environment into a structured format suitable for input to the transformer model.
    
    The output is a dictionary with two keys:
    - "cards": A tensor of shape (52, feature_dim) containing features for each card in the deck.
        Each card's features include:
        - Suit (0-3)
        - Rank (0-12)
        - Location (0=unknown, 1=in hand, 2=in current trick, 3=already played)
        - Played by (0=played by me, 1=played by player on left, 2=played by partner, 3=played by player on right, 4=not played)
        - Played order (0 if not played, otherwise the order in which it was played)
        - Legal (1 if the card is legal to play, 0 otherwise)
    - "global": A tensor containing global features about the current game state.
        Each feature is encoded as an integer:
        - Trump suit (0-3)
        - Trick position (0-3)
        - Lead suit (0-3, with 4 representing no lead suit)
        - Trick winner (0=opposing team, 1=own team, 2=none yet)
        - Current player (0-3)
        - Trumps at least (4 values representing the minimum number of trumps each player has, normalized by 6)
        - Count from deadwood (4 values representing the number of cards each player has played from deadwood, normalized by 6)
        - Count in widow (number of cards in the widow, normalized by 5)
        - Void (4 values representing whether each player is void in each suit for each player, 0 or 1). 
          A player could be void in a suit, but other players might not know that yet, so this is based on the current player's knowledge of voids.
    """
    cards = []
    player = obs["player"]

    for suit_i, suit in enumerate(SUITS):
        for rank_i, rank in enumerate(RANKS):
            idx = suit_i * 13 + rank_i

            # -------------------------
            # Location encoding
            # 0 = unknown
            # 1 = in hand
            # 2 = in current trick
            # 3 = already played
            # -------------------------
            location = 0
            played_by = 4  # default to not played
            if obs["hand"][idx] == 1:
                location = 1
            elif obs["trick"][idx] > 0:
                location = 2
            elif obs["cards_played"][idx] > 0:
                location = 3
                played_by = int(obs["cards_played_by"][idx] - player) % 4  # 0=played by me, 1=played by player on left, 2=played by partner, 3=played by player on right               

            played_order = obs["cards_played"][idx]

            legal = idx in obs["legal_actions"]

            cards.append([
                suit_i,
                rank_i,
                location,
                played_by,
                played_order,
                int(legal)
            ])

    cards = torch.tensor(cards, dtype=torch.long)

    # Adjust trumps_at_least and count_from_deadwood to be in player order starting from current player
    trumps_at_least = obs["trumps_at_least"][player:] + obs["trumps_at_least"][:player]
    count_from_deadwood = obs["count_from_dead_wood"][player:] + obs["count_from_dead_wood"][:player]
    void = list(obs["void"][player*4:]) + list(obs["void"][:player*4])

    global_features_list = [
        obs["trump"],
        obs["trick_pos"],
        obs["lead_suit"] if obs["lead_suit"] != -1 else 4,
        obs["trick_winner"] if obs["trick_winner"] != -1 else 2,
        player,
    ] + list(trumps_at_least) + list(count_from_deadwood) + [obs["count_in_widow"]] + list(void)
    global_features = torch.tensor(global_features_list, dtype=torch.long)

    return {
        "cards": cards,
        "global": global_features
    }