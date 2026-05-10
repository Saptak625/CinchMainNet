import torch
import numpy as np

def obs_to_tensor(obs):
    """
    Convert the observation dictionary to a tensor.
    The observation dictionary contains the following keys:
     - "hand": a binary vector of length 52 indicating which cards are in the player's hand
     - "trick": a normalized integer vector of length 52 indicating which cards have been played in the current trick
     - "cards_played": a float vector (normalized by 1/24) of length 52 indicating which cards have been played in the game
     - "trump": a normalized integer from 0 to 3 indicating the trump suit (0=hearts, 1=diamonds, 2=clubs, 3=spades)
     - "player": a normalized integer from 0 to 3 indicating the player's position (0=first player, 1=second player, 2=third player, 3=fourth player)
     - "void": a flattened binary vector of length 16 indicating which suits the various players are void in based on the cards that have been played (4 suits x 4 players)
     - "trick_pos": a normalized integer from 0 to 3 indicating the player's position in the current trick (0=first to play, 1=second to play, etc.)
     - "lead_suit": a binary vector of length 5 indicating the lead suit of the current trick (one-hot encoding with an extra index for no lead suit)
     - "trick_winner": A binary vector of length 3 indicating the winner of the current trick (0=opposing team, 1=current team, 2=neither).
     - "trick_value": A normalized float indicating the value of the current trick based on the cards played so far. (Indicates whether the trick is currently a junk trick or a high value trick)
     - "can_win_mask": A binary vector of length 52 indicating which legal cards in the player's hand can currently win the trick based on the cards that have been played so far.
     - "trumps_at_start": a normalized integer list of length 4 indicating how many trump cards each player had before drawing from the deck
     - "count_from_deck": a normalized integer list of length 4 indicating how many cards each player has drawn from the deck
     - "count_from_dead_wood": a normalized integer list of length 4 indicating how many cards each player has drawn from the dead wood
     - "count_in_widow": an normalized integer indicating how many cards are currently in the widow (partially important in case certain point trump cards are missing from the game)
    """
    lead = obs["lead_suit"]
    lead_vec = np.zeros(5)
    if lead != -1:
        lead_vec[lead] = 1
    else:
        lead_vec[4] = 1

    trick_winner = obs["trick_winner"]
    trick_winner_vec = np.zeros(3)
    if trick_winner != -1:
        trick_winner_vec[trick_winner] = 1
    else:
        trick_winner_vec[2] = 1

    # print("Len of hand vector:", len(obs["hand"]))
    # print("Len of trick vector:", len(obs["trick"]))
    # print("Len of cards_played vector:", len(obs["cards_played"]))
    # print("Len of trump vector:", len(np.eye(4)[obs["trump"]]))
    # print("Len of player vector:", len(np.eye(4)[obs["player"]]))
    # print("Len of void vector:", len(obs["void"]))
    # print("Len of trick_pos vector:", len(np.array([obs["trick_pos"]])))
    # print("Len of lead_suit vector:", len(lead_vec))
    # print("Len of trumps_at_start vector:", len(np.array(obs["trumps_at_start"])))
    # print("Len of count_from_deck vector:", len(np.array(obs["count_from_deck"])))
    # print("Len of count_from_dead_wood vector:", len(np.array(obs["count_from_dead_wood"])))
    # print("Len of count_in_widow vector:", len(np.array([obs["count_in_widow"]])))

    final_tensor = torch.tensor(np.concatenate([
        obs["hand"],
        obs["trick"] / 3.0,
        obs["cards_played"] / 24.0, # Each of the 4 players has 6 cards.
        np.eye(4)[obs["trump"]] / 3.0,
        np.eye(4)[obs["player"]] / 3.0,
        obs["void"],
        np.array([obs["trick_pos"]]) / 3.0,
        lead_vec,
        trick_winner_vec,
        np.array([obs["trick_value"]]),
        np.array(obs["can_win_mask"]),
        np.array(obs["trumps_at_start"]) / 6.0,
        np.array(obs["count_from_deck"]) / 6.0,
        np.array(obs["count_from_dead_wood"]) / 6.0,
        np.array([obs["count_in_widow"]]) / 5.0 # There can be at most 5 cards in the widow. 52 - 11 drawn (13 of trump in game before) - 36 from initial drawing = 5
    ]), dtype=torch.float32)
    return final_tensor