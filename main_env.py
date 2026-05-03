import numpy as np
import random

SUITS = ["H", "D", "C", "S"] # Hearts, Diamonds, Clubs, Spades
RANKS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]

def create_deck():
    """
    Creates a standard 52-card deck.
    
    Returns:
        list: A list of tuples representing the cards in the deck.
    """
    return [(s, r) for s in SUITS for r in RANKS]

def card_to_index(card):
    """
    Converts a card tuple to an index from 0 to 51.
    
    Args:
        card (tuple): A tuple representing a card, e.g. ("H", "A").
    
    Returns:
        int: The index of the card in the deck.
    """
    s, r = card
    return SUITS.index(s) * 13 + RANKS.index(r)

class CinchMainEnv:
    def __init__(self):
        """
        Initializes the Cinch environment.
        
        The environment simulates a simplified version of the Cinch card game. 
        It manages the state of the game, including the players' hands, the current trick, and the trump suit. 
        The environment provides methods for resetting the game, taking actions, and determining legal moves.
        """

        self.num_players = 4
        self.reset()

    def reset(self):
        """
        Resets the environment to the initial state for a new episode.
        This method shuffles the deck, deals the cards to the players, and randomly selects a trump suit.
        
        Returns:
            dict: The initial observation of the environment, including the player's hand, the trump suit, and the current trick.
        """

        deck = create_deck()
        random.shuffle(deck)
        initial_hands = [deck[i*9:(i+1)*9] for i in range(4)]
        self.trump = random.choice(SUITS)
        # print(f"Trump suit for this game: {self.trump}")
        # print("Initial hands (before discarding non-trump cards):")
        # for i, hand in enumerate(initial_hands):
            # print(f"Player {i}: {hand}")
        # Discard the all non-trump cards.
        discarded_cards = []
        self.hands = [[] for _ in range(4)]
        self.count_from_deck = []
        self.count_from_dead_wood = []
        start_ind = 36
        for i in range(4):
            hand = initial_hands[i]
            non_trump_cards = [c for c in hand if c[0] != self.trump]
            discarded_cards.extend(non_trump_cards)
            self.hands[i] = [c for c in hand if c[0] == self.trump]
            # Everyone must start with 6 cards.
            if len(self.hands[i]) < 6:
                # Get cards from the remaining deck until we have 6 cards in hand.
                if start_ind + (6 - len(self.hands[i])) >= 52:
                    # Get as many cards as we can from the remaining deck, then shuffle the discarded cards and use them to fill up the rest of the hand.
                    # print(f"Player {i} has only {len(self.hands[i])} trump cards. Dealing from remaining deck and then shuffled discarded cards to fill up hand.")
                    # print(f"Remaining deck before dealing to player {i}: {[c[1] + c[0] for c in deck[start_ind:]]}")
                    self.hands[i].extend(deck[start_ind:])
                    self.count_from_deck.append(len(deck) - start_ind)
                    start_ind = 52
                    random.shuffle(discarded_cards)
                    # print(f"Discarded cards shuffled to deal to player {i}: {[c[1] + c[0] for c in discarded_cards[:6 - len(self.hands[i])]]}")
                    self.count_from_dead_wood.append(6 - len(self.hands[i]))
                    self.hands[i].extend(discarded_cards[:6 - len(self.hands[i])])
                else:
                    # print(f"Player {i} has only {len(self.hands[i])} trump cards. Dealing from remaining deck to fill up hand.")
                    # print(f"Dealing to player {i} from remaining deck: {[c[1] + c[0] for c in deck[start_ind:start_ind + (6 - len(self.hands[i]))]]}")
                    cards_to_deal = (6 - len(self.hands[i]))
                    self.hands[i].extend(deck[start_ind:start_ind + cards_to_deal])
                    self.count_from_deck.append(cards_to_deal)
                    start_ind += cards_to_deal
            elif len(self.hands[i]) > 6:
                # print(f"Player {i} has {len(self.hands[i])} trump cards. Discarding down to 6 cards.")
                self.hands[i] = self.hands[i][:6] # Technically up to the user to decide, but this is such a rare case that it shouldn't matter much.
                discarded_cards.extend(hand[6:])
        self.count_in_widow = 52 - start_ind
        # print("Count from deck:", self.count_from_deck)
        # print("Count from dead wood:", self.count_from_dead_wood)
        # print("Cards in widow:", self.count_in_widow)
        self.starting_hands = [list(hand) for hand in self.hands]  # Keep a copy of the initial hands for reward calculation
        self.cards_played = np.zeros(52)
        self.void = np.zeros((4, 4))  # player x suit
        self.card_num = 0

        self.current_player = 0
        self.trick = []
        self.rewards = [0] * 4
        self.cards_won = [[] for _ in range(4)]
        self.done = False

        return self._get_obs()

    def _get_obs(self):
        """
        Constructs the observation for the current player, including their hand, the trump suit, and the current trick.
        
        Returns:
            dict: A dictionary containing the player's hand as a binary vector, the trump suit as an index, and the current trick as a binary vector.
        """

        hand_vec = np.zeros(52)
        for c in self.hands[self.current_player]:
            hand_vec[card_to_index(c)] = 1

        obs = {
            "hand": hand_vec,
            "trump": SUITS.index(self.trump),
            "trick": self._encode_trick(),
            "player": self.current_player,
            "cards_played": self.cards_played,
            "trick_pos": len(self.trick),
            "lead_suit": SUITS.index(self.trick[0][0]) if self.trick else -1,
            "void": self.void.flatten(),
            "count_from_deck": self.count_from_deck,
            "count_from_dead_wood": self.count_from_dead_wood,
            "count_in_widow": self.count_in_widow
        }
        return obs

    def _encode_trick(self):
        """
        Encodes the current trick as a binary vector indicating which cards have been played in the trick.
        
        Returns:
            np.array: A binary vector of length 52 where each index corresponds to a card, and the value is 1 if the card is in the current trick, otherwise 0.
        """
        vec = np.zeros(52)
        for c in self.trick:
            vec[card_to_index(c)] = 1
        return vec

    def legal_actions(self):
        """
        Determines the legal actions for the current player.
        
        Returns:
            list: A list of indices representing the legal cards that can be played.
        """
        any_of_suit = any(c[0] == self.trick[0][0] for c in self.hands[self.current_player]) if self.trick else False
        if any_of_suit and self.trick:
            lead_suit = self.trick[0][0]
            return [card_to_index(c) for c in self.hands[self.current_player] if c[0] == lead_suit]
        return [card_to_index(c) for c in self.hands[self.current_player]]

    def step(self, action):
        """
        Executes the given action (playing a card) and updates the environment state accordingly.
        
        Args:
            action (int): The index of the card to be played by the current player.
        Returns:
            tuple: A tuple containing the new observation, the rewards for all players, a boolean 
            indicating if the episode is done, and an empty info dictionary.
        """
        card = self._index_to_card(action)

        if self.trick:
            lead_suit = self.trick[0][0]
            if card[0] != lead_suit:
                self.void[self.current_player][SUITS.index(lead_suit)] = 1

        self.hands[self.current_player].remove(card)
        self.trick.append(card)
        self.card_num += 1
        self.cards_played[action] = self.card_num

        if len(self.trick) == 4:
            winner = self._resolve_trick()
            self.current_player = winner
            self.trick = []
        else:
            self.current_player = (self.current_player + 1) % 4

        if all(len(h) == 0 for h in self.hands):
            self.done = True
            self._final_rewards()

        return self._get_obs(), self.rewards, self.done, {}

    def _index_to_card(self, idx):
        """
        Converts an index back to a card tuple.
        
        Args:
            idx (int): The index of the card in the deck.
        Returns:
            tuple: A tuple representing the card, e.g. ("H", "A").
        """
        return (SUITS[idx // 13], RANKS[idx % 13])

    def _resolve_trick(self):
        """
        Determines the winner of the current trick based on the cards played.
        
        Returns:
            int: The index of the player who won the trick.
        """
        any_trump = any(c[0] == self.trump for c in self.trick)
        if any_trump:
            trump_cards = [c for c in self.trick if c[0] == self.trump]
            best = max(trump_cards, key=lambda c: RANKS.index(c[1]))
        else:
            lead_suit = self.trick[0][0]
            lead_cards = [c for c in self.trick if c[0] == lead_suit]
            best = max(lead_cards, key=lambda c: RANKS.index(c[1]))
        winner = (self.current_player + self.trick.index(best) + 1) % 4
        self.cards_won[winner].extend(self.trick)
        self._trick_rewards(winner)
        return winner
    
    def _trick_rewards(self, winner):
        """
        Calculates the rewards for the current trick.
        """
        # Check if any points cards were played in the trick and assign rewards accordingly.
        points_cards = {"10": 10, "J": 1, "Q": 2, "K": 3, "A": 4}
        
        # See if there are any points cards in the trick
        trick_point_cards = [c for c in self.trick if c[1] in points_cards]
        trick_points = [points_cards.get(c[1], 0) for c in self.trick]
        total_points = sum(trick_points)

        # See if the ace, jack, or a 2 of trump was played in the trick
        trump_cards = [c for c in self.trick if c[0] == self.trump and c[1] in ["A", "J", "2"]] # guaranteed point cards

        if len(trump_cards) > 0 or total_points > 0:
            # Assign rewards to the winner and their partner and penalize the opponents if they gave any points to the winner.
            # print(f"Trick won by player {winner} with cards {[c[1] + c[0] for c in self.trick]}. Total points in trick: {total_points}. Point cards in trick: {[c[1] + c[0] for c in trump_cards]}")

            self.rewards[winner] += total_points
            self.rewards[winner] += 20 * len(trump_cards)  # Give extra reward for point cards in trick

            for c in trick_point_cards + trump_cards:
                ind = self.trick.index(c)
                player_who_played = (self.current_player + ind + 1) % 4
                is_partner_of_winner = (player_who_played - winner) % 4 == 2
                if not is_partner_of_winner and player_who_played != winner:
                    # print(f"Player {player_who_played} gave points to player {winner} by playing {c[1] + c[0]}. Penalizing player {player_who_played}.")
                    self.rewards[player_who_played] -= points_cards.get(c[1], 0)
                    self.rewards[player_who_played] -= 20 * (1 if c[1] in ["A", "J", "2"] else 0)  # Penalize for giving trump points
                elif is_partner_of_winner:
                    # print(f"Player {player_who_played} is a partner of player {winner} and played {c[1] + c[0]}")
                    self.rewards[player_who_played] += points_cards.get(c[1], 0)
                    self.rewards[player_who_played] += 20 * (1 if c[1] in ["A", "J", "2"] else 0)  # Reward for giving trump points to partner
        else:
            # Nothing of note was played, so no rewards to anyone for trick.
            return


    def _final_rewards(self):
        """
        Calculates the final reward for the game. Assigns rewards for lowest and game accordingly.
        """
        # print("Cards won by each player:")
        # for i in range(4):
            # print(f"Player {i}: {[c[1] + c[0] for c in self.cards_won[i]]}")
        all_trump_cards = [c for c in self.cards_won[0] + self.cards_won[1] + self.cards_won[2] + self.cards_won[3] if c[0] == self.trump]
        lowest_trump = min(all_trump_cards, key=lambda c: RANKS.index(c[1])) if all_trump_cards else None
        
        # Give 100 points for lowest trump
        ace_of_trump = (self.trump, "A")
        if lowest_trump:
            for i in range(4):
                if lowest_trump in self.cards_won[i]:
                    # print(f"Player {i} has the lowest trump: {lowest_trump}. Awarding 100 points to player {i} and their partner.")
                    self.rewards[i] += 100
                    self.rewards[(i + 2) % 4] += 100

        # Give 100 points for ace of trump
        if ace_of_trump in all_trump_cards:
            for i in range(4):
                if ace_of_trump in self.cards_won[i]:
                    # print(f"Player {i} has the ace of trump: {ace_of_trump}. Awarding 100 points to player {i} and their partner.")
                    self.rewards[i] += 100
                    self.rewards[(i + 2) % 4] += 100

        # Give 100 points for jack of trump
        jack_of_trump = (self.trump, "J")
        if jack_of_trump in all_trump_cards:
            for i in range(4):
                if jack_of_trump in self.cards_won[i]:
                    # print(f"Player {i} has the jack of trump: {jack_of_trump}. Awarding 100 points to player {i} and their partner.")
                    self.rewards[i] += 100
                    self.rewards[(i + 2) % 4] += 100

        # Give 100 points for game (most points in cards won)
        sum_points = [0] * 4
        game_dict = {"10": 10, "J": 1, "Q": 2, "K": 3, "A": 4}
        for i in range(4):
            for c in self.cards_won[i]:
                if c[1] in game_dict:
                    sum_points[i] += game_dict[c[1]]
        # print("Points from cards won by each player:", sum_points)
        points_per_team = [sum_points[0] + sum_points[2], sum_points[1] + sum_points[3]]
        if points_per_team[0] > points_per_team[1]:
            # print("Team 0 (Players 0 and 2) wins the game. Awarding 100 points to players 0 and 2.")
            self.rewards[0] += 100
            self.rewards[2] += 100
        elif points_per_team[1] > points_per_team[0]:
            # print("Team 1 (Players 1 and 3) wins the game. Awarding 100 points to players 1 and 3.")
            self.rewards[1] += 100
            self.rewards[3] += 100

if __name__ == "__main__":
    env = CinchEnv()
    obs = env.reset()

    # print("Initial Observation:", obs)
    # print("Trump Suit:", SUITS[obs["trump"]])
    # print("Player 0's Hand:", [RANKS[i % 13] + SUITS[i // 13] for i in np.where(obs["hand"] == 1)[0]])
    # print("Player 1's Hand:", [rank + suit for suit, rank in env.hands[1]])
    # print("Player 2's Hand:", [rank + suit for suit, rank in env.hands[2]])
    # print("Player 3's Hand:", [rank + suit for suit, rank in env.hands[3]])

    # Play out a full round.
    count = 0
    while not env.done:
        legal = env.legal_actions()
        # print(f"\nPlayer {env.current_player} legal actions: {[''.join(env._index_to_card(a)[::-1]) for a in legal]}")
        action = random.choice(legal)
        card = env._index_to_card(action)
        # print(f"Player {env.current_player} played {card[1]}{card[0]}")
        obs, rewards, done, _ = env.step(action)
        count += 1
        # print("\nFinal Rewards:", rewards)
        if count == 4:
            # print('=' * 20, "Trick completed.", '=' * 20)
            count = 0
