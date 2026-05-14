import torch
import torch.nn as nn

class CinchTransformer(nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512
    ):
        """
        A transformer-based neural network for the Cinch agent.
        
        Args:
            d_model: The dimension of the model (embedding size).
            nhead: The number of attention heads in the transformer.
            num_layers: The number of transformer encoder layers.
            dim_feedforward: The dimension of the feedforward network in the transformer.
            
        The network architecture is as follows:
        - Card feature embeddings:
            - Suit embedding: Embedding layer for the suit of the card (4 suits, embedding size 16)
            - Rank embedding: Embedding layer for the rank of the card (13 ranks, embedding size 32)
            - Location embedding: Embedding layer for the location of the card (hand, trick, played, legal - 4 locations, embedding size 16)
            - Played by embedding: Embedding layer for who played the card (5 values: played by me, player on left, partner, player on right, not played; embedding size 16)
            - Played order: Fully connected layer to embed the order in which the card was played in the trick (normalized by 24, embedding size 16)
            - Legal action embedding: Embedding layer for whether the card is a legal action (2 values, embedding size 8)
            
        - Global feature embeddings:
            - Trump suit embedding: Embedding layer for the trump suit (4 suits, embedding size d_model)
            - Trick size embedding: Embedding layer for the number of cards in the current trick (0-4, embedding size d_model)
            - Lead suit embedding: Embedding layer for the suit led in the current trick (4 suits, embedding size d_model)
            - Trick winner embedding: Embedding layer for the winner of the current trick (3 values, embedding size d_model)
            - Player embedding: Embedding layer for the player (4 players, embedding size d_model)
            - Trumps at least: Fully connected layer to embed the minimum number of trumps each player has (normalized by 6, embedding size d_model)
            - Count from deadwood: Fully connected layer to embed the number of cards each player has played from deadwood (normalized by 6, embedding size d_model)
            - Count in widow: Fully connected layer to embed the number of cards in the widow (normalized by 5, embedding size d_model)
            - Void: Fully connected layer to embed whether each player is void in each suit (4 values, embedding size d_model)

        - Transformer encoder with num_layers layers, d_model dimension, nhead attention heads, and dim_feedforward feedforward dimension.
        - Policy head: A fully connected layer from d_model to 1 to produce the logit for each card.
        - Value head: A fully connected layer from d_model to 1 to produce the state value.
        """
        super().__init__()

        # ============================================================
        # Card embeddings
        # ============================================================
        self.suit_emb = nn.Embedding(4, 16) # 0-3 for the four suits [Hearts, Diamonds, Clubs, Spades]
        self.rank_emb = nn.Embedding(13, 32) # 0-12 for 2-A
        self.location_emb = nn.Embedding(4, 16) # 0 = unknown, 1 = in hand, 2 = in current trick, 3 = already played
        self.played_by_emb = nn.Embedding(5, 16) # 0=played by me, 1=played by player on left, 2=played by partner, 3=played by player on right, 4=not played
        self.play_order_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        ) # Normalize play order by dividing by 24 (max number of cards that can be played before this card)
        self.legal_emb = nn.Embedding(2, 8) # 0 not legal, 1 legal

        # 16 + 32 + 16 + 16 + 16 + 8 = 104
        self.input_proj = nn.Linear(104, d_model) # Project card features to model dimension

        # ============================================================
        # Global embeddings
        # ============================================================
        self.trump_emb = nn.Embedding(4, d_model) # 0-3 for the four suits [Hearts, Diamonds, Clubs, Spades]
        self.trick_size_emb = nn.Embedding(4, d_model) # 0-3 for the number of cards in the current trick
        self.lead_suit_emb = nn.Embedding(5, d_model) # 0-3 for the suit led in the current trick, 4 if no lead suit yet
        self.trick_winner_emb = nn.Embedding(3, d_model) # 0=opposing team won the trick, 1=own team won the trick, 2=no winner yet
        self.player_emb = nn.Embedding(4, d_model) # 0-3 for the player index (relative to who started the round)
        self.trumps_at_least_fc = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        ) # Number of trump cards each player has given what was shown pre-deal (0-6, normalized by 6). Order will always be me, player on left, partner, player on right.
        self.count_from_deadwood_fc = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        ) # Number of cards drawn from deadwood for each player (0-6, normalized by 6). Order will always be me, player on left, partner, player on right.
        self.widow_fc = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        ) # Number of cards in the widow (0-5, normalized by 5)
        self.void_fc = nn.Sequential(
            nn.Linear(16, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        ) # Void (4 values representing whether each player is void in each suit for each player, 0 or 1). 
        # Order will always be me, player on left, partner, player on right, and within each player order is Hearts, Diamonds, Clubs, Spades.

        # ============================================================
        # CLS token
        # ============================================================
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        # ============================================================
        # Positional embeddings
        # ============================================================
        # 53 tokens = 1 CLS + 52 cards
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 53, d_model) * 0.02
        )

        # ============================================================
        # Transformer
        # ============================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ============================================================
        # Policy head
        # ============================================================
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # ============================================================
        # Value head
        # ============================================================
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

         # ============================================================
        # Initialization
        # ============================================================
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, cards, global_features):
        # ============================================================
        # Card features
        # ============================================================
        suit = cards[..., 0]
        rank = cards[..., 1]
        location = cards[..., 2]
        played_by = cards[..., 3]
        played_order = cards[..., 4].float().unsqueeze(-1)
        legal = cards[..., 5]

        x = torch.cat([
            self.suit_emb(suit),
            self.rank_emb(rank),
            self.location_emb(location),
            self.played_by_emb(played_by),
            self.play_order_fc(played_order / 24.0),
            self.legal_emb(legal)
        ], dim=-1)

        x = self.input_proj(x)

        # ============================================================
        # Global features
        # ============================================================
        trump = global_features[..., 0]
        trick_size = global_features[..., 1]
        lead_suit = global_features[..., 2]
        trick_winner = global_features[..., 3]
        current_player = global_features[..., 4]
        trumps_at_least = global_features[..., 5:9]
        count_from_deadwood = global_features[..., 9:13]
        count_in_widow = global_features[..., 13].float().unsqueeze(-1)
        void = global_features[..., 14:30]

        # ============================================================
        # Build global context
        # ============================================================
        global_context = (
            self.trump_emb(trump)
            + self.trick_size_emb(trick_size)
            + self.lead_suit_emb(lead_suit)
            + self.player_emb(current_player)
            + self.trick_winner_emb(trick_winner)
            + self.trumps_at_least_fc(trumps_at_least / 6.0)
            + self.count_from_deadwood_fc(count_from_deadwood / 6.0)
            + self.widow_fc(count_in_widow / 5.0)
            + self.void_fc(void.float())
        )

        # ============================================================
        # Create CLS token
        # ============================================================
        batch_size = x.size(0) if x.dim() == 3 else 1

        pos_embedding = self.pos_embedding
        if batch_size == 1:
            cls_token = self.cls_token.squeeze(0) # Shape: (1, d_model) -> (d_model,)
            pos_embedding = pos_embedding.squeeze(0) # Shape: (1, 53, d_model) -> (53, d_model)
        else:
            cls_token = self.cls_token.expand(
                batch_size,
                -1,
                -1
            )
        cls_token = cls_token + global_context.unsqueeze(-2) # Add global context to CLS token

        # ============================================================
        # Combine sequence
        # ============================================================
        x = torch.cat([
            cls_token,
            x
        ], dim=-2) # Shape: (batch_size, 53, d_model) where the first token is CLS and the rest are cards

        # ============================================================
        # Add positional embeddings
        # ============================================================
        x = x + pos_embedding

        # ============================================================
        # Transformer
        # ============================================================
        x = self.transformer(x)

        # ============================================================
        # Separate CLS and card tokens
        # ============================================================
        if batch_size == 1:
            cls_output = x[0, :] # Shape: (d_model,)
            card_outputs = x[1:, :] # Shape: (52, d_model)
        else:
            cls_output = x[:, 0]
            card_outputs = x[:, 1:]

        # ============================================================
        # Policy logits
        # ============================================================
        logits = self.policy_head(card_outputs).squeeze(-1)

        # ============================================================
        # Mask illegal actions
        # ============================================================
        legal_mask = legal.bool()
        logits = logits.masked_fill(
            ~legal_mask,
            -1e9
        )

        # ============================================================
        # Value
        # ============================================================
        value = self.value_head(cls_output).squeeze(-1)
        return logits, value