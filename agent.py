import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, size):
        """
        A simple residual block with two fully connected layers and a skip connection.
        
        Args:
            size: The size of the input and output tensors for the block.
        """
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, x):
        """
        A simple residual block with two fully connected layers and a skip connection.
        
        Args:
            x: Input tensor of shape (batch_size, size)
        Returns:
            Output tensor of shape (batch_size, size)
        """
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(self.norm(x + h))


class CinchAgent(nn.Module):
    def __init__(self, input_size, hidden=512):
        """
        A simple feedforward neural network with residual connections for the Cinch agent.
        
        Args:
            input_size: The size of the input tensor (observation space).
            hidden: The size of the hidden layers.
        
        The network architecture is as follows:
        - Fully connected layer from input_size to hidden, followed by ReLU activation
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Policy head: fully connected layer from hidden to 52 (action logits)
        - Value head: fully connected layer from hidden to 1 (state value)
        """
        super().__init__()

        self.fc_in = nn.Linear(input_size, hidden)

        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.res3 = ResidualBlock(hidden)
        self.res4 = ResidualBlock(hidden)
        self.res5 = ResidualBlock(hidden)

        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 52)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Network architecture:
        - Fully connected layer from input_size to hidden, followed by ReLU activation
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Policy head: 
            - Fully connected layer from hidden to hidden, followed by ReLU activation
            - Fully connected layer from hidden to 52 (action logits)
        - Value head:
            - Fully connected layer from hidden to hidden, followed by ReLU activation
            - Fully connected layer from hidden to 1 (state value)
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            logits: Output tensor of shape (batch_size, 52) representing the action logits
            value: Output tensor of shape (batch_size, 1) representing the state value
        """
        x = F.relu(self.fc_in(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        logits = self.policy(x)
        value = self.value(x)

        return logits, value