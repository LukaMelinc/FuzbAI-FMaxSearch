import torch.nn as nn

class ActorCriticNet(nn.Module):
    """
    A simple Actor-Critic network.
    It outputs both action_mean (the policy) and value (the critic).
    """
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),    # 9 -> 128
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # 128 -> 128
            nn.ReLU(), 
            nn.Linear(hidden_size, act_dim),     # 128 -> 4
            #nn.Tanh()   # NOTE: Temporarely, later change the approach
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),    # 9 -> 128
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # 128 -> 128
            nn.ReLU(),
            nn.Linear(hidden_size, 1)           # 128 -> 1
        )

    def forward(self, x):
        "Takes in the actin and calculates the log-probabilities of actions "
        "and estimated value of the action"
        action_mean = self.actor(x)
        value = self.critic(x)
        return action_mean, value