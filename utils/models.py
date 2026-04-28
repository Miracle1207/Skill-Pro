# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_mlp(m):
    """
    Simple weight init for linear layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ActionValueCritic(nn.Module):
    """
    Q_ψ(s, g, a): estimate expected return given state, Skills and action.

    Inputs (batched):
        state_emb:    [B, state_dim]
        guidance_emb: [B, guidance_dim]
        action_emb:   [B, action_dim]

    Output:
        q_values: [B]  (scalar per sample)
    """

    def __init__(self, args):
        super().__init__()

        state_dim = args.state_dim
        guidance_dim = args.guidance_dim
        action_dim = args.action_dim
        hidden_dim = getattr(args, "critic_hidden_dim", 256)

        input_dim = state_dim + guidance_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.apply(_init_mlp)

    def forward(self, state_emb, guidance_emb, action_emb):
        """
        Args:
            state_emb:    tensor [B, state_dim]
            guidance_emb: tensor [B, guidance_dim]
            action_emb:   tensor [B, action_dim]
        Returns:
            q_values: tensor [B]
        """
        x = torch.cat([state_emb, guidance_emb, action_emb], dim=-1)
        q = self.net(x)             # [B, 1]
        return q.squeeze(-1)        # [B]


class StateValueCritic(nn.Module):
    """
    V_ω(s): estimate state value under the current policy & selector.

    Input:
        state_emb: [B, state_dim]

    Output:
        v_values: [B]
    """

    def __init__(self, args):
        super().__init__()

        state_dim = args.state_dim
        hidden_dim = getattr(args, "critic_hidden_dim", 256)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.apply(_init_mlp)

    def forward(self, state_emb):
        """
        Args:
            state_emb: tensor [B, state_dim]
        Returns:
            v_values: tensor [B]
        """
        v = self.net(state_emb)     # [B, 1]
        return v.squeeze(-1)
