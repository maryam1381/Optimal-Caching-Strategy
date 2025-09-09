# src/model.py
"""
model.py

MLP factory and model definition for the CVaR-based D2D caching project.

Provides:
 - class MLPNet(nn.Module): flexible MLP that outputs unconstrained caching scores y in R^M
 - function create_mlp(M, device, lr, weight_decay, ...): returns (model, optimizer)

Design notes
 - Final layer is linear (no activation). Projection into feasible set is handled
   by the projection utility in losses.py.
 - Default initializations: Kaiming for hidden layers, Xavier for final layer.
 - LayerNorm is optional (defaults to True). Dropout optional.
 - Returned optimizer is Adam with user-specified lr and weight decay.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim


class MLPNet(nn.Module):
    """
    Flexible MLP for mapping measurement q -> unconstrained caching scores y in R^M.

    Args:
        in_dim: input dimension (typically M + number of additional measurements)
        out_dim: output dimension M (one score per file)
        hidden_dims: list/tuple of hidden layer widths, e.g. [256,256,128]
        use_layernorm: whether to apply LayerNorm after linear (defaults True)
        dropout: dropout probability (0.0 disables)
        activation: nn.Module class for activation (defaults ReLU)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Tuple[int, ...]] = (256, 256, 128),
        use_layernorm: bool = True,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        prev = in_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h

        # final linear layer -> outputs raw real scores (no activation)
        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights:
         - Kaiming normal for all Linear layers except final
         - Xavier uniform for final Linear
         - biases to zero
        """
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        if len(linear_layers) == 0:
            return

        # All but last: kaiming
        for m in linear_layers[:-1]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Last linear: Xavier uniform
        last = linear_layers[-1]
        nn.init.xavier_uniform_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        q: tensor of shape (batch_size, in_dim) or (in_dim,) (unsqueezed automatically).
        Returns: y tensor of shape (batch_size, out_dim) or (out_dim,) if input was 1D.
        """
        is_1d = False
        if q.dim() == 1:
            q = q.unsqueeze(0)
            is_1d = True

        y = self.net(q)

        return y.squeeze(0) if is_1d else y


def create_mlp(
    M: int,
    in_extra: int = 1,
    hidden_dims: Optional[Tuple[int, ...]] = None,
    use_layernorm: bool = True,
    dropout: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    activation: nn.Module = nn.ReLU,
) -> Tuple[nn.Module, optim.Optimizer]:
    """
    Factory that constructs an MLP and its optimizer.

    Args:
        M: number of files (output dimension)
        in_extra: number of additional measurements appended to the M-dimensional popularity vector (default 1)
                  Typically measurement vector q has length M + in_extra.
        hidden_dims: hidden layers sizes. If None, choose based on M:
                     - M <= 64 : (128,128)
                     - else    : (256,256,128)
        use_layernorm: whether to use LayerNorm
        dropout: dropout probability
        lr: learning rate for Adam
        weight_decay: weight decay for Adam
        device: torch.device or None (model moved to device if specified)
        activation: activation class (default ReLU)

    Returns:
        (model, optimizer)
    """
    if hidden_dims is None:
        hidden_dims = (128, 128) if M <= 64 else (256, 256, 128)

    in_dim = M + in_extra
    model = MLPNet(in_dim=in_dim, out_dim=M, hidden_dims=tuple(hidden_dims),
                   use_layernorm=use_layernorm, dropout=dropout, activation=activation)

    if device is not None:
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer


if __name__ == "__main__":
    # simple smoke test
    torch.manual_seed(0)
    M = 100
    in_extra = 1
    model, opt = create_mlp(M=M, in_extra=in_extra, lr=1e-3, weight_decay=1e-4, device=None)
    print("MLP created:", model)
    x = torch.randn(M + in_extra)
    y = model(x)   # (M,)
    print("Output shape:", y.shape, "sum:", y.sum().item())
