"""Mixture-of-Experts (MoE) FFN with top-K routing and load balancing.

Replaces standard FFN in DiT blocks. Enables large total parameter count (e.g. 30B)
while only activating a fraction per forward pass (e.g. 3–4B).

Design follows:
- DeepSeek-MoE / Mamoda2.5: fine-grained experts with shared expert
- Switch Transformer: top-K routing with auxiliary load balancing loss

Usage in vision/audio branch blocks:
    self.ffn = MoEFFN(dim, num_experts=32, top_k=2)  # replaces MLP(dim, 4*dim, dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """Mixture-of-Experts feed-forward network.

    Each token is routed to top-K experts via a learned router.
    A shared expert processes all tokens (always active, like DeepSeek).
    An auxiliary load balancing loss prevents routing collapse.

    Total params: N_experts × (2 × dim × hidden_dim) + shared_expert
    Activated params: (top_k + shared) × (2 × dim × hidden_dim) ≈ 1/8 of dense

    Args:
        dim: Input/output dimension.
        num_experts: Number of expert FFNs (default: 32).
        top_k: Number of active experts per token (default: 2).
        expert_dim_ratio: Hidden dim ratio per expert (default: 1.0).
            With 32 experts and ratio=1.0, total FFN params = 32 × 2dim² ≈ 64dim².
            A standard FFN (ratio=4.0) has 8dim².
            So 32 experts × ratio=1.0 gives ~8× more capacity than dense.
        shared_expert: If True, add a shared expert that processes all tokens (DeepSeek-style).
        router_bias: If True, use a bias term in the router (improves initial uniformity).
        router_z_loss_weight: Weight for router z-loss (stabilizes training).
        load_balance_weight: Weight for auxiliary load balancing loss.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 32,
        top_k: int = 2,
        expert_dim_ratio: float = 1.0,
        shared_expert: bool = True,
        router_bias: bool = True,
        router_z_loss_weight: float = 0.001,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.load_balance_weight = load_balance_weight
        self.router_z_loss_weight = router_z_loss_weight

        expert_hidden = int(dim * expert_dim_ratio)

        # ── Router ──────────────────────────────────────────────────
        self.router = nn.Linear(dim, num_experts, bias=router_bias)
        # Initialize router with small weights for uniform initial routing
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02 / math.sqrt(dim))
        if router_bias:
            nn.init.zeros_(self.router.bias)

        # ── Expert FFNs ─────────────────────────────────────────────
        # Each expert: dim → expert_hidden → dim (GELU activation)
        self.experts = nn.ModuleList([
            _Expert(dim, expert_hidden) for _ in range(num_experts)
        ])

        # ── Shared Expert (DeepSeek-style, always active) ──────────
        if shared_expert:
            self.shared = _Expert(dim, expert_hidden * top_k)
        else:
            self.shared = None

        # ── Tracking ────────────────────────────────────────────────
        self.register_buffer("_expert_counts", torch.zeros(num_experts))
        self.register_buffer("_total_tokens", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tokens (B, N, D).

        Returns:
            Tuple of (output, aux_losses_dict).
            aux_losses_dict contains "load_balance_loss", "router_z_loss".
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)

        # ── Router logits ───────────────────────────────────────────
        router_logits = self.router(x_flat)  # (B*N, num_experts)

        # Router z-loss (DeepSeek): log(sum(exp(z)))² stabilizes router
        router_z_loss = (
            torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            * self.router_z_loss_weight
        )

        # ── Top-K routing ───────────────────────────────────────────
        routing_weights, routing_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # (B*N, top_k)
        routing_weights = F.softmax(routing_weights, dim=-1)

        # ── Dispatch to experts ─────────────────────────────────────
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = routing_indices[:, k]  # (B*N,)
            weight = routing_weights[:, k]      # (B*N,)

            for eid in range(self.num_experts):
                mask = (expert_idx == eid)
                n_tokens = mask.sum().item()
                if n_tokens == 0:
                    continue

                token_indices = mask.nonzero(as_tuple=True)[0]
                expert_input = x_flat[token_indices]  # (n_tokens, D)
                expert_output = self.experts[eid](expert_input)

                # Weighted combination
                output[token_indices] += expert_output * weight[token_indices].unsqueeze(-1)

        # ── Shared expert ───────────────────────────────────────────
        if self.shared is not None:
            output = output + self.shared(x_flat)

        output = output.reshape(B, N, D)

        # ── Load balancing loss ─────────────────────────────────────
        # Switch Transformer auxiliary loss: encourage uniform expert usage
        # f = fraction of tokens dispatched to each expert
        # P = mean router probability for each expert
        # loss = N_experts * sum(f * P)
        with torch.no_grad():
            dispatch_mask = F.one_hot(
                routing_indices, num_classes=self.num_experts
            ).float()  # (B*N, top_k, num_experts)
            f = dispatch_mask.mean(dim=(0, 1))  # (num_experts,)
            # Track expert usage (move buffers to correct device if needed)
            counts = dispatch_mask.sum(dim=(0, 1))
            if self._expert_counts.device != counts.device:
                self._expert_counts = self._expert_counts.to(counts.device)
                self._total_tokens = self._total_tokens.to(counts.device)
            self._expert_counts += counts
            self._total_tokens += (B * N * self.top_k)

        # Mean router probability for each expert
        router_probs = F.softmax(router_logits, dim=-1)  # (B*N, num_experts)
        P = router_probs.mean(dim=0)  # (num_experts,)

        load_balance_loss = (
            self.num_experts * (f * P).sum() * self.load_balance_weight
        )

        aux_losses = {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
        }
        return output, aux_losses

    def reset_stats(self):
        """Reset load balancing statistics."""
        self._expert_counts.zero_()
        self._total_tokens.zero_()

    def get_expert_usage(self) -> torch.Tensor:
        """Get fraction of tokens dispatched to each expert."""
        total = self._total_tokens.item()
        if total == 0:
            return torch.zeros(self.num_experts)
        return self._expert_counts / total


class _Expert(nn.Module):
    """Single expert FFN: dim → hidden → dim with GELU."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        # Kaiming init for better gradient flow
        nn.init.kaiming_normal_(self.w1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.w2.weight, nonlinearity="linear")
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


class MoEWrapper(nn.Module):
    """Convenience wrapper: adds MoE aux losses to the main loss dict.

    Usage:
        moe = MoEWrapper(MoEFFN(dim, num_experts=32, top_k=2))
        output, aux_losses = moe(x)
        total_loss = main_loss + aux_losses["load_balance_loss"] + aux_losses["router_z_loss"]
    """

    def __init__(self, moe_ffn: MoEFFN):
        super().__init__()
        self.moe = moe_ffn
        self.cumulative_aux_losses = {"load_balance_loss": 0.0, "router_z_loss": 0.0}
        self._step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, aux = self.moe(x)
        for k in self.cumulative_aux_losses:
            self.cumulative_aux_losses[k] += aux[k].item()
        self._step += 1
        return out

    def get_aux_losses(self, reset: bool = True) -> dict[str, float]:
        """Get averaged auxiliary losses and optionally reset."""
        n = max(self._step, 1)
        result = {k: v / n for k, v in self.cumulative_aux_losses.items()}
        if reset:
            self.cumulative_aux_losses = {k: 0.0 for k in self.cumulative_aux_losses}
            self._step = 0
        return result
