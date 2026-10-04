"""
Sharpness-Aware Minimization (SAM) optimizer wrapper.

Wraps a base optimizer (e.g. AdamW) and performs a two-step update per
batch: an ascent step that perturbs parameters toward the worst-case
point within an L2 ball of radius `rho` (the "sharpness" direction),
followed by a descent step of the base optimizer computed at that
perturbed point. This biases training toward flatter minima, which
generalizes better on small, imbalanced datasets like ours.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization", ICLR 2021.

Usage (see trainer.py): requires TWO forward+backward passes per
optimizer step, so when `use_sam=True` gradient accumulation is
bypassed (one SAM step == one full-batch step).
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    SAM optimizer wrapper around any base torch.optim.Optimizer class.

    Args:
        params: model parameters or param groups (as passed to any optimizer)
        base_optimizer: an UNINSTANTIATED optimizer class, e.g. torch.optim.AdamW
        rho: neighborhood size for the sharpness-aware ascent step
        **base_optimizer_kwargs: forwarded to base_optimizer(...)
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, **base_optimizer_kwargs):
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, **base_optimizer_kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Ascent step: perturb params toward the local worst-case direction."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                perturbation = p.grad * scale.to(p.device)
                p.add_(perturbation)
                self.state[p]["perturbation"] = perturbation

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Descent step: undo the perturbation, then apply the base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "perturbation" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["perturbation"])
                del self.state[p]["perturbation"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)

    def step(self, closure=None):
        raise RuntimeError(
            "SAM requires two explicit passes — call first_step() after the "
            "first backward(), then second_step() after the second backward(). "
            "Do not call step() directly."
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
