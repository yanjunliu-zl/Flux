"""ODE Schedulers for flow matching inference.

Provides Euler (1st order) and Heun (2nd order) integration methods.
"""

import torch
import torch.nn as nn


class EulerScheduler(nn.Module):
    """Euler method (1st order) for ODE integration.

    Simplest integration method. Good for many steps (>=50).
    """

    def __init__(self):
        super().__init__()

    def step(
        self,
        z_v: torch.Tensor,
        z_a: torch.Tensor,
        v_pred: torch.Tensor,
        a_pred: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single Euler step.

        Args:
            z_v, z_a: Current state.
            v_pred, a_pred: Predicted velocity.
            dt: Step size.

        Returns:
            Next state.
        """
        return z_v + v_pred * dt, z_a + a_pred * dt


class HeunScheduler(nn.Module):
    """Heun's method (2nd order Runge-Kutta) for ODE integration.

    More accurate than Euler for the same number of steps.
    Typically reduces required steps from 50 to 25-30.
    """

    def __init__(self):
        super().__init__()

    def step(
        self,
        z_v: torch.Tensor,
        z_a: torch.Tensor,
        v_pred: torch.Tensor,
        a_pred: torch.Tensor,
        dt: float,
        v_pred2: torch.Tensor,
        a_pred2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single Heun step (trapezoidal rule).

        Args:
            z_v, z_a: Current state.
            v_pred, a_pred: Velocity at current state.
            dt: Step size.
            v_pred2, a_pred2: Velocity at predicted next state.

        Returns:
            Next state (trapezoidal average).
        """
        return (
            z_v + 0.5 * (v_pred + v_pred2) * dt,
            z_a + 0.5 * (a_pred + a_pred2) * dt,
        )
