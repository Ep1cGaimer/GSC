"""
Conformal Prediction Escalation: Single-pass uncertainty quantification
using MAPIE for mathematically guaranteed confidence intervals.

Replaces expensive ensemble critics (N forward passes) with a single
model + calibration dataset approach.

Coverage guarantee: If calibrated on exchangeable data, the prediction
interval covers the true value with probability ≥ (1 - alpha).
"""

import numpy as np
from typing import Optional


class CriticEstimatorWrapper:
    """Wraps the RL critic network into a scikit-learn-compatible interface
    for MAPIE. The critic is already trained — we just need predict().
    """

    def __init__(self, critic_model, device="cpu"):
        self.critic = critic_model
        self.device = device

    def fit(self, X, y):
        """No-op. Critic is already trained."""
        return self

    def predict(self, X):
        """Run critic inference on state observations.
        
        Args:
            X: numpy array [batch, obs_dim]
        Returns:
            numpy array of value predictions [batch]
        """
        import torch
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            values = self.critic(x_tensor).squeeze(-1)
            return values.cpu().numpy()


class ConformalEscalation:
    """Conformal prediction wrapper for intelligent human escalation.
    
    Uses MAPIE MapieRegressor with Jackknife+ method for tight,
    distribution-free prediction intervals.
    
    When the interval width exceeds a threshold, the system
    escalates to a human operator — indicating the agent is
    uncertain about the value of the current state.
    """

    def __init__(self, critic_model=None, alpha: float = 0.1,
                 escalation_threshold: float = None, device: str = "cpu"):
        """
        Args:
            critic_model: Trained critic network (nn.Module)
            alpha: Significance level (0.1 = 90% coverage)
            escalation_threshold: Width threshold for escalation.
                If None, auto-calibrated from calibration set.
            device: torch device
        """
        self.alpha = alpha
        self.escalation_threshold = escalation_threshold
        self.device = device
        self.is_calibrated = False
        self.mapie = None
        self.calibration_widths = []

        if critic_model is not None:
            self.estimator = CriticEstimatorWrapper(critic_model, device)
        else:
            self.estimator = None

    def calibrate(self, calibration_states: np.ndarray, calibration_rewards: np.ndarray):
        """Calibrate conformal prediction on held-out data.
        
        Args:
            calibration_states: [n_samples, obs_dim] — states never seen during training
            calibration_rewards: [n_samples] — actual rewards for those states
        """
        try:
            from mapie.regression import MapieRegressor
        except ImportError:
            print("WARNING: MAPIE not installed. Using fallback uncertainty estimation.")
            self._calibrate_fallback(calibration_states, calibration_rewards)
            return

        self.mapie = MapieRegressor(
            estimator=self.estimator,
            method="plus",     # Jackknife+ for tighter intervals
            cv="prefit",       # Critic is already trained — don't refit
        )
        self.mapie.fit(calibration_states, calibration_rewards)

        # Compute calibration widths for auto-threshold
        _, intervals = self.mapie.predict(calibration_states, alpha=self.alpha)
        widths = intervals[:, 1, 0] - intervals[:, 0, 0]
        self.calibration_widths = widths.tolist()

        # Auto-set threshold at 90th percentile of calibration widths
        if self.escalation_threshold is None:
            self.escalation_threshold = float(np.percentile(widths, 90))

        self.is_calibrated = True

    def _calibrate_fallback(self, states: np.ndarray, rewards: np.ndarray):
        """Fallback calibration when MAPIE is not available.
        Uses simple residual-based intervals."""
        if self.estimator is not None:
            predictions = self.estimator.predict(states)
            residuals = np.abs(rewards - predictions)
            self.fallback_quantile = float(np.percentile(residuals, (1 - self.alpha) * 100))
        else:
            self.fallback_quantile = float(np.std(rewards) * 2)

        if self.escalation_threshold is None:
            self.escalation_threshold = self.fallback_quantile * 2

        self.is_calibrated = True

    def should_escalate(self, state: np.ndarray) -> tuple[bool, float, float]:
        """Check if the current state should be escalated to a human.
        
        Args:
            state: Single observation [obs_dim]
            
        Returns:
            (escalate: bool, interval_width: float, threshold: float)
        """
        if not self.is_calibrated:
            # Before calibration, always escalate (fail-safe)
            return True, float('inf'), 0.0

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state

        if self.mapie is not None:
            pred, intervals = self.mapie.predict(state_2d, alpha=self.alpha)
            width = float(intervals[0, 1, 0] - intervals[0, 0, 0])
        else:
            # Fallback
            width = self.fallback_quantile * 2  # Approximate

        escalate = width > self.escalation_threshold
        return escalate, width, self.escalation_threshold

    def get_metrics(self) -> dict:
        """Get current conformal prediction metrics for dashboard."""
        return {
            "is_calibrated": self.is_calibrated,
            "alpha": self.alpha,
            "threshold": self.escalation_threshold or 0.0,
            "calibration_samples": len(self.calibration_widths),
            "median_width": float(np.median(self.calibration_widths)) if self.calibration_widths else 0.0,
        }
