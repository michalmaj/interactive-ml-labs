"""Challenge mode for the Gradient Descent Playground demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ml_lab_core import AlgorithmSnapshot

DEFAULT_TARGET_LOSS: Final[float] = 1.0
DEFAULT_MAX_STEPS: Final[int] = 80

STATUS_ACTIVE: Final[str] = "active"
STATUS_SUCCESS: Final[str] = "success"
STATUS_FAILED: Final[str] = "failed"


@dataclass(frozen=True, slots=True)
class LossChallengeConfig:
    """Configuration for a loss-based challenge.

    Attributes:
        target_loss: Loss value that must be reached or beaten.
        max_steps: Maximum number of allowed gradient descent steps.
    """

    target_loss: float = DEFAULT_TARGET_LOSS
    max_steps: int = DEFAULT_MAX_STEPS


@dataclass(frozen=True, slots=True)
class ChallengeResult:
    """Result of evaluating the current algorithm state against the challenge."""

    status: str
    target_loss: float
    current_loss: float
    current_step: int
    max_steps: int
    steps_remaining: int
    message: str

    @property
    def success(self) -> bool:
        """Return whether the challenge has been completed successfully."""
        return self.status == STATUS_SUCCESS

    @property
    def failed(self) -> bool:
        """Return whether the challenge has failed."""
        return self.status == STATUS_FAILED


class LossChallenge:
    """Challenge requiring the model to reach a target loss within a step limit."""

    def __init__(self, config: LossChallengeConfig | None = None) -> None:
        """Initialize the challenge."""
        self._config = config or LossChallengeConfig()
        _validate_config(self._config)

    @property
    def config(self) -> LossChallengeConfig:
        """Return challenge configuration."""
        return self._config

    def evaluate(self, snapshot: AlgorithmSnapshot) -> ChallengeResult:
        """Evaluate challenge state using the current algorithm snapshot."""
        current_loss = float(snapshot.metrics["loss"])
        current_step = snapshot.iteration
        steps_remaining = max(0, self._config.max_steps - current_step)

        if current_loss <= self._config.target_loss:
            return ChallengeResult(
                status=STATUS_SUCCESS,
                target_loss=self._config.target_loss,
                current_loss=current_loss,
                current_step=current_step,
                max_steps=self._config.max_steps,
                steps_remaining=steps_remaining,
                message="Challenge completed. The target loss was reached.",
            )

        if current_step >= self._config.max_steps:
            return ChallengeResult(
                status=STATUS_FAILED,
                target_loss=self._config.target_loss,
                current_loss=current_loss,
                current_step=current_step,
                max_steps=self._config.max_steps,
                steps_remaining=0,
                message="Challenge failed. The step limit was reached.",
            )

        return ChallengeResult(
            status=STATUS_ACTIVE,
            target_loss=self._config.target_loss,
            current_loss=current_loss,
            current_step=current_step,
            max_steps=self._config.max_steps,
            steps_remaining=steps_remaining,
            message="Try to reduce the loss before the step limit is reached.",
        )


def _validate_config(config: LossChallengeConfig) -> None:
    """Validate challenge configuration."""
    if config.target_loss <= 0.0:
        msg = "target_loss must be greater than 0."
        raise ValueError(msg)

    if config.max_steps <= 0:
        msg = "max_steps must be greater than 0."
        raise ValueError(msg)
