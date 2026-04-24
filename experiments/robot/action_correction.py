"""Utilities for inference-time action correction in robot evaluation loops."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ActionCorrectionConfig:
    """Configuration for optional closed-loop action correction."""

    enabled: bool = False
    use_smoothing: bool = False
    use_gripper_stabilization: bool = False
    use_stagnation_detection: bool = False
    smoothing_alpha: float = 0.5
    stagnation_window: int = 5
    stagnation_threshold: float = 0.01
    retry_cooldown: int = 0
    gripper_hold_steps: int = 1
    gripper_flip_tolerance: int = 0


class ActionCorrector:
    """Inference-time action corrector with a default no-op implementation."""

    def __init__(self, config: ActionCorrectionConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Reset per-episode internal state."""
        self._prev_action: Optional[np.ndarray] = None
        self._prev_raw_action: Optional[np.ndarray] = None
        self._prev_raw_gripper_state: Optional[int] = None
        self._recent_action_norms = []
        self._last_retry_step = -1
        self._last_applied_smoothing = False
        self._last_applied_gripper_stabilization = False
        self._raw_action_delta_norm_sum = 0.0
        self._raw_action_delta_count = 0
        self._prev_gripper_state: Optional[int] = None
        self._last_gripper_change_step = -1
        self._raw_gripper_flip_count = 0
        self._held_gripper_flip_count = 0
        self._recent_gripper_flips = []
        self._prev_image: Optional[np.ndarray] = None
        self._recent_image_deltas = []
        self._recent_action_magnitudes = []
        self._recent_gripper_states = []
        self._last_detected_stagnation = False
        self._stagnation_trigger_count = 0
        self._stagnation_steps = []

    def correct(
        self,
        action: np.ndarray,
        observation: Dict[str, Any],
        task_description: str,
        step_idx: int,
    ) -> np.ndarray:
        """Return a corrected action.

        The default implementation is intentionally conservative and leaves the
        action unchanged unless correction features are explicitly enabled.
        """
        action_array = np.array(action, copy=True)
        self._last_applied_smoothing = False
        self._last_applied_gripper_stabilization = False
        self._last_detected_stagnation = False

        if action_array.shape != (7,):
            action_array = np.asarray(action_array).reshape(-1)

        self._update_raw_action_metrics(action_array)

        if not self.config.enabled:
            return action_array

        if self.config.use_stagnation_detection:
            self._update_stagnation_state(action_array, observation, step_idx)

        if self.config.use_smoothing and self._prev_action is not None:
            alpha = float(self.config.smoothing_alpha)
            smoothed_action = np.array(action_array, copy=True)
            smoothed_action[:-1] = alpha * action_array[:-1] + (1.0 - alpha) * self._prev_action[:-1]
            action_array = smoothed_action
            self._last_applied_smoothing = True

        if self.config.use_gripper_stabilization:
            action_array = self._stabilize_gripper(action_array, step_idx)

        self._prev_action = np.array(action_array, copy=True)
        return action_array

    def _update_raw_action_metrics(self, action: np.ndarray) -> None:
        """Track raw model action changes for ablation, independent of correction."""
        if self._prev_raw_action is not None:
            raw_action_delta = float(np.linalg.norm(action - self._prev_raw_action))
            self._raw_action_delta_norm_sum += raw_action_delta
            self._raw_action_delta_count += 1

        raw_gripper_state = self._to_gripper_state(action[-1])
        if self._prev_raw_gripper_state is not None and raw_gripper_state != self._prev_raw_gripper_state:
            self._raw_gripper_flip_count += 1

        self._prev_raw_action = np.array(action, copy=True)
        self._prev_raw_gripper_state = raw_gripper_state

    def _update_stagnation_state(self, action: np.ndarray, observation: Dict[str, Any], step_idx: int) -> None:
        """Track simple history signals and flag likely stagnation."""
        window = max(1, int(self.config.stagnation_window))
        action_magnitude = float(np.linalg.norm(action[:-1]))
        gripper_state = self._to_gripper_state(action[-1])
        image_delta = self._compute_image_delta(observation.get("full_image"))

        self._append_window(self._recent_action_norms, action_magnitude, window)
        self._append_window(self._recent_action_magnitudes, action_magnitude, window)
        self._append_window(self._recent_gripper_states, gripper_state, window)
        if image_delta is not None:
            self._append_window(self._recent_image_deltas, image_delta, window)

        if len(self._recent_image_deltas) < window or len(self._recent_action_magnitudes) < window:
            return

        avg_image_delta = float(np.mean(self._recent_image_deltas[-window:]))
        avg_action_magnitude = float(np.mean(self._recent_action_magnitudes[-window:]))
        gripper_unchanged = len(set(self._recent_gripper_states[-window:])) == 1
        action_is_active = avg_action_magnitude > self.config.stagnation_threshold
        image_is_static = avg_image_delta < self.config.stagnation_threshold

        if image_is_static and action_is_active and gripper_unchanged:
            self._last_detected_stagnation = True
            self._stagnation_trigger_count += 1
            self._stagnation_steps.append(step_idx)

    def _compute_image_delta(self, image: Optional[np.ndarray]) -> Optional[float]:
        if image is None:
            return None

        image_array = np.asarray(image, dtype=np.float32)
        if self._prev_image is None:
            self._prev_image = image_array
            return None

        image_delta = float(np.mean(np.abs(image_array - self._prev_image)) / 255.0)
        self._prev_image = image_array
        return image_delta

    @staticmethod
    def _append_window(values, value, window: int) -> None:
        values.append(value)
        if len(values) > window:
            values.pop(0)

    def _stabilize_gripper(self, action: np.ndarray, step_idx: int) -> np.ndarray:
        """Reduce high-frequency gripper flips before env-specific mapping."""
        action_array = np.array(action, copy=True)
        current_gripper_state = self._to_gripper_state(action_array[-1])

        if self._prev_gripper_state is None:
            self._prev_gripper_state = current_gripper_state
            self._last_gripper_change_step = step_idx
            return action_array

        if current_gripper_state != self._prev_gripper_state:
            steps_since_change = step_idx - self._last_gripper_change_step
            self._recent_gripper_flips.append(step_idx)

            if self._should_hold_gripper(steps_since_change):
                action_array[-1] = self._prev_action[-1] if self._prev_action is not None else action_array[-1]
                self._last_applied_gripper_stabilization = True
                self._held_gripper_flip_count += 1
                return action_array

            self._prev_gripper_state = current_gripper_state
            self._last_gripper_change_step = step_idx

        return action_array

    def _should_hold_gripper(self, steps_since_change: int) -> bool:
        hold_steps = max(0, int(self.config.gripper_hold_steps))
        if steps_since_change < hold_steps:
            return True

        tolerance = max(0, int(self.config.gripper_flip_tolerance))
        if tolerance == 0:
            return False

        window = max(1, int(self.config.stagnation_window))
        recent_cutoff = self._recent_gripper_flips[-1] - window
        recent_flip_count = sum(1 for flip_step in self._recent_gripper_flips if flip_step > recent_cutoff)
        return recent_flip_count > tolerance

    @staticmethod
    def _to_gripper_state(gripper_value: float) -> int:
        return 1 if float(gripper_value) >= 0.5 else 0

    def get_episode_metrics(self) -> Dict[str, float]:
        """Return aggregate correction metrics for the current episode."""
        avg_delta = 0.0
        if self._raw_action_delta_count > 0:
            avg_delta = self._raw_action_delta_norm_sum / self._raw_action_delta_count

        return {
            "avg_action_delta_norm": avg_delta,
            "num_action_delta_steps": float(self._raw_action_delta_count),
            "gripper_flip_count": float(self._raw_gripper_flip_count),
            "held_gripper_flip_count": float(self._held_gripper_flip_count),
            "stagnation_trigger_count": float(self._stagnation_trigger_count),
        }

    def get_debug_state(self) -> Dict[str, Any]:
        """Expose internal state for logging/debugging."""
        return {
            "config": asdict(self.config),
            "has_prev_action": self._prev_action is not None,
            "recent_action_norms": list(self._recent_action_norms),
            "last_retry_step": self._last_retry_step,
            "last_applied_smoothing": self._last_applied_smoothing,
            "last_applied_gripper_stabilization": self._last_applied_gripper_stabilization,
            "gripper_flip_count": self._raw_gripper_flip_count,
            "held_gripper_flip_count": self._held_gripper_flip_count,
            "last_detected_stagnation": self._last_detected_stagnation,
            "stagnation_trigger_count": self._stagnation_trigger_count,
            "stagnation_steps": list(self._stagnation_steps[:20]),
            "recent_image_deltas": list(self._recent_image_deltas),
        }
