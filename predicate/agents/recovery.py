"""
Recovery: Checkpoint and rollback mechanisms for automation recovery.

This module provides state tracking and recovery mechanisms for when
automation gets off-track. Key concepts:

- RecoveryCheckpoint: A snapshot of known-good state (URL, step, digest)
- RecoveryState: Tracks checkpoints and manages recovery attempts

Recovery flow:
1. After each successful step verification, record a checkpoint
2. If verification fails repeatedly, attempt recovery to last checkpoint
3. Navigate back to checkpoint URL and re-verify
4. If recovery succeeds, resume from checkpoint step
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RecoveryCheckpoint:
    """
    Checkpoint for rollback recovery.

    Created after each successful step verification to enable rollback
    if subsequent steps fail.

    Attributes:
        url: The URL at this checkpoint
        step_index: The step index that was just completed (0-indexed)
        snapshot_digest: Hash of the snapshot for state verification
        timestamp: When the checkpoint was created
        predicates_passed: Labels of predicates that passed at this checkpoint
    """

    url: str
    step_index: int
    snapshot_digest: str
    timestamp: datetime
    predicates_passed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure predicates_passed is a list
        if self.predicates_passed is None:
            self.predicates_passed = []


@dataclass
class RecoveryState:
    """
    Tracks recovery state for rollback mechanism.

    Checkpoints are created after each successful step verification.
    Recovery can be attempted when steps fail repeatedly.

    Attributes:
        checkpoints: List of recorded checkpoints (most recent last)
        recovery_attempts_used: Number of recovery attempts consumed
        max_recovery_attempts: Maximum allowed recovery attempts
        current_recovery_target: The checkpoint being recovered to (if any)
        max_checkpoints: Maximum checkpoints to retain (bounds memory)

    Example:
        state = RecoveryState(max_recovery_attempts=2)

        # After successful step
        state.record_checkpoint(
            url="https://shop.com/cart",
            step_index=2,
            snapshot_digest="abc123",
            predicates_passed=["url_contains('/cart')"],
        )

        # On repeated failure
        if state.can_recover():
            checkpoint = state.consume_recovery_attempt()
            # Navigate to checkpoint.url and resume
    """

    checkpoints: list[RecoveryCheckpoint] = field(default_factory=list)
    recovery_attempts_used: int = 0
    max_recovery_attempts: int = 2
    current_recovery_target: RecoveryCheckpoint | None = None
    max_checkpoints: int = 10

    def record_checkpoint(
        self,
        url: str,
        step_index: int,
        snapshot_digest: str,
        predicates_passed: list[str] | None = None,
    ) -> RecoveryCheckpoint:
        """
        Record a successful checkpoint.

        Called after step verification passes to enable future rollback.

        Args:
            url: Current page URL
            step_index: Index of the step that just completed
            snapshot_digest: Hash of current snapshot for verification
            predicates_passed: Labels of predicates that passed

        Returns:
            The created RecoveryCheckpoint
        """
        checkpoint = RecoveryCheckpoint(
            url=url,
            step_index=step_index,
            snapshot_digest=snapshot_digest,
            timestamp=datetime.now(),
            predicates_passed=predicates_passed or [],
        )
        self.checkpoints.append(checkpoint)

        # Keep only last N checkpoints to bound memory
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints :]

        return checkpoint

    def get_recovery_target(self) -> RecoveryCheckpoint | None:
        """
        Get the most recent checkpoint for recovery.

        Returns:
            Most recent RecoveryCheckpoint, or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def get_checkpoint_at_step(self, step_index: int) -> RecoveryCheckpoint | None:
        """
        Get checkpoint at a specific step index.

        Args:
            step_index: The step index to find

        Returns:
            RecoveryCheckpoint at that step, or None if not found
        """
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.step_index == step_index:
                return checkpoint
        return None

    def get_checkpoint_before_step(self, step_index: int) -> RecoveryCheckpoint | None:
        """
        Get the most recent checkpoint before a given step.

        Args:
            step_index: The step index to find checkpoint before

        Returns:
            Most recent checkpoint with step_index < given index, or None
        """
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.step_index < step_index:
                return checkpoint
        return None

    def can_recover(self) -> bool:
        """
        Check if recovery is still possible.

        Returns:
            True if recovery attempts remain and checkpoints exist
        """
        return (
            self.recovery_attempts_used < self.max_recovery_attempts
            and len(self.checkpoints) > 0
        )

    def consume_recovery_attempt(self) -> RecoveryCheckpoint | None:
        """
        Consume a recovery attempt and return target checkpoint.

        Increments recovery_attempts_used and sets current_recovery_target.

        Returns:
            The checkpoint to recover to, or None if recovery not possible
        """
        if not self.can_recover():
            return None

        self.recovery_attempts_used += 1
        self.current_recovery_target = self.get_recovery_target()
        return self.current_recovery_target

    def clear_recovery_target(self) -> None:
        """Clear the current recovery target after recovery completes."""
        self.current_recovery_target = None

    def reset(self) -> None:
        """Reset recovery state for a new run."""
        self.checkpoints.clear()
        self.recovery_attempts_used = 0
        self.current_recovery_target = None

    def pop_checkpoint(self) -> RecoveryCheckpoint | None:
        """
        Remove and return the most recent checkpoint.

        Useful when recovery fails and we want to try an earlier checkpoint.

        Returns:
            The removed checkpoint, or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None
        return self.checkpoints.pop()

    @property
    def last_successful_url(self) -> str | None:
        """Get the URL of the most recent successful checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1].url

    @property
    def last_successful_step(self) -> int | None:
        """Get the step index of the most recent successful checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1].step_index

    def __len__(self) -> int:
        """Return number of checkpoints."""
        return len(self.checkpoints)
