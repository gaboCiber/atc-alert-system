"""Núcleo del sistema de alertas."""

from .state_manager import StateManager, StateTransaction, Transaction
from .state_projection import (
    ProjectedSeparation,
    ProjectedState,
    ProjectedTrajectory,
    ProjectedWaypoint,
    StateProjector,
)

__all__ = [
    "StateManager",
    "StateTransaction",
    "Transaction",
    "ProjectedState",
    "ProjectedTrajectory",
    "ProjectedWaypoint",
    "ProjectedSeparation",
    "StateProjector",
]
