"""High-level actions and deterministic baselines on a shared decision clock."""

from enum import IntEnum

from src.simulation.state import Observation


class Action(IntEnum):
    KEEP = 0
    LOCAL = 1
    GLOBAL = 2
    LOCAL_REOPTIMIZATION = 3


def never(observation: Observation) -> Action:
    return Action.KEEP


def always(observation: Observation) -> Action:
    return Action.GLOBAL if observation.event_types else Action.KEEP


def periodic(observation: Observation) -> Action:
    return Action.GLOBAL if observation.tick else Action.KEEP


def threshold(observation: Observation) -> Action:
    return Action.GLOBAL if observation.predicted_delay_s >= 300 else Action.KEEP


def multi_feature(observation: Observation) -> Action:
    urgent = bool(observation.waiting_urgent)
    serious = (observation.priority_at_risk > 0 or observation.at_risk >= 2
               or observation.predicted_overtime_s >= 180)
    if urgent or serious:
        return Action.GLOBAL
    quiet = observation.since_replan_s is None or observation.since_replan_s >= 600
    if quiet and observation.at_risk and (
        observation.congestion_factor >= 1.3 or observation.predicted_delay_s >= 120
        or "service_delay" in observation.event_types
    ):
        return Action.LOCAL
    return Action.KEEP


POLICIES = {"never": never, "always": always, "periodic": periodic,
            "threshold": threshold, "multi_feature": multi_feature}
