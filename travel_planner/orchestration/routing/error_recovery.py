"""Centralized error and interruption handling for the workflow."""

from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def handle_error(state: TravelPlanningState) -> TravelPlanningState:
    """Record the failure and checkpoint the state without retrying it here."""
    from travel_planner.orchestration.serialization.checkpoint import (
        save_state_checkpoint,
    )

    error_message = state.error or "Unknown error"
    failed_node = state.error_node or "unknown node"

    state.conversation_history.append(
        {
            "role": "system",
            "content": f"Error occurred in {failed_node}: {error_message}",
        }
    )

    try:
        checkpoint_id = save_state_checkpoint(state)
        state.travel_checkpoint_id = checkpoint_id
        logger.error(
            "Handled error in %s: %s. Created checkpoint: %s",
            failed_node,
            error_message,
            checkpoint_id,
        )
    except Exception as checkpoint_error:
        # Never replace the original workflow error with a checkpoint error.
        logger.exception(
            "Failed to create error checkpoint after %s: %s",
            failed_node,
            checkpoint_error,
        )

    return state


def handle_interruption(state: TravelPlanningState) -> TravelPlanningState:
    """Checkpoint an interrupted workflow."""
    from travel_planner.orchestration.serialization.checkpoint import (
        save_state_checkpoint,
    )

    if not state.interrupted:
        state.mark_interrupted("User requested interruption")

    try:
        checkpoint_id = save_state_checkpoint(state)
        state.travel_checkpoint_id = checkpoint_id
    except Exception as checkpoint_error:
        logger.exception("Failed to checkpoint interruption: %s", checkpoint_error)
        checkpoint_id = state.travel_checkpoint_id or "unavailable"

    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                f"Workflow interrupted: {state.interruption_reason}. "
                f"Checkpoint ID: {checkpoint_id}"
            ),
        }
    )

    logger.info(
        "Handled interruption: %s. Created checkpoint: %s",
        state.interruption_reason,
        checkpoint_id,
    )
    return state
