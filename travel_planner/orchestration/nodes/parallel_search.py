"""
Parallel search node implementation for the travel planning workflow.

This module defines functions for setting up and combining results from
flight, accommodation, and transportation searches.
"""

from typing import Any

from travel_planner.orchestration.nodes.accommodation_search import (
    accommodation_task,
)
from travel_planner.orchestration.nodes.flight_search import (
    flight_search_task,
)
from travel_planner.orchestration.nodes.transportation_planning import (
    transportation_task,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def create_parallel_search_branch():
    """
    Return the three search tasks used by the planning workflow.

    LangGraph handles parallel execution through graph fan-out rather than
    the old ParallelBranch API.
    """
    return {
        "flight_search": flight_search_task,
        "accommodation_search": accommodation_task,
        "transportation_planning": transportation_task,
    }


def combine_search_results(state: TravelPlanningState) -> TravelPlanningState:
    """
    Combine results from the parallel search tasks.

    The individual parallel nodes return their results using separate
    state-update keys, which avoids multiple parallel nodes attempting
    to write to the same LangGraph state key.
    """
    logger.info("Combining results from parallel search")

    state.update_stage(WorkflowStage.PARALLEL_SEARCH_COMPLETED)

    # Read the results produced by the parallel search nodes.
    flight_result = getattr(state, "flight_search_result", None)
    accommodation_result = getattr(state, "accommodation_search_result", None)
    transportation_result = getattr(
        state,
        "transportation_search_result",
        None,
    )

    # Log which searches completed.
    completed_searches = []

    if flight_result is not None:
        completed_searches.append("flights")

    if accommodation_result is not None:
        completed_searches.append("accommodations")

    if transportation_result is not None:
        completed_searches.append("transportation")

    logger.info(
        "Parallel searches completed: %s",
        ", ".join(completed_searches) if completed_searches else "none",
    )

    # Preserve the results in the conversation history for traceability.
    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                "Completed parallel search: "
                f"{'flights' if flight_result is not None else 'no flights'}, "
                f"{'accommodations' if accommodation_result is not None else 'no accommodations'}, "
                f"{'transportation' if transportation_result is not None else 'no transportation'}"
            ),
        }
    )

    return state