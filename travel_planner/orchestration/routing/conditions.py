"""
Routing conditions for the travel planner workflow.

The routing layer decides whether a failed workflow stage can be retried.
Only known transient failures are retried; configuration, validation, model,
and provider-quota failures terminate cleanly instead of creating recovery
loops.
"""

from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage

MAX_ERROR_COUNT = 3
HUMAN_INTERVENTION_ERROR_THRESHOLD = 2


def query_research_needed(state: TravelPlanningState) -> str:
    if not state.query or not state.query.destination:
        return "research_destination"
    return "flight_search"


def has_error(state: TravelPlanningState) -> str:
    if state.error or state.current_stage == WorkflowStage.ERROR:
        return "true"
    return "false"


def error_recoverable(state: TravelPlanningState) -> str:
    """Return true only for failures that are safe to retry."""
    error_text = str(state.error or "").lower()

    # Daily/provider quota and request-limit failures are not fixed by
    # immediately invoking the same node again.
    non_recoverable_markers = (
        "429",
        "rate limit",
        "rate-limit",
        "quota",
        "resource_exhausted",
        "too many requests",
        "requests/day",
        "requests per day",
        "daily limit",
        "permission denied",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "authentication",
        "validationerror",
        "validation error",
        "attributeerror",
        "typeerror",
        "keyerror",
        "importerror",
        "module not found",
        "no field",
        "has no field",
    )
    if any(marker in error_text for marker in non_recoverable_markers):
        return "false"

    if state.error_count > MAX_ERROR_COUNT:
        return "false"

    # Retry only failures that are normally transient.
    transient_markers = (
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "connection aborted",
        "server disconnected",
        "clientconnectorerror",
        "temporarily unavailable",
        "service unavailable",
        "502",
        "503",
        "504",
        "gateway timeout",
        "network error",
    )
    if not any(marker in error_text for marker in transient_markers):
        return "false"

    error_stage = state.error_node or (
        str(state.previous_stage) if state.previous_stage else "unknown"
    )
    return "true" if state.should_retry(error_stage) else "false"


def recover_to_stage(state: TravelPlanningState) -> str:
    """Translate the failed stage/node into a concrete LangGraph node."""
    stage_to_node = {
        WorkflowStage.START: "analyze_query",
        WorkflowStage.QUERY_ANALYZED: "research_destination",
        WorkflowStage.DESTINATION_RESEARCHED: "flight_search",
        WorkflowStage.FLIGHTS_SEARCHED: "combine_search_results",
        WorkflowStage.ACCOMMODATION_SEARCHED: "combine_search_results",
        WorkflowStage.TRANSPORTATION_PLANNED: "combine_search_results",
        WorkflowStage.PARALLEL_SEARCH_COMPLETED: "plan_activities",
        WorkflowStage.ACTIVITIES_PLANNED: "manage_budget",
        WorkflowStage.BUDGET_MANAGED: "generate_final_plan",
    }

    # Prefer the actual failed node. This matters for the three parallel
    # search branches: a failed accommodation search must not restart flights.
    failed_node = state.error_node
    valid_nodes = {
        "analyze_query",
        "research_destination",
        "flight_search",
        "accommodation_search",
        "transportation_planning",
        "combine_search_results",
        "plan_activities",
        "manage_budget",
        "generate_final_plan",
    }

    if failed_node in valid_nodes:
        node_name = failed_node
    else:
        previous_stage = state.previous_stage or WorkflowStage.START
        if not isinstance(previous_stage, WorkflowStage):
            try:
                previous_stage = WorkflowStage(previous_stage)
            except ValueError:
                previous_stage = WorkflowStage.START
        node_name = stage_to_node.get(previous_stage, "analyze_query")

    state.error = None
    state.error_node = None

    state.conversation_history.append(
        {
            "role": "system",
            "content": f"Recovering from transient error, retrying {node_name}",
        }
    )

    return node_name


def needs_human_intervention(state: TravelPlanningState) -> str:
    if state.guidance_requested:
        return "true"
    if state.error_count >= HUMAN_INTERVENTION_ERROR_THRESHOLD:
        return "true"
    if state.interrupted:
        return "true"
    return "false"


def continue_after_intervention(state: TravelPlanningState) -> str:
    state.guidance_requested = False
    if state.interrupted:
        return "END"
    return_stage = str(state.previous_stage) if state.previous_stage else "analyze_query"
    state.conversation_history.append(
        {
            "role": "system",
            "content": f"Continuing workflow at {return_stage} after human intervention",
        }
    )
    return return_stage


def plan_complete(state: TravelPlanningState) -> bool:
    if state.current_stage == WorkflowStage.COMPLETE:
        return True
    if not state.plan:
        return False
    required_fields = [
        state.plan.destination,
        state.plan.flights,
        state.plan.accommodation,
        state.plan.activities,
        state.plan.transportation,
        state.plan.budget,
    ]
    return all(field is not None for field in required_fields)
