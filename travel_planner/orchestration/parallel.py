"""
Parallel execution capabilities for the travel planner system.

This module provides concurrent execution of travel-planning agents while
maintaining compatibility with the project's existing state model and tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from travel_planner.agents.base import BaseAgent
from travel_planner.orchestration.states.workflow_stages import (
    PARALLEL_SEARCH_COMPLETED,
)
from travel_planner.utils.logging import get_logger


if TYPE_CHECKING:
    from travel_planner.orchestration.states.planning_state import (
        TravelPlanningState,
    )


TravelPlanningState = Any

T = TypeVar("T")
UpdateFunction = Callable[[T], T]

logger = get_logger(__name__)


class ParallelTask(str, Enum):
    """Types of tasks that can execute in parallel."""

    FLIGHT_SEARCH = "flight_search"
    ACCOMMODATION = "accommodation"
    TRANSPORTATION = "transportation"
    ACTIVITIES = "activities"
    BUDGET = "budget"


class ParallelResult(BaseModel):
    """Result produced by a parallel task."""

    task_type: ParallelTask
    result: dict[str, Any]
    error: str | None = None
    completed: bool = False


async def execute_in_parallel(
    tasks: list[tuple[BaseAgent, dict[str, Any]]],
    state: TravelPlanningState,
) -> dict[str, Any]:
    """
    Execute multiple agent tasks concurrently.

    Args:
        tasks: List of ``(agent, parameters)`` tuples.
        state: Current travel-planning state.

    Returns:
        Dictionary keyed by agent name.

    Example:
        {
            "FlightSearchAgent": {
                "result": {...},
                "error": None,
                "retries": 0,
            }
        }
    """

    async def execute_task(
        agent: BaseAgent,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one agent task."""

        try:
            logger.info(
                "Starting parallel task: %s",
                agent.name,
            )

            # Some existing agents implement process() as async while
            # test mocks may implement it synchronously.
            result = agent.process(
                **params,
                context=state,
            )

            if asyncio.iscoroutine(result):
                result = await result

            logger.info(
                "Completed parallel task: %s",
                agent.name,
            )

            return {
                agent.name: {
                    "result": result,
                    "error": None,
                    "retries": 0,
                }
            }

        except Exception as exc:
            logger.error(
                "Error in parallel task %s: %s",
                agent.name,
                exc,
            )

            return {
                agent.name: {
                    "result": None,
                    "error": str(exc),
                    "retries": 0,
                }
            }

    coroutines = [
        execute_task(agent, params)
        for agent, params in tasks
    ]

    try:
        results = await asyncio.gather(
            *coroutines
        )

        combined_results: dict[str, Any] = {}

        for result in results:
            combined_results.update(result)

        return combined_results

    except asyncio.TimeoutError:
        logger.error(
            "Parallel execution timed out"
        )

        return {
            "error": "Execution timeout exceeded"
        }


async def parallel_search_tasks(
    state: TravelPlanningState,
) -> TravelPlanningState:
    """
    Execute flight, accommodation, transportation and activity tasks
    concurrently.

    This function is retained for backwards compatibility with the original
    orchestration API.
    """

    from travel_planner.agents.accommodation import (
        AccommodationAgent,
    )
    from travel_planner.agents.activity_planning import (
        ActivityPlanningAgent,
    )
    from travel_planner.agents.flight_search import (
        FlightSearchAgent,
    )
    from travel_planner.agents.transportation import (
        TransportationAgent,
    )

    from travel_planner.data.models import TravelPlan

    logger.info(
        "Setting up parallel search tasks"
    )

    working_state = state.model_copy(
        deep=True
    )

    try:
        agents = {
            "flight": FlightSearchAgent(),
            "accommodation": AccommodationAgent(),
            "transportation": TransportationAgent(),
            "activity": ActivityPlanningAgent(),
        }

        tasks = [
            (
                agents["flight"],
                {
                    "query": working_state.query,
                    "timeout": 60,
                },
            ),
            (
                agents["accommodation"],
                {
                    "query": working_state.query,
                    "timeout": 60,
                },
            ),
            (
                agents["transportation"],
                {
                    "query": working_state.query,
                    "timeout": 45,
                },
            ),
            (
                agents["activity"],
                {
                    "query": working_state.query,
                    "preferences": working_state.preferences,
                    "timeout": 60,
                },
            ),
        ]

        logger.info(
            "Executing %s tasks in parallel",
            len(tasks),
        )

        async with asyncio.timeout(180):

            results = await execute_in_parallel(
                tasks,
                working_state,
            )

        if (
            "error" in results
            and not any(
                key != "error"
                for key in results
            )
        ):
            working_state.error = (
                f"Parallel execution error: "
                f"{results['error']}"
            )

            if working_state.plan is None:
                working_state.plan = TravelPlan()

            if not working_state.plan.alerts:
                working_state.plan.alerts = []

            working_state.plan.alerts.append(
                f"Error in parallel search: "
                f"{results['error']}"
            )

            working_state.current_stage = "error"

            return working_state

        updated_state = merge_parallel_results(
            working_state,
            results,
        )

        updated_state.current_stage = (
            PARALLEL_SEARCH_COMPLETED
        )

        logger.info(
            "Parallel search tasks completed successfully"
        )

        return updated_state

    except TimeoutError:

        logger.error(
            "Parallel search tasks timed out"
        )

        working_state.error = (
            "Parallel search timed out"
        )

        if working_state.plan is None:
            working_state.plan = TravelPlan()

        if not working_state.plan.alerts:
            working_state.plan.alerts = []

        working_state.plan.alerts.append(
            "Search operations timed out. "
            "Some results may be incomplete."
        )

        working_state.current_stage = "error"

        return working_state

    except Exception as exc:

        logger.error(
            "Unexpected error in parallel search: %s",
            exc,
        )

        working_state.error = (
            f"Unexpected error: {exc!s}"
        )

        if working_state.plan is None:
            working_state.plan = TravelPlan()

        if not working_state.plan.alerts:
            working_state.plan.alerts = []

        working_state.plan.alerts.append(
            f"Unexpected error in search: {exc!s}"
        )

        working_state.current_stage = "error"

        return working_state


def merge_parallel_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:
    """
    Merge traditional parallel-agent results into the travel plan.

    A deep copy is used so that callers do not unexpectedly mutate their
    original Pydantic state.
    """

    updated_state = state.model_copy(
        deep=True
    )

    updated_state = _ensure_plan_initialized(
        updated_state
    )

    updated_state = _process_flight_results(
        updated_state,
        results,
    )

    updated_state = _process_accommodation_results(
        updated_state,
        results,
    )

    updated_state = _process_transportation_results(
        updated_state,
        results,
    )

    updated_state = _process_activity_results(
        updated_state,
        results,
    )

    updated_state = _process_parallel_errors(
        updated_state,
        results,
    )

    return updated_state


def _ensure_plan_initialized(
    state: TravelPlanningState,
) -> TravelPlanningState:

    from travel_planner.data.models import TravelPlan

    if state.plan is None:
        state.plan = TravelPlan()

    if not state.plan.alerts:
        state.plan.alerts = []

    return state


def _process_flight_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:

    agent_result = results.get(
        "FlightSearchAgent"
    )

    if not agent_result:
        return state

    result = agent_result.get("result")

    if not result:
        return state

    if "flights" in result:
        state.plan.flights = result["flights"]

    elif "flight_options" in result:
        state.plan.flights = result[
            "flight_options"
        ]

    return state


def _process_accommodation_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:

    agent_result = results.get(
        "AccommodationAgent"
    )

    if not agent_result:
        return state

    result = agent_result.get("result")

    if not result:
        return state

    if "accommodations" in result:
        state.plan.accommodation = result[
            "accommodations"
        ]

    elif "accommodation_options" in result:
        state.plan.accommodation = result[
            "accommodation_options"
        ]

    return state


def _process_transportation_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:

    agent_result = results.get(
        "TransportationAgent"
    )

    if not agent_result:
        return state

    result = agent_result.get("result")

    if not result:
        return state

    if "transportation" in result:
        state.plan.transportation = result[
            "transportation"
        ]

    elif "transportation_options" in result:
        state.plan.transportation = result[
            "transportation_options"
        ]

    return state


def _process_activity_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:

    agent_result = results.get(
        "ActivityPlanningAgent"
    )

    if not agent_result:
        return state

    result = agent_result.get("result")

    if not result:
        return state

    if "activities" in result:
        state.plan.activities = result[
            "activities"
        ]

    elif "daily_itineraries" in result:
        state.plan.activities = result[
            "daily_itineraries"
        ]

    return state


def _process_parallel_errors(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:

    errors: list[str] = []

    for agent_name, result in results.items():

        if not isinstance(result, dict):
            continue

        error = result.get("error")

        if error:
            errors.append(
                f"{agent_name}: {error}"
            )

    if errors:

        if not state.plan.alerts:
            state.plan.alerts = []

        state.plan.alerts.extend(
            errors
        )

    return state


def combine_parallel_branch_results(
    state: TravelPlanningState,
    branch_results: dict[str, ParallelResult],
) -> TravelPlanningState:
    """
    Combine results generated by LangGraph-style parallel branches.
    """

    updated_state = state.model_copy(
        deep=True
    )

    updated_state = _ensure_plan_initialized(
        updated_state
    )

    if not _validate_branch_results(
        branch_results
    ):
        logger.warning(
            "No results from parallel branch execution"
        )

        return updated_state

    results_by_task = (
        _organize_branch_results(
            branch_results
        )
    )

    _process_branch_flight_results(
        updated_state,
        results_by_task,
    )

    _process_branch_accommodation_results(
        updated_state,
        results_by_task,
    )

    _process_branch_transportation_results(
        updated_state,
        results_by_task,
    )

    _process_branch_activity_results(
        updated_state,
        results_by_task,
    )

    _process_branch_budget_results(
        updated_state,
        results_by_task,
    )

    _process_branch_errors(
        updated_state,
        results_by_task,
    )

    _update_workflow_stage(
        updated_state
    )

    return updated_state


def _validate_branch_results(
    branch_results: dict[str, ParallelResult],
) -> bool:

    if not branch_results:
        return False

    if branch_results.get("result"):
        return True

    for value in branch_results.values():

        if (
            isinstance(value, ParallelResult)
            and value.completed
            and not value.error
        ):
            return True

    # Also accept completed/error-only branch results.
    return any(
        isinstance(value, ParallelResult)
        for value in branch_results.values()
    )


def _organize_branch_results(
    branch_results: dict[str, ParallelResult],
) -> dict[ParallelTask, ParallelResult]:

    results_by_task: dict[
        ParallelTask,
        ParallelResult,
    ] = {}

    for value in branch_results.values():

        if isinstance(
            value,
            ParallelResult,
        ):
            results_by_task[
                value.task_type
            ] = value

    return results_by_task


def _process_branch_flight_results(
    state: TravelPlanningState,
    results_by_task: dict[
        ParallelTask,
        ParallelResult,
    ],
) -> TravelPlanningState:

    result = results_by_task.get(
        ParallelTask.FLIGHT_SEARCH
    )

    if (
        result
        and result.completed
        and not result.error
    ):
        state.plan.flights = (
            result.result.get(
                "flight_options",
                result.result.get(
                    "flights",
                    [],
                ),
            )
        )

    return state


def _process_branch_accommodation_results(
    state: TravelPlanningState,
    results_by_task: dict[
        ParallelTask,
        ParallelResult,
    ],
) -> TravelPlanningState:

    result = results_by_task.get(
        ParallelTask.ACCOMMODATION
    )

    if (
        result
        and result.completed
        and not result.error
    ):
        state.plan.accommodation = (
            result.result.get(
                "accommodations",
                result.result.get(
                    "accommodation_options",
                    [],
                ),
            )
        )

    return state


def _process_branch_transportation_results(
    state: TravelPlanningState,
    results_by_task: dict[
        ParallelTask,
        ParallelResult,
    ],
) -> TravelPlanningState:

    result = results_by_task.get(
        ParallelTask.TRANSPORTATION
    )

    if (
        result
        and result.completed
        and not result.error
    ):
        state.plan.transportation = (
            result.result.get(
                "transportation_options",
                result.result.get(
                    "transportation",
                    {},
                ),
            )
        )

    return state


def _process_branch_activity_results(
    state: TravelPlanningState,
    results_by_task: dict[
        ParallelTask,
        ParallelResult,
    ],
) -> TravelPlanningState:

    result = results_by_task.get(
        ParallelTask.ACTIVITIES
    )

    if (
        result
        and result.completed
        and not result.error
    ):
        state.plan.activities = (
            result.result.get(
                "daily_itineraries",
                result.result.get(
                    "activities",
                    {},
                ),
            )
        )

    return state


def _process_branch_budget_results(
    state: TravelPlanningState,
    results_by_task: dict[
        ParallelTask,
        ParallelResult,
    ],
) -> TravelPlanningState:

    result = results_by_task.get(
        ParallelTask.BUDGET
    )

    if (
        result
        and result.completed
        and not result.error
    ):
        state.plan.budget = (
            result.result.get(
                "report",
                result.result.get(
                    "budget",
                    {},
                ),
            )
        )

    return state


def _process_branch_errors(
    state: TravelPlanningState,
    results_by_task: dict[
        ParallelTask,
        ParallelResult,
    ],
) -> TravelPlanningState:

    errors: list[str] = []

    for task_type, result in results_by_task.items():

        if result.error:
            errors.append(
                f"{task_type.value}: "
                f"{result.error}"
            )

    if errors:

        if not state.plan.alerts:
            state.plan.alerts = []

        state.plan.alerts.extend(
            errors
        )

    return state


def _update_workflow_stage(
    state: TravelPlanningState,
) -> TravelPlanningState:

    state.current_stage = (
        PARALLEL_SEARCH_COMPLETED
    )

    return state


def create_parallel_search_branch(
    state: TravelPlanningState | None = None,
) -> list[Callable[..., Any]]:
    """
    Return the current native-LangGraph parallel search nodes.
    """

    from travel_planner.orchestration.nodes.accommodation_search import (
        accommodation_task,
    )
    from travel_planner.orchestration.nodes.flight_search import (
        flight_search_task,
    )
    from travel_planner.orchestration.nodes.transportation_planning import (
        transportation_task,
    )

    return [
        flight_search_task,
        accommodation_task,
        transportation_task,
    ]


def create_parallel_task(
    task_type: ParallelTask,
    agent: BaseAgent,
    params: dict[str, Any] | None = None,
) -> tuple[BaseAgent, dict[str, Any]]:
    """Create a task tuple."""

    return (
        agent,
        params or {},
    )


async def run_parallel_tasks(
    tasks: list[tuple[BaseAgent, dict[str, Any]]],
    state: TravelPlanningState,
) -> dict[str, Any]:
    """Alias for execute_in_parallel()."""

    return await execute_in_parallel(
        tasks,
        state,
    )