"""
Flight search node implementation for the travel planning workflow.
"""

from typing import Any

from travel_planner.agents.flight_search import FlightSearchAgent
from travel_planner.orchestration.nodes.base_node import (
    create_node_function,
    build_agent_input,
)
from travel_planner.orchestration.states.planning_state import (
    TravelPlanningState,
)
from travel_planner.orchestration.states.workflow_stages import (
    WorkflowStage,
)
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


# Sequential flight-search node.
flight_search = create_node_function(
    agent_class=FlightSearchAgent,
    task_name="flight_search",
    complete_stage=WorkflowStage.FLIGHTS_SEARCHED
)


async def flight_search_task(
    state: TravelPlanningState,
) -> dict[str, Any]:
    """
    Execute flight search as a parallel branch.

    IMPORTANT:
    This function returns only the flight-specific result.
    It does not return TravelPlanningState, preventing parallel
    branches from overwriting common state fields such as query.
    """

    from travel_planner.orchestration.parallel import (
        ParallelResult,
        ParallelTask,
    )

    try:
        agent = FlightSearchAgent()

        user_query = build_agent_input(state)

        result = await agent.run(
            user_query
        )

        return {
            "flight_search_result": ParallelResult(
                task_type=ParallelTask.FLIGHT_SEARCH,
                result=result,
                completed=True,
            )
        }

    except Exception as e:
        logger.error(
            f"Error in flight search task: {e!s}"
        )

        return {
            "flight_search_result": ParallelResult(
                task_type=ParallelTask.FLIGHT_SEARCH,
                result={},
                error=str(e),
                completed=False,
            )
        }