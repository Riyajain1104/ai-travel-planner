"""
Accommodation search node implementation for the travel planning workflow.
"""

from typing import Any

from travel_planner.agents.accommodation import AccommodationAgent
from travel_planner.orchestration.nodes.base_node import (
    create_node_function,
    get_user_query,
)
from travel_planner.orchestration.states.planning_state import (
    TravelPlanningState,
)
from travel_planner.orchestration.states.workflow_stages import (
    WorkflowStage,
)
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


# Sequential accommodation-search node.
accommodation_search = create_node_function(
    agent_class=AccommodationAgent,
    task_name="accommodation_search",
    complete_stage=WorkflowStage.ACCOMMODATION_SEARCHED
)


async def accommodation_task(
    state: TravelPlanningState,
) -> dict[str, Any]:
    """
    Execute accommodation search as a parallel branch.

    Only the accommodation-specific result is returned.
    """

    from travel_planner.orchestration.parallel import (
        ParallelResult,
        ParallelTask,
    )

    try:
        agent = AccommodationAgent()

        user_query = get_user_query(state)

        result = await agent.run(
            user_query
        )

        return {
            "accommodation_search_result": ParallelResult(
                task_type=ParallelTask.ACCOMMODATION,
                result=result,
                completed=True,
            )
        }

    except Exception as e:
        logger.error(
            f"Error in accommodation task: {e!s}"
        )

        return {
            "accommodation_search_result": ParallelResult(
                task_type=ParallelTask.ACCOMMODATION,
                result={},
                error=str(e),
                completed=False,
            )
        }