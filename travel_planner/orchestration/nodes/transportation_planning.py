"""
Transportation planning node implementation for the travel planning workflow.
"""

from typing import Any

from travel_planner.agents.transportation import (
    TransportationAgent,
)
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


# Sequential transportation-planning node.
transportation_planning = create_node_function(
    agent_class=TransportationAgent,
    task_name="transportation_planning",
    complete_stage=WorkflowStage.TRANSPORTATION_PLANNED
)


async def transportation_task(
    state: TravelPlanningState,
) -> dict[str, Any]:
    """
    Execute transportation planning as a parallel branch.

    Only the transportation-specific result is returned.
    """

    from travel_planner.orchestration.parallel import (
        ParallelResult,
        ParallelTask,
    )

    try:
        agent = TransportationAgent()

        user_query = get_user_query(state)

        result = await agent.run(
            user_query
        )

        return {
            "transportation_search_result": ParallelResult(
                task_type=ParallelTask.TRANSPORTATION,
                result=result,
                completed=True,
            )
        }

    except Exception as e:
        logger.error(
            f"Error in transportation task: {e!s}"
        )

        return {
            "transportation_search_result": ParallelResult(
                task_type=ParallelTask.TRANSPORTATION,
                result={},
                error=str(e),
                completed=False,
            )
        }