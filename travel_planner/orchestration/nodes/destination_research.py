"""
Destination research node implementation for the travel planning workflow.

This module defines the function that researches travel destinations
using the destination research agent.
"""

from travel_planner.agents.destination_research import DestinationResearchAgent
from travel_planner.orchestration.nodes.base_node import (
    AgentTaskParams,
    execute_agent_task,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


async def destination_research(state: TravelPlanningState) -> TravelPlanningState:
    """
    Research destination information based on user query.

    Args:
        state: Current travel planning state

    Returns:
        Updated travel planning state with destination information
    """

    def result_formatter(result):
        destination_details = result.get("destination_details", {})
        destination_name = destination_details.get("name", "Unknown destination")
        return f"Destination researched: {destination_name}"

    def result_processor(state, result):
        destination_details = result.get("destination_details", {})
        if state.plan:
            state.plan.destination = destination_details

    params = AgentTaskParams(
        state=state,
        agent=DestinationResearchAgent(),
        task_name="destination_research",
        complete_stage=WorkflowStage.DESTINATION_RESEARCHED,
        result_formatter=result_formatter,
        result_processor=result_processor,
    )

    return await execute_agent_task(params)
