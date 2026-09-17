"""
Final plan generation node implementation for the travel planning workflow.

This module defines the function that generates the final travel plan by
integrating all components and creating a comprehensive output.
"""

from travel_planner.agents.orchestrator import OrchestratorAgent
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


async def generate_final_plan(
    state: TravelPlanningState,
) -> TravelPlanningState:
    """
    Generate the final travel plan with all components.

    Args:
        state: Current travel planning state

    Returns:
        Updated travel planning state with complete plan.
    """
    from travel_planner.orchestration.serialization.checkpoint import (
        save_state_checkpoint,
    )

    try:
        logger.info("Generating final travel plan")

        # The specialized agents have already populated state.plan. Do not
        # send the TravelPlanningState back through the orchestrator: its
        # public API accepts text/list input, and doing so causes a second
        # Gemini extraction call and can corrupt typed plan data.
        result = {"final_plan": state.plan}
        state.update_stage(WorkflowStage.COMPLETE)

        # Create and save a checkpoint of the final state.
        checkpoint_id = save_state_checkpoint(state)
        state.travel_checkpoint_id = checkpoint_id

        # Add completion info to conversation history.
        state.conversation_history.append(
            {
                "role": "system",
                "content": (
                    "Travel planning completed successfully. "
                    f"Final plan saved as checkpoint {checkpoint_id}"
                ),
            }
        )

        # Record task result.
        state.add_task_result("generate_final_plan", result)

        logger.info(
            f"Final plan generated successfully. Checkpoint ID: {checkpoint_id}"
        )
        return state

    except Exception as e:
        logger.error(f"Error during final plan generation: {e!s}")
        state.mark_error(f"Error during final plan generation: {e!s}")

        if state.should_retry("generate_final_plan"):
            logger.info("Will retry final plan generation")

        return state