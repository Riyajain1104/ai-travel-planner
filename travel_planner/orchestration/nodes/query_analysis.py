"""
Query analysis node implementation for the travel planning workflow.

This module analyzes the user's travel query with the orchestrator agent,
then propagates the extracted travel requirements into TravelPlanningState.
"""

from datetime import date
from typing import Any

from travel_planner.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorContext,
    TravelRequirements,
)
from travel_planner.data.models import TravelQuery
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.helpers import generate_session_id
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def _to_date(value: Any) -> date | None:
    """Convert a supported date value into a date object."""
    if value is None or value == "":
        return None

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            logger.warning("Could not parse date value: %s", value)

    return None


def _requirements_to_query(
    existing_query: TravelQuery | None,
    requirements: TravelRequirements,
) -> TravelQuery:
    """
    Merge extracted TravelRequirements into the workflow's TravelQuery.

    Existing values are preserved whenever the orchestrator did not extract
    a replacement value.
    """
    existing = existing_query or TravelQuery()

    origin = getattr(requirements, "origin", None)
    destination = getattr(requirements, "destination", None)

    start_date = getattr(requirements, "start_date", None)
    end_date = getattr(requirements, "end_date", None)

    duration_days = getattr(requirements, "duration_days", None)

    budget = getattr(requirements, "budget", None)
    currency = getattr(requirements, "currency", None)
    num_travelers = getattr(requirements, "num_travelers", None)

    accommodation_preferences = getattr(
        requirements,
        "accommodation_preferences",
        None,
    )

    transportation_preferences = getattr(
        requirements,
        "transportation_preferences",
        None,
    )

    activity_preferences = getattr(
        requirements,
        "activity_preferences",
        None,
    )

    dietary_restrictions = getattr(
        requirements,
        "dietary_restrictions",
        None,
    )

    accessibility_needs = getattr(
        requirements,
        "accessibility_needs",
        None,
    )

    additional_notes = getattr(
        requirements,
        "additional_notes",
        None,
    )

    # Preserve the original user query.
    raw_query = existing.raw_query

    # TravelQuery currently exposes budget_range rather than a single
    # TravelRequirements.budget field.
    budget_range = existing.budget_range

    if budget is not None:
        budget_range = {
            "min": float(budget),
            "max": float(budget),
        }

    # Combine the extracted preference/restriction information into the
    # existing requirements field rather than introducing a new state field.
    extracted_requirements: dict[str, Any] = {}

    if accommodation_preferences:
        extracted_requirements["accommodation_preferences"] = (
            accommodation_preferences
        )

    if transportation_preferences:
        extracted_requirements["transportation_preferences"] = (
            transportation_preferences
        )

    if activity_preferences:
        extracted_requirements["activity_preferences"] = (
            activity_preferences
        )

    if dietary_restrictions:
        extracted_requirements["dietary_restrictions"] = (
            dietary_restrictions
        )

    if accessibility_needs:
        extracted_requirements["accessibility_needs"] = (
            accessibility_needs
        )

    if additional_notes:
        extracted_requirements["additional_notes"] = additional_notes

    if duration_days is not None:
        extracted_requirements["duration_days"] = duration_days

    if currency:
        extracted_requirements["currency"] = currency

    merged_requirements = dict(existing.requirements or {})
    merged_requirements.update(extracted_requirements)

    return TravelQuery(
        raw_query=raw_query,
        destination=destination or existing.destination,
        origin=origin or existing.origin,
        departure_date=(
            _to_date(start_date)
            if start_date is not None
            else existing.departure_date
        ),
        return_date=(
            _to_date(end_date)
            if end_date is not None
            else existing.return_date
        ),
        travelers=(
            int(num_travelers)
            if num_travelers is not None
            else existing.travelers
        ),
        budget_range=budget_range,
        purpose=existing.purpose,
        requirements=merged_requirements or None,
    )


async def query_analysis(state: TravelPlanningState) -> TravelPlanningState:
    """
    Analyze the user query and propagate extracted requirements into state.

    Args:
        state: Current travel planning state.

    Returns:
        Updated travel planning state.
    """
    logger.info("Starting query analysis")

    if state.query is None:
        state.query = TravelQuery(
            raw_query=(
                state.conversation_history[-1].get("content", "")
                if state.conversation_history
                else ""
            )
        )

    # Create a fresh orchestrator context for query analysis.
    context = OrchestratorContext(
        session_id=generate_session_id(),
        planning_stage="initial",
        travel_requirements=TravelRequirements(),
    )

    # Send only the structured query representation to the orchestrator.
    # Do NOT pass TravelPlanningState itself.
    user_input = state.query.model_dump_json()

    logger.info(
        "Running query analysis for: %s",
        state.query.raw_query,
    )

    try:
        result = await OrchestratorAgent().run(
            user_input,
            context,
        )
    except Exception as exc:
        logger.error(
            "Query analysis failed: %s",
            exc,
        )
        raise

    # The orchestrator returns its updated context.
    returned_context = result.get("context")

    if isinstance(returned_context, OrchestratorContext):
        requirements = returned_context.travel_requirements

        if isinstance(requirements, TravelRequirements):
            state.query = _requirements_to_query(
                state.query,
                requirements,
            )

            logger.info(
                "Extracted requirements: origin=%s, destination=%s, "
                "travelers=%s, budget=%s, dates=%s to %s",
                state.query.origin,
                state.query.destination,
                state.query.travelers,
                state.query.budget_range,
                state.query.departure_date,
                state.query.return_date,
            )
        else:
            logger.warning(
                "Orchestrator returned no valid TravelRequirements"
            )
    else:
        logger.warning(
            "Orchestrator result did not contain a valid context"
        )

    # Move the workflow to the analyzed stage only after the query has
    # been processed.
    state.update_stage(WorkflowStage.QUERY_ANALYZED)

    destination = (
        state.query.destination
        if state.query and state.query.destination
        else "Unknown"
    )

    logger.info(
        "Query analyzed. Destination: %s",
        destination,
    )

    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                f"Query analyzed: {destination}"
                if destination != "Unknown"
                else "Query analyzed: Destination research needed"
            ),
        }
    )

    state.add_task_result(
        "query_analysis",
        result,
    )

    return state
