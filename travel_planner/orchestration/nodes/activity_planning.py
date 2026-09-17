"""
Activity planning node implementation for the travel planning workflow.
"""

from typing import Any

from travel_planner.agents.activity_planning import (
    ActivityPlanningAgent,
)
from travel_planner.orchestration.nodes.base_node import (
    AgentTaskParams,
    execute_agent_task,
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


async def activity_planning(
    state: TravelPlanningState,
) -> TravelPlanningState:
    """
    Plan activities and create daily itineraries.
    """

    def result_formatter(
        result: dict[str, Any],
    ) -> str:
        daily_itineraries = result.get(
            "daily_itineraries",
            {},
        )

        num_days = len(
            daily_itineraries
        )

        return (
            f"Activities planned for "
            f"{num_days} days"
        )

    def result_processor(
        state: TravelPlanningState,
        result: dict[str, Any],
    ) -> None:
        if state.plan and "daily_itineraries" in result:
            # Convert the activity agent's internal dataclasses into the
            # Pydantic models used by TravelPlan. This avoids serializer
            # warnings and keeps dates as real ``date`` objects.
            from datetime import date
            from travel_planner.data.models import (
                Activity as PlanActivity,
                DailyItinerary as PlanDailyItinerary,
            )

            normalized = {}
            for index, (date_key, day) in enumerate(
                (result.get("daily_itineraries") or {}).items(),
                start=1,
            ):
                raw_date = getattr(day, "date", date_key)
                day_date = (
                    raw_date
                    if isinstance(raw_date, date)
                    else date.fromisoformat(str(raw_date))
                )

                activities = []
                for scheduled in getattr(day, "activities", []):
                    source = getattr(scheduled, "activity", scheduled)
                    try:
                        raw_type = getattr(source, "type", "sightseeing")
                        raw_type = getattr(raw_type, "value", raw_type)
                        activity_type_map = {
                            "attraction": "sightseeing",
                            "tour": "sightseeing",
                            "museum": "sightseeing",
                            "outdoor": "adventure",
                            "food_and_drink": "culinary",
                            "entertainment": "relaxation",
                            "wellness": "relaxation",
                            "cultural": "cultural",
                            "adventure": "adventure",
                            "shopping": "shopping",
                            "sightseeing": "sightseeing",
                            "relaxation": "relaxation",
                            "culinary": "culinary",
                        }
                        normalized_type = activity_type_map.get(str(raw_type).lower(), "sightseeing")
                        activities.append(
                            PlanActivity(
                                name=str(getattr(source, "name", "Activity")),
                                type=normalized_type,
                                description=str(getattr(source, "description", "")),
                                location=str(getattr(source, "location", "")),
                                duration_minutes=int(getattr(source, "duration_minutes", 0) or 0),
                                cost=float(getattr(source, "price", 0.0) or 0.0),
                                currency=str(getattr(source, "currency", "INR") or "INR"),
                                booking_required=bool(getattr(source, "booking_required", False)),
                                booking_link=getattr(source, "booking_url", None) or None,
                                highlights=list(getattr(source, "tags", []) or []),
                            )
                        )
                    except Exception:
                        logger.exception("Could not normalize an activity for %s", date_key)

                normalized[str(date_key)] = PlanDailyItinerary(
                    date=day_date,
                    day_number=index,
                    activities=activities,
                    notes=getattr(day, "notes", None) or None,
                    weather_forecast=getattr(day, "weather_forecast", None),
                )

            state.plan.activities = normalized

    params = AgentTaskParams(
        state=state,
        agent=ActivityPlanningAgent(),
        task_name="activity_planning",
        complete_stage=WorkflowStage.ACTIVITIES_PLANNED,
        result_formatter=result_formatter,
        result_processor=result_processor,
    )

    return await execute_agent_task(
        params
    )


async def activities_task(
    state: TravelPlanningState,
) -> dict[str, Any]:
    """
    Execute activity planning as a parallel branch.

    Only the activity-specific result is returned.
    """

    from travel_planner.orchestration.parallel import (
        ParallelResult,
        ParallelTask,
    )

    try:
        agent = ActivityPlanningAgent()

        user_query = get_user_query(state)

        result = await agent.run(
            user_query
        )

        return {
            "activities_search_result": ParallelResult(
                task_type=ParallelTask.ACTIVITIES,
                result=result,
                completed=True,
            )
        }

    except Exception as e:
        logger.error(
            f"Error in activities task: {e!s}"
        )

        return {
            "activities_search_result": ParallelResult(
                task_type=ParallelTask.ACTIVITIES,
                result={},
                error=str(e),
                completed=False,
            )
        }
