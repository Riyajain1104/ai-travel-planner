"""
Regression tests for budget management fallback behavior.
"""

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from travel_planner.data.models import (
    Activity,
    ActivityType,
    DailyItinerary,
    TravelPlan,
    TravelQuery,
)
from travel_planner.orchestration.nodes.budget_management import budget_management
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage


@pytest.mark.asyncio
async def test_budget_fallback_includes_itinerary_activity_costs():
    """Local budget fallback should calculate spent from scheduled activities."""

    state = TravelPlanningState(
        query=TravelQuery(
            raw_query="Plan a trip to Jaipur",
            budget_range={"min": 10000, "max": 15000},
            travelers=2,
        ),
        plan=TravelPlan(
            activities={
                "2026-10-10": DailyItinerary(
                    date=date(2026, 10, 10),
                    day_number=1,
                    activities=[
                        Activity(
                            name="Amber Fort",
                            type=ActivityType.SIGHTSEEING,
                            description="Visit Amber Fort",
                            location="Jaipur",
                            duration_minutes=120,
                            cost=200,
                            currency="INR",
                        ),
                        Activity(
                            name="Jal Mahal",
                            type=ActivityType.SIGHTSEEING,
                            description="Visit Jal Mahal",
                            location="Jaipur",
                            duration_minutes=60,
                            cost=0,
                            currency="INR",
                        ),
                    ],
                ),
            }
        ),
        current_stage=WorkflowStage.ACTIVITIES_PLANNED,
    )

    quota_result = {
        "error": "429 RESOURCE_EXHAUSTED: quota exceeded"
    }

    with patch(
        "travel_planner.orchestration.nodes.budget_management.BudgetManagementAgent.run",
        new=AsyncMock(return_value=quota_result),
    ):
        result = await budget_management(state)

    assert result.current_stage == WorkflowStage.BUDGET_MANAGED
    assert result.plan.budget is not None
    assert result.plan.budget.total_budget == pytest.approx(15000.0)
    assert result.plan.budget.spent == pytest.approx(200.0)
    assert result.plan.budget.remaining == pytest.approx(14800.0)
