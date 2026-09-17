"""
Budget management node implementation for the travel planning workflow.

This module defines functions for managing the travel budget using the budget
management agent, both for individual execution and as part of parallel processing.
"""

from travel_planner.agents.budget_management import BudgetManagementAgent
from travel_planner.orchestration.nodes.base_node import (
    AgentTaskParams,
    build_agent_input,
    execute_agent_task,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def _build_local_budget_result(state: TravelPlanningState) -> dict:
    """Build a deterministic budget when the LLM budget agent is unavailable."""
    from travel_planner.agents.budget_management import BudgetAllocation, BudgetContext, ExpenseCategory
    query = state.query
    budget = 0.0
    currency = "INR"
    travelers = 1
    if query and query.budget_range:
        # TravelQuery stores budget_range as a plain dict after LangGraph/Pydantic
        # state serialization, so support both dict and model-like values.
        budget_range = query.budget_range
        if isinstance(budget_range, dict):
            budget = float(
                budget_range.get("max")
                or budget_range.get("min")
                or 0.0
            )
        else:
            budget = float(
                getattr(budget_range, "max", None)
                or getattr(budget_range, "min", None)
                or 0.0
            )
        requirements = query.requirements or {}
        currency = str(requirements.get("currency", "INR") or "INR")
        travelers = int(query.travelers or 1)
    context = BudgetContext(total_budget=budget, currency=currency, traveler_count=travelers)
    percentages = {ExpenseCategory.ACCOMMODATION:30, ExpenseCategory.FLIGHTS:25, ExpenseCategory.FOOD:20, ExpenseCategory.ACTIVITIES:15, ExpenseCategory.TRANSPORTATION:5, ExpenseCategory.SHOPPING:3, ExpenseCategory.MISCELLANEOUS:2}
    for category, percentage in percentages.items():
        context.allocations[category] = BudgetAllocation(category=category, amount=budget*percentage/100, currency=currency, percentage=percentage)
    return {"context": context, "allocations": context.allocations, "expenses": [], "recommendations": [], "alerts": ["Budget calculated locally because the Gemini budget service was unavailable."], "report": "Local fallback budget allocation used; no LLM-generated recommendations were available."}


async def budget_management(state: TravelPlanningState) -> TravelPlanningState:
    """
    Manage and optimize the budget for the trip.

    Args:
        state: Current travel planning state

    Returns:
        Updated travel planning state with budget information
    """

    def result_formatter(result):
        if isinstance(result, dict):
            budget_report = result.get("report", {})
            if isinstance(budget_report, dict):
                total_budget = budget_report.get("total_budget", "Unknown")
                return f"Budget plan created with total: {total_budget}"
            return f"Budget plan created: {str(budget_report)[:500]}"

        return f"Budget plan created: {str(result)[:500]}"

    def result_processor(state, result):
        if not state.plan or not isinstance(result, dict):
            return

        # BudgetManagementAgent returns a human-readable report string plus
        # structured context. Store the structured values in BudgetSummary
        # instead of assigning the report text to a typed Pydantic field.
        context = result.get("context")
        if context is None:
            return

        from travel_planner.data.models import BudgetSummary

        total_budget = float(getattr(context, "total_budget", 0.0) or 0.0)
        # Calculate known planned expenses from the budget context.
        spent = float(
            sum(
                float(getattr(expense, "amount", 0.0) or 0.0)
                for expense in getattr(context, "expenses", [])
            )
        )

        # The local fallback may not contain structured expenses.
        # In that case, include the costs of activities already scheduled
        # in the generated itinerary.
        if spent == 0.0 and state.plan.activities:
            activity_spend = 0.0

            for itinerary in state.plan.activities.values():
                for activity in getattr(itinerary, "activities", []):
                    cost = getattr(activity, "cost", None)
                    if cost is not None:
                        activity_spend += float(cost or 0.0)

            spent = activity_spend
        breakdown = {}
        category_map = {
            "flights": "flights",
            "accommodation": "accommodation",
            "transportation": "local_transportation",
            "local_transportation": "local_transportation",
            "activities": "activities",
            "food": "food",
            "shopping": "shopping",
            "miscellaneous": "miscellaneous",
        }
        for category, allocation in getattr(context, "allocations", {}).items():
            amount = float(getattr(allocation, "amount", 0.0) or 0.0)
            raw_category = getattr(category, "value", category)
            normalized_category = category_map.get(str(raw_category), str(raw_category))
            breakdown[normalized_category] = amount

        state.plan.budget = BudgetSummary(
            total_budget=total_budget,
            currency=str(getattr(context, "currency", "INR") or "INR"),
            spent=spent,
            remaining=total_budget - spent,
            breakdown=breakdown,
            notes=str(result.get("report", ""))[:2000] or None,
            saving_recommendations=[
                str(rec) for rec in getattr(context, "recommendations", [])
            ],
        )


    params = AgentTaskParams(
        state=state,
        agent=BudgetManagementAgent(),
        task_name="budget_management",
        complete_stage=WorkflowStage.BUDGET_MANAGED,
        result_formatter=result_formatter,
        result_processor=result_processor,
    )

    result = await BudgetManagementAgent().run(build_agent_input(state))
    if isinstance(result, dict) and result.get("error"):
        error_text = str(result["error"]).lower()
        if any(marker in error_text for marker in ("429", "quota", "resource_exhausted", "503", "service unavailable", "temporarily unavailable")):
            logger.warning("Budget agent unavailable (%s); using local budget fallback.", result["error"])
            result = _build_local_budget_result(state)
    if isinstance(result, dict) and not result.get("error"):
        result_processor(state, result)
        state.current_stage = WorkflowStage.BUDGET_MANAGED
        state.error = None
        state.error_node = None
        return state
    return await execute_agent_task(params)


async def budget_task(state: TravelPlanningState) -> dict[str, any]:
    """
    Execute budget management task (usually runs after other tasks are complete).

    Args:
        state: Current travel planning state

    Returns:
        Dictionary with task results
    """
    from travel_planner.orchestration.parallel import ParallelResult, ParallelTask

    try:
        agent = BudgetManagementAgent()
        result = await agent.run(state)

        return {
            "result": ParallelResult(
                task_type=ParallelTask.BUDGET, result=result, completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in budget task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.BUDGET, result={}, error=str(e), completed=False
            )
        }
