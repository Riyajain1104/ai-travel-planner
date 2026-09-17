"""
Graph builder for the travel planner workflow.

This module builds the LangGraph state graph and defines the workflow
transitions, including native fan-out/fan-in parallel search execution.
"""

from langgraph.graph import END, START, StateGraph

from travel_planner.orchestration.nodes.accommodation_search import (
    accommodation_task,
)
from travel_planner.orchestration.nodes.activity_planning import activity_planning
from travel_planner.orchestration.nodes.budget_management import budget_management
from travel_planner.orchestration.nodes.destination_research import (
    destination_research,
)
from travel_planner.orchestration.nodes.final_plan import generate_final_plan
from travel_planner.orchestration.nodes.flight_search import flight_search_task
from travel_planner.orchestration.nodes.parallel_search import combine_search_results
from travel_planner.orchestration.nodes.query_analysis import query_analysis
from travel_planner.orchestration.nodes.transportation_planning import (
    transportation_task,
)
from travel_planner.orchestration.routing.conditions import (
    error_recoverable,
    has_error,
    query_research_needed,
    recover_to_stage,
)
from travel_planner.orchestration.routing.error_recovery import (
    handle_error,
    handle_interruption,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def route_after_query_analysis(state: TravelPlanningState):
    """
    Decide the next workflow step after query analysis.

    Errors go to error handling. Otherwise, the query determines whether
    destination research is required.
    """
    if has_error(state) == "true":
        return "handle_error"

    return query_research_needed(state)


def route_after_destination_research(state: TravelPlanningState):
    """
    Route destination research either to error handling or to all
    three parallel search branches.
    """
    if has_error(state) == "true":
        return "handle_error"

    return [
        "flight_search",
        "accommodation_search",
        "transportation_planning",
    ]


def route_after_search(state: TravelPlanningState):
    """
    Route a completed search branch.

    Successful branches continue to the fan-in node. Errors go to the
    centralized error handler.
    """
    if has_error(state) == "true":
        return "handle_error"

    return "combine_search_results"


def route_after_node(state: TravelPlanningState, success_target: str):
    """
    Generic router for sequential workflow nodes.

    Errors are sent to the error handler; successful execution continues
    to the supplied target.
    """
    if has_error(state) == "true":
        return "handle_error"

    return success_target


def create_planning_graph():
    """
    Create and compile the travel planning workflow.

    The graph uses native LangGraph fan-out/fan-in execution for flight,
    accommodation, and transportation searches.
    """
    logger.info("Creating planning graph")

    workflow = StateGraph(TravelPlanningState)

    # ------------------------------------------------------------------
    # Register workflow nodes
    # ------------------------------------------------------------------

    workflow.add_node("analyze_query", query_analysis)
    workflow.add_node("research_destination", destination_research)

    # Parallel search branches
    workflow.add_node("flight_search", flight_search_task)
    workflow.add_node("accommodation_search", accommodation_task)
    workflow.add_node("transportation_planning", transportation_task)

    # Fan-in node
    workflow.add_node("combine_search_results", combine_search_results)

    # Remaining workflow
    workflow.add_node("plan_activities", activity_planning)
    workflow.add_node("manage_budget", budget_management)
    workflow.add_node("generate_final_plan", generate_final_plan)

    # Error/interruption handling
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("handle_interruption", handle_interruption)

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    workflow.add_edge(START, "analyze_query")

    # ------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "analyze_query",
        route_after_query_analysis,
        {
            "handle_error": "handle_error",
            "research_destination": "research_destination",
            "flight_search": "flight_search",
        },
    )

    # ------------------------------------------------------------------
    # Destination research -> parallel fan-out
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "research_destination",
        route_after_destination_research,
        {
            "handle_error": "handle_error",
            "flight_search": "flight_search",
            "accommodation_search": "accommodation_search",
            "transportation_planning": "transportation_planning",
        },
    )

    # ------------------------------------------------------------------
    # Parallel fan-in
    #
    # Each branch has exactly ONE successful route to the combine node.
    # LangGraph waits for all three upstream branches before executing
    # the fan-in node.
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "flight_search",
        route_after_search,
        {
            "handle_error": "handle_error",
            "combine_search_results": "combine_search_results",
        },
    )

    workflow.add_conditional_edges(
        "accommodation_search",
        route_after_search,
        {
            "handle_error": "handle_error",
            "combine_search_results": "combine_search_results",
        },
    )

    workflow.add_conditional_edges(
        "transportation_planning",
        route_after_search,
        {
            "handle_error": "handle_error",
            "combine_search_results": "combine_search_results",
        },
    )

    # ------------------------------------------------------------------
    # Continue after parallel search
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "combine_search_results",
        lambda state: route_after_node(state, "plan_activities"),
        {
            "handle_error": "handle_error",
            "plan_activities": "plan_activities",
        },
    )

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "plan_activities",
        lambda state: route_after_node(state, "manage_budget"),
        {
            "handle_error": "handle_error",
            "manage_budget": "manage_budget",
        },
    )

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "manage_budget",
        lambda state: route_after_node(state, "generate_final_plan"),
        {
            "handle_error": "handle_error",
            "generate_final_plan": "generate_final_plan",
        },
    )

    # ------------------------------------------------------------------
    # Final plan
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "generate_final_plan",
        lambda state: route_after_node(state, END),
        {
            "handle_error": "handle_error",
            END: END,
        },
    )

                   

    # ------------------------------------------------------------------
    # Error recovery
    # ------------------------------------------------------------------

    def route_after_error(state: TravelPlanningState):
        """
        Route recoverable errors back to the appropriate workflow stage.

        LangGraph conditional edges must resolve to registered node names,
        so the recovery stage returned by recover_to_stage() is used
        directly as the destination name.
        """
        if error_recoverable(state) != "true":
            return END

        return recover_to_stage(state)

    workflow.add_conditional_edges(
        "handle_error",
        route_after_error,
        {
            "analyze_query": "analyze_query",
            "research_destination": "research_destination",
            "flight_search": "flight_search",
            "accommodation_search": "accommodation_search",
            "transportation_planning": "transportation_planning",
            "combine_search_results": "combine_search_results",
            "plan_activities": "plan_activities",
            "manage_budget": "manage_budget",
            "generate_final_plan": "generate_final_plan",
            END: END,
        },
    )
    logger.info("Planning graph created and compiled")

    return workflow.compile()