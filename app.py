import asyncio
from datetime import date

import streamlit as st

from travel_planner.data.models import TravelQuery
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.workflow import TravelWorkflow


st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="",
    layout="wide",
)


def format_currency(value, currency="INR"):
    return f"{currency} {float(value):,.0f}"


async def generate_plan(
    origin,
    destination,
    departure_date,
    return_date,
    travelers,
    budget_min,
    budget_max,
):
    query = TravelQuery(
        raw_query=(
            f"Plan a trip from {origin} to {destination} "
            f"for {travelers} travelers from "
            f"{departure_date} to {return_date}"
        ),
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        travelers=travelers,
        budget_range={
            "min": float(budget_min),
            "max": float(budget_max),
        },
    )

    state = TravelPlanningState(query=query)

    workflow = TravelWorkflow()

    return await workflow.execute(state)


def display_plan(state):
    plan = state.plan

    if not plan:
        st.error("No travel plan was generated.")
        return

    destination = plan.destination or {}

    if isinstance(destination, dict):
        destination_name = destination.get("name", "Your Destination")
    else:
        destination_name = str(destination)

    st.success(f"Travel plan generated for {destination_name}")

    if plan.overview:
        st.markdown("##  Trip Overview")
        st.write(plan.overview)

    if plan.flights:
        st.markdown("##  Flights")

        for flight in plan.flights:
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Route**")
                    st.write(
                        f"{flight.departure_location} ? "
                        f"{flight.arrival_location}"
                    )

                with col2:
                    st.write("**Airline**")
                    st.write(flight.airline)

                with col3:
                    st.write("**Price**")
                    st.write(
                        format_currency(
                            flight.price,
                            flight.currency,
                        )
                    )

    if plan.accommodation:
        st.markdown("##  Accommodation")

        for accommodation in plan.accommodation:
            with st.container(border=True):
                st.subheader(accommodation.name)
                st.write(accommodation.address)
                st.write(
                    f"**Price:** "
                    f"{format_currency(accommodation.total_price, accommodation.currency)}"
                )

    if plan.activities:
        st.markdown("##  Daily Itinerary")

        days = sorted(
            plan.activities.values(),
            key=lambda item: item.day_number,
        )

        for day in days:
            with st.expander(
                f"Day {day.day_number} — {day.date}",
                expanded=True,
            ):
                for activity in day.activities:
                    cost = (
                        format_currency(activity.cost, activity.currency)
                        if activity.cost is not None
                        else "Free"
                    )

                    st.markdown(
                        f"### {activity.name}"
                    )
                    st.write(activity.description)
                    st.write(f" {activity.location}")
                    st.write(f" {cost}")

    if plan.budget:
        st.markdown("##  Budget Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Budget",
                format_currency(
                    plan.budget.total_budget,
                    plan.budget.currency,
                ),
            )

        with col2:
            st.metric(
                "Estimated Spend",
                format_currency(
                    plan.budget.spent,
                    plan.budget.currency,
                ),
            )

        with col3:
            st.metric(
                "Remaining",
                format_currency(
                    plan.budget.remaining,
                    plan.budget.currency,
                ),
            )

        if plan.budget.breakdown:
            st.markdown("### Budget Breakdown")

            for category, amount in plan.budget.breakdown.items():
                category_name = (
                    category.value.replace("_", " ").title()
                    if hasattr(category, "value")
                    else str(category).replace("_", " ").title()
                )
                st.write(
                    f"**{category_name}:** "
                    f"{format_currency(amount, plan.budget.currency)}"
                )
    if plan.recommendations:
        st.markdown("##  Recommendations")

        for recommendation in plan.recommendations:
            st.write(f"• {recommendation}")

    if plan.alerts:
        st.markdown("##  Alerts")

        for alert in plan.alerts:
            st.warning(alert)


st.title(" AI Travel Planner")
st.caption(
    "Multi-agent travel planning powered by LangGraph and Gemini"
)

st.markdown(
    """
    Plan a trip with AI-generated destination research, flights,
    accommodation, activities, and budget planning.
    """
)

with st.form("travel_form"):
    st.markdown("###  Trip Details")

    col1, col2 = st.columns(2)

    with col1:
        origin = st.text_input(
            "From",
            value="Delhi",
            placeholder="e.g. Delhi",
        )

    with col2:
        destination = st.text_input(
            "Destination",
            value="Jaipur",
            placeholder="e.g. Jaipur",
        )

    col1, col2 = st.columns(2)

    with col1:
        departure_date = st.date_input(
            "Departure Date",
            value=date(2026, 10, 10),
        )

    with col2:
        return_date = st.date_input(
            "Return Date",
            value=date(2026, 10, 12),
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        travelers = st.number_input(
            "Travelers",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
        )

    with col2:
        budget_min = st.number_input(
            "Minimum Budget (INR)",
            min_value=0,
            value=10000,
            step=1000,
        )

    with col3:
        budget_max = st.number_input(
            "Maximum Budget (INR)",
            min_value=0,
            value=15000,
            step=1000,
        )

    submitted = st.form_submit_button(
        " Generate Travel Plan",
        use_container_width=True,
    )

if submitted:
    if not origin.strip() or not destination.strip():
        st.error("Please enter both origin and destination.")

    elif return_date < departure_date:
        st.error("Return date must be after departure date.")

    elif budget_max < budget_min:
        st.error("Maximum budget must be greater than minimum budget.")

    else:
        with st.spinner(
            " AI agents are planning your trip..."
        ):
            try:
                state = asyncio.run(
                    generate_plan(
                        origin=origin.strip(),
                        destination=destination.strip(),
                        departure_date=departure_date,
                        return_date=return_date,
                        travelers=travelers,
                        budget_min=budget_min,
                        budget_max=budget_max,
                    )
                )

                display_plan(state)

            except Exception as exc:
                st.error(
                    "The travel planner encountered an error."
                )
                st.exception(exc)


st.divider()

st.caption(
    "AI Travel Planner • LangGraph • Gemini • Supabase"
)
