"""
Flight Search Agent for the travel planner system.

This module implements the specialized agent responsible for searching,
comparing, and recommending flight options for the travel itinerary.
"""

from dataclasses import dataclass, field
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import Field

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils import (
    AgentExecutionError,
    AgentLogger,
    format_price,
    handle_errors,
    safe_serialize,
    with_retry,
)


class CabinClass(str, Enum):
    """Flight cabin classes."""

    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


@dataclass
class FlightLeg:
    """A single flight leg (segment)."""

    airline: str
    flight_number: str
    departure_airport: str
    departure_time: str
    arrival_airport: str
    arrival_time: str
    duration_minutes: int
    aircraft: str | None = None


@dataclass
class FlightOption:
    """A flight option with one or more legs."""

    id: str
    price: float
    currency: str
    cabin_class: CabinClass
    legs: list[FlightLeg]
    layover_count: int
    total_duration_minutes: int
    baggage_allowance: str | None = None
    refundable: bool = False
    changeable: bool = False
    eco_friendly: bool = False
    amenities: list[str] = field(default_factory=list)

    @property
    def formatted_price(self) -> str:
        """Get the formatted price with currency symbol."""

        return format_price(
            self.price,
            self.currency,
        )

    @property
    def formatted_duration(self) -> str:
        """Get the formatted total duration as hours and minutes."""

        hours, minutes = divmod(
            self.total_duration_minutes,
            60,
        )

        return f"{hours}h {minutes}m"


class FlightSearchContext(AgentContext):
    """Context for the flight search agent."""

    origin: str = ""
    destination: str = ""
    departure_date: str | None = None
    return_date: str | None = None
    travelers: int = 1
    cabin_class: CabinClass = CabinClass.ECONOMY
    max_price: float | None = None
    currency: str = "USD"
    preferred_airlines: list[str] = Field(default_factory=list)
    flight_options: list[FlightOption] = Field(default_factory=list)
    selected_flight: FlightOption | None = None
    search_params: dict[str, Any] = Field(default_factory=dict)
    search_results_raw: dict[str, Any] = Field(default_factory=dict)


class FlightSearchAgent(BaseAgent[FlightSearchContext]):
    """
    Specialized agent for flight search and booking.

    This agent is responsible for:
    1. Searching multiple flight booking sites for optimal options
    2. Filtering results based on user preferences
    3. Providing price comparisons and recommendations
    4. Monitoring flight prices if needed
    5. Presenting flight options with key details
    """

    def __init__(self, config: AgentConfig | None = None):
        """Initialize the flight search agent."""

        default_config = AgentConfig(
            name="Flight Search",
            instructions=(
                "You are an AI flight search specialist for travel planning. "
                "Your expertise is in finding and comparing flight options across "
                "multiple airlines and booking platforms. Provide comprehensive flight "
                "information including prices, times, layovers, and amenities. "
                "Consider user preferences for airlines, times, cabin class, and budget. "
                "Present options clearly with pros and cons to help users make informed decisions."
            ),
            model="gemini-3.6-flash",
            tools=[],
        )

        super().__init__(
            config or default_config,
            FlightSearchContext,
        )

        self.logger = AgentLogger(self.name)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: FlightSearchContext | None = None,
    ) -> dict[str, Any]:
        """Run the flight search agent."""

        self.logger.info(
            "Running flight search agent with input: "
            f"{input_data if isinstance(input_data, str) else '...'}"
        )

        if context is None:
            context = FlightSearchContext()

        try:
            result = await self.process(
                input_data,
                context,
            )

            return {
                "context": context,
                "result": result,
            }

        except Exception as e:
            error_msg = f"Error in flight search agent: {e!s}"

            self.logger.error(error_msg)

            raise AgentExecutionError(
                error_msg,
                self.name,
                original_error=e,
            ) from e

    @handle_errors(error_cls=AgentExecutionError)
    async def process(
        self,
        input_data: str | list[dict[str, Any]],
        context: FlightSearchContext,
    ) -> dict[str, Any]:
        """Process the flight search request."""

        self._prepare_messages(input_data)

        if (
            not context.origin
            or not context.destination
            or not context.departure_date
        ):
            await self._extract_search_params(
                input_data,
                context,
            )

        self.logger.info(
            f"Processing flight search for: "
            f"{context.origin} to {context.destination}"
        )

        search_results = await self._search_flights(
            context
        )

        ranked_options = await self._rank_flight_options(
            search_results,
            context,
        )

        context.flight_options = ranked_options

        summary = await self._generate_options_summary(
            ranked_options,
            context,
        )

        return {
            "flight_options": [
                self._format_flight_option(option)
                for option in ranked_options
            ],
            "summary": summary,
        }

    async def _extract_search_params(
        self,
        input_data: str | list[dict[str, Any]],
        context: FlightSearchContext,
    ) -> None:
        """Extract flight parameters, preferring structured workflow state."""

        self.logger.info("Extracting flight search parameters")

        user_input = (
            input_data
            if isinstance(input_data, str)
            else self._get_latest_user_input(input_data)
        )

        # Downstream workflow nodes receive a JSON payload containing the
        # structured TravelQuery. Use it directly so CLI values are not
        # replaced by demo defaults or another model interpretation.
        try:
            payload = json.loads(user_input) if isinstance(user_input, str) else {}
            travel_query = payload.get("travel_query", {})
            if isinstance(travel_query, dict):
                context.origin = travel_query.get("origin") or context.origin
                context.destination = travel_query.get("destination") or context.destination
                context.departure_date = travel_query.get("departure_date") or context.departure_date
                context.return_date = travel_query.get("return_date") or context.return_date
                context.travelers = int(travel_query.get("travelers") or context.travelers)

                budget = travel_query.get("budget_range")
                if isinstance(budget, dict):
                    maximum = budget.get("max")
                    if maximum is not None:
                        context.max_price = float(maximum)

                if any((context.origin, context.destination, context.departure_date)):
                    return
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

        # Fall back to model extraction only for genuinely unstructured input.
        extraction_prompt = (
            "Extract flight search parameters from the user's input. "
            "Include origin, destination, dates, number of travelers, "
            "cabin class, and any airline preferences. "
            "If information is missing, keep the current values. "
            "Format the output as a structured JSON object."
        )

        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": user_input},
        ]
        if context.model_dump(exclude_none=True):
            messages.append({
                "role": "system",
                "content": "Current parameters: " + safe_serialize(context),
            })

        await self._call_model(messages)

        # Do not invent a real-world route when required data is absent.
        if not context.origin or not context.destination:
            raise AgentExecutionError(
                "Flight search requires both origin and destination.",
                self.name,
            )

        if not context.departure_date:
            raise AgentExecutionError(
                "Flight search requires a departure date.",
                self.name,
            )

    async def _search_flights(
        self,
        context: FlightSearchContext,
    ) -> list[dict[str, Any]]:
        """Search for flights based on the context parameters."""

        self.logger.info(
            f"Searching flights from "
            f"{context.origin} to {context.destination}"
        )

        # Demo/mock flight data.
        # Replace this method later with a real flight API.
        mock_flights = [
            {
                "id": "F1",
                "airline": "Gamma Airways",
                "price": 350.0,
                "currency": context.currency,
                "cabin_class": context.cabin_class.value,
                "departure_time": "08:00",
                "arrival_time": "11:30",
                "duration_minutes": 210,
                "layovers": [],
                "baggage": "1 checked bag included",
                "refundable": True,
                "eco_friendly": True,
            },
            {
                "id": "F2",
                "airline": "Beta Airlines",
                "price": 280.0,
                "currency": context.currency,
                "cabin_class": context.cabin_class.value,
                "departure_time": "14:15",
                "arrival_time": "19:45",
                "duration_minutes": 330,
                "layovers": ["ORD"],
                "baggage": "Carry-on only",
                "refundable": False,
                "eco_friendly": False,
            },
            {
                "id": "F3",
                "airline": "Alpha Airlines",
                "price": 420.0,
                "currency": context.currency,
                "cabin_class": context.cabin_class.value,
                "departure_time": "10:30",
                "arrival_time": "13:45",
                "duration_minutes": 195,
                "layovers": [],
                "baggage": "2 checked bags included",
                "refundable": True,
                "eco_friendly": True,
            },
        ]

        context.search_results_raw = {
            "flights": mock_flights
        }

        return mock_flights

    async def _rank_flight_options(
        self,
        search_results: list[dict[str, Any]],
        context: FlightSearchContext,
    ) -> list[FlightOption]:
        """Rank and convert flight options."""

        self.logger.info(
            f"Ranking {len(search_results)} flight options"
        )

        flight_options: list[FlightOption] = []

        for result in search_results:
            legs = [
                FlightLeg(
                    airline=result["airline"],
                    flight_number=f"{result['airline'][0:2]}123",
                    departure_airport=context.origin,
                    departure_time=(
                        f"{context.departure_date}T"
                        f"{result['departure_time']}"
                    ),
                    arrival_airport=context.destination,
                    arrival_time=(
                        f"{context.departure_date}T"
                        f"{result['arrival_time']}"
                    ),
                    duration_minutes=result["duration_minutes"],
                )
            ]

            layovers = result.get(
                "layovers",
                [],
            )

            option = FlightOption(
                id=result["id"],
                price=result["price"],
                currency=result["currency"],
                cabin_class=CabinClass(
                    result["cabin_class"]
                ),
                legs=legs,
                layover_count=len(layovers),
                total_duration_minutes=result[
                    "duration_minutes"
                ],
                baggage_allowance=result.get(
                    "baggage"
                ),
                refundable=result.get(
                    "refundable",
                    False,
                ),
                changeable=result.get(
                    "changeable",
                    False,
                ),
                eco_friendly=result.get(
                    "eco_friendly",
                    False,
                ),
            )

            flight_options.append(option)

        # Demo ranking: cheapest first.
        flight_options.sort(
            key=lambda x: x.price
        )

        return flight_options

    async def _generate_options_summary(
        self,
        options: list[FlightOption],
        context: FlightSearchContext,
    ) -> str:
        """Generate a human-readable summary of flight options."""

        self.logger.info(
            "Generating flight options summary"
        )

        if not options:
            return (
                "No flight options found matching "
                "your criteria."
            )

        summary_prompt = (
            f"Summarize the following {len(options)} flight options "
            f"from {context.origin} to {context.destination} "
            f"on {context.departure_date}. Highlight the best value, "
            "fastest option, and any notable features or drawbacks. "
            "Be concise but informative."
        )

        options_text = "\n\n".join(
            [
                f"Option {i + 1}: "
                f"{option.legs[0].airline} - "
                f"{option.formatted_price}\n"
                f"Departure: {option.legs[0].departure_time} - "
                f"Arrival: {option.legs[0].arrival_time}\n"
                f"Duration: {option.formatted_duration} - "
                f"Layovers: {option.layover_count}\n"
                f"Baggage: "
                f"{option.baggage_allowance or 'Not specified'}\n"
                f"Refundable: {option.refundable} - "
                f"Eco-friendly: {option.eco_friendly}"
                for i, option in enumerate(options[:5])
            ]
        )

        messages = [
            {
                "role": "system",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": summary_prompt,
            },
            {
                "role": "system",
                "content": options_text,
            },
        ]

        response = await self._call_model(
            messages
        )

        return response.get(
            "content",
            "Flight options summary not available.",
        )

    def _format_flight_option(
        self,
        option: FlightOption,
    ) -> dict[str, Any]:
        """Format a flight option for display."""

        legs_formatted = []

        for leg in option.legs:
            legs_formatted.append(
                {
                    "airline": leg.airline,
                    "flight_number": leg.flight_number,
                    "departure": {
                        "airport": leg.departure_airport,
                        "time": leg.departure_time,
                    },
                    "arrival": {
                        "airport": leg.arrival_airport,
                        "time": leg.arrival_time,
                    },
                    "duration": (
                        f"{leg.duration_minutes // 60}h "
                        f"{leg.duration_minutes % 60}m"
                    ),
                }
            )

        return {
            "id": option.id,
            "price": {
                "amount": option.price,
                "currency": option.currency,
                "formatted": option.formatted_price,
            },
            "cabin_class": option.cabin_class.value,
            "legs": legs_formatted,
            "layovers": option.layover_count,
            "duration": option.formatted_duration,
            "baggage": option.baggage_allowance,
            "refundable": option.refundable,
            "eco_friendly": option.eco_friendly,
            "amenities": option.amenities,
        }

    def _get_latest_user_input(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Extract the latest user input."""

        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get(
                    "content",
                    "",
                )

        return ""

    @with_retry(max_attempts=3)
    async def _call_model(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call Gemini through the OpenAI-compatible API."""

        self.logger.info(
            f"Calling model with {len(messages)} messages"
        )

        self.logger.log_llm_input(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )

        try:
            # BaseAgent uses the synchronous OpenAI client.
            # Do NOT await this call.
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            self.logger.log_llm_output(
                model=self.config.model,
                response=response,
            )

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content

                return {
                    "content": content or "",
                }

            return {
                "content": "No response generated.",
            }

        except Exception as e:
            self.logger.error(
                f"Error calling model: {e!s}"
            )
            raise