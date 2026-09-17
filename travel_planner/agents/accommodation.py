"""
Accommodation Agent for the travel planner system.

This module implements the specialized agent responsible for searching,
comparing, and recommending accommodation options for the travel itinerary.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import Field

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils.error_handling import with_retry
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class AccommodationType(str, Enum):
    """Types of accommodation."""

    HOTEL = "hotel"
    APARTMENT = "apartment"
    HOSTEL = "hostel"
    RESORT = "resort"
    VILLA = "villa"
    GUESTHOUSE = "guesthouse"


@dataclass
class AccommodationOption:
    """A single accommodation option."""

    id: str
    name: str
    type: AccommodationType
    location: str
    price_per_night: float
    currency: str
    rating: float | None = None
    amenities: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    description: str = ""
    address: str = ""
    booking_url: str = ""
    refundable: bool = False
    reviews_count: int = 0

    @property
    def formatted_price(self) -> str:
        """Get the formatted price with currency symbol."""
        if self.currency == "USD":
            return f"${self.price_per_night:.2f}"
        elif self.currency == "EUR":
            return f"€{self.price_per_night:.2f}"
        else:
            return f"{self.price_per_night:.2f} {self.currency}"


class AccommodationSearchContext(AgentContext):
    """Context for the accommodation search agent."""

    destination: str = ""
    check_in_date: str | None = None
    check_out_date: str | None = None
    guests: int = 1
    rooms: int = 1
    accommodation_type: AccommodationType | None = None
    max_price: float | None = None
    amenities: list[str] = Field(default_factory=list)
    accommodation_options: list[AccommodationOption] = Field(default_factory=list)
    selected_accommodation: AccommodationOption | None = None
    search_params: dict[str, Any] = Field(default_factory=dict)
    search_results_raw: dict[str, Any] = Field(default_factory=dict)


class AccommodationAgent(BaseAgent[AccommodationSearchContext]):
    """
    Specialized agent for accommodation search and booking.

    This agent is responsible for:
    1. Searching multiple accommodation booking sites for optimal options
    2. Filtering results based on user preferences
    3. Providing comparisons and recommendations
    4. Supporting booking capabilities when needed
    5. Presenting options with key details
    """

    def __init__(self, config: AgentConfig | None = None):
        """Initialize the accommodation search agent."""

        default_config = AgentConfig(
            name="accommodation_agent",
            instructions="""
            You are a specialized agent focused on finding the best accommodation
            options.

            Your goal is to research, compare, and recommend accommodations that match
            the traveler's preferences and budget. You should consider factors like
            location, amenities, reviews, and value.
            """,
            model="gemini-3.6-flash",
            tools=[],
        )

        config = config or default_config
        super().__init__(config, AccommodationSearchContext)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: AccommodationSearchContext | None = None,
    ) -> Any:
        """Run the accommodation search agent."""

        try:
            if context is None:
                context = AccommodationSearchContext()

            result = await self.process(input_data, context)
            return result

        except Exception as e:
            error_msg = f"Error in accommodation search agent: {e!s}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def process(
        self,
        input_data: str | list[dict[str, Any]],
        context: AccommodationSearchContext,
    ) -> dict[str, Any]:
        """Process the accommodation search request."""

        if not context.search_params:
            await self._extract_search_parameters(input_data, context)

        search_results = await self._search_accommodations(context)

        ranked_options = await self._rank_accommodation_options(
            search_results,
            context,
        )

        context.accommodation_options = ranked_options

        summary = await self._generate_options_summary(
            ranked_options,
            context,
        )

        return {
            "context": context,
            "accommodations": ranked_options,
            "summary": summary,
        }

    async def _extract_search_parameters(
        self,
        input_data: str | list[dict[str, Any]],
        context: AccommodationSearchContext,
    ) -> None:
        """Extract accommodation search parameters from user input."""

        extraction_prompt = (
            "Please extract accommodation search parameters from the following "
            "user input. Include destination, check-in and check-out dates, "
            "number of guests, room preferences, accommodation type, price "
            "range, and required amenities. Format your response as a JSON "
            "object.\n\nUser input: {input}"
        )

        user_input = (
            input_data
            if isinstance(input_data, str)
            else self._get_latest_user_input(input_data)
        )

        messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": extraction_prompt.format(input=user_input),
            },
        ]

        if context.search_params:
            messages.append(
                {
                    "role": "system",
                    "content": f"Current parameters: {context.search_params}",
                }
            )

        await self._call_model(messages)

        # Demo/default values.
        context.destination = "Paris, France"
        context.check_in_date = "2025-06-15"
        context.check_out_date = "2025-06-22"
        context.guests = 2
        context.rooms = 1
        context.accommodation_type = AccommodationType.HOTEL
        context.max_price = 300.0
        context.amenities = ["wifi", "breakfast", "pool"]

        context.search_params = {
            "destination": context.destination,
            "check_in_date": context.check_in_date,
            "check_out_date": context.check_out_date,
            "guests": context.guests,
            "rooms": context.rooms,
            "accommodation_type": context.accommodation_type,
            "max_price": context.max_price,
            "amenities": context.amenities,
        }

    async def _search_accommodations(
        self,
        context: AccommodationSearchContext,
    ) -> list[dict[str, Any]]:
        """Search for accommodations based on the context parameters."""

        mock_accommodations = [
            {
                "id": "hotel1",
                "name": "Grand Hotel Paris",
                "type": "hotel",
                "location": "Paris, France",
                "price_per_night": 250.0,
                "currency": "EUR",
                "rating": 4.5,
                "amenities": ["wifi", "breakfast", "pool", "spa"],
                "description": "Luxury hotel in the heart of Paris",
                "address": "1 Rue de Rivoli, 75001 Paris, France",
                "refundable": True,
                "reviews_count": 1250,
            },
            {
                "id": "apartment1",
                "name": "Eiffel Tower View Apartment",
                "type": "apartment",
                "location": "Paris, France",
                "price_per_night": 180.0,
                "currency": "EUR",
                "rating": 4.3,
                "amenities": ["wifi", "kitchen", "washer"],
                "description": "Cozy apartment with stunning views of the Eiffel Tower",
                "address": "15 Avenue de la Bourdonnais, 75007 Paris, France",
                "refundable": False,
                "reviews_count": 320,
            },
            {
                "id": "hotel2",
                "name": "Boutique Hotel Marais",
                "type": "hotel",
                "location": "Paris, France",
                "price_per_night": 210.0,
                "currency": "EUR",
                "rating": 4.7,
                "amenities": ["wifi", "breakfast", "bar"],
                "description": "Charming boutique hotel in the historic Marais district",
                "address": "25 Rue des Archives, 75004 Paris, France",
                "refundable": True,
                "reviews_count": 850,
            },
        ]

        context.search_results_raw = {"results": mock_accommodations}

        return mock_accommodations

    async def _rank_accommodation_options(
        self,
        search_results: list[dict[str, Any]],
        context: AccommodationSearchContext,
    ) -> list[AccommodationOption]:
        """Rank and convert accommodation options."""

        accommodation_options: list[AccommodationOption] = []

        for result in search_results:
            option = AccommodationOption(
                id=result["id"],
                name=result["name"],
                type=AccommodationType(result["type"]),
                location=result["location"],
                price_per_night=result["price_per_night"],
                currency=result["currency"],
                rating=result.get("rating"),
                amenities=result.get("amenities", []),
                images=result.get("images", []),
                description=result.get("description", ""),
                address=result.get("address", ""),
                booking_url=result.get("booking_url", ""),
                refundable=result.get("refundable", False),
                reviews_count=result.get("reviews_count", 0),
            )
            accommodation_options.append(option)

        accommodation_options.sort(
            key=lambda x: (
                -x.rating if x.rating is not None else 0,
                x.price_per_night,
            )
        )

        return accommodation_options[:5]

    async def _generate_options_summary(
        self,
        options: list[AccommodationOption],
        context: AccommodationSearchContext,
    ) -> str:
        """Generate a human-readable summary of accommodation options."""

        summary_prompt = (
            "Create a summary of the following accommodation options for "
            "{destination}. Highlight key features, price differences, and "
            "which options best match the traveler's preferences for "
            "{amenities}. Max budget is {max_price} {currency} per night. "
            "Stay dates: {check_in} to {check_out}.\n\nOptions:\n\n{options_text}"
        )

        options_text = "\n\n".join(
            [
                f"Option {i + 1}: {option.name}\n"
                f"Type: {option.type.value}\n"
                f"Location: {option.location}\n"
                f"Price: {option.formatted_price} per night\n"
                f"Rating: {option.rating} ({option.reviews_count} reviews)\n"
                f"Amenities: {', '.join(option.amenities)}\n"
                f"Description: {option.description}\n"
                f"Refundable: {'Yes' if option.refundable else 'No'}"
                for i, option in enumerate(options[:5])
            ]
        )

        messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": summary_prompt.format(
                    destination=context.destination,
                    amenities=", ".join(context.amenities),
                    max_price=context.max_price,
                    currency=options[0].currency if options else "EUR",
                    check_in=context.check_in_date,
                    check_out=context.check_out_date,
                    options_text=options_text,
                ),
            },
        ]

        response = await self._call_model(messages)

        return response.get("content", "")

    def _format_accommodation_option(
        self,
        option: AccommodationOption,
    ) -> dict[str, Any]:
        """Format an accommodation option for display."""

        return {
            "id": option.id,
            "name": option.name,
            "type": option.type.value,
            "location": option.location,
            "price": option.formatted_price,
            "rating": f"{option.rating}/5" if option.rating else "Not rated",
            "amenities": option.amenities,
            "description": option.description,
            "refundable": option.refundable,
        }

    def _get_latest_user_input(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Extract the latest user input."""

        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")

        return ""

    @with_retry(max_attempts=3)
    async def _call_model(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call the configured model."""

        logger.debug(f"Calling model with messages: {messages}")

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        logger.debug(f"Model response: {response}")

        content = response.choices[0].message.content

        return {"content": content}