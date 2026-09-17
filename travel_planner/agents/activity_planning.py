"""
Activity Planning Agent for the travel planner system.

This module implements the specialized agent responsible for researching,
scheduling, and recommending activities and attractions for the travel itinerary.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any
from pydantic import Field
from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils.error_handling import with_retry
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)

# Constants
MAX_ACTIVITIES_PER_DAY = 3


class ActivityType(str, Enum):
    """Types of activities."""

    ATTRACTION = "attraction"
    TOUR = "tour"
    MUSEUM = "museum"
    OUTDOOR = "outdoor"
    FOOD = "food_and_drink"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    CULTURAL = "cultural"
    ADVENTURE = "adventure"
    WELLNESS = "wellness"


class WeatherCondition(str, Enum):
    """Types of weather conditions."""

    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    WINDY = "windy"
    HOT = "hot"
    COLD = "cold"


@dataclass
class Activity:
    """A single activity or attraction."""

    id: str
    name: str
    type: ActivityType
    location: str
    description: str
    price: float
    currency: str
    duration_minutes: int
    opening_hours: dict[str, dict[str, time]] = field(default_factory=dict)
    booking_required: bool = False
    booking_url: str = ""
    weather_dependent: bool = False
    suitable_weather: list[WeatherCondition] = field(default_factory=list)
    rating: float | None = None
    reviews_count: int = 0
    images: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    accessibility_features: list[str] = field(default_factory=list)

    @property
    def formatted_price(self) -> str:
        """Get the formatted price with currency symbol."""
        if self.currency == "USD":
            return f"${self.price:.2f}"
        elif self.currency == "EUR":
            return f"€{self.price:.2f}"
        else:
            return f"{self.price:.2f} {self.currency}"

    @property
    def formatted_duration(self) -> str:
        """Get the formatted duration as hours and minutes."""
        hours, minutes = divmod(self.duration_minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


@dataclass
class ScheduledActivity:
    """An activity scheduled for a specific date and time."""

    activity: Activity
    date: str
    start_time: str
    end_time: str
    notes: str = ""


@dataclass
class DailyItinerary:
    """Itinerary for a single day."""

    date: str
    activities: list[ScheduledActivity] = field(default_factory=list)
    weather_forecast: dict[str, Any] | None = None
    notes: str = ""
    total_cost: float = 0.0
    currency: str = "EUR"



class ActivityPlanningContext(AgentContext):
    """Context for the activity planning agent."""

    destination: str = ""
    start_date: str | None = None
    end_date: str | None = None
    traveler_count: int = 1
    currency: str = "INR"
    interests: list[str] = Field(default_factory=list)
    has_children: bool = False
    has_accessibility_needs: bool = False
    budget_per_day: float | None = None
    accommodation_location: str = ""
    available_activities: list[Activity] = Field(default_factory=list)
    daily_itineraries: dict[str, DailyItinerary] = Field(default_factory=dict)
    weather_forecasts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    search_params: dict[str, Any] = Field(default_factory=dict)
    excluded_activity_types: list[ActivityType] = Field(default_factory=list)


class ActivityPlanningAgent(BaseAgent[ActivityPlanningContext]):
    """
    Specialized agent for activity planning.

    This agent is responsible for:
    1. Researching activities and attractions at the destination
    2. Creating daily itineraries with scheduled activities
    3. Considering factors like opening hours, travel time, and weather
    4. Balancing must-see attractions with personalized experiences
    5. Ensuring activities are within budget constraints
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the activity planning agent.

        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="activity_planning_agent",
            instructions="""
            You are a specialized activity planning agent.
            Your goal is to research, recommend, and schedule activities and attractions
            that match the traveler's interests and preferences. Create logical daily
            itineraries that account for location, opening hours, travel time, 
            and budget.
            """,
            model="gemini-3.6-flash",
            tools=[],  # No tools initially, they would be added in a real implementation
        )

        config = config or default_config
        super().__init__(config, ActivityPlanningContext)

        # Add tools for specific activity planning functionality
        # These would typically be implemented as part of the full system
        # self.add_tool(search_activities)
        # self.add_tool(check_opening_hours)
        # self.add_tool(get_weather_forecast)
        # self.add_tool(calculate_travel_time)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: ActivityPlanningContext | None = None,
    ) -> Any:
        """
        Run the activity planning agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional activity planning context

        Returns:
            Updated context and activity planning results
        """
        try:
            # Initialize context if not provided
            if not context:
                context = ActivityPlanningContext()

            # Process the input
            result = await self.process(input_data, context)
            return result
        except Exception as e:
            error_msg = f"Error in activity planning agent: {e!s}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def process(
        self, input_data: str | list[dict[str, Any]], context: ActivityPlanningContext
    ) -> dict[str, Any]:
        """
        Process the activity planning request.

        Args:
            input_data: User input or conversation history
            context: Activity planning context

        Returns:
            Activity planning results
        """
        # Extract activity preferences if not already set
        if not context.search_params:
            await self._extract_activity_preferences(input_data, context)

        # Get available activities at the destination
        if not context.available_activities:
            context.available_activities = await self._research_activities(context)

        # Get weather forecasts for the trip dates
        if not context.weather_forecasts and context.start_date and context.end_date:
            context.weather_forecasts = await self._get_weather_forecasts(context)

        # Create daily itineraries
        daily_itineraries = await self._create_daily_itineraries(context)
        context.daily_itineraries = daily_itineraries

        # Generate an itinerary summary
        itinerary_summary = await self._generate_itinerary_summary(context)

        return {
            "context": context,
            "daily_itineraries": daily_itineraries,
            "summary": itinerary_summary,
        }

    async def _extract_activity_preferences(
        self,
        input_data: str | list[dict[str, Any]],
        context: ActivityPlanningContext,
    ) -> None:
        """Populate activity-planning context from structured workflow input."""
        from datetime import datetime, timedelta

        from travel_planner.utils.input_parser import (
            get_preferences,
            get_travel_query,
            query_dates,
        )

        query = get_travel_query(input_data)
        preferences = get_preferences(input_data)
        requirements = query.get("requirements") or {}

        context.destination = query.get("destination") or context.destination
        context.currency = (
            (query.get("requirements") or {}).get("currency")
            or context.currency
        )
        context.traveler_count = int(
            query.get("travelers") or context.traveler_count or 1
        )

        start_date, end_date = query_dates(query)
        if not start_date:
            start_date = (
                datetime.now() + timedelta(days=1)
            ).strftime("%Y-%m-%d")

        if not end_date:
            duration = int(query.get("duration_days") or 1)
            end_date = (
                datetime.strptime(start_date, "%Y-%m-%d")
                + timedelta(days=max(duration - 1, 0))
            ).strftime("%Y-%m-%d")

        context.start_date = start_date
        context.end_date = end_date

        interests = requirements.get("activity_preferences")
        if isinstance(interests, list):
            context.interests = [str(item) for item in interests]

        if not context.interests:
            context.interests = preferences.get("special_interests", [])

        budget_range = query.get("budget_range")
        if isinstance(budget_range, dict):
            total_budget = budget_range.get("max")
            duration = max(
                (datetime.strptime(context.end_date, "%Y-%m-%d")
                 - datetime.strptime(context.start_date, "%Y-%m-%d")).days + 1,
                1,
            )
            if total_budget is not None:
                context.budget_per_day = float(total_budget) / duration

        accessibility = (
            requirements.get("accessibility_needs")
            or preferences.get("accessibility_requirements")
        )
        if accessibility:
            context.has_accessibility_needs = True

        excluded = requirements.get("excluded_activity_types", [])
        for value in excluded:
            try:
                context.excluded_activity_types.append(ActivityType(value))
            except (ValueError, TypeError):
                continue

        context.search_params = {
            "destination": context.destination,
            "start_date": context.start_date,
            "end_date": context.end_date,
            "traveler_count": context.traveler_count,
            "interests": context.interests,
            "has_children": context.has_children,
            "has_accessibility_needs": context.has_accessibility_needs,
            "budget_per_day": context.budget_per_day,
            "accommodation_location": context.accommodation_location,
            "excluded_activity_types": [
                item.value for item in context.excluded_activity_types
            ],
        }

        if not context.destination:
            raise ValueError("Activity planning requires a destination.")

    async def _research_activities(
        self,
        context: ActivityPlanningContext,
    ) -> list[Activity]:
        """Build a destination-aware activity catalogue.

        The catalogue is deterministic so the application remains useful when
        an external activity-search provider or the LLM is unavailable. Prices
        are planning estimates, not live ticket quotes.
        """
        currency = "INR" if context.destination else context.currency
        destination = context.destination.strip()

        catalogues = {
            "jaipur": [
                Activity("jai-amber", "Amber Fort", ActivityType.ATTRACTION,
                         "Amer, Jaipur", "Explore Jaipur's hilltop fort complex and palace courtyards.",
                         200.0, currency, 150, tags=["sightseeing", "history", "culture"]),
                Activity("jai-city-palace", "City Palace", ActivityType.ATTRACTION,
                         "Old City, Jaipur", "Visit the City Palace complex and its royal collections.",
                         300.0, currency, 120, tags=["sightseeing", "history", "culture"]),
                Activity("jai-hawa", "Hawa Mahal", ActivityType.ATTRACTION,
                         "Badi Choupad, Jaipur", "See Jaipur's iconic Palace of Winds and explore the surrounding old city.",
                         200.0, currency, 75, tags=["sightseeing", "history", "culture"]),
                Activity("jai-jantar", "Jantar Mantar", ActivityType.MUSEUM,
                         "Old City, Jaipur", "Explore the historic astronomical instruments at Jantar Mantar.",
                         200.0, currency, 75, tags=["history", "science", "culture"]),
                Activity("jai-albert", "Albert Hall Museum", ActivityType.MUSEUM,
                         "Ram Niwas Garden, Jaipur", "Explore art, crafts and historical collections in the museum.",
                         200.0, currency, 120, tags=["museum", "history", "culture"]),
                Activity("jai-jal", "Jal Mahal Viewpoint", ActivityType.OUTDOOR,
                         "Amer Road, Jaipur", "Stop by Man Sagar Lake for views of Jal Mahal from the lakeside road.",
                         0.0, currency, 45, weather_dependent=True,
                         suitable_weather=[WeatherCondition.SUNNY, WeatherCondition.CLOUDY],
                         tags=["nature", "sightseeing", "photography"]),
                Activity("jai-nahargarh", "Nahargarh Fort Sunset", ActivityType.OUTDOOR,
                         "Nahargarh, Jaipur", "Take in panoramic views of Jaipur from the Aravalli hills.",
                         100.0, currency, 120, weather_dependent=True,
                         suitable_weather=[WeatherCondition.SUNNY, WeatherCondition.CLOUDY],
                         tags=["nature", "photography", "sightseeing"]),
                Activity("jai-johari", "Johari Bazaar", ActivityType.SHOPPING,
                         "Johari Bazaar, Jaipur", "Browse traditional jewellery, textiles and handicrafts in the old city market.",
                         0.0, currency, 120, tags=["shopping", "culture"]),
                Activity("jai-food", "Rajasthani Food Experience", ActivityType.FOOD,
                         "Jaipur", "Try regional dishes such as dal baati churma and other Rajasthani specialties.",
                         500.0, currency, 90, tags=["food", "culinary", "culture"]),
                Activity("jai-cultural", "Rajasthani Cultural Evening", ActivityType.CULTURAL,
                         "Jaipur", "Experience regional music, dance and traditional cuisine at an evening cultural venue.",
                         700.0, currency, 150, tags=["culture", "food", "entertainment"]),
            ],
        }

        activities = catalogues.get(destination.lower())
        if activities is None:
            # Useful deterministic fallback for destinations without a curated
            # catalogue; real search integrations can replace this later.
            activities = [
                Activity("generic-sightseeing", "Local Heritage & Sightseeing", ActivityType.ATTRACTION,
                         destination, f"Explore major heritage and sightseeing attractions in {destination}.",
                         0.0, currency, 120, tags=["sightseeing", "history", "culture"]),
                Activity("generic-culture", "Local Cultural Experience", ActivityType.CULTURAL,
                         destination, f"Discover local culture, traditions and history in {destination}.",
                         0.0, currency, 120, tags=["culture", "history"]),
                Activity("generic-food", "Local Food Experience", ActivityType.FOOD,
                         destination, f"Try representative local food and regional specialties in {destination}.",
                         0.0, currency, 90, tags=["food", "culinary"]),
                Activity("generic-outdoor", "Outdoor Exploration", ActivityType.OUTDOOR,
                         destination, f"Explore an outdoor area or scenic location in {destination}.",
                         0.0, currency, 90, weather_dependent=True,
                         suitable_weather=[WeatherCondition.SUNNY, WeatherCondition.CLOUDY],
                         tags=["nature", "relaxation"]),
            ]

        filtered = [
            item for item in activities
            if item.type not in context.excluded_activity_types
            and (not context.interests or any(tag.lower() in {i.lower() for i in context.interests} for tag in item.tags))
        ]
        return filtered or [item for item in activities if item.type not in context.excluded_activity_types]

    async def _get_weather_forecasts(
        self, context: ActivityPlanningContext
    ) -> dict[str, dict[str, Any]]:
        """
        Get weather forecasts for the destination during the trip dates.

        Args:
            context: Activity planning context

        Returns:
            Dictionary mapping dates to weather forecasts
        """
        # In a real implementation, this would call a weather API
        # For demonstration, we'll create some mock forecasts

        forecasts = {}

        if not context.start_date or not context.end_date:
            return forecasts

        start = datetime.strptime(context.start_date, "%Y-%m-%d")
        end = datetime.strptime(context.end_date, "%Y-%m-%d")
        current_date = start

        # Generate forecasts for each day
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")

            # Create a mock forecast (in a real implementation, this would come from an API)
            # We'll alternate between sunny and cloudy for simplicity
            is_sunny = (current_date - start).days % 2 == 0

            forecasts[date_str] = {
                "condition": WeatherCondition.SUNNY
                if is_sunny
                else WeatherCondition.CLOUDY,
                "temperature_celsius": 22 if is_sunny else 19,
                "temperature_fahrenheit": 72 if is_sunny else 66,
                "precipitation_chance": 10 if is_sunny else 40,
                "wind_speed_kmh": 10 if is_sunny else 15,
            }

            current_date += timedelta(days=1)

        return forecasts

    async def _create_daily_itineraries(
        self, context: ActivityPlanningContext
    ) -> dict[str, DailyItinerary]:
        """
        Create daily itineraries for the trip dates.

        Args:
            context: Activity planning context

        Returns:
            Dictionary mapping dates to daily itineraries
        """
        if (
            not context.start_date
            or not context.end_date
            or not context.available_activities
        ):
            return {}

        itineraries = {}

        start = datetime.strptime(context.start_date, "%Y-%m-%d")
        end = datetime.strptime(context.end_date, "%Y-%m-%d")
        current_date = start

        # Create an itinerary for each day
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            day_of_week = current_date.strftime("%A")

            # Get weather forecast if available
            weather = context.weather_forecasts.get(date_str)

            # Create a new daily itinerary
            itinerary = DailyItinerary(
                date=date_str, weather_forecast=weather, currency=context.currency
            )

            # Filter activities based on day of week (opening hours) and weather.
            # An empty opening-hours mapping means no schedule restriction was
            # supplied, so the activity remains eligible.
            suitable_activities = []
            for activity in context.available_activities:
                opening_hours = activity.opening_hours.get(day_of_week, {})

                if opening_hours:
                    is_open = (
                        opening_hours.get("open") != time(0, 0)
                        or opening_hours.get("close") != time(0, 0)
                    )
                    if not is_open:
                        continue

                if activity.weather_dependent and weather:
                    if weather.get("condition") not in activity.suitable_weather:
                        continue

                suitable_activities.append(activity)

            # For a real implementation, we would create a logical daily schedule based on:
            # - Location proximity (to minimize travel time)
            # - Opening hours
            # - Activity durations
            # - Budget constraints

            # For this demo, we'll create a simple schedule with morning, afternoon, and evening activities

            # Morning activity (9:00 - 12:00)
            morning_activities = [
                a
                for a in suitable_activities
                if a.type
                in [ActivityType.MUSEUM, ActivityType.ATTRACTION, ActivityType.TOUR]
            ]
            if (
                morning_activities
                and len(itinerary.activities) < MAX_ACTIVITIES_PER_DAY
            ):
                day_index = (current_date - start).days
                selected = morning_activities[day_index % len(morning_activities)]
                suitable_activities.remove(selected)

                itinerary.activities.append(
                    ScheduledActivity(
                        activity=selected,
                        date=date_str,
                        start_time="09:00",
                        end_time="12:00",
                        notes="Visit in the morning to avoid crowds",
                    )
                )

                itinerary.total_cost += selected.price

            # Lunch break

            # Afternoon activity (14:00 - 17:00)
            afternoon_activities = [
                a
                for a in suitable_activities
                if a.type
                in [ActivityType.OUTDOOR, ActivityType.CULTURAL, ActivityType.SHOPPING]
            ]
            if (
                afternoon_activities
                and len(itinerary.activities) < MAX_ACTIVITIES_PER_DAY
            ):
                day_index = (current_date - start).days
                selected = afternoon_activities[day_index % len(afternoon_activities)]
                suitable_activities.remove(selected)

                itinerary.activities.append(
                    ScheduledActivity(
                        activity=selected,
                        date=date_str,
                        start_time="14:00",
                        end_time="17:00",
                        notes="Afternoon exploration",
                    )
                )

                itinerary.total_cost += selected.price

            # Evening activity (19:00 - 21:00)
            evening_activities = [
                a
                for a in suitable_activities
                if a.type in [ActivityType.FOOD, ActivityType.ENTERTAINMENT]
            ]
            if (
                evening_activities
                and len(itinerary.activities) < MAX_ACTIVITIES_PER_DAY
            ):
                day_index = (current_date - start).days
                selected = evening_activities[day_index % len(evening_activities)]

                itinerary.activities.append(
                    ScheduledActivity(
                        activity=selected,
                        date=date_str,
                        start_time="19:00",
                        end_time="21:00",
                        notes="Evening entertainment",
                    )
                )

                itinerary.total_cost += selected.price

            # Check budget constraints
            if context.budget_per_day and itinerary.total_cost > context.budget_per_day:
                # In a real implementation, we would adjust the itinerary to fit the budget
                itinerary.notes += (
                    f"\nNote: This day's activities exceed your "
                    f"daily budget of {context.budget_per_day} EUR."
                )

            # Add weather note
            if weather:
                itinerary.notes += (
                    f"\nWeather forecast: {weather.get('condition')}, "
                    f"{weather.get('temperature_celsius')}°C "
                    f"({weather.get('temperature_fahrenheit')}°F)"
                )

            # Add the itinerary to the dictionary
            itineraries[date_str] = itinerary

            # Move to the next day
            current_date += timedelta(days=1)

        return itineraries

    async def _generate_itinerary_summary(
        self,
        context: ActivityPlanningContext,
    ) -> str:
        """Generate a concise deterministic itinerary summary."""
        if not context.daily_itineraries:
            return (
                f"No dated activities could be scheduled for "
                f"{context.destination}."
            )

        total_cost = sum(
            itinerary.total_cost
            for itinerary in context.daily_itineraries.values()
        )
        activity_count = sum(
            len(itinerary.activities)
            for itinerary in context.daily_itineraries.values()
        )

        return (
            f"Planned {activity_count} activities across "
            f"{len(context.daily_itineraries)} day(s) in "
            f"{context.destination}. Estimated activity cost: "
            f"{total_cost:.2f} {context.currency}."
        )

    async def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call the OpenAI API with the given messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Model response
        """
        # Log inputs for debugging
        logger.debug(f"Calling model with messages: {messages}")

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Log the response
        logger.debug(f"Model response: {response}")

        # Extract the content from the response
        content = response.choices[0].message.content
        return {"content": content}
