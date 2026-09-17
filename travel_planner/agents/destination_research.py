"""
Destination Research Agent for the travel planner system.

This module implements the specialized agent responsible for researching
destination information, analyzing travel advisories, providing weather
insights, and identifying points of interest for potential travel destinations.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils import (
    AgentExecutionError,
    AgentLogger,
    handle_errors,
    with_retry,
)
from travel_planner.utils.rate_limiting import rate_limited


@dataclass
class DestinationInfo:
    """Information about a travel destination."""

    name: str
    country: str
    description: str = ""
    weather: dict[str, Any] = field(default_factory=dict)
    best_times_to_visit: list[str] = field(default_factory=list)
    points_of_interest: list[dict[str, Any]] = field(default_factory=list)
    local_transportation: list[dict[str, Any]] = field(default_factory=list)
    travel_advisories: list[dict[str, Any]] = field(default_factory=list)
    visa_requirements: str = ""
    language: str = ""
    currency: str = ""
    timezone: str = ""
    cost_index: float = 0.0


class DestinationContext(AgentContext):
    """Context for the destination research agent."""

    query: str = ""
    destinations: list[DestinationInfo] = Field(default_factory=list)
    selected_destination: DestinationInfo | None = None
    travel_dates: dict[str, str] = Field(default_factory=dict)
    search_results: dict[str, Any] = Field(default_factory=dict)


class DestinationResearchAgent(BaseAgent[DestinationContext]):
    """
    Specialized agent for researching travel destinations.

    This agent is responsible for:
    1. Analyzing user preferences to suggest appropriate destinations
    2. Researching detailed information about destinations
    3. Checking travel advisories and visa requirements
    4. Providing weather and seasonal information
    5. Identifying key points of interest and activities
    """

    def __init__(self, config: AgentConfig | None = None):
        """Initialize the destination research agent."""

        default_config = AgentConfig(
            name="Destination Research",
            instructions=(
                "You are an AI destination research specialist for travel planning. "
                "Your expertise is in providing comprehensive, accurate information about "
                "travel destinations worldwide. Research and analyze destinations based on "
                "user preferences, provide detailed information about points of interest, "
                "local travel conditions, weather patterns, and travel advisories. "
                "Your goal is to help travelers make informed decisions about their destinations."
            ),
            model="gemini-3.6-flash",
            tools=[],
        )

        super().__init__(config or default_config, DestinationContext)
        self.logger = AgentLogger(self.name)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: DestinationContext | None = None,
    ) -> dict[str, Any]:
        """
        Run the destination research agent with the provided input and context.
        """

        self.logger.info(
            "Running destination research agent with input: "
            f"{input_data if isinstance(input_data, str) else '...'}"
        )

        if context is None:
            context = DestinationContext()

        if isinstance(input_data, str):
            context.query = input_data

        try:
            result = await self.process(input_data, context)

            return {
                "context": context,
                "result": result,
            }

        except Exception as e:
            error_msg = f"Error in destination research agent: {e!s}"
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
        context: DestinationContext,
    ) -> dict[str, Any]:
        """Process the destination research request."""

        self.logger.info(
            f"Processing destination research for query: {context.query}"
        )

        self._prepare_messages(input_data)

        if not context.selected_destination:
            result = await self._suggest_destinations(context)
        else:
            result = await self._research_destination(
                context.selected_destination.name,
                context,
            )

        return result

    async def _suggest_destinations(
        self,
        context: DestinationContext,
    ) -> dict[str, Any]:
        """Suggest destinations based on user preferences."""

        self.logger.info(
            f"Suggesting destinations for query: {context.query}"
        )

        suggestion_prompt = (
            "Based on the user's preferences, suggest 3-5 suitable travel destinations. "
            "For each destination, provide a brief description explaining why it matches "
            "their preferences, the best time to visit, and any notable attractions. "
            "Format the output as a structured JSON object."
        )

        messages = [
            {
                "role": "system",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": context.query,
            },
            {
                "role": "system",
                "content": suggestion_prompt,
            },
        ]

        response = await self._call_model(messages)

        return {
            "suggestions": response.get("content", ""),
        }

    async def _research_destination(
        self,
        destination: str,
        context: DestinationContext,
    ) -> dict[str, Any]:
        """Research detailed information about a specific destination."""

        self.logger.info(
            f"Researching destination: {destination}"
        )

        research_prompt = (
            f"Provide comprehensive information about {destination} as a travel destination. "
            "Include details about the location, weather, best times to visit, main attractions, "
            "local transportation options, visa requirements, local currency, language, and any "
            "relevant travel advisories. Format the output as a structured JSON object."
        )

        messages = [
            {
                "role": "system",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": f"Research {destination} as a travel destination",
            },
            {
                "role": "system",
                "content": research_prompt,
            },
        ]

        response = await self._call_model(messages)

        return {
            "research": response.get("content", ""),
        }

    @with_retry(max_attempts=3)
    @rate_limited("gemini")
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
            # Therefore this call must NOT be awaited.
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