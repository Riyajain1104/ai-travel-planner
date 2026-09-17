"""
Base agent class for the travel planner system.

This module implements the foundational agent class used by all
specialized travel-planning agents.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from openai import OpenAI
from pydantic import BaseModel


T = TypeVar("T")


class AgentContext(BaseModel):
    """Base context shared by agents."""

    pass


class TravelPlannerAgentError(Exception):
    """Base exception for all agent-related errors."""

    pass


class InvalidConfigurationException(TravelPlannerAgentError):
    """Raised when an agent configuration is invalid."""

    pass


@dataclass
class AgentConfig:
    """Configuration for a travel-planning agent."""

    name: str
    instructions: str

    # Keep the original public default for backwards compatibility.
    # Individual production agents can explicitly use Gemini models.
    model: str = "gpt-4o"

    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[Any] = field(default_factory=list)


class BaseAgent(Generic[T]):
    """
    Base class for all travel-planner agents.

    The application uses Google's Gemini API through its
    OpenAI-compatible endpoint. The provider is therefore controlled
    by the client configuration while the AgentConfig model remains
    backwards compatible with the original project.
    """

    def __init__(
        self,
        config: AgentConfig,
        context_type: type[T] | None = None,
    ):
        """
        Initialize an agent.

        Args:
            config: Agent configuration.
            context_type: Optional context model.
        """

        self.config = config

        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            ),
        )

        self.context_type = (
            context_type or AgentContext
        )

        self._validate_config()

    @property
    def name(self) -> str:
        """Return the agent name."""

        return self.config.name

    @property
    def instructions(self) -> str:
        """Return the agent instructions."""

        return self.config.instructions

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: T | None = None,
    ) -> Any:
        """
        Run the agent.

        Specialized agents should override this method.
        """

        raise NotImplementedError(
            "Subclasses must implement run method"
        )

    async def process(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Process an agent request.

        Specialized agents should override this method.
        """

        raise NotImplementedError(
            "Subclasses must implement process method"
        )

    def _validate_config(self) -> bool:
        """Validate the agent configuration."""

        if not self.config.name:
            raise InvalidConfigurationException(
                "Agent name cannot be empty"
            )

        if not self.config.instructions:
            raise InvalidConfigurationException(
                "Agent instructions cannot be empty"
            )

        if not self.config.model:
            raise InvalidConfigurationException(
                "Agent model cannot be empty"
            )

        return True

    def _prepare_messages(
        self,
        input_data: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert input into chat-completion messages.
        """

        if isinstance(input_data, str):

            return [
                {
                    "role": "system",
                    "content": self.instructions,
                },
                {
                    "role": "user",
                    "content": input_data,
                },
            ]

        if not input_data:

            return [
                {
                    "role": "system",
                    "content": self.instructions,
                }
            ]

        if input_data[0].get("role") != "system":

            return [
                {
                    "role": "system",
                    "content": self.instructions,
                },
                *input_data,
            ]

        return input_data