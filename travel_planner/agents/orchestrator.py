"""
Orchestrator agent for the travel planner system.

This module implements the orchestrator agent that coordinates the activities
of all specialized agents, manages the travel planning workflow, and ensures
proper handoffs between different components of the system.
"""
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import Field

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils import (
    AgentExecutionError,
    AgentLogger,
    APIError,
    handle_errors,
    safe_serialize,
)
from travel_planner.utils.rate_limiting import rate_limit_manager

class PlanningStage(str, Enum):
    """Stages of the travel planning process."""

    INITIAL = "initial"
    DESTINATION_RESEARCH = "destination_research"
    FLIGHT_SEARCH = "flight_search"
    ACCOMMODATION_SEARCH = "accommodation_search"
    TRANSPORTATION_PLANNING = "transportation_planning"
    ACTIVITY_PLANNING = "activity_planning"
    BUDGET_MANAGEMENT = "budget_management"
    FINAL_ITINERARY = "final_itinerary"


@dataclass
class TravelRequirements:
    """User's travel requirements."""

    origin: str | None = None
    destination: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    duration_days: int | None = None
    budget: float | None = None
    currency: str = "USD"
    num_travelers: int = 1
    accommodation_preferences: list[str] = field(default_factory=list)
    transportation_preferences: list[str] = field(default_factory=list)
    activity_preferences: list[str] = field(default_factory=list)
    dietary_restrictions: list[str] = field(default_factory=list)
    accessibility_needs: list[str] = field(default_factory=list)
    additional_notes: str | None = None


class OrchestratorContext(AgentContext):
    """Context for the orchestrator agent."""

    session_id: str
    planning_stage: PlanningStage = PlanningStage.INITIAL

    travel_requirements: TravelRequirements = Field(
        default_factory=TravelRequirements
    )

    destination_details: dict[str, Any] = Field(default_factory=dict)
    flight_options: list[dict[str, Any]] = Field(default_factory=list)
    accommodation_options: list[dict[str, Any]] = Field(default_factory=list)
    transportation_options: list[dict[str, Any]] = Field(default_factory=list)
    activity_options: list[dict[str, Any]] = Field(default_factory=list)

    budget_allocation: dict[str, float] = Field(default_factory=dict)
    selected_options: dict[str, Any] = Field(default_factory=dict)
    user_feedback: dict[str, Any] = Field(default_factory=dict)
    final_itinerary: dict[str, Any] = Field(default_factory=dict)

    conversation_history: list[dict[str, Any]] = Field(default_factory=list)


class OrchestratorAgent(BaseAgent[OrchestratorContext]):
    """
    Orchestrator agent that coordinates the overall travel planning process.

    This agent is responsible for:
    1. Managing the travel planning workflow
    2. Coordinating communication between specialized agents
    3. Maintaining the master context of the planning session
    4. Ensuring all user requirements are met
    5. Handling exceptions and fallbacks from other agents
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the orchestrator agent.

        Args:
            config: Configuration for the agent (optional)
        """

        default_config = AgentConfig(
            name="Travel Orchestrator",
            instructions=(
                "You are an AI travel planning orchestrator. Your job is to guide the overall "
                "travel planning process by coordinating specialized agents for destination research, "
                "flight search, accommodation booking, transportation arrangements, activity planning, "
                "and budget management. Maintain a coherent plan that satisfies all user requirements "
                "while optimizing for budget, convenience, and user preferences."
            ),
            model="gemini-3.6-flash",
        )

        super().__init__(config or default_config, OrchestratorContext)

        self.logger = AgentLogger(self.name)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: OrchestratorContext | None = None,
    ) -> dict[str, Any]:
        """
        Run the orchestrator agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional orchestrator context

        Returns:
            Updated orchestrator context and response
        """

        self.logger.info(
            f"Running orchestrator agent with input: "
            f"{input_data if isinstance(input_data, str) else '...'}"
        )

        if context is None:
            from travel_planner.utils.helpers import generate_session_id

            context = OrchestratorContext(
                session_id=generate_session_id()
            )

        # Add user input to conversation history
        if isinstance(input_data, str):
            context.conversation_history.append(
                {
                    "role": "user",
                    "content": input_data,
                }
            )

        # Process the input based on the current planning stage
        try:
            response = await self.process(input_data, context)

            # Add agent response to conversation history
            if isinstance(response, dict) and "content" in response:
                context.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": response["content"],
                    }
                )

            return {
                "context": context,
                "response": response,
            }

        except Exception as e:
            error_msg = f"Error in orchestrator agent: {e!s}"
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
        context: OrchestratorContext,
    ) -> dict[str, Any]:
        """
        Process the input based on the current planning stage.
        """

        self.logger.info(
            f"Processing input in stage: {context.planning_stage}"
        )

        # Initial stage: use one Gemini call to extract structured
        # requirements. This avoids wasting a second model request.
        if context.planning_stage == PlanningStage.INITIAL:
            updated_requirements = await self._extract_requirements(
                input_data,
                context.travel_requirements,
            )

            context.travel_requirements = updated_requirements
            context.planning_stage = PlanningStage.DESTINATION_RESEARCH

            response = {
                "content": (
                    "Travel requirements extracted successfully. "
                    f"Destination: {updated_requirements.destination or 'Unknown'}, "
                    f"Travelers: {updated_requirements.num_travelers}, "
                    f"Budget: "
                    f"{updated_requirements.budget or 'Not specified'} "
                    f"{updated_requirements.currency}."
                )
            }

            self.logger.info(
                "Initial requirements extracted without an additional "
                "Gemini call."
            )

            return response

        # For later stages, use the model to determine the next step.
        messages = self._prepare_messages(input_data)

        messages.append(
            {
                "role": "system",
                "content": (
                    f"Current planning stage: {context.planning_stage}. "
                    f"Travel requirements: "
                    f"{safe_serialize(context.travel_requirements)}. "
                    "Use the context information to determine the next "
                    "steps in the planning process."
                ),
            }
        )

        response = await self._call_model(messages)

        await self._update_planning_stage(
            context,
            response,
        )

        return response

    async def _extract_requirements(
        self,
        input_data: str | list[dict[str, Any]],
        current_requirements: TravelRequirements,
    ) -> TravelRequirements:
        """
        Extract travel requirements from user input.

        Gemini is used for flexible extraction, with a deterministic fallback
        for common travel-query patterns. Existing requirements are preserved
        when the user does not provide replacement values.
        """

        import json
        import re

        self.logger.info("Extracting travel requirements")

        # ---------------------------------------------------------
        # Get the actual user text
        # ---------------------------------------------------------
        payload: dict[str, Any] | None = None

        if isinstance(input_data, str):
            user_input = input_data.strip()
            # The workflow normally passes a serialized TravelQuery. Keep the
            # structured fields so CLI arguments are not lost when raw_query is
            # empty.
            try:
                parsed_payload = json.loads(user_input)
                if isinstance(parsed_payload, dict):
                    payload = parsed_payload
                    if payload.get("raw_query"):
                        user_input = str(payload["raw_query"]).strip()
                    else:
                        user_input = ""
            except (TypeError, json.JSONDecodeError):
                pass
        elif isinstance(input_data, list):
            user_input = self._get_latest_user_input(input_data).strip()
        else:
            user_input = str(input_data).strip()

        # If the caller supplied a structured TravelQuery (for example through
        # the CLI), use those values directly. This avoids an unnecessary Gemini
        # request and, more importantly, preserves origin/destination/travelers/
        # budget even when raw_query is empty.
        if payload is not None and not user_input:
            budget_range = payload.get("budget_range")
            requirements_data = payload.get("requirements") or {}
            if not isinstance(requirements_data, dict):
                requirements_data = {}

            budget = None
            if isinstance(budget_range, dict):
                budget = budget_range.get("max")
                if budget is None:
                    budget = budget_range.get("min")
            elif isinstance(budget_range, (int, float)):
                budget = budget_range

            structured_values = {
                "origin": payload.get("origin"),
                "destination": payload.get("destination"),
                "start_date": payload.get("departure_date"),
                "end_date": payload.get("return_date"),
                "budget": budget,
                "currency": requirements_data.get("currency") or "INR",
                "num_travelers": payload.get("travelers"),
                "duration_days": requirements_data.get("duration_days"),
                "accommodation_preferences": requirements_data.get(
                    "accommodation_preferences", []
                ),
                "transportation_preferences": requirements_data.get(
                    "transportation_preferences", []
                ),
                "activity_preferences": requirements_data.get(
                    "activity_preferences", []
                ),
                "dietary_restrictions": requirements_data.get(
                    "dietary_restrictions", []
                ),
                "accessibility_needs": requirements_data.get(
                    "accessibility_needs", []
                ),
                "additional_notes": requirements_data.get("additional_notes"),
            }

            if any(
                value not in (None, "", [])
                for value in structured_values.values()
            ):
                return TravelRequirements(
                    origin=structured_values["origin"]
                    or current_requirements.origin,
                    destination=structured_values["destination"]
                    or current_requirements.destination,
                    start_date=structured_values["start_date"]
                    or current_requirements.start_date,
                    end_date=structured_values["end_date"]
                    or current_requirements.end_date,
                    duration_days=structured_values["duration_days"]
                    or current_requirements.duration_days,
                    budget=(
                        float(structured_values["budget"])
                        if structured_values["budget"] is not None
                        else current_requirements.budget
                    ),
                    currency=(
                        str(structured_values["currency"])
                        if structured_values["currency"]
                        else current_requirements.currency
                    ),
                    num_travelers=(
                        int(structured_values["num_travelers"])
                        if structured_values["num_travelers"] is not None
                        else current_requirements.num_travelers
                    ),
                    accommodation_preferences=structured_values[
                        "accommodation_preferences"
                    ] or current_requirements.accommodation_preferences,
                    transportation_preferences=structured_values[
                        "transportation_preferences"
                    ] or current_requirements.transportation_preferences,
                    activity_preferences=structured_values[
                        "activity_preferences"
                    ] or current_requirements.activity_preferences,
                    dietary_restrictions=structured_values[
                        "dietary_restrictions"
                    ] or current_requirements.dietary_restrictions,
                    accessibility_needs=structured_values[
                        "accessibility_needs"
                    ] or current_requirements.accessibility_needs,
                    additional_notes=structured_values["additional_notes"]
                    or current_requirements.additional_notes,
                )

        if not user_input:
            self.logger.warning("No user input available for requirement extraction")
            return current_requirements

        # ---------------------------------------------------------
        # Helper: deterministic extraction
        # ---------------------------------------------------------
        def fallback_extract() -> dict[str, Any]:
            text = user_input
            lower = text.lower()

            extracted: dict[str, Any] = {
                "origin": None,
                "destination": None,
                "start_date": None,
                "end_date": None,
                "duration_days": None,
                "budget": None,
                "currency": None,
                "num_travelers": None,
                "accommodation_preferences": [],
                "transportation_preferences": [],
                "activity_preferences": [],
                "dietary_restrictions": [],
                "accessibility_needs": [],
                "additional_notes": None,
            }

            # -----------------------------------------------------
            # Travelers
            # -----------------------------------------------------
            traveler_patterns = [
                r"for\s+(\d+)\s+(?:travell?ers?|people|persons?)",
                r"(\d+)\s+(?:travell?ers?|people|persons?)",
                r"party\s+of\s+(\d+)",
            ]

            for pattern in traveler_patterns:
                match = re.search(pattern, lower)
                if match:
                    extracted["num_travelers"] = int(match.group(1))
                    break

            # -----------------------------------------------------
            # Duration
            # Examples:
            #   3-day trip
            #   3 days trip
            #   trip for 3 days
            # -----------------------------------------------------
            duration_patterns = [
                r"(\d+)\s*[- ]?\s*day(?:s)?\s+(?:trip|tour|vacation|holiday)",
                r"(?:trip|tour|vacation|holiday)\s+(?:for\s+)?(\d+)\s*days?",
                r"(\d+)\s*days?",
            ]

            for pattern in duration_patterns:
                match = re.search(pattern, lower)
                if match:
                    extracted["duration_days"] = int(match.group(1))
                    break

            # -----------------------------------------------------
            # Origin and destination
            # Examples:
            #   from Delhi to Jaipur
            #   Delhi to Jaipur
            # -----------------------------------------------------
            route_match = re.search(
                r"\bfrom\s+(.+?)\s+to\s+(.+?)(?=\s+for\s+\d+\s+(?:day|days|travell?ers?|people)|"
                r"\s+with\s+(?:a\s+)?budget|\s+on\s+|\s*$)",
                text,
                re.IGNORECASE,
            )

            if route_match:
                extracted["origin"] = route_match.group(1).strip(" ,.")
                extracted["destination"] = route_match.group(2).strip(" ,.")
            else:
                simple_route_match = re.search(
                    r"\bfrom\s+([A-Za-z][A-Za-z .'-]+?)\s+to\s+([A-Za-z][A-Za-z .'-]+?)(?=\s|$)",
                    text,
                    re.IGNORECASE,
                )

                if simple_route_match:
                    extracted["origin"] = simple_route_match.group(1).strip(" ,.")
                    extracted["destination"] = simple_route_match.group(2).strip(" ,.")

            # -----------------------------------------------------
            # Budget and currency
            # Examples:
            #   budget of 15000 INR
            #   budget of ₹15000
            #   budget 15000
            #   INR 15000
            # -----------------------------------------------------
            currency_patterns = [
                (r"\bINR\b|₹|rs\.?|rupees?", "INR"),
                (r"\bUSD\b|\$", "USD"),
                (r"\bEUR\b|€", "EUR"),
                (r"\bGBP\b|£", "GBP"),
            ]

            detected_currency = None

            for pattern, currency in currency_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected_currency = currency
                    break

            budget_patterns = [
                r"budget(?:\s+of|\s+is|\s*:)?\s*(?:₹|rs\.?|inr|usd|\$|eur|€|gbp|£)?\s*([\d,]+(?:\.\d+)?)",
                r"(?:₹|rs\.?|inr|usd|\$|eur|€|gbp|£)\s*([\d,]+(?:\.\d+)?)",
            ]

            for pattern in budget_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted["budget"] = float(match.group(1).replace(",", ""))
                    break

            if detected_currency:
                extracted["currency"] = detected_currency

            # -----------------------------------------------------
            # Date extraction
            # Keep this intentionally conservative.
            # Gemini handles natural-language dates better.
            # -----------------------------------------------------
            iso_date_matches = re.findall(
                r"\b(20\d{2}-\d{2}-\d{2})\b",
                text,
            )

            if iso_date_matches:
                extracted["start_date"] = iso_date_matches[0]

                if len(iso_date_matches) > 1:
                    extracted["end_date"] = iso_date_matches[1]

            # -----------------------------------------------------
            # Common activity preferences
            # -----------------------------------------------------
            activity_keywords = {
                "history": "history",
                "historical": "history",
                "culture": "culture",
                "cultural": "culture",
                "food": "food",
                "architecture": "architecture",
                "shopping": "shopping",
                "adventure": "adventure",
                "nature": "nature",
                "beach": "beach",
                "nightlife": "nightlife",
            }

            for keyword, value in activity_keywords.items():
                if keyword in lower and value not in extracted["activity_preferences"]:
                    extracted["activity_preferences"].append(value)

            return extracted

        # ---------------------------------------------------------
        # Gemini extraction
        # ---------------------------------------------------------
        extraction_prompt = """
    Extract the travel requirements from the user's input.

    Return ONLY a valid JSON object with these fields:

    {
      "origin": null,
      "destination": null,
      "start_date": null,
      "end_date": null,
      "duration_days": null,
      "budget": null,
      "currency": null,
      "num_travelers": null,
      "accommodation_preferences": [],
      "transportation_preferences": [],
      "activity_preferences": [],
      "dietary_restrictions": [],
      "accessibility_needs": [],
      "additional_notes": null
    }

    Rules:
    - Extract the departure/origin city if provided.
    - Extract the destination city if provided.
    - Extract the number of travelers.
    - Extract the total trip budget as a number.
    - Extract the currency.
    - If the user says "5-day trip", set duration_days to 5.
    - If explicit dates are provided, extract them.
    - Do not invent missing information.
    - For missing values, use null or an empty list.
    - Return JSON only. No markdown.
    """

        messages = [
            {
                "role": "system",
                "content": extraction_prompt,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]

        if current_requirements and any(
            getattr(current_requirements, field, None)
            for field in [
                "origin",
                "destination",
                "start_date",
                "end_date",
                "duration_days",
                "budget",
                "currency",
                "num_travelers",
            ]
        ):
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Existing requirements. Preserve these values when "
                        "the user did not provide a replacement: "
                        f"{safe_serialize(current_requirements)}"
                    ),
                }
            )

        extracted: dict[str, Any] | None = None

        try:
            response = await self._call_model(messages)
            content = response.get("content", "") or ""

            cleaned_content = content.strip()

            # -----------------------------------------------------
            # Remove markdown fences
            # -----------------------------------------------------
            if "```" in cleaned_content:
                fenced_match = re.search(
                    r"```(?:json)?\s*(.*?)\s*```",
                    cleaned_content,
                    re.DOTALL | re.IGNORECASE,
                )

                if fenced_match:
                    cleaned_content = fenced_match.group(1).strip()

            # -----------------------------------------------------
            # First attempt: entire response is JSON
            # -----------------------------------------------------
            try:
                parsed = json.loads(cleaned_content)

                if isinstance(parsed, dict):
                    extracted = parsed

            except json.JSONDecodeError:
                # -------------------------------------------------
                # Second attempt: locate the first JSON object
                # -------------------------------------------------
                object_match = re.search(
                    r"\{.*\}",
                    cleaned_content,
                    re.DOTALL,
                )

                if object_match:
                    try:
                        parsed = json.loads(object_match.group(0))

                        if isinstance(parsed, dict):
                            extracted = parsed

                    except json.JSONDecodeError:
                        extracted = None

            if extracted is None:
                self.logger.warning(
                    "Gemini response was not valid JSON. "
                    "Using deterministic requirement extraction."
                )

        except Exception as exc:
            self.logger.warning(
                f"Gemini requirement extraction failed: {exc!s}. "
                "Using deterministic requirement extraction."
            )

        # ---------------------------------------------------------
        # Deterministic fallback
        # ---------------------------------------------------------
        fallback = fallback_extract()

        if extracted is None:
            extracted = fallback
        else:
            # Fill missing Gemini fields from deterministic extraction.
            for key, value in fallback.items():
                if extracted.get(key) in (None, "", []):
                    if value not in (None, "", []):
                        extracted[key] = value

        # ---------------------------------------------------------
        # Merge with existing requirements
        # ---------------------------------------------------------
        def choose(
            key: str,
            existing: Any = None,
        ) -> Any:
            value = extracted.get(key)

            if value not in (None, "", []):
                return value

            return existing

        # Normalize model/fallback values before constructing the dataclass.
        # Gemini occasionally returns an empty string for numeric fields.
        raw_budget = extracted.get("budget")
        if raw_budget in (None, ""):
            extracted["budget"] = None
        else:
            try:
                extracted["budget"] = float(str(raw_budget).replace(",", "").strip())
            except (TypeError, ValueError):
                extracted["budget"] = getattr(current_requirements, "budget", None)

        raw_travelers = extracted.get("num_travelers")
        if raw_travelers in (None, ""):
            extracted["num_travelers"] = None
        else:
            try:
                extracted["num_travelers"] = int(raw_travelers)
            except (TypeError, ValueError):
                extracted["num_travelers"] = getattr(current_requirements, "num_travelers", 1)

        try:
            updated = TravelRequirements(
                origin=choose(
                    "origin",
                    getattr(current_requirements, "origin", None),
                ),
                destination=choose(
                    "destination",
                    getattr(current_requirements, "destination", None),
                ),
                start_date=choose(
                    "start_date",
                    getattr(current_requirements, "start_date", None),
                ),
                end_date=choose(
                    "end_date",
                    getattr(current_requirements, "end_date", None),
                ),
                duration_days=choose(
                    "duration_days",
                    getattr(current_requirements, "duration_days", None),
                ),
                budget=(
                    extracted["budget"]
                    if extracted.get("budget") is not None
                    else getattr(current_requirements, "budget", None)
                ),
                currency=choose(
                    "currency",
                    getattr(current_requirements, "currency", "INR"),
                ),
                num_travelers=(
                    extracted["num_travelers"]
                    if extracted.get("num_travelers") is not None
                    else getattr(current_requirements, "num_travelers", 1)
                ),
                accommodation_preferences=choose(
                    "accommodation_preferences",
                    getattr(current_requirements, "accommodation_preferences", []),
                ),
                transportation_preferences=choose(
                    "transportation_preferences",
                    getattr(current_requirements, "transportation_preferences", []),
                ),
                activity_preferences=choose(
                    "activity_preferences",
                    getattr(current_requirements, "activity_preferences", []),
                ),
                dietary_restrictions=choose(
                    "dietary_restrictions",
                    getattr(current_requirements, "dietary_restrictions", []),
                ),
                accessibility_needs=choose(
                    "accessibility_needs",
                    getattr(current_requirements, "accessibility_needs", []),
                ),
                additional_notes=choose(
                    "additional_notes",
                    getattr(current_requirements, "additional_notes", None),
                ),
            )

            self.logger.info(
                "Travel requirements extracted successfully: "
                f"{safe_serialize(updated)}"
            )

            return updated

        except Exception as exc:
            self.logger.error(
                f"Could not construct TravelRequirements: {exc!s}"
            )
            return current_requirements

    async def _update_planning_stage(
        self,
        context: OrchestratorContext,
        response: dict[str, Any],
    ) -> None:
        """
        Update the planning stage based on the agent's response.

        Args:
            context: Orchestrator context
            response: Agent response
        """

        current_stage = context.planning_stage

        stage_progression = {
            PlanningStage.INITIAL: PlanningStage.DESTINATION_RESEARCH,
            PlanningStage.DESTINATION_RESEARCH: PlanningStage.FLIGHT_SEARCH,
            PlanningStage.FLIGHT_SEARCH: PlanningStage.ACCOMMODATION_SEARCH,
            PlanningStage.ACCOMMODATION_SEARCH: (
                PlanningStage.TRANSPORTATION_PLANNING
            ),
            PlanningStage.TRANSPORTATION_PLANNING: (
                PlanningStage.ACTIVITY_PLANNING
            ),
            PlanningStage.ACTIVITY_PLANNING: (
                PlanningStage.BUDGET_MANAGEMENT
            ),
            PlanningStage.BUDGET_MANAGEMENT: (
                PlanningStage.FINAL_ITINERARY
            ),
        }

        if current_stage in stage_progression:
            context.planning_stage = stage_progression[current_stage]

            self.logger.info(
                f"Updating planning stage from "
                f"{current_stage} to {context.planning_stage}"
            )

    def _get_latest_user_input(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """
        Extract the latest user input from a list of messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Latest user input text
        """

        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")

        return ""

    async def _call_model(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Call the Gemini-compatible API with the given messages.
        """

        self.logger.info(
            f"Calling model with {len(messages)} messages"
        )

        self.logger.log_llm_input(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )

        try:
            limiter = rate_limit_manager.get_limiter("gemini")

            # Fail fast when our local quota/rate-limit guard says another
            # Gemini request should not be made. Waiting and immediately
            # acquiring again used to create repeated recovery loops.
            if not await limiter.acquire():
                stats = limiter.get_quota_stats()
                raise APIError(
                    (
                        "Gemini request blocked by the local rate-limit "
                        f"guard. Remaining daily quota: {stats['remaining']}. "
                        "Please wait for the quota window to reset."
                    ),
                    "gemini",
                    status_code=429,
                )

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
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
                    "content": content,
                }

            return {
                "content": "No response generated.",
            }

        except Exception as e:
            self.logger.error(
                f"Error calling model: {e!s}"
            )
            raise