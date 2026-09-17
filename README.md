# AI Travel Planner — Gemini + LangGraph

A customized multi-agent travel planning application built on top of the open-source **OpenAI Agents Travel Graph** project by Bjorn Melin. This version adapts the original architecture for **Google Gemini**, strengthens LangGraph state handling, adds quota-safe fallbacks, and provides destination-aware itinerary generation.

> **Portfolio project:** This repository is a customized/extended version of the original project. The original project and its MIT license are credited below.

## What this project does

The application accepts a travel request such as:

```text
Delhi → Jaipur
10 Oct 2026 → 12 Oct 2026
2 travelers
Budget: ₹10,000–₹15,000
```

It then orchestrates specialized stages for query analysis, flight search, parallel research, activity planning, budget management, and final-plan generation through a LangGraph workflow.

## Key customizations

Compared with the original project, this portfolio version includes:

- **Google Gemini integration** through the OpenAI-compatible API interface.
- **LangGraph 0.4.x compatibility fixes** and native graph construction.
- **Structured travel-query handling** for origin, destination, dates, travelers, and budget.
- **Activity-model normalization** between internal agent types and the final Pydantic travel-plan schema.
- **Destination-aware itinerary catalogue**, including curated Jaipur activities with descriptions and planning estimates.
- **Multi-day activity rotation** so the same generic itinerary is not repeated for every day.
- **Quota-safe Gemini handling** for `429` quota exhaustion and `503` temporary unavailability.
- **Local budget fallback** so the workflow can still complete when the LLM budget agent is unavailable.
- **Checkpointed workflow execution** and improved error recovery.
- **Tests and Python 3.12 compatibility** maintained during the refactor.

## Architecture

```text
User Travel Request
        │
        ▼
┌─────────────────────┐
│ Query Analysis      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────┐
│        LangGraph Workflow           │
│                                     │
│  Flight Search   Parallel Research │
│         │              │            │
│         └──────┬───────┘            │
│                ▼                    │
│        Activity Planning            │
│                │                    │
│                ▼                    │
│        Budget Management            │
│                │                    │
│                ▼                    │
│         Final Travel Plan           │
└─────────────────────────────────────┘
```

### Main technologies

- Python 3.12
- LangGraph
- Google Gemini
- Pydantic
- Supabase
- Tavily
- Firecrawl
- uv
- pytest

## Example output

For a Jaipur trip, the planner can generate destination-specific activities such as:

- Amber Fort
- City Palace
- Hawa Mahal
- Jal Mahal viewpoint
- Nahargarh Fort sunset
- Johari Bazaar
- Rajasthani food experience

Activity prices shown by the curated catalogue are **planning estimates**, not guaranteed live ticket or restaurant prices.

## Gemini quota fallback

The application is designed to degrade gracefully when Gemini is unavailable.

If Gemini returns a quota or temporary-service error, the workflow can continue using deterministic local logic for supported stages rather than terminating the entire travel plan.

Example:

```text
Budget agent unavailable; using local budget fallback.
Final plan generated successfully.
```

This makes the application more resilient during development on limited API quotas.

## Setup

### 1. Clone your repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ai-travel-planner
```

### 2. Install dependencies

Install [uv](https://docs.astral.sh/uv/) if necessary, then run:

```powershell
uv sync
```

### 3. Configure environment variables

Create a local `.env` file. **Never commit it to GitHub.**

Example structure:

```env
GEMINI_API_KEY=your_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

The repository `.gitignore` excludes `.env` and local virtual-environment files.

### 4. Run tests

```powershell
uv run python -m compileall -q travel_planner
uv run python -m pytest -q
```

### 5. Run the planner

```powershell
uv run python -m travel_planner.main `
  --origin "Delhi" `
  --destination "Jaipur" `
  --travelers 2 `
  --budget "10000-15000" `
  --departure-date "2026-10-10" `
  --return-date "2026-10-12" `
  --log-level INFO
```

For command-line options:

```powershell
uv run python -m travel_planner.main --help
```

## Testing status

The development workflow has been tested locally with the Delhi → Jaipur scenario. The workflow reaches the `COMPLETE` state and successfully falls back to local budget logic when the Gemini free-tier quota is exhausted.

Live external services remain dependent on their respective API credentials, availability, and quotas.

## Project structure

```text
travel_planner/
├── agents/              # Specialized AI agents
├── data/                # Data models and persistence
├── orchestration/       # LangGraph workflow and nodes
├── utils/               # Logging, rate limiting, helpers
└── main.py              # CLI entry point

tests/                   # Automated tests
pyproject.toml            # Project/dependency configuration
uv.lock                  # Locked dependencies
```

## Attribution

This project is a customized/extended adaptation of:

**Bjorn Melin — OpenAI Agents Travel Graph**

Original repository:
https://github.com/BjornMelin/openai-agents-travel-graph

The original project describes a multi-agent travel planning system using OpenAI Agents SDK and LangGraph. This repository preserves attribution to that work while documenting the modifications made in this version.

## License

The original project is released under the **MIT License**. The MIT license text and copyright attribution are retained in this repository in `LICENSE`.

See `LICENSE` for the complete terms.

## Disclaimer

Travel prices, activity costs, availability, and recommendations can change. Information produced by this application should be verified against current provider information before making bookings or financial decisions.
