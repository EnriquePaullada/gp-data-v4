# GP Data v4

AI-powered conversational sales assistant built with a three-agent architecture. Qualifies leads, extracts BANT signals, and generates contextual responses through a structured pipeline. Designed for text-based communication channels (SMS, WhatsApp, web chat).

Built with Python 3.12, FastAPI, PydanticAI, MongoDB, and Twilio.

## Architecture

```
Inbound Message ──► Security ──► Buffer ──► Queue ──► Classifier ──► Director ──► Executor ──► Response
                      │            │                      │              │             │
                  Signature     10s burst             Extract        Strategy       Message
                  + Rate Limit  + concat               BANT         Decision      Generation
```

**Classifier** -- High-precision intent classification and BANT signal extraction (GPT-4o-mini)

**Director** -- Strategic decision-making with deterministic hard-gates + LLM reasoning (GPT-4o)

**Executor** -- Channel-optimized message generation with persona, tone, and compliance enforcement (GPT-4o-mini)

## CI/CD & Git Workflow

GitHub Actions runs the full test suite against a MongoDB service container on every push and PR. See `.github/workflows/ci.yml`.

```
feature branch ──► develop ──► main (via PR with merge commit)
```

- Feature branches merge into `develop` with merge commits (preserves individual commit history)
- `develop` merges into `main` via pull request
- Branches deleted after merge
- Conventional commits: `feat:`, `fix:`, `chore:`, `test:`, `refactor:`

## Key Features

- **Three-agent pipeline** with structured output contracts between stages
- **Circuit breaker** with language-aware fallback responses for graceful LLM degradation
- **Message buffering** -- concatenates burst messages (10s window) before processing
- **Async message queue** with retry logic, dead letter queue, and concurrency control
- **Proactive follow-ups** -- scheduled re-engagement with escalating urgency
- **Human handoff** -- escalation to sales team via Slack with state machine tracking
- **Security validation** -- prompt injection detection, PII filtering, profanity/hate speech, SQLi/XSS
- **Rate limiting** with spike detection and automatic banning
- **Webhook signature verification** (HMAC-SHA256)
- **Phone normalization** -- E.164 with Mexico mobile edge case handling
- **Context pruning** -- smart conversation history management to prevent token overflow
- **Cost tracking** with per-agent attribution and hourly/daily budget enforcement
- **Prometheus metrics** -- counters, gauges, histograms across the full pipeline
- **Structured observability** -- JSON logging with business event tracking

## Project Structure

```
src/
├── agents/                  # Three-agent pipeline
│   ├── classifier_agent.py  # Intent classification + BANT extraction
│   ├── director_agent.py    # Strategic decision routing
│   └── executor_agent.py    # Message generation with persona
├── api/
│   ├── main.py              # FastAPI app, lifespan, background workers
│   ├── models/              # Webhook payload models
│   └── routes/              # Health, webhooks, metrics endpoints
├── core/
│   └── conversation_orchestrator.py  # Pipeline coordinator
├── models/                  # Domain models (Lead, Message, BANT signals)
├── repositories/            # MongoDB persistence layer
├── message_queue/           # Async queue with buffering and workers
├── services/                # Messaging, follow-up scheduler, handoff
└── utils/                   # Circuit breaker, rate limiter, security, metrics, cost tracking
```

## Development

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager
- MongoDB instance
- Twilio account
- OpenAI API key

### Setup

```bash
cp .env.example .env         # Configure environment variables
uv sync                      # Install dependencies
uv run pytest -q             # Run test suite (414 tests)
uv run uvicorn src.api.main:app --reload  # Start dev server
```

### Testing

```bash
uv run pytest -q             # All tests
uv run pytest tests/agents/  # Agent tests only
uv run ruff check src/       # Linting
```

## Configuration

All configuration is managed through environment variables. See `.env.example` for the full reference including:

- LLM model selection per agent
- Circuit breaker thresholds
- Message buffer timing
- Rate limiting windows
- Cost budget limits
- Follow-up scheduling
- Security validation toggles

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.12, asyncio |
| API | FastAPI |
| AI Agents | PydanticAI, OpenAI GPT-4o / GPT-4o-mini |
| Database | MongoDB (Motor async driver) |
| Messaging | Twilio |
| Notifications | Slack webhooks |
| Metrics | Prometheus |
| Package Manager | UV |
| CI | GitHub Actions |
| Container | Docker (multi-stage build) |
| Linting | Ruff |
