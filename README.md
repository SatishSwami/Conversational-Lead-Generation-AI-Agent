# Conversational Lead Generation AI Agent

> Production-oriented conversational AI agent for product discovery, intent detection, RAG-based knowledge retrieval, and structured lead capture.

**Live API:** https://autostream-ai-agent-4gt2.onrender.com

---

## Overview

The Conversational Lead Generation AI Agent is a stateful conversational system designed to turn product-related conversations into qualified leads.

The agent can:

- Understand conversational user intent
- Answer product and pricing questions using a local knowledge base
- Retrieve relevant information using deterministic TF-IDF-based RAG
- Maintain multi-turn conversation state using LangGraph
- Detect high-intent prospects
- Collect lead information progressively
- Persist conversations, messages, and captured leads
- Expose the agent through a FastAPI REST API
- Protect API endpoints using API-key authentication
- Support WhatsApp webhook integration
- Run inside a Docker container
- Handle LLM/provider failures gracefully
- Provide request IDs and latency logging for observability

The project is designed as a standalone professional engineering project with a focus on reliability, security, maintainability, and deployment readiness.

---

## Architecture

```text
                         ┌──────────────────────┐
                         │       Client         │
                         │ CLI / REST / WhatsApp│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │  Authentication      │
                         │  Request Validation  │
                         │  Request ID / Logs   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     LangGraph        │
                         │     Agent State      │
                         └──────────┬───────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
       ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
       │ Intent         │  │ TF-IDF RAG     │  │ Lead Collection│
       │ Classification │  │ Pipeline       │  │ Flow           │
       └────────────────┘  └───────┬────────┘  └───────┬────────┘
                                   │                    │
                                   ▼                    ▼
                         ┌──────────────────┐  ┌──────────────────┐
                         │ Local Knowledge  │  │ Lead Capture     │
                         │ Base             │  │ Service          │
                         └──────────────────┘  └──────────────────┘

                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Configured LLM       │
                         │ Google / OpenAI /     │
                         │ Anthropic             │
                         └──────────────────────┘

                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ SQLAlchemy Database  │
                         │ Conversations        │
                         │ Messages             │
                         │ Leads                │
                         └──────────────────────┘
Agent Workflow

Every incoming message enters the LangGraph state machine.

User Message
     │
     ▼
Classify Intent
     │
     ├── Casual Greeting
     │      └── Generate Response
     │
     ├── Product / Pricing Inquiry
     │      └── Retrieve Relevant Context
     │              └── Generate Response
     │
     └── High-Intent Lead
            └── Activate Lead Collection
                    └── Collect Lead Information

The graph maintains conversation state across turns, allowing the agent to continue an ongoing lead-collection conversation instead of treating every request as an independent interaction.

Intent Classification

The system distinguishes between three primary conversational categories:

CASUAL_GREETING
PRODUCT_INQUIRY
HIGH_INTENT_LEAD

Intent classification is performed before routing the request through the graph.

The resulting intent determines whether the system should:

respond conversationally,
retrieve information from the knowledge base,
or activate the lead-capture workflow.
RAG Pipeline

The project uses a deterministic TF-IDF retrieval pipeline instead of an external vector database.

The knowledge base is intentionally small and structured, so a lightweight local retrieval system provides:

deterministic results,
simple deployment,
low infrastructure overhead,
easy debugging,
no dependency on an external vector database.

The retrieval pipeline:

Loads the local knowledge base.
Tokenizes documents and queries.
Calculates TF-IDF relevance.
Applies query/entity/phrase matching.
Applies intent-aware relevance boosts.
Ranks candidate documents.
Returns the most relevant context to the response-generation node.

The default retrieval limit is three relevant results.

Lead Capture

When a user demonstrates high purchase intent, the agent activates the lead-collection workflow.

The system progressively collects:

Name
  ↓
Email
  ↓
Creator Platform

The information is collected conversationally rather than through a single form.

Once sufficient information has been collected, the lead is persisted through the lead repository.

The system is designed to avoid triggering lead capture prematurely for ordinary greetings or low-intent questions.

Database Persistence

The application uses SQLAlchemy for durable application data.

Conversation

Stores:

session ID
creation timestamp
update timestamp
Message

Stores:

conversation/session
role
message content
timestamp
Lead

Stores:

session ID
name
email
creator platform
timestamps

SQLite is used by default for local development.

The database layer is structured so that PostgreSQL can be configured through:

DATABASE_URL=...

The active LangGraph state is maintained separately from durable business data.

This allows the application to use active in-memory graph state during execution while persisting important conversation and lead information in the database.

API
Health Check
GET /health

Returns application and database health information.

Example:

{
  "status": "ok",
  "service": "AutoStream AI Agent",
  "version": "1.0.0",
  "database": "ok"
}
Chat
POST /chat

Requires:

X-API-Key: <API_KEY>

Request:

{
  "session_id": "user_001",
  "message": "What plans do you offer?"
}

Response:

{
  "session_id": "user_001",
  "response": "...",
  "intent": "PRODUCT_INQUIRY",
  "lead_captured": false,
  "turn_count": 1
}
Reset Session
POST /chat/reset?session_id=user_001

Requires API authentication.

Metrics
GET /metrics

Requires API authentication.

WhatsApp
GET /whatsapp
POST /whatsapp

The WhatsApp endpoints support Meta webhook verification and incoming WhatsApp messages.

API Security

Production API endpoints require an API key.

The implementation uses:

X-API-Key authentication
constant-time comparison using hmac.compare_digest
Pydantic request validation
session ID validation
message length limits
production enforcement when no API key is configured
WhatsApp HMAC-SHA256 signature verification
no request-body logging
no API-token logging

The public health endpoint remains accessible without authentication so deployment platforms can perform health checks.

Security Verification

The deployed API was explicitly tested for:

unauthenticated /chat requests → 401 Unauthorized
invalid API keys → 401 Unauthorized
authenticated /chat requests reaching the application successfully
Request Observability

Each API request receives a request ID.

If a valid X-Request-ID is supplied, it is preserved.

Otherwise, the application generates a UUID-based request ID.

The application records:

request_id
HTTP method
path
HTTP status
request duration

Example:

Request completed |
request_id=... |
method=POST |
path=/chat |
status=200 |
duration_ms=...

User message contents and authentication secrets are not logged.

Reliability

LLM calls are wrapped with controlled error handling.

If the configured LLM provider becomes temporarily unavailable, the API does not expose the underlying provider exception directly to the user.

Instead, the agent returns a controlled fallback response.

This protects the API from crashing because of temporary external provider failures.

LLM Provider Abstraction

LLM construction is isolated behind a provider factory.

Supported providers:

Google Gemini
OpenAI
Anthropic

Provider selection is controlled through environment configuration:

LLM_PROVIDER=google
LLM_MODEL=<model-name>

This keeps provider-specific model construction outside the core LangGraph workflow and makes provider switching easier.

Docker

The application includes a production-oriented Dockerfile.

The container:

uses Python 3.11
installs dependencies without retaining pip cache
runs as a non-root appuser
exposes port 8000
includes a Docker health check
keeps secrets outside the image
copies only required application components
Build
docker build -t autostream-agent:latest .
Run
docker run -d \
  --name autostream-agent \
  -p 8000:8000 \
  --env-file .env \
  autostream-agent:latest
Deployment

The application is deployed as a Docker-based web service on Render.

Live API

https://autostream-ai-agent-4gt2.onrender.com

Health Check

https://autostream-ai-agent-4gt2.onrender.com/health

Deployment flow:

GitHub
   ↓
production-upgrade branch
   ↓
Render
   ↓
Docker build
   ↓
FastAPI / Uvicorn
   ↓
HTTPS API

The public deployment has been verified with:

public HTTPS health check
database health check
authenticated /chat request
invalid API-key rejection
Docker production testing
Environment Configuration

Create a local .env file.

Example:

ENV=development
PORT=8000

LLM_PROVIDER=google
LLM_MODEL=<your-model>

GOOGLE_API_KEY=<your-key>

DATABASE_URL=sqlite:///./data/autostream.db

API_KEY=<your-api-key>

LOG_LEVEL=INFO

Other supported providers:

LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=<your-key>

or:

LLM_PROVIDER=openai
OPENAI_API_KEY=<your-key>

Never commit .env files or API keys to Git.

Local Setup
1. Clone
git clone https://github.com/SatishSwami/Conversational-Lead-Generation-AI-Agent.git
cd Conversational-Lead-Generation-AI-Agent
2. Create Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate

Linux/macOS:

python -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure Environment

Create .env with the required LLM provider API key and application configuration.

5. Run CLI
python main.py

Optional provider override:

python main.py --provider google
6. Run API
uvicorn webhook_server:app --host 0.0.0.0 --port 8000
Testing

The project currently contains 83 automated tests covering:

agent behavior
intent classification
RAG retrieval
lead collection
graph routing
API behavior
API security
database persistence
reliability
configuration
observability

Run:

pytest -q

Expected result:

83 passed
Example Conversation
User:
What plans do you offer?

Agent:
The Basic plan is $29/month and the Pro plan is $79/month...

A high-intent conversation can transition into:

User:
I want to get started.

Agent:
I'd be happy to help. What's your name?

User:
Alex Johnson

Agent:
What's the best email address to reach you?

User:
alex@example.com

Agent:
Which platform do you primarily create for?

User:
YouTube

The captured lead can then be persisted in the application database.

Project Structure
.
├── agent/
│   ├── __init__.py
│   ├── graph.py
│   ├── intent_classifier.py
│   ├── nodes.py
│   ├── rag_pipeline.py
│   └── state.py
│
├── core/
│   ├── __init__.py
│   ├── config.py
│   └── llm.py
│
├── database/
│   ├── __init__.py
│   ├── connection.py
│   ├── models.py
│   └── repository.py
│
├── tools/
│   ├── __init__.py
│   └── lead_capture.py
│
├── utils/
│   ├── logger.py
│   └── session_manager.py
│
├── autostream_kb.json
├── main.py
├── webhook_server.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── pytest.ini
└── README.md
Design Decisions
Decision	Reason
LangGraph	Explicit stateful conversational workflow
TF-IDF RAG	Small static knowledge base does not require a vector database
Intent-aware retrieval	Improves retrieval for targeted queries such as pricing
SQLAlchemy	Structured persistence and database portability
SQLite default	Simple local development
PostgreSQL compatibility	Allows deployment with a PostgreSQL database
LLM provider factory	Keeps provider-specific construction isolated
FastAPI	Lightweight typed REST API
API-key authentication	Protects application endpoints
Request IDs	Enables request tracing
Docker	Reproducible deployment environment
Non-root container	Reduces container privilege
Graceful LLM fallback	Prevents provider failures from crashing the API
Production Considerations

The current implementation is intentionally lightweight while following production-oriented engineering practices.

Potential future improvements include:

PostgreSQL as the primary production database
Redis-backed distributed session state
centralized log aggregation
metrics and distributed tracing
background job processing
rate limiting
automated CI/CD
dedicated secret management
persistent vector retrieval for larger knowledge bases
automated LLM response evaluation

These are future scalability considerations and are not claimed as currently implemented features.

License

MIT


### Now do this

After saving the file, run:

```powershell
git diff --check