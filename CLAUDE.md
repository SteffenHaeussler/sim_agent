# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- `uv install` - Install dependencies
- `uv install --dev` - Install with dev dependencies

### Running the Service
- `make dev` - Run FastAPI app in development mode (port 5055)
- `make prod` - Run FastAPI app in production mode
- `make run Q="question" M="mode"` - Run via CLI with a specific question and mode
- `make up` - Start Docker containers
- `make down` - Stop Docker containers

### Testing
- `make test` or `make tests` - Run all tests (excludes evals)
- `make coverage` - Run tests with coverage report
- `uv run python -m pytest tests/ -s -v --envfile=.env.tests` - Full test command
- Individual test files can be run with: `uv run python -m pytest path/to/test_file.py`

### Evaluation Commands
- `make eval_sql` - Run all SQL agent evaluations
- `make eval_tool` - Run all tool agent evaluations
- SQL evaluations: `eval_sql_aggregate`, `eval_sql_construct`, `eval_sql_e2e`, `eval_sql_filter`, `eval_sql_grounding`, `eval_sql_join`, `eval_sql_pre_check`
- Tool evaluations: `eval_tool_e2e`, `eval_tool_enhance`, `eval_tool_pre_check`, `eval_tool_post_check`, `eval_tool_ir`, `eval_tool_tools`

## Architecture Overview

This is an agentic AI framework for internal question answering systems, following Domain Driven Design principles from "Architecture Patterns with Python" (Cosmic Python book).

### Core Architecture Components

**Domain Layer** (`src/agent/domain/`):
- `model.py` - BaseAgent state machine that processes commands through stages
- `commands.py` - Command objects (Question, Check, Retrieve, Rerank, Enhance, UseTools, LLMResponse, FinalCheck)
- `events.py` - Event objects for notifications and responses
- `sql_model.py` - SQL-specific agent implementation
- `scenario_model.py` - Scenario-specific agent implementation

**Service Layer** (`src/agent/service_layer/`):
- `messagebus.py` - Message bus for command/event handling
- `handlers.py` - Command and event handlers with dependency injection

**Adapters** (`src/agent/adapters/`):
- `adapter.py` - Abstract adapter interface
- `llm.py` - LLM integrations
- `rag.py` - RAG (Retrieval Augmented Generation) functionality
- `agent_tools.py` - Tool integrations for the agent
- `notifications.py` - Notification systems (CLI, Slack, WebSocket)
- `database.py` - Database connections and operations

**Tools** (`src/agent/adapters/tools/`):
- Tool implementations for data retrieval, conversion, and analysis
- Each tool inherits from `base.py` Tool class

### Agent Flow
The BaseAgent follows a state machine pattern:
1. Question → Check (guardrails)
2. Check → Retrieve (from knowledge base)
3. Retrieve → Rerank (documents)
4. Rerank → Enhance (question via LLM)
5. Enhance → UseTools (agent tools)
6. UseTools → LLMResponse (final generation)
7. LLMResponse → FinalCheck (guardrails)
8. FinalCheck → Evaluation (complete)

### Key Features
- **Observability**: Integrated tracing with Langfuse and OpenTelemetry
- **Real-time Communication**: WebSocket support for live updates
- **Notifications**: Multi-channel notifications (CLI, Slack, WebSocket)
- **Evaluation Framework**: Comprehensive evaluation suite for different components
- **Dependency Injection**: Bootstrap system with handler dependency injection

### Entry Points
- **FastAPI App**: `src/agent/entrypoints/app.py` - Web API with WebSocket support
- **CLI**: `src/agent/entrypoints/main.py` - Command line interface

### Configuration
- Environment-based configuration in `src/agent/config.py`
- Prompts defined in YAML files in `src/agent/prompts/`
- Uses `.env` files for environment variables
- Test environment uses `.env.tests` for test-specific configuration

### Testing Structure
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for adapters
- `tests/e2e/` - End-to-end tests
- `evals/` - Evaluation tests with performance benchmarks and quality assessments
