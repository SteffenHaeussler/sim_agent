# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- `uv install` - Install dependencies
- `uv install --dev` - Install with dev dependencies

### Running the Service
- `make dev` - Run FastAPI app in development mode (port 5055)
- `make prod` - Run FastAPI app in production mode
- `make run Q="question"` - Run via CLI with a specific question
- `make up` - Start Docker containers
- `make down` - Stop Docker containers

### Testing
- `make test` or `make tests` - Run all tests (excludes evals)
- `make coverage` - Run tests with coverage report
- `uv run python -m pytest tests/ -s -v --envfile=.env.tests` - Full test command
- Individual test files can be run with: `uv run python -m pytest path/to/test_file.py`

### Evaluation Commands
- `make eval` - Run all evaluation tests
- `make eval_e2e` - End-to-end evaluations
- `make eval_enhance` - Enhancement evaluations
- `make eval_pre_check` - Pre-check evaluations
- `make eval_post_check` - Post-check evaluations
- `make eval_ir` - Information retrieval evaluations
- `make eval_tool_agent` - Tool agent evaluations

## Architecture Overview

This is an agentic AI framework for internal question answering systems, following Domain Driven Design principles from "Architecture Patterns with Python" (Cosmic Python book).

### Core Architecture Components

**Domain Layer** (`src/agent/domain/`):
- `model.py` - BaseAgent state machine that processes commands through stages
- `commands.py` - Command objects (Question, Check, Retrieve, Rerank, Enhance, UseTools, LLMResponse, FinalCheck)
- `events.py` - Event objects for notifications and responses

**Service Layer** (`src/agent/service_layer/`):
- `messagebus.py` - Message bus for command/event handling
- `handlers.py` - Command and event handlers with dependency injection

**Adapters** (`src/agent/adapters/`):
- `adapter.py` - Abstract adapter interface
- `llm.py` - LLM integrations
- `rag.py` - RAG (Retrieval Augmented Generation) functionality
- `agent_tools.py` - Tool integrations for the agent
- `notifications.py` - Notification systems (CLI, Slack, WebSocket)

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

### Testing Structure
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for adapters
- `tests/e2e/` - End-to-end tests
- `tests/evals/` - Evaluation tests with JSON test data