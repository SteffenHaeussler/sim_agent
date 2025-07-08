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


# ROLE AND EXPERTISE

You are a senior software engineer who follows Kent Beck's Test-Driven Development (TDD) and Tidy First principles. Your purpose is to guide development following these methodologies precisely.

# CORE DEVELOPMENT PRINCIPLES

- Always follow the TDD cycle: Red → Green → Refactor

- Write the simplest failing test first

- Implement the minimum code needed to make tests pass

- Refactor only after tests are passing

- Follow Beck's "Tidy First" approach by separating structural changes from behavioral changes

- Maintain high code quality throughout development

# TDD METHODOLOGY GUIDANCE

- Start by writing a failing test that defines a small increment of functionality

- Use meaningful test names that describe behavior (e.g., "shouldSumTwoPositiveNumbers")

- Make test failures clear and informative

- Write just enough code to make the test pass - no more

- Once tests pass, consider if refactoring is needed

- Repeat the cycle for new functionality

# TIDY FIRST APPROACH

- Separate all changes into two distinct types:

1. STRUCTURAL CHANGES: Rearranging code without changing behavior (renaming, extracting methods, moving code)

2. BEHAVIORAL CHANGES: Adding or modifying actual functionality

- Never mix structural and behavioral changes in the same commit

- Always make structural changes first when both are needed

- Validate structural changes do not alter behavior by running tests before and after

# COMMIT DISCIPLINE

- Only commit when:

1. ALL tests are passing

2. ALL compiler/linter warnings have been resolved

3. The change represents a single logical unit of work

4. Commit messages clearly state whether the commit contains structural or behavioral changes

- Use small, frequent commits rather than large, infrequent ones

# CODE QUALITY STANDARDS

- Eliminate duplication ruthlessly

- Express intent clearly through naming and structure

- Make dependencies explicit

- Keep methods small and focused on a single responsibility

- Minimize state and side effects

- Use the simplest solution that could possibly work

# REFACTORING GUIDELINES

- Refactor only when tests are passing (in the "Green" phase)

- Use established refactoring patterns with their proper names

- Make one refactoring change at a time

- Run tests after each refactoring step

- Prioritize refactorings that remove duplication or improve clarity

# EXAMPLE WORKFLOW

When approaching a new feature:

1. Write a simple failing test for a small part of the feature

2. Implement the bare minimum to make it pass

3. Run tests to confirm they pass (Green)

4. Make any necessary structural changes (Tidy First), running tests after each change

5. Commit structural changes separately

6. Add another test for the next small increment of functionality

7. Repeat until the feature is complete, committing behavioral changes separately from structural ones

Follow this process precisely, always prioritizing clean, well-tested code over quick implementation.

Always write one test at a time, make it run, then improve structure. Always run all the tests (except long-running tests) each time.

