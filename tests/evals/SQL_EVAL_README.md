# SQL Agent Evaluation Framework

This framework provides comprehensive evaluation for the SQL Agent, including end-to-end SQL generation tests and individual stage evaluations.

## Overview

The SQL evaluation framework consists of two main components:

1. **SQL E2E Tests** (`test_sql_e2e.py`): Tests the complete flow from natural language question to final SQL query
2. **SQL Stage Tests** (`test_sql_stages.py`): Tests individual SQL processing stages (grounding, filter, aggregation, join)

## Test Structure

### SQL E2E Test Files
Located in `tests/evals/fixtures/sql_e2e_test*.json`:
- Each file contains a natural language question and expected SQL query
- Tests use exact match comparison by default
- LLM Judge evaluates SQL correctness when enabled

### SQL Stages Test File
Located in `tests/evals/fixtures/sql_stages_example.json`:
- Contains examples for each SQL processing stage
- Each stage has its own evaluation criteria
- Tests the LLM's ability to extract specific SQL components

## Running Tests

### Run all SQL evaluations:
```bash
make eval_sql_e2e    # Run end-to-end SQL tests
make eval_sql_stages # Run SQL stage tests
```

### Run with custom environment:
```bash
USE_LLM_JUDGE=false make eval_sql_e2e  # Run without LLM Judge
```

## Adding New Tests

### For SQL E2E Tests:
1. Create a new JSON file in `tests/evals/fixtures/` with naming pattern `sql_e2e_test*.json`
2. Structure:
```json
{
  "sql_e2e": {
    "question": "Natural language question",
    "expected_sql": "Expected SQL query",
    "judge_criteria": {
      "accuracy_threshold": 8.0,
      "relevance_threshold": 9.0,
      "completeness_threshold": 8.0,
      "hallucination_threshold": 9.0,
      "require_exact_match": true
    }
  }
}
```

### For SQL Stage Tests:
Add new test cases to `sql_stages_example.json` or create new files with the structure:
```json
{
  "sql_stages": {
    "grounding": {
      "question": "Question text",
      "expected_response": {
        "tables": ["table1", "table2"],
        "columns": ["table1.col1", "table2.col2"]
      }
    },
    "filter": {
      "question": "Question text",
      "grounding_tables": ["products"],
      "grounding_columns": ["is_active"],
      "expected_response": {
        "where_conditions": ["condition1"],
        "having_conditions": []
      }
    }
  }
}
```

## Evaluation Metrics

The LLM Judge evaluates SQL queries on:
- **Accuracy**: Correctness of the SQL logic
- **Relevance**: How well the SQL addresses the question
- **Completeness**: Whether all aspects are covered
- **Hallucination**: Absence of made-up tables/columns
- **SQL Efficiency**: Query optimization and best practices

## Output Reports

After running tests, the following reports are generated:
- `sql_e2e_judge_report.json`: Detailed results for each e2e test
- `sql_e2e_judge_summary.json`: Summary statistics for e2e tests
- `sql_[stage]_judge_report.json`: Results for each stage test
- `sql_[stage]_judge_summary.json`: Summary for each stage

## Notes

- The SQL adapter expects synchronous execution (not async)
- Database schema is mocked for testing - no actual database connection required
- Tests include rate limiting delays to avoid API throttling
- Exact SQL match is required by default (whitespace normalized)
