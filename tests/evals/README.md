# Evaluation Test Framework Documentation

## Overview

The evaluation test framework provides comprehensive testing capabilities for the agentic AI system with LLM-as-a-Judge evaluation, structured validation, and detailed metrics collection.

## Key Features

### 1. LLM Judge Evaluation
- Semantic evaluation of LLM-generated responses
- Multi-dimensional scoring (accuracy, relevance, completeness, hallucination)
- Configurable thresholds for pass/fail determination
- Test-type specific evaluation criteria

### 2. Test Categories

#### End-to-End (E2E) Tests
- Full system integration testing via API
- Tests complete question-answering flow
- Location: `tests/evals/e2e/`

#### Tool Agent Tests
- Tests agent's ability to use tools correctly
- Validates tool selection and execution
- Location: `tests/evals/tool_agent/`

#### Information Retrieval (IR) Tests
- Tests RAG and document retrieval
- Validates ranking and relevance scoring
- Location: `tests/evals/ir/`

#### Enhancement Tests
- Tests LLM question enhancement
- Validates enhanced questions maintain intent
- Location: `tests/evals/enhance/`

#### Pre-Check Tests
- Tests input validation guardrails
- Ensures inappropriate requests are blocked
- Location: `tests/evals/pre_check/`

#### Post-Check Tests
- Tests output validation guardrails
- Ensures responses meet safety requirements
- Location: `tests/evals/post_check/`

## Test Data Format

### Basic Format
```json
{
    "question": "What is the daily maximum value of PI-P0017 in April 2025?",
    "response": "The daily maximum value of PI-P0017 in April 2025 is approximately 1.59."
}
```

### Enhanced Format with Judge Criteria
```json
{
    "question": "What is the daily maximum value of PI-P0017 in April 2025?",
    "response": "The daily maximum value of PI-P0017 in April 2025 is approximately 1.59.",
    "judge_criteria": {
        "accuracy_threshold": 8,
        "relevance_threshold": 9,
        "completeness_threshold": 8,
        "hallucination_threshold": 9,
        "require_exact_match": false,
        "allow_additional_context": true
    }
}
```

## LLM Judge Scoring Dimensions

### 1. Accuracy (0-10)
- Factual correctness of the response
- Alignment with expected information
- Numerical precision (when applicable)

### 2. Relevance (0-10)
- How well the response addresses the question
- Staying on topic
- Direct answer to what was asked

### 3. Completeness (0-10)
- Whether all parts of the question are addressed
- No critical information missing
- Comprehensive coverage

### 4. Hallucination (0-10)
- 10 = No hallucination
- 0 = Severe hallucination
- Checks for made-up information

### 5. Format Compliance (0-10) [Optional]
- For structured outputs only
- JSON structure correctness
- Required fields presence
- Data type matching

## Running Evaluation Tests

### Run All Evaluations
```bash
make eval
```

### Run Specific Test Types
```bash
make eval_e2e          # End-to-end tests
make eval_tool_agent   # Tool agent tests
make eval_ir          # Information retrieval tests
make eval_enhance     # Enhancement tests
make eval_pre_check   # Pre-check guardrail tests
make eval_post_check  # Post-check guardrail tests
```

### Run Enhanced Tests with Judge
```bash
# Run E2E tests with LLM judge
uv run python -m pytest tests/evals/test_e2e_with_judge.py -v

# Run tool agent tests with judge
uv run python -m pytest tests/evals/test_tool_agent_with_judge.py -v
```

## Adding New Test Cases

### 1. Create Test Data File
Create a JSON file in the appropriate subdirectory:
```json
{
    "question": "Your test question here",
    "response": "Expected response",
    "judge_criteria": {
        "accuracy_threshold": 8,
        "relevance_threshold": 8,
        "completeness_threshold": 7,
        "hallucination_threshold": 9
    }
}
```

### 2. File Naming Convention
- Use descriptive names: `eval_<feature>_<number>.json`
- Examples: `eval_get_data_1.json`, `eval_plot_data_2.json`

### 3. Test Type Guidelines

#### E2E Tests
- Test complete user interactions
- Include realistic questions and responses
- Set high thresholds (8-9) for accuracy and relevance

#### Tool Agent Tests
- Test specific tool functionality
- Include expected data formats
- Enable format_compliance scoring

#### IR Tests
- Test document retrieval accuracy
- Include expected document metadata
- Focus on relevance scoring

## Response Validation

### Using Response Validators
```python
from tests.evals.response_validators import ResponseValidator

# Validate a numeric response
result = ResponseValidator.validate_response(
    response="1.59",
    expected_type="numeric"
)

if result.valid:
    print(f"Parsed value: {result.parsed_data.value}")
else:
    print(f"Validation errors: {result.errors}")
```

### Available Validators
- `NumericResponse`: For numeric values with units
- `DataPointResponse`: For single data points
- `TimeSeriesResponse`: For time series data
- `StatisticalResponse`: For statistical analysis results
- `ComparisonResponse`: For comparison results
- `PlotResponse`: For visualization outputs
- `IRResponse`: For information retrieval results
- `GuardrailResponse`: For guardrail check results

## Metrics and Reporting

### Metrics Collection
The framework automatically collects:
- Test execution time
- Pass/fail status
- LLM judge scores
- Error messages and stack traces

### Generated Reports

#### 1. Test Reports
- `e2e_judge_report.json`: Detailed E2E test results
- `tool_agent_judge_report.json`: Tool agent test results

#### 2. Summary Reports
- `e2e_judge_summary.json`: Aggregated E2E metrics
- `tool_agent_judge_summary.json`: Aggregated tool metrics

#### 3. HTML Reports
Generated in `test_metrics/` directory:
- Visual representation of test results
- Score distributions
- Common failure patterns

### Accessing Metrics
```python
from tests.evals.metrics_collector import MetricsCollector

collector = MetricsCollector()

# Add test results
collector.add_test_result(
    test_id="test_001",
    test_type="e2e",
    passed=True,
    duration=2.5,
    judge_result=judge_result
)

# Generate summary
summary = collector.generate_summary("e2e")

# Save reports
collector.save_metrics()
collector.generate_report()
```

## Best Practices

### 1. Test Data Quality
- Use realistic questions and responses
- Include edge cases and error scenarios
- Vary complexity levels

### 2. Judge Criteria Settings
- Set thresholds based on test importance
- Use stricter criteria for critical functionality
- Allow flexibility for creative responses

### 3. Test Maintenance
- Regularly review failed tests
- Update expected responses as system improves
- Add new tests for new features

### 4. Performance Optimization
- Remove hardcoded sleeps where possible
- Use parallel test execution
- Cache expensive operations

## Troubleshooting

### Common Issues

#### 1. Judge Evaluation Failures
- Check if LLM configuration is correct
- Verify judge criteria thresholds
- Review the reasoning in judge results

#### 2. Test Timeouts
- Increase timeout values for slow operations
- Check for network connectivity issues
- Verify service endpoints are accessible

#### 3. JSON Parsing Errors
- Validate JSON syntax in test files
- Ensure proper escaping of special characters
- Check for trailing commas

### Debug Mode
Run tests with verbose output:
```bash
uv run python -m pytest tests/evals/test_e2e_with_judge.py -v -s
```

## Future Enhancements

1. **Multi-Model Judging**: Use multiple LLMs for consensus
2. **Automated Test Generation**: Generate tests from user queries
3. **Performance Benchmarking**: Track latency trends
4. **A/B Testing Support**: Compare different implementations
5. **Integration with CI/CD**: Automated evaluation on commits
