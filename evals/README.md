# Simplified Evaluation Framework

This is a minimal evaluation framework with only essential functionality.

## Core Components

### 1. Base Test Class (`base_eval_simple.py`)
- Simple base class for evaluation tests
- Supports LLM judge evaluation or simple comparison
- Writes results to JSON files in `reports/` directory
- No database integration

### 2. LLM Judge (`llm_judge_simple.py`)
- Evaluates responses using LLM
- Scores on 4 dimensions: accuracy, relevance, completeness, hallucination
- Configurable thresholds for pass/fail
- Simple prompt without complex test-type specific logic

### 3. Utilities (`utils.py`)
- `load_yaml_fixtures()` - Load YAML test fixtures (for SQL tests)
- `load_json_fixtures()` - Load JSON test fixtures (for tool agent tests)

## Usage

```python
from evals.base_eval_simple import BaseEvaluationTest
from evals.utils import load_json_fixtures

class TestMyFeature(BaseEvaluationTest):
    RUN_TYPE = "my_feature"
    TEST_TYPE = "my_feature"

    def test_example(self):
        # Run your test
        actual = my_function(question)

        # Evaluate with judge
        self.evaluate_with_judge(
            fixture_name="test_1",
            question="What is 2+2?",
            expected_response="4",
            actual_response=actual,
            test_data={"judge_criteria": {}}
        )
```

## Environment Variables
- `USE_LLM_JUDGE` - Set to "true" to use LLM judge, "false" for simple comparison

## What's Been Removed
- Database integration (EvaluationRepository, ORM models)
- Complex test utilities (metrics collection, response validators)
- Test-type specific judge prompts
- Batch evaluation features
- Complex reporting

## To Add Later (if needed)
- Database persistence
- Advanced metrics and analytics
- Test-specific judge prompts
- Parallel test execution
- HTML reports
