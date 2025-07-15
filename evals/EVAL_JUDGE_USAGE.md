# Evaluation Framework with LLM Judge

The evaluation framework has been simplified to integrate LLM Judge functionality directly into the original test files.

## Usage

### Running Tests Without Judge (Default)
```bash
make eval  # Run all evaluation tests without judge
```

### Running Tests With LLM Judge
```bash
USE_LLM_JUDGE=true make eval  # Run all evaluation tests with judge
```

### Running Specific Test Areas
```bash
# Without judge
make eval_e2e
make eval_tool_agent
make eval_ir
make eval_enhance
make eval_pre_check
make eval_post_check

# With judge
USE_LLM_JUDGE=true make eval_e2e
USE_LLM_JUDGE=true make eval_tool_agent
# etc...
```

## How It Works

Each test file now includes:
1. Optional import of judge components based on `USE_LLM_JUDGE` environment variable
2. Conditional judge evaluation alongside standard assertions
3. Separate report generation for judge results
4. Summary statistics in `teardown_class` when judge is enabled

## Benefits

- **Simplicity**: One test file per test area
- **Flexibility**: Tests work with or without judge
- **Non-intrusive**: No changes needed to existing test logic
- **Backward Compatible**: Tests run normally without the flag

## Judge Reports

When `USE_LLM_JUDGE=true`, additional reports are generated:
- `{test_type}_judge_report.json`: Detailed results for each test
- `{test_type}_judge_summary.json`: Aggregate statistics and failed test details
