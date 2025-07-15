"""Response validation models for structured test evaluation."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class NumericResponse(BaseModel):
    """Validator for numeric responses."""

    value: Union[float, int]
    unit: Optional[str] = None
    precision: Optional[int] = Field(
        default=None, description="Expected decimal places"
    )

    @field_validator("value")
    def validate_numeric(cls, v):
        """Ensure value is numeric."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"Expected numeric value, got {type(v)}")
        return v


class DataPointResponse(BaseModel):
    """Validator for data point responses."""

    tag: str = Field(description="Data point identifier")
    value: Union[float, int, str]
    timestamp: Optional[str] = None
    quality: Optional[str] = None


class TimeSeriesResponse(BaseModel):
    """Validator for time series data responses."""

    tag: str
    start_time: str
    end_time: str
    data_points: List[DataPointResponse]
    aggregation: Optional[str] = None


class StatisticalResponse(BaseModel):
    """Validator for statistical analysis responses."""

    mean: float
    standard_deviation: float = Field(alias="std_dev")
    minimum: float = Field(alias="min")
    maximum: float = Field(alias="max")
    percentile_25: float = Field(alias="p25")
    percentile_50: float = Field(alias="p50", description="Median")
    percentile_75: float = Field(alias="p75")
    count: Optional[int] = None


class ComparisonResponse(BaseModel):
    """Validator for comparison responses."""

    items: Dict[str, Any] = Field(description="Items being compared")
    comparison_type: str = Field(description="Type of comparison performed")
    results: Dict[str, Any] = Field(description="Comparison results")
    summary: Optional[str] = None


class PlotResponse(BaseModel):
    """Validator for plot/visualization responses."""

    plot_type: str = Field(description="Type of plot generated")
    data: Optional[str] = Field(default=None, description="Base64 encoded plot data")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Plot metadata"
    )

    @field_validator("data")
    def validate_base64(cls, v):
        """Validate base64 encoding if present."""
        if v is not None:
            import base64

            try:
                base64.b64decode(v)
            except Exception:
                raise ValueError("Invalid base64 encoding")
        return v


class IRResponse(BaseModel):
    """Validator for information retrieval responses."""

    id: str
    tag: str
    name: str
    text: str = Field(description="Retrieved text/description")
    score: float = Field(ge=-100, le=100, description="Relevance score")
    area: Optional[str] = None
    location: Optional[str] = None


class GuardrailResponse(BaseModel):
    """Validator for guardrail check responses."""

    approved: bool
    reason: Optional[str] = None
    risk_level: Optional[str] = Field(
        default=None, pattern="^(low|medium|high|critical)$"
    )


class ValidationResult(BaseModel):
    """Result of response validation."""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    parsed_data: Optional[Any] = None


class ResponseValidator:
    """Utility class for validating responses against expected formats."""

    @staticmethod
    def validate_response(
        response: Any, expected_type: str, strict: bool = False
    ) -> ValidationResult:
        """
        Validate a response against expected type.

        Args:
            response: The response to validate
            expected_type: Expected response type
            strict: Whether to enforce strict validation

        Returns:
            ValidationResult with validation status and details
        """

        validators = {
            "numeric": NumericResponse,
            "data_point": DataPointResponse,
            "time_series": TimeSeriesResponse,
            "statistical": StatisticalResponse,
            "comparison": ComparisonResponse,
            "plot": PlotResponse,
            "ir": IRResponse,
            "guardrail": GuardrailResponse,
        }

        result = ValidationResult(valid=True)

        # Get appropriate validator
        validator_class = validators.get(expected_type)
        if not validator_class:
            result.valid = False
            result.errors.append(f"Unknown response type: {expected_type}")
            return result

        try:
            # Parse response
            if isinstance(response, str):
                import json

                try:
                    response_data = json.loads(response)
                except json.JSONDecodeError:
                    # For simple string responses, wrap in appropriate structure
                    if expected_type == "numeric":
                        try:
                            value = float(response)
                            response_data = {"value": value}
                        except ValueError:
                            result.valid = False
                            result.errors.append(
                                f"Cannot parse '{response}' as numeric"
                            )
                            return result
                    else:
                        response_data = response
            else:
                response_data = response

            # Validate with model
            parsed = validator_class.model_validate(response_data)
            result.parsed_data = parsed

        except Exception as e:
            result.valid = False
            result.errors.append(f"Validation failed: {str(e)}")

            # Add warnings for non-strict mode
            if not strict:
                result.warnings.append(
                    "Consider updating response format to match schema"
                )

        return result

    @staticmethod
    def compare_responses(
        actual: Any, expected: Any, response_type: str, tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Compare actual vs expected responses with type-specific logic.

        Args:
            actual: Actual response
            expected: Expected response
            response_type: Type of response for comparison logic
            tolerance: Numeric tolerance for float comparisons

        Returns:
            Comparison results with match status and differences
        """

        comparison = {"matches": True, "differences": [], "similarity_score": 1.0}

        # Type-specific comparison logic
        if response_type == "numeric":
            try:
                actual_val = float(actual) if isinstance(actual, str) else actual
                expected_val = (
                    float(expected) if isinstance(expected, str) else expected
                )

                if abs(actual_val - expected_val) > tolerance:
                    comparison["matches"] = False
                    comparison["differences"].append(
                        {
                            "field": "value",
                            "actual": actual_val,
                            "expected": expected_val,
                            "difference": abs(actual_val - expected_val),
                        }
                    )
                    comparison["similarity_score"] = 1 - min(
                        abs(actual_val - expected_val) / expected_val, 1
                    )

            except (ValueError, TypeError) as e:
                comparison["matches"] = False
                comparison["differences"].append({"error": str(e)})
                comparison["similarity_score"] = 0

        # Add more type-specific comparisons as needed

        return comparison
