"""SQLAlchemy ORM models for evaluation database."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    ARRAY,
    JSON,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import uuid

Base = declarative_base()


class EvaluationRunORM(Base):
    """ORM model for evaluation runs."""

    __tablename__ = "evaluation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_type = Column(String(50), nullable=False, index=True)
    evaluation_category = Column(String(50))
    stage = Column(String(50))
    git_commit = Column(String(40), index=True)
    branch = Column(String(100))
    started_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    completed_at = Column(DateTime)
    total_tests = Column(Integer, default=0)
    passed_tests = Column(Integer, default=0)
    failed_tests = Column(Integer, default=0)
    model_name = Column(String(100))
    model_temperature = Column(Float)
    prompt_version = Column(String(50))
    fixtures_used = Column(ARRAY(Text))
    meta_data = Column(JSON)  # Renamed from metadata to avoid SQLAlchemy reserved word
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    test_results = relationship(
        "TestResultORM", back_populates="evaluation_run", cascade="all, delete-orphan"
    )
    metrics = relationship(
        "EvaluationMetricORM",
        back_populates="evaluation_run",
        cascade="all, delete-orphan",
    )


class TestResultORM(Base):
    """ORM model for individual test results."""

    __tablename__ = "test_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    test_name = Column(String(200), nullable=False, index=True)
    test_type = Column(String(50))
    question = Column(Text)
    expected_response = Column(Text)
    actual_response = Column(Text)
    passed = Column(Boolean, nullable=False, index=True)
    execution_time_ms = Column(Integer)
    judge_scores = Column(JSON)
    judge_reasoning = Column(Text)
    error_message = Column(Text)
    meta_data = Column(JSON)  # Renamed from metadata to avoid SQLAlchemy reserved word
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    evaluation_run = relationship("EvaluationRunORM", back_populates="test_results")
    tool_agent_result = relationship(
        "ToolAgentResultORM",
        back_populates="test_result",
        uselist=False,
        cascade="all, delete-orphan",
    )
    sql_test_result = relationship(
        "SQLTestResultORM",
        back_populates="test_result",
        uselist=False,
        cascade="all, delete-orphan",
    )


class EvaluationMetricORM(Base):
    """ORM model for aggregated evaluation metrics."""

    __tablename__ = "evaluation_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metric_type = Column(String(50), nullable=False, index=True)
    average_score = Column(Float)
    min_score = Column(Float)
    max_score = Column(Float)
    std_deviation = Column(Float)
    percentile_25 = Column(Float)
    percentile_50 = Column(Float)
    percentile_75 = Column(Float)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    evaluation_run = relationship("EvaluationRunORM", back_populates="metrics")


class ToolAgentResultORM(Base):
    """ORM model for tool agent specific results."""

    __tablename__ = "tool_agent_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_result_id = Column(
        UUID(as_uuid=True),
        ForeignKey("test_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    tools_used = Column(ARRAY(Text))
    tool_outputs = Column(JSON)
    execution_delay_ms = Column(Integer)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    test_result = relationship("TestResultORM", back_populates="tool_agent_result")


class SQLTestResultORM(Base):
    """ORM model for SQL test specific results."""

    __tablename__ = "sql_test_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_result_id = Column(
        UUID(as_uuid=True),
        ForeignKey("test_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    stage = Column(String(50))
    schema_context = Column(JSON)
    sql_query = Column(Text)
    query_plan = Column(Text)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    test_result = relationship("TestResultORM", back_populates="sql_test_result")


def get_evaluation_session(connection_string: str):
    """Create a session factory for the evaluation database."""
    engine = create_engine(connection_string)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    return DBSession
