# TODO - Agentic AI Framework Improvements

## Critical Evaluation Summary

This document outlines the priority improvements identified through code analysis of the agentic AI framework.

## 🚨 Immediate Priorities (Critical Issues)

### 1. Database Safety & Reliability
- [ ] Add SQLAlchemy connection pooling to prevent resource exhaustion
- [ ] Fix silent failures in `database.py` that return `None` instead of raising exceptions
- [ ] Implement retry logic with exponential backoff for transient DB failures
- [ ] Add query validation and parameterization checks to prevent SQL injection
- [ ] Add connection health checks and monitoring

### 2. WebSocket/SSE Memory Leaks
- [ ] Fix `connected_clients` dict in `notifications.py` - implement cleanup for disconnected clients
- [ ] Add proper error handling and cleanup for async operations
- [ ] Implement backpressure handling to prevent memory exhaustion
- [ ] Use weak references or TTL for client tracking
- [ ] Add circuit breaker pattern for failing connections

### 3. RouterAdapter Bug
- [ ] Fix `collect_new_events()` method to yield events instead of returning a list
- [ ] Ensure proper iterator contract implementation
- [ ] Add tests for event collection across multiple adapters

## 🔧 High-Impact Refactoring (Week 1-2)

### 1. Extract Duplicate Code
- [ ] Extract `convert_schema` method duplicated in SQLAgentAdapter and ScenarioAdapter
- [ ] Create base adapter class or utility module for shared functionality
- [ ] Refactor common patterns across adapters

### 2. Improve Test Coverage
- [ ] Increase adapter layer coverage from 26% to >80%
- [ ] Add integration tests for database connection failures
- [ ] Add tests for concurrent WebSocket scenarios
- [ ] Implement timeout and retry behavior tests
- [ ] Add chaos engineering tests for failure scenarios
- [ ] Create property-based tests for command validation

### 3. Implement Caching Layer
- [ ] Add Redis/in-memory caching for embedding operations
- [ ] Implement LLM response caching with appropriate TTL
- [ ] Add cache invalidation strategies
- [ ] Monitor cache hit rates

## 🏗️ Architectural Improvements (Week 3-4)

### 1. Error Handling Standardization
- [ ] Create custom exception hierarchy for different failure types
- [ ] Add correlation IDs to all errors for tracing
- [ ] Standardize error propagation patterns
- [ ] Replace generic `except Exception` with specific handlers
- [ ] Implement structured error logging with context

### 2. Adapter Pattern Cleanup
- [ ] Split large adapter classes into smaller, focused components
- [ ] Reduce adapter knowledge of domain internals (Feature Envy)
- [ ] Implement proper adapter isolation
- [ ] Create adapter-specific event queues

### 3. Async/Sync Boundary Management
- [ ] Define clear async/sync boundaries
- [ ] Fix mixed sync/async code in notifications
- [ ] Consider full async migration for I/O operations
- [ ] Implement proper event loop management

## 📊 Observability & Monitoring (Month 2)

### 1. Comprehensive Tracing
- [ ] Add OpenTelemetry spans for all error paths
- [ ] Implement database query tracing
- [ ] Add custom business metrics
- [ ] Create performance dashboards

### 2. Metrics Implementation
- [ ] Query success rate monitoring
- [ ] Latency percentiles (p50, p95, p99)
- [ ] Resource utilization metrics
- [ ] SLI/SLO implementation

## 🔒 Security Enhancements

### 1. SQL Security
- [ ] Implement query whitelisting for allowed operations
- [ ] Add execution time limits
- [ ] Implement row count limits
- [ ] Add query audit logging

### 2. API Security
- [ ] Add rate limiting for endpoints
- [ ] Implement request validation
- [ ] Add authentication/authorization checks where needed

## 📈 Performance Optimizations

### 1. Query Optimization
- [ ] Batch similar embedding requests
- [ ] Implement async HTTP client for external calls
- [ ] Add database query optimization
- [ ] Profile and optimize hot paths

### 2. Resource Management
- [ ] Implement connection pooling for all external services
- [ ] Add resource limits and quotas
- [ ] Optimize memory usage in large result sets

## 📝 Code Quality Improvements

### 1. Reduce Complexity
- [ ] Refactor `create_prompt` method using factory pattern
- [ ] Replace large switch statements with polymorphic dispatch
- [ ] Extract validation logic to separate classes
- [ ] Simplify complex conditional logic

### 2. Documentation
- [ ] Add missing docstrings for public methods
- [ ] Create architecture decision records (ADRs)
- [ ] Document error handling strategies
- [ ] Add API documentation

## 🧪 Testing Strategy Improvements

### 1. Test Coverage Goals
- [ ] Domain layer: Maintain >95% coverage
- [ ] Service layer: Achieve >90% coverage
- [ ] Adapter layer: Increase to >80% coverage
- [ ] Add load/stress testing suite

### 2. Test Quality
- [ ] Ensure single behavior per test
- [ ] Add contract tests for external dependencies
- [ ] Implement test data builders
- [ ] Add mutation testing

## 📅 Implementation Timeline

### Week 1: Critical Fixes
- Fix RouterAdapter bug
- Implement database connection pooling
- Fix WebSocket memory leaks

### Week 2: Safety & Testing
- Add SQL injection prevention
- Implement critical error path tests
- Basic caching implementation

### Week 3: Standardization
- Error handling standardization
- Monitoring implementation
- Code duplication removal

### Month 2: Architecture
- Adapter refactoring
- Performance optimizations
- Comprehensive testing suite

## ✅ Success Metrics

- Zero critical security vulnerabilities
- >80% test coverage across all layers
- <100ms p95 latency for cached operations
- Zero memory leaks in production
- <0.1% error rate for API calls

## 📚 Technical Debt Items

- Consider migrating from Pandas to Polars for better performance
- Evaluate replacing synchronous HTTP clients with httpx
- Consider event sourcing for audit requirements
- Evaluate GraphQL for more flexible API queries
