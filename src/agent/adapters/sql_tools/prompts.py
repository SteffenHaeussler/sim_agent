"""
Simplified prompt templates for SQL generation agents.
"""

GROUNDING_AGENT_PROMPT = """
Map the user's question to database tables and columns.

USER QUESTION: {question}

DATABASE SCHEMA:
{schema_info}

Analyze the question and identify:
1. Which tables are relevant to answer this question
2. Which columns from those tables are needed
3. Assign confidence scores (0.0 to 1.0) based on how certain you are

Focus on finding the most relevant tables and columns for this question.
"""

FILTER_AGENT_PROMPT = """
Extract WHERE clause conditions from the user's question.

USER QUESTION: {question}
AVAILABLE COLUMNS: {mapped_columns}

Look for filtering words like "where", "with", "from", dates, numbers, and text values.
Identify any conditions that would limit or filter the data returned.

If no filters are found, return empty conditions.
"""

JOIN_INFERENCE_AGENT_PROMPT = """
Determine which tables need to be joined and how.

USER QUESTION: {question}
REQUIRED TABLES: {required_tables}
RELATIONSHIPS: {schema_relationships}

Use foreign key relationships to connect the required tables.
If only one table is needed, no joins are required.
Choose appropriate join types (INNER, LEFT, RIGHT) based on the question context.
"""

AGGREGATION_AGENT_PROMPT = """
Detect if this question needs aggregation functions (COUNT, SUM, AVG, etc.).

USER QUESTION: {question}
AVAILABLE COLUMNS: {mapped_columns}

Look for words like "total", "count", "average", "how many", "sum", "by category", etc.
Determine if the question asks for:
- Counting records (COUNT)
- Summing values (SUM)
- Averaging values (AVG)
- Finding maximum/minimum (MAX/MIN)
- Grouping data (GROUP BY)
"""

SQL_CONSTRUCTION_AGENT_PROMPT = """
Build the final SQL query from the analyzed components.

USER QUESTION: {question}

COMPONENTS:
- Tables/Columns: {grounding_results}
- Filters: {filter_results}
- Joins: {join_results}
- Aggregations: {aggregation_results}

Build a complete, syntactically correct SQL query that answers the user's question.
Combine all the components (SELECT, FROM, JOIN, WHERE, GROUP BY, etc.) as needed.
"""

VALIDATION_AGENT_PROMPT = """
Validate the generated SQL query for correctness.

USER QUESTION: {question}
GENERATED SQL: {sql_query}
SCHEMA: {schema_info}

Check for:
- Syntax errors
- Missing tables/columns in schema
- Logical issues
- Whether the query answers the original question

Provide a confidence score (0.0 to 1.0) and any corrected SQL if needed.
"""