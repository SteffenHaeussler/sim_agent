"""
Simple example usage of the SQL Tools system.
"""

from src.agent.adapters.sql_tools.controller import create_sql_controller


def main():
    """Example usage of the SQL tools system."""

    # Create the controller
    controller = create_sql_controller()

    # Example questions
    questions = [
        "How many customers do we have?",
        "What are the top 5 selling products?",
        "Show me orders from 2024",
        "What is the average order value?",
    ]

    print("=== Simple SQL Tools Demo ===\n")

    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")

        try:
            result = controller.process_question(question)

            print(f"SQL: {result.sql_query}")
            print(f"Valid: {result.validation_passed}")
            print(f"Confidence: {result.confidence:.2f}")
            print()

        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
