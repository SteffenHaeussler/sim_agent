from src.agent.domain import commands, guardrails, rag


# think, if multiple questions can be aggreagted into a Task class
class Agent:
    def __init__(self, question):
        if not question or not question.question:
            raise ValueError("Question is required to enhance")

        self.question = question.question
        self.q_id = question.q_id
        self.enhancement = None
        self.tool_answer = None
        self.response = None

        self.cls_guard = guardrails.Guardrails(self)
        self.cls_rag = rag.RAG(self)
        self.events = []

    def check(self, question):
        if not question:
            raise ValueError("Question is required to enhance")

        check = self.cls_guard.check(question)
        self.cls_guard.check = check

        return None

    def use_tools(self, answer):
        if self.enhancement is None:
            raise ValueError("Enhancement is required to use tools")

        tool_answer = commands.UseTools(
            question=self.question, answer=answer, q_id=self.q_id
        )
        self.tool_answer = tool_answer

        return None

    def rerank(self, question):
        if not question:
            raise ValueError("Question is required to enhance")

        rerank = self.cls_rag.rerank(question)
        self.cls_rag.rerank = rerank

        return None

    def retrieve(self, question):
        if not question:
            raise ValueError("Question is required to enhance")

        retrieve = self.cls_rag.retrieve(question)
        self.cls_rag.retrieve = retrieve

        return None
