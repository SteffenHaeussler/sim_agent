from src.agent.domain import commands, events


class BaseAgent:
    def __init__(self, question: commands.Question):
        if not question or not question.question:
            raise ValueError("Question is required to enhance")

        self.question = question.question
        self.q_id = question.q_id
        self.enhancement = None
        self.tool_answer = None
        self.response = None

        self.is_answered = False
        self.previous_command = None

        # self.cls_guard = guardrails.Guardrails(self)
        # self.cls_rag = rag.RAG(self)
        self.events = []

    # def check(self, question):
    #     if not question:
    #         raise ValueError("Question is required to enhance")

    #     check = self.cls_guard.check(question)
    #     self.cls_guard.check = check

    #     return None

    def update_final_answer(self, answer):
        if self.tool_answer is None:
            raise ValueError("Tool answer is required to finalize")

        self.is_answered = True

        response = events.Response(
            question=self.question,
            answer=answer,
            q_id=self.q_id,
        )

        self.response = response

        return None

    def update_tool_answer(self, answer):
        # if self.enhancement is None:
        #     raise ValueError("Enhancement is required to use tools")

        tool_answer = commands.UseTools(
            question=self.question, answer=answer, q_id=self.q_id
        )
        self.tool_answer = tool_answer

        return tool_answer

    # def rerank(self, question):
    #     if not question:
    #         raise ValueError("Question is required to enhance")

    #     rerank = self.cls_rag.rerank(question)
    #     self.cls_rag.rerank = rerank

    #     return rerank

    # def retrieve(self, question):
    #     if not question:
    #         raise ValueError("Question is required to enhance")

    #     retrieve = self.cls_rag.retrieve(question)
    #     self.cls_rag.retrieve = retrieve

    #     return retrieve

    def _update_state(self, command: commands.Command, response: str) -> str:
        if self.previous_command is type(command):
            self.is_answered = True
            self.response = events.FailedRequest(
                question=self.question,
                exception=response,
            )

        else:
            self.previous_command = type(command)

        return None

    def update(self, command: commands.Command, response: str) -> str:
        self._update_state(command, response)

        if self.is_answered:
            return None

        # if type(command) is commands.Check:
        #     new_command = self.check(response)
        if type(command) is commands.Question:
            new_command = self.update_tool_answer(response)
        # elif type(command) is commands.Retrieve:
        #     new_command = self.retrieve(response)
        # elif type(command) is commands.Rerank:
        #     new_command = self.rerank(response)
        elif type(command) is commands.UseTools:
            new_command = self.update_final_answer(response)
        else:
            raise NotImplementedError(
                f"Not implemented yet for BaseAgent: {type(command)}"
            )

        return new_command
