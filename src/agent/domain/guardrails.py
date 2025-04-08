class Guardrails:
    def __init__(self, parent):
        self.parent = parent

    def check(self, text):
        is_okay = False

        if text:
            is_okay = True

        return is_okay
