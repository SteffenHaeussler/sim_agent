class RAG:
    def __init__(self, parent):
        self.parent = parent
        self.retrieve = None
        self.rerank = None

    def rerank(self, text):
        return text

    def retrieve(self, text):
        return text
