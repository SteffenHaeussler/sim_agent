from abc import ABC


class AbstractDB(ABC):
    def __init__(self):
        pass

    def read(self):
        pass

    def write(self):
        pass
