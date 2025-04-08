from smolagents import Tool


class BaseTool(Tool):
    """
    Basic class for all tools

    Methods:
    Args:
        kwargs: Additional parameters
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
