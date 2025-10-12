class ConfigurationError(BaseException):
    """Base class for configuration errors.

    Subclasses BaseException to avoid being caught by the default
    exception handler in the ToolNode.
    """
