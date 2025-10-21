import logging
from typing import Optional, Union, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LoggingConfig

__all__ = [
    'apply_logging_config',
    'normalize_level', 
    'init_logging',
    'get_logger'
]


def normalize_level(level: Union[int, str]) -> int:
    """Convert string level to int."""
    if isinstance(level, str):
        return getattr(logging, level.upper())
    return level


def apply_logging_config(config: 'LoggingConfig') -> None:
    """Apply a LoggingConfig to Python's logging system."""
    root = logging.getLogger()
    root.setLevel(normalize_level(config.level))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(config.format, datefmt=config.date_format)

    # Console handler
    if config.console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(normalize_level(config.console_level))
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    # File handler
    if config.file_enabled and config.file_path:
        file_handler = logging.FileHandler(
            config.file_path,
            mode=config.file_mode,
            encoding=config.file_encoding
        )
        file_handler.setLevel(normalize_level(config.file_level))
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Per-logger level overrides
    if config.logger_levels:
        for logger_name, level in config.logger_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(normalize_level(level))

    # Suppress noisy loggers
    if config.suppress_loggers:
        for logger_name in config.suppress_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
            logger.addHandler(logging.NullHandler())


def init_logging(
    level: Union[int, str] = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    log_dir: Optional[str] = None,
    log_filename_prefix: str = "lightmem",
    log_filename_format: str = "{prefix}_{timestamp}.log",
    timestamp_format: str = "%Y%m%d_%H%M%S",
    console_level: Union[int, str] = logging.INFO,
    file_level: Union[int, str] = logging.DEBUG,
    file_mode: str = "a",
    force: bool = True,
    logger_levels: Optional[Dict[str, Union[int, str]]] = None,
    suppress_loggers: Optional[List[str]] = None
) -> 'LoggingConfig':
    """Initialize logging configuration for LightMem (convenience function).
    
    Args:
        level: Root logging level.
        fmt: Log message format string.
        datefmt: Date format for timestamps.
        log_dir: Directory for auto-generated log files. If provided, file logging is enabled.
        log_filename_prefix: Prefix for auto-generated filenames (default: "lightmem").
        log_filename_format: Format string for auto-generated filenames.
        timestamp_format: Timestamp format for auto-generated filenames.
        console_level: Logging level for console output.
        file_level: Logging level for file output.
        file_mode: File open mode ('a' for append, 'w' for overwrite).
        force: If True, remove existing handlers before configuration.
        logger_levels: Optional dict mapping logger names to their specific levels.
        suppress_loggers: Optional list of logger names to suppress.
                         If None, defaults to ["httpcore", "openai", "urllib3", "httpx"].
    
    Returns:
        The created LoggingConfig instance (in case caller wants to modify it later).
    
    """
    from .base import LoggingConfig
    
    config = LoggingConfig(
        level=level,
        format=fmt,
        date_format=datefmt,
        console_enabled=True,
        console_level=console_level,
        file_enabled=log_dir is not None,
        log_dir=log_dir,
        log_filename_prefix=log_filename_prefix,
        log_filename_format=log_filename_format,
        timestamp_format=timestamp_format,
        file_level=file_level,
        file_mode=file_mode,
        file_encoding='utf-8',
        logger_levels=logger_levels,
        suppress_loggers=suppress_loggers,
        force_reconfigure=force
    )
    
    config.apply()
    return config


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for the given name."""
    if name is None:
        name = "lightmem"
    return logging.getLogger(name)