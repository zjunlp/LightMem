from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Union, Dict, List, Literal
import logging
import os
from datetime import datetime

class LoggingConfig(BaseModel):
    """Logging configuration for LightMem."""
    
    level: Union[int, str] = Field(
        default=logging.INFO,
        description="Root logging level"
    )
    
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format string"
    )
    
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for timestamps"
    )
    
    console_enabled: bool = Field(
        default=True,
        description="Whether to enable console logging"
    )
    
    console_level: Union[int, str] = Field(
        default=logging.INFO,
        description="Logging level for console output"
    )
    
    file_enabled: bool = Field(
        default=False,
        description="Whether to enable file logging"
    )
    
    log_dir: Optional[str] = Field(
        default=None,
        description="Directory for log files. Auto-generates timestamped log files when file_enabled=True."
    )
    
    log_filename_prefix: str = Field(
        default="lightmem",
        description="Prefix for auto-generated log filenames"
    )
    
    log_filename_format: str = Field(
        default="{prefix}_{timestamp}.log",
        description="Format for auto-generated log filenames. Available placeholders: {prefix}, {timestamp}"
    )
    
    timestamp_format: str = Field(
        default="%Y%m%d_%H%M%S",
        description="Timestamp format for auto-generated log filenames"
    )
    
    file_level: Union[int, str] = Field(
        default=logging.DEBUG,
        description="Logging level for file output"
    )
    
    file_mode: Literal["a", "w"] = Field(
        default="a",
        description="File open mode"
    )
    
    file_encoding: str = Field(
        default="utf-8",
        description="File encoding"
    )
    
    logger_levels: Optional[Dict[str, Union[int, str]]] = Field(
        default=None,
        description="Per-logger level overrides"
    )
    
    suppress_loggers: Optional[List[str]] = Field(
        default_factory=lambda: ["httpcore", "openai", "urllib3", "httpx"],
        description="List of logger names to suppress (set to WARNING level)"
    )
    
    force_reconfigure: bool = Field(
        default=False,
        description="Remove existing handlers before configuration"
    )
    
    # Internal field: resolved file path (computed from log_dir)
    _resolved_file_path: Optional[str] = None
    
    @field_validator('level', 'console_level', 'file_level')
    @classmethod
    def validate_log_level(cls, v: Union[int, str]) -> Union[int, str]:
        """Validate log level."""
        if isinstance(v, int):
            valid_levels = [logging.NOTSET, logging.DEBUG, logging.INFO, 
                          logging.WARNING, logging.ERROR, logging.CRITICAL]
            if v not in valid_levels:
                raise ValueError(f"Invalid numeric log level: {v}")
            return v
        
        if isinstance(v, str):
            v_upper = v.upper()
            valid_names = ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if v_upper not in valid_names:
                raise ValueError(f"Invalid log level: {v}. Must be one of {valid_names}")
            return v_upper
        
        raise ValueError(f"Log level must be int or str, got {type(v)}")
    
    @model_validator(mode='after')
    def resolve_file_path(self) -> 'LoggingConfig':
        """
        Resolve the actual log file path from log_dir.
        Simple logic: log_dir + auto-generated filename with timestamp.
        """
        if not self.file_enabled:
            return self
        
        if self.log_dir is None:
            raise ValueError("log_dir must be provided when file_enabled=True")
        
        # Create directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime(self.timestamp_format)
        filename = self.log_filename_format.format(
            prefix=self.log_filename_prefix,
            timestamp=timestamp
        )
        
        # Store resolved path
        self._resolved_file_path = os.path.join(self.log_dir, filename)
        
        return self
    
    def apply(self) -> None:
        """Apply this logging configuration to Python's logging system."""
        from .utils import apply_logging_config
        apply_logging_config(self)
    
    @property
    def file_path(self) -> Optional[str]:
        """Get the resolved file path."""
        return self._resolved_file_path
    
    class Config:
        arbitrary_types_allowed = True