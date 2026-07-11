import logging
from typing import Optional
from rich.logging import RichHandler


def get_logger(name: str = "auto_ml_env", level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    if level is None:
        level = logging.INFO

    logger.setLevel(level)
    logger.propagate = False

    rich_handler = RichHandler(
        level=level,
        rich_tracebacks=True,
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    rich_handler.setFormatter(formatter)

    logger.addHandler(rich_handler)

    return logger
