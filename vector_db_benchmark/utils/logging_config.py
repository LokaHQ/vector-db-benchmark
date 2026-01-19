import sys

from loguru import logger


def setup_logger():
    """Initializes the loguru logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    return logger


log = setup_logger()
