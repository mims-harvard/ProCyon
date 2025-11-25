import os
import sys

LOG_LEVEL = os.getenv("PROCYON_LOG_LEVEL")


def add_stdout_to_logger(logger, log_level):
    logger.add(sys.stdout, level=log_level)

    return logger


def construct_logger(log_level: str = "INFO"):
    from loguru import logger

    logger.remove()
    log_level = log_level
    logger = add_stdout_to_logger(logger, log_level)

    return logger


logger = construct_logger(LOG_LEVEL)

logger.info(f"Log level: {LOG_LEVEL}")
