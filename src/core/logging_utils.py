from __future__ import annotations

import logging

from .paths import OUTPUT_DIR

LOGGER_NAME = 'qmt_assistant'


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, '_qmt_configured', False):
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False

    log_dir = OUTPUT_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / 'application.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger._qmt_configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    suffix = name.replace('src.', '')
    return logging.getLogger(f'{LOGGER_NAME}.{suffix}')
