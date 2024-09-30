import logging


def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up logger.

    Args:
        name (str): The name of the logger.
        log_file (str): The log file to write the logs to.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
