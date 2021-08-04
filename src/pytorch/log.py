import logging
import os
import sys

_log = logging.getLogger(__name__)

def setup_logging_config(log_level=logging.INFO, filename=None):
    """
    Setup basic logging config.
    """
    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()

    kwargs = {
        "level": log_level,
        "fmt": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    # Add new FileHandler which logs to a file
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(kwargs["fmt"], kwargs["datefmt"]))

    # Add a StreamHander which logs to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(kwargs["fmt"], kwargs["datefmt"]))

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

def setup_full_logging(
    full_experiment_dir: str,
    log_name: str = "nfd.log",
    log_level=logging.INFO,
):
    """
    Setup logging.
    """
    log_fname = os.path.join(full_experiment_dir, log_name)
    setup_logging_config(filename=log_fname, log_level=log_level)
    _log.debug(f"Writing logs to {log_fname}")
