import os
import logging
import json
from logging import Logger

_loggers = {}

def get_logger(actor_name: str) -> Logger:
    """Returns a logger for the given actor, creating one if necessary."""
    global _loggers
    if actor_name in _loggers:
        return _loggers[actor_name]
    else:
        # Ensure the logs folder exists
        logs_dir = "logs/actor_logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        # Create and configure the logger
        logger = logging.getLogger(actor_name)
        logger.setLevel(logging.INFO)
        # Create a file handler for this actor
        file_handler = logging.FileHandler(os.path.join(logs_dir, f"{actor_name}.log"), mode="w")
        file_handler.setLevel(logging.INFO)
        # Create a formatter that includes a timestamp
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        _loggers[actor_name] = logger
        return logger

def log_actor_action(actor_name: str, action: str, details: dict) -> None:
    """
    Logs an action for a given actor.
    
    :param actor_name: Name (ID) of the actor.
    :param action: The name of the action (e.g., "change_logic", "set_yellow_logic", "extend_logic").
    :param details: A dictionary containing the actor details to log.
    """
    logger = get_logger(actor_name)
    # Combine the action with the details
    log_entry = {"action": action, **details}
    # Log the structured entry as a JSON string for readability
    logger.info(json.dumps(log_entry))