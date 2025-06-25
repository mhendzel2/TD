import logging
import sys

def setup_logging(app):
    # Remove all default handlers
    for handler in app.logger.handlers:
        app.logger.removeHandler(handler)

    # Set up a simple console handler
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    app.logger.addHandler(console_handler)

    # Set logging level based on environment (e.g., DEBUG for development)
    if app.config.get("DEBUG", False):
        app.logger.setLevel(logging.DEBUG)
    else:
        app.logger.setLevel(logging.INFO)

    app.logger.info("Logging configured for personal use.")


