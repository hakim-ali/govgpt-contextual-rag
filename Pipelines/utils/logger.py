import logging

# Configure logging only once here
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for verbose logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# This function returns a logger instance with module name
def get_logger(name: str = __name__):
    return logging.getLogger(name)
