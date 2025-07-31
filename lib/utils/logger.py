import logging
import sys

# Create logger
logger = logging.getLogger('HAQ')
logger.setLevel(logging.DEBUG)

# Remove default handlers to avoid duplicate logs
logger.handlers.clear()

# Create console handler with formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create detailed formatter for console
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

# Prevent propagation to avoid duplicate logging
logger.propagate = False
