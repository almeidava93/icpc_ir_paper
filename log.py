import logging

def get_logger(name) -> logging.Logger:
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    return logger