import logging

def setup_loggers(name, log_file, format, level=logging.INFO,):
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(format)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger