import logging

GLOBAL_LOG_LEVEL = logging.DEBUG


class ColoredFormatter(logging.Formatter):
    COLOR_CODES = {
        "DEBUG": "\033[94m",  # blue
        "INFO": "\033[92m",  # green
        "WARNING": "\033[93m",  # yellow
        "ERROR": "\033[91m",  # red
        "CRITICAL": "\033[91m",  # red
        "RESET": "\033[0m",  # reset color
    }

    def format(self, record):
        msg = super().format(record)
        color = self.COLOR_CODES.get(record.levelname, self.COLOR_CODES["RESET"])
        return f'{color}{msg}{self.COLOR_CODES["RESET"]}'


def configure_logger(name):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(GLOBAL_LOG_LEVEL)

        ch = logging.StreamHandler()
        ch.setLevel(GLOBAL_LOG_LEVEL)

        formatter = ColoredFormatter(
            "[%(asctime)s] - file: %(name)s - level: [%(levelname)s] : %(message)s"
        )
        ch.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(ch)

    return logger


def set_log_level(level):
    assert level in ["info", "error", "debug"]
    _level_dict = {"info": logging.INFO, "error": logging.ERROR, "debug": logging.DEBUG}
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = _level_dict[level]
    logging.getLogger().setLevel(GLOBAL_LOG_LEVEL)
