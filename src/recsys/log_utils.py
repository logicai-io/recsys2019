import logging
import sys


def get_logger():
    file_handler = logging.FileHandler(filename="recsys.log")
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger("RECSYS")
    return logger
