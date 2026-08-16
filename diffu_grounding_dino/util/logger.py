"""Logger that writes to stdout on rank 0 and to a per-rank file everywhere."""

import functools
import logging
import os
import sys
from typing import Optional


@functools.lru_cache(maxsize=None)
def setup_logger(
    output: Optional[str] = None,
    distributed_rank: int = 0,
    name: str = "diffugdino",
    color: bool = True,
) -> logging.Logger:
    """Create (once per argument combination) a configured logger.

    Args:
        output: path to a log file, or a directory in which ``log.txt`` is created.
        distributed_rank: only rank 0 writes to stdout; other ranks get
            ``log.rank<N>.txt`` so their tracebacks are not lost.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    plain = logging.Formatter("[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s", "%Y-%m-%d %H:%M:%S")

    if distributed_rank == 0:
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setLevel(logging.DEBUG)
        stream.setFormatter(_ColorFormatter() if color else plain)
        logger.addHandler(stream)

    if output:
        filename = output if output.endswith((".txt", ".log")) else os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = f"{filename}.rank{distributed_rank}"
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        file_handler = logging.StreamHandler(open(filename, "a", encoding="utf-8"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(plain)
        logger.addHandler(file_handler)

    return logger


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",
        logging.INFO: "\033[0m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self):
        super().__init__("[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s", "%Y-%m-%d %H:%M:%S")

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{super().format(record)}{self.RESET}"
