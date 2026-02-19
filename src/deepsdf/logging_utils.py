import logging
from datetime import datetime
from pathlib import Path


def setup_training_logger(save_dir):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("deepsdf_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_path / f"train_{timestamp}.log"

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file