import datetime
import logging


def setup_logger(name: str, level: str) -> logging.Logger:
    """
    Function to set up the logger
    :param name: name of the logger
    :param level: logging level, can be DEBUG, INFO, WARNING, ERROR, CRITICAL
    :return: logger object
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = f"logs/{date}_{name}.log"
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def save_to_csv(data: list, filename: str) -> None:
    """
    Save data to a csv file
    :param data: list of data to be saved
    :param filename: name of the csv file
    """
    with open(filename, "a+") as f:
        for d in data[:-1]:
            f.write(f"{d},")
        f.write(f"{data[-1]}\n")
