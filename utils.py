import datetime
import logging
import os


def setup_logger(name: str, level: str | int, folder_path: str) -> logging.Logger:
    """
    Function to set up the logger
    :param name: name of the logger
    :param level: logging level, can be DEBUG (10), INFO (20), WARNING (30), ERROR (40), CRITICAL (50)
    :param folder_path: path to the folder where the logs will be saved
    :return: logger object
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = f"{folder_path}/{date}_{name}.log"
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def create_folders(folder_paths: list[str]) -> None:
    """
    Create the folders provided in the list if they do not exist
    :param folder_paths: list of strings containing the folder paths
    """
    for folder in folder_paths:
        if not os.path.exists(folder):
            os.makedirs(folder)


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

def get_values_from_csv(
    filename: str,
    column_name: str,
    target_date: str = None,
    last_n: int = None,
    first_n: int = None
) -> list[float | str] | float | str | None:
    """
    Extract values from a CSV file by date, or get the last/first n values of a column.

    :param filename: path to the CSV file
    :param column_name: column to extract ('date', 'growth', 'weight', 'weight_g', 'humidity', 'light')
    :param target_date: optional, date to search for
    :param last_n: optional, number of last rows to extract
    :param first_n: optional, number of first rows to extract
    :return:
        - if target_date is set: single value (float or str),
        - if last_n or first_n is set: list of values,
        - else: None
    """
    # Column name to index
    columns = {
        "date": 0,
        "growth": 1,
        "weight": 2,
        "weight_g": 3,
        "humidity": 4,
        "light": 5
    }

    col_index = columns.get(column_name)
    if col_index is None:
        raise ValueError(f"Invalid column name: {column_name}")

    with open(filename, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]
    if not lines:
        return [] if (last_n or first_n) else None
        
    if target_date:
        for line in lines:
            parts = line.strip().split(",")
            if parts and parts[0] == target_date:
                value = parts[col_index]
                return value if column_name == "date" else float(value)

    if last_n:
        selected_lines = lines[-last_n:]
    elif first_n:
        selected_lines = lines[:first_n]
    else:
        return None

    values = []
    for line in selected_lines:
    for line in selected_lines:
        parts = line.strip('\x00').strip().split(",")
        if len(parts) > col_index:
            val = parts[col_index]
            values.append(val if column_name == "date" else float(val))

    return values
