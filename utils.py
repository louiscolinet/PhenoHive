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
    Create the folders provided in the list if they do not exist.
    
    :param folder_paths: list of strings containing the folder paths
    """
    for folder in folder_paths:
        try:
            # Vérifier si un fichier existe déjà avec le même nom
            if os.path.exists(folder) and not os.path.isdir(folder):
                print(f"Erreur : Un fichier du même nom existe déjà -> {folder}")
                continue

            # Créer le dossier en évitant les erreurs si un autre processus l'a créé entre-temps
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            print(f"Erreur lors de la création du dossier {folder}: {e}")



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
