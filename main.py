"""
Main file to run the station
This script starts the main loop of the station, and handles the different menus and measurements
"""
from PhenoHiveStation import PhenoHiveStation
from utils import setup_logger, create_folders
import time
import datetime
import RPi.GPIO as GPIO
import argparse
import configparser
import logging

CONFIG_FILE = "config.ini"
LOGGER = None

def main() -> None:
    """
    Main function, initialise the station and start the main loop
    """
    LOGGER.info("Initializing the station")
    try:
        station = PhenoHiveStation.get_instance()  # Initialize the station
        station.running = int(station.parser['Station']['running'])
    except Exception as e:
        LOGGER.critical(f"Error while initializing the station: {type(e).__name__}: {e}")
        raise e

    n_round = 0
    error_count = 0
    while True:
        try:
            station.disp.show_menu()
            handle_main_menu(station, n_round)
        except Exception as e:
            error_count += 1
            station.register_error(exception=e)
            if error_count > 10:
                # Reached an unhandled error threshold, exiting the program
                LOGGER.critical("Critical: too many exception raised, exiting.")
                raise RuntimeError("Too many exception raised, exiting. Check logs for more details.")
            else:
                time.sleep(5)


def handle_main_menu(station: PhenoHiveStation, n_round: int) -> None:
    """
    Function to handle the button presses in the main menu
    :param station: station object
    :param n_round: number of measurement rounds done
    """
    if not GPIO.input(station.BUT_LEFT):
        station.disp.show_cal_prev_menu()
        time.sleep(1)
        handle_configuration_menu(station)

    if not GPIO.input(station.BUT_RIGHT) or station.running:
        station.parser['Station']['running'] = "1"
        station.running = 1
        with open(CONFIG_FILE, 'w') as configfile:
            station.parser.write(configfile)
        time.sleep(1)
        handle_measurement_loop(station, n_round)


def handle_configuration_menu(station: PhenoHiveStation) -> None:
    """
    Configuration menu
    :param station: station object
    """
    while True:
        if not GPIO.input(station.BUT_RIGHT):
            handle_preview_loop(station)
            time.sleep(1)
            break

        if not GPIO.input(station.BUT_LEFT):
            station.disp.show_calib_menu()
            time.sleep(1)
            handle_calibration_menu(station)
            time.sleep(1)
            break


def handle_preview_loop(station: PhenoHiveStation) -> None:
    """
    Preview loop: takes a preview photo and displays it on the screen to check the camera position
    :param station: station object
    """
    while True:
        GPIO.output(station.LED, GPIO.HIGH)
        path_img = station.save_photo(preview=True, time_to_wait=1)
        station.disp.show_image(path_img)
        if not GPIO.input(station.BUT_RIGHT):
            GPIO.output(station.LED, GPIO.LOW)
            break

def handle_calibration_menu(station: PhenoHiveStation) -> None:
    """
    Calibration menu
    :param station: station object
    """
    while True:
        if not GPIO.input(station.BUT_RIGHT):
            handle_imag_param_calibration_menu(station)
            time.sleep(1)
            break

        if not GPIO.input(station.BUT_LEFT):
            handle_weight_calibration_menu(station)
            time.sleep(1)
            break

def handle_imag_param_calibration_menu(station: PhenoHiveStation) -> None:
    """
    Image parameters calibration loop.
    .......
    :param station: station object
    """
    print("entrée menu calib")
    inc = 1
    sigma = float(station.sigma)
    kernel_size = int(station.kernel_size)
    channel = str(station.channel)
    image_path=str(station.image_path)
    time.sleep(1)
    station.disp.show_img_param_menu(sigma, kernel_size, inc)
    station.best_score = 0.0
    
    while True:
        if not GPIO.input(station.BUT_RIGHT):
            print("bouton droit")
            time.sleep(1)
            break
        if not GPIO.input(station.BUT_LEFT):
            print("bouton gauche")
            if inc > 10:
                break
            elif inc % 2 == 0:
                print("calcul")
                sigma, kernel_size = station.calib_img_param(image_path, channel, sigma, kernel_size, inc)
                station.parser['image_arg']["sigma"] = str(sigma)
                station.parser['image_arg']["kernel_size"] = str(kernel_size)
                with open("config.ini", "w") as configfile:
                    station.parser.write(configfile)
                station.sigma = sigma
                station.kernel_size = kernel_size
                station.disp.show_img_param_menu(sigma, kernel_size, inc-1)
                inc += 1
                time.sleep(1)
            else:
                print("photo")
                GPIO.output(station.LED, GPIO.HIGH)
                image_path = station.save_photo(img_name = 'img_calib')
                GPIO.output(station.LED, GPIO.LOW)
                station.disp.show_photo_taken(inc-1)
                time.sleep(1)
                station.disp.show_img_param_menu(sigma, kernel_size, inc-1)
                inc += 1


def handle_weight_calibration_menu(station: PhenoHiveStation) -> None:
    """
    Weight calibration loop.
    This function takes the tare value and computes the calibration coefficient when the left button is pressed.
    :param station: station object
    """
    station.tare = station.get_weight(20)[0]
    if station.tare == -1.0:
        LOGGER.error("Error: Invalid tare measurement. Aborting calibration.")
        return
    station.parser['cal_coef']["tare"] = str(station.tare)
    with open(CONFIG_FILE, 'w') as configfile:
        station.parser.write(configfile)
    raw_weight = 0
    weight_g = 0
    calib_or_test = 0
    while True:
        station.disp.show_weight_cal_menu(raw_weight, round(weight_g, 2), station.tare, calib_or_test)
        if not GPIO.input(station.BUT_RIGHT):
            break
        if not GPIO.input(station.BUT_LEFT):
            calib_or_test += 1
            raw_weight = station.get_weight()[0]
            if raw_weight == -1.0:
                LOGGER.error("Error: Invalid raw weight measurement. Aborting calibration.")
                return
            try:
                reference_weight = float(station.parser['cal_coef']["calibration_weight"])
                if raw_weight - station.tare == 0:
                    LOGGER.error("Error: raw_weight - tare is zero, cannot compute calibration coefficient.")
                    return
                if calib_or_test < 3:
                    load_cell_cal = reference_weight / (raw_weight - station.tare)
                    # Save the calibration coefficient in the config file
                    station.parser['cal_coef']["load_cell_cal"] = str(load_cell_cal)
                    with open(CONFIG_FILE, 'w') as configfile:
                        station.parser.write(configfile)
                weight_g = (raw_weight - station.tare) * load_cell_cal
                time.sleep(1)
            except (KeyError, ValueError) as e:
                LOGGER.error(f"Error in calibration: {e}")
                return



def handle_status_menu(station: PhenoHiveStation) -> bool:
    """
    Status menu: display the current status of the station
    :param station: station object
    :return: True if the measurement loop should continue, False otherwise
    """
    time.sleep(0.5)
    while True:
        station.disp.show_status()
        if not GPIO.input(station.BUT_RIGHT):
            # Resume
            time.sleep(1)
            return True
        if not GPIO.input(station.BUT_LEFT):
            # Stop
            time.sleep(1)
            return False


def handle_measurement_loop(station: PhenoHiveStation, n_round: int) -> None:
    """
    Measurement loop, displays the measurement menu and handles the measurements cycles
    :param station: station object
    :param n_round: number of measurement rounds done
    """
    LOGGER.debug("Entering measurement loop")
    growth_value = 0.0
    weight = 0.0
    humidity = 0
    light = 0.0
    time_delta = datetime.timedelta(seconds=station.time_interval)
    time_now = datetime.datetime.now()
    time_nxt_measure = time_now + time_delta
    continue_measurements = True
    while continue_measurements:
        time_now = datetime.datetime.now()
        station.disp.show_measuring_menu(round(weight*station.load_cell_cal, 2), round(growth_value, 2),
                                         round(humidity, 2), time_now.strftime("%Y/%m/%d %H:%M:%S"),
                                         time_nxt_measure.strftime("%H:%M:%S"), n_round)

        if time_now >= time_nxt_measure:
            LOGGER.info("Measuring time reached, starting measurement")
            station.disp.show_collecting_data("")
            time_nxt_measure = datetime.datetime.now() + time_delta
            growth_value, weight, humidity, light = station.measurement_pipeline()
            n_round += 1

        if not GPIO.input(station.BUT_RIGHT):
            # Stop the measurements
            station.parser['Station']['running'] = "0"
            station.running = 0
            with open(CONFIG_FILE, 'w') as configfile:
                station.parser.write(configfile)
            time.sleep(1)
            break

        if not GPIO.input(station.BUT_LEFT):
            continue_measurements = handle_status_menu(station)
            if not continue_measurements:
                # Stop the measurements
                station.parser['Station']['running'] = "0"
                station.running = 0
                with open(CONFIG_FILE, 'w') as configfile:
                    station.parser.write(configfile)
                break
            time.sleep(1)
    time.sleep(1)


if __name__ == "__main__":
    # Parse arguments
    arg_parser = argparse.ArgumentParser(description='Définition du niveau de log')
    arg_parser.add_argument('-l', '--logger', type=str, help='Niveau de log (DEBUG, INFO, WARNING, ERROR,'
                                                             'CRITICAL). Défaut = DEBUG', default='DEBUG')
    args = arg_parser.parse_args()

    # Read configuration file and create folders if they do not exist
    config_parser = configparser.ConfigParser()
    config_parser.read(CONFIG_FILE)
    paths = [
        config_parser['Paths']['data_folder'],
        config_parser['Paths']['image_folder'],
        config_parser['Paths']['log_folder']
    ]
    create_folders(paths)

    # Setup logger
    log_level_map = {
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    try:
        LOGGER = setup_logger(name="PhenoHive", level=log_level_map[args.logger],
                              folder_path=config_parser['Paths']['log_folder'])
    except KeyError:
        LOGGER = setup_logger(name="PhenoHive", level=logging.DEBUG,
                              folder_path=config_parser['Paths']['log_folder'])

    main()
