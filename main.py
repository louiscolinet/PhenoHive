"""
Main file to run the station
This script starts the main loop of the station, and handles the different menus and measurements
"""
from PhenoStation import PhenoStation
from utils import setup_logger
import time
import datetime
import RPi.GPIO as GPIO
import argparse
import logging

CONFIG_FILE = "config.ini"
LOGGER = None


def main() -> None:
    """
    Main function, initialise the station and start the main loop
    """
    LOGGER.info("Initializing the station")
    try:
        station = PhenoStation.get_instance()  # Initialize the station
        running = int(station.parser['Station']['running'])
    except Exception as e:
        LOGGER.critical(f"Error while initializing the station: {type(e).__name__}: {e}")
        raise e

    n_round = 0
    error_count = 0
    while True:
        try:
            station.disp.show_menu()
            handle_button_presses(station, running, n_round)
        except Exception as e:
            error_count += 1
            station.register_error(exception=e)
            if error_count > 10:
                # Reached an unhandled error threshold, exiting the program
                LOGGER.critical("Critical: too many exception raised, exiting.")
                raise RuntimeError("Too many exception raised, exiting. Check logs for more details.")
            else:
                time.sleep(5)


def handle_button_presses(station: PhenoStation, running: int, n_round: int) -> None:
    """
    Function to handle the button presses in the main menu
    :param station: station object
    :param running: flag to indicate if the station is running (1) or not (0)
    :param n_round: number of measurement rounds done
    """
    if not GPIO.input(station.BUT_LEFT):
        station.disp.show_cal_prev_menu()
        time.sleep(1)
        handle_configuration_menu(station)

    if not GPIO.input(station.BUT_RIGHT) or running:
        station.parser['Station']['running'] = "1"
        with open(CONFIG_FILE, 'w') as configfile:
            station.parser.write(configfile)
        time.sleep(1)
        handle_measurement_loop(station, n_round)


def handle_configuration_menu(station: PhenoStation) -> None:
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
            handle_calibration_loop(station)
            time.sleep(1)
            break


def handle_preview_loop(station: PhenoStation) -> None:
    """
    Preview loop
    :param station: station object
    """
    while True:
        path_img = station.save_photo(preview=True, time_to_wait=1)
        station.disp.show_image(path_img)
        if not GPIO.input(station.BUT_RIGHT):
            break


def handle_calibration_loop(station: PhenoStation) -> None:
    """
    Calibration loop.
    This function takes the tare value and displays the current weight on the screen
    :param station: station object
    """
    station.tare = station.get_weight(20)[0]
    station.parser['cal_coef']["tare"] = str(station.tare)
    with open(CONFIG_FILE, 'w') as configfile:
        station.parser.write(configfile)
    weight = 0
    while True:
        station.disp.show_cal_menu(weight, station.tare)
        if not GPIO.input(station.BUT_LEFT):
            weight = station.get_weight()[0]
        if not GPIO.input(station.BUT_RIGHT):
            break


def handle_status_loop(station: PhenoStation) -> bool:
    """
    Status menu: display the current status of the station
    :param station: station object
    :return: True if the measurement loop should continue, False otherwise
    """
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


def handle_measurement_loop(station: PhenoStation, n_round: int) -> None:
    """
    Measurement loop
    :param station: station object
    :param n_round: number of measurement rounds done
    """
    LOGGER.debug("Entering measurement loop")
    growth_value = 0.0
    weight = 0.0
    time_delta = datetime.timedelta(seconds=station.time_interval)
    time_now = datetime.datetime.now()
    time_nxt_measure = time_now + time_delta
    continue_measurements = True
    while continue_measurements:
        time_now = datetime.datetime.now()
        station.disp.show_measuring_menu(round(weight, 2), round(growth_value, 2),
                                         time_now.strftime("%Y/%m/%d %H:%M:%S"),
                                         time_nxt_measure.strftime("%H:%M:%S"), n_round)

        if time_now >= time_nxt_measure:
            LOGGER.info("Measuring time reached, starting measurement")
            station.disp.show_collecting_data("")
            growth_value, weight = station.measurement_pipeline()
            time_nxt_measure = datetime.datetime.now() + time_delta
            n_round += 1

        if not GPIO.input(station.BUT_RIGHT):
            # Stop the measurements
            station.parser['Station']['running'] = "0"
            with open(CONFIG_FILE, 'w') as configfile:
                station.parser.write(configfile)
            time.sleep(1)
            break

        if not GPIO.input(station.BUT_LEFT):
            continue_measurements = handle_status_loop(station)
            if not continue_measurements:
                # Stop the measurements
                station.parser['Station']['running'] = "0"
                with open(CONFIG_FILE, 'w') as configfile:
                    station.parser.write(configfile)
                break
            time.sleep(1)
    time.sleep(1)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Définition du niveau de log')
    parser.add_argument('-l', '--logger', type=str, help='Niveau de log (DEBUG, INFO, WARNING, ERROR,'
                                                         'CRITICAL). Défaut = DEBUG', default='DEBUG')
    args = parser.parse_args()

    # Setup logger
    log_level_map = {
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    try:
        LOGGER = setup_logger("PhenoStation", level=log_level_map[args.logger])
    except KeyError:
        LOGGER = setup_logger("PhenoStation", level=logging.DEBUG)

    main()
