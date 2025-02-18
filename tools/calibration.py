"""
HX711 calibration script
"""
from hx711 import HX711
import statistics
import configparser


def get_weight(hx: HX711, n: int = 20) -> float:
    """
    Get the weight from the load cell (median of n measurements)
    :param hx: HX711 controller
    :param n: number of measurements to take (default: 20)
    :return: the median of the measurements
    """
    measurements = hx.get_raw_data(n)
    if not measurements:
        raise RuntimeError("No data received from the load cell")

    median = statistics.median(measurements)
    std_dev = statistics.stdev(measurements)
    print(f"Weight list: {measurements}\n"
          f"Median: {median}\n"
          f"Std dev: {std_dev}")
    return median


def calibration_mode(hx: HX711) -> float:
    """
    Calibration mode, get the real weight and calculate the calibration coefficient with the measured weight
    :param hx: HX711 controller
    :return: the calibration coefficient
    """
    real_weight = input("Enter the real weight (in g): ")
    print("Weighting")
    measured = get_weight(hx, 20) - tare
    coef = float(real_weight) / measured
    print(f"Calibration coefficient: {coef}")
    return coef


def measuring_mode(hx: HX711, c: float) -> None:
    """
    Measuring mode, convert the measured weight to grams using the calibration coefficient
    :param hx: HX711 controller
    :param c: calibration coefficient
    """
    k = 0
    while k != -1:
        k = input("Enter the number of loops to average the weight, -1 to exit:")
        if k == "-1":
            return
        print("Measuring")
        measured = get_weight(hx, int(k)) - tare
        # Calculate the weight in g
        weight = measured * c
        print(f"Weight: {weight} g")


if __name__ == "__main__":
    """ Main function, load the configuration file and starts the different modes """
    path_to_config = input("Enter the path to the config file (default: '../config.ini'): ")

    if path_to_config == "":
        path_to_config = "../config.ini"

    parser = configparser.ConfigParser()
    parser.read(path_to_config)
    coefficient = float(parser["cal_coef"]["load_cell_cal"])

    # Initialize the HX711
    hx711 = HX711(dout_pin=5, pd_sck_pin=6)
    print("Resetting HX711")
    hx711.reset()
    print("Taring")
    tare = get_weight(hx711)
    print(f"Tare value: {tare}")

    mode = input("Run in calibration mode? (y/n): ")
    if mode == "y":
        # Calibration mode
        print("Calibration mode")
        coefficient = calibration_mode(hx711)
        # Save the calibration coefficient in '../config.ini'
        parser["cal_coef"]["load_cell_cal"] = str(coefficient)
        with open(path_to_config, 'w') as f:
            parser.write(f)
        exit(0)
    else:
        # Normal mode
        print(f"Calibration coefficient: {coefficient}")
        measuring_mode(hx711, coefficient)
