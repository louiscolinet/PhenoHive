
import base64
import configparser
import os
import numpy as np
from plantcv import plantcv as pcv
import cv2
import csv
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import statistics
import time
import threading
import Adafruit_GPIO.SPI as SPI
import ST7735 as TFT
import hx711
import Adafruit_MCP3008
import RPi.GPIO as GPIO
import logging
import shutil
import multiprocessing as mp
from datetime import datetime, timezone, timedelta
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from picamera2 import Picamera2, Preview
from image_processing import get_total_length, get_segment_list
from utils import setup_logger, create_folders, save_to_csv, get_values_from_csv
from show_display import Display

CONFIG_FILE = "config.ini"
LOGGER = logging.getLogger("PhenoHiveStation")
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DATE_FORMAT_FILE = "%Y-%m-%dT%H-%M-%SZ"  # Date format for file names (no ':', which is not illegal in Windows)


class PhenoHiveStation:
    """
    PhenoHiveStation class, contains all the variables and functions of the station.
    It functions as a singleton. Use PhenoHiveStation.get_instance() method to get an instance.
    """
    # Instance class variable for singleton
    __instance = None

    # Station constants
    WIDTH = -1
    HEIGHT = -1
    SPEED_HZ = -1
    DC = -1
    RST = -1
    SPI_PORT = -1
    SPI_DEVICE = -1
    LED = -1
    BUT_LEFT = -1
    BUT_RIGHT = -1
    HUM = -1
    EN_HUM = -1

    # Station variables
    parser = None
    token = ""
    org = ""
    bucket = ""
    url = ""
    station_id = ""
    image_path = ""
    csv_path = ""
    pot_limit = -1
    channel = ""
    kernel_size = -1
    fill_size = -1
    time_interval = -1
    load_cell_cal = -1.0
    tare = -1.0
    status = -1
    last_error = ("", "")
    running = 0
    best_score = -np.inf
    last_data_send_time = None

    @staticmethod
    def get_instance() -> 'PhenoHiveStation':
        """
        Static access method to create a new instance of the station if not already initialised.
        Otherwise, return the current instance.
        :return: A PhenoHiveStation instance
        """
        if PhenoHiveStation.__instance is None:
            PhenoHiveStation()
        return PhenoHiveStation.__instance

    def __init__(self) -> None:
        """
        Initialize the station
        :raises RuntimeError: If trying to instantiate a new PhenoHiveStation if one was already instantiated
                                (use get_instance() instead)
        """
        if PhenoHiveStation.__instance is not None:
            raise RuntimeError("PhenoHiveStation class is a singleton. Use PhenoHiveStation.get_instance() to "
                               "initiate it.")
        else:
            PhenoHiveStation.__instance = self

        # Parser initialisation
        self.parser = configparser.ConfigParser()

        # Parse Config.ini file
        self.parse_config_file(CONFIG_FILE)
        self.status = 0  # 0: idle, 1: measuring, -1: error
        LOGGER = setup_logger(name="PhenoHiveStation", level=logging.DEBUG,
                              folder_path=self.parser['Paths']['log_folder'])

        threads = [
            threading.Thread(target=self.init_display),
            threading.Thread(target=self.init_influxdb),
            threading.Thread(target=self.init_camera_button),
            threading.Thread(target=self.init_load),
            threading.Thread(target=self.init_button),
            threading.Thread(target=self.init_data),
            threading.Thread(target=self.init_adc)
        ]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def parse_config_file(self, path: str) -> None:
        """
        Parse the config file at the given path and initialise the station's variables with the values
        :param path: the path to the config file
        :raises RuntimeError: If the config file could not be parsed
        """
        if self.parser is None:
            self.parser = configparser.ConfigParser()
            
        try:
            self.parser.read(path)
        except configparser.ParsingError as e:
            LOGGER.error(f"Failed to parse config file: {type(e).__name__}: {e}")
            raise RuntimeError(f"Failed to parse config file {e}")

        self.token = str(self.parser["InfluxDB"]["token"])
        self.org = str(self.parser["InfluxDB"]["org"])
        self.bucket = str(self.parser["InfluxDB"]["bucket"])
        self.url = str(self.parser["InfluxDB"]["url"])
        self.station_id = str(self.parser["Station"]["ID"])
        self.image_path = str(self.parser["Paths"]["image_folder"])
        self.csv_path = str(self.parser["Paths"]["csv_path"])
        self.pot_limit = int(self.parser["image_arg"]["pot_limit"])
        self.channel = str(self.parser["image_arg"]["channel"])
        self.kernel_size = int(self.parser["image_arg"]["kernel_size"])
        self.sigma = float(self.parser["image_arg"]["sigma"])
        self.fill_size = int(self.parser["image_arg"]["fill_size"])
        self.time_interval = int(self.parser["time_interval"]["time_interval"])
        self.WIDTH = int(self.parser["Display"]["width"])
        self.HEIGHT = int(self.parser["Display"]["height"])
        self.SPEED_HZ = int(self.parser["Display"]["speed_hz"])
        self.DC = int(self.parser["Display"]["dc"])
        self.RST = int(self.parser["Display"]["rst"])
        self.SPI_PORT = int(self.parser["Display"]["spi_port"])
        self.SPI_DISPLAY = int(self.parser["Display"]["spi_display"])
        self.SPI_ADC = int(self.parser["ADC"]["spi_adc"])
        self.load_cell_cal = float(self.parser["cal_coef"]["load_cell_cal"])
        self.tare = float(self.parser["cal_coef"]["tare"])
        self.LED = int(self.parser["Camera"]["led"])
        self.BUT_LEFT = int(self.parser["Buttons"]["left"])
        self.BUT_RIGHT = int(self.parser["Buttons"]["right"])
        self.HUM = int(self.parser["Humidity"]["humidity_channel"])
        self.EN_HUM = int(self.parser["Humidity"]["humidity_enable"])
        self.LIGHT = int(self.parser["Light"]["light_channel"])
        self.best_score = float(self.parser["image_arg"]["best_score"])

    def init_display(self):
        # Screen initialisation
        LOGGER.debug("Initialising screen")
        self.st7735 = TFT.ST7735(
            self.DC,
            rst=self.RST,
            spi=SPI.SpiDev(
                self.SPI_PORT,
                self.SPI_DISPLAY,
                max_speed_hz=self.SPEED_HZ
            )
        )
        self.disp = Display(self)
        self.disp.show_image("assets/logo_elia.jpg")

    def init_influxdb(self):
        # InfluxDB client initialization
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.connected = self.client.ping()
        self.last_connection = datetime.now().strftime(DATE_FORMAT)
        LOGGER.debug(f"InfluxDB client initialised with url : {self.url}, org : {self.org} and token : {self.token}" +
                     f", Ping returned : {self.connected}")

    def init_camera_button(self):
        # Camera and LED init
        self.cam = Picamera2()
        GPIO.setwarnings(False)
        GPIO.setup(self.LED, GPIO.OUT)
        GPIO.output(self.LED, GPIO.LOW)
        time.sleep(1)
        GPIO.output(self.LED, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(self.LED, GPIO.LOW)
        time.sleep(1)
        GPIO.output(self.LED, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(self.LED, GPIO.LOW)
        
    def init_load(self):
        # Hx711
        self.hx = DebugHx711(dout_pin=5, pd_sck_pin=6)
        try:
            LOGGER.debug("Resetting HX711")
            self.hx.reset()
        except hx711.GenericHX711Exception as e:
            self.register_error(type(e)(f"Error while resetting HX711 : {e}"))
        else:
            LOGGER.debug("HX711 reset")
        
    def init_adc(self):
        # MCP3008
        
        self.mcp = Adafruit_MCP3008.MCP3008(
            spi=SPI.SpiDev(
                self.SPI_PORT,  # Utilization of the same port than screen
                self.SPI_ADC,  
                max_speed_hz=self.SPEED_HZ # Utilization of the same frequency than screen
            )
        )  

    def init_button(self):
        GPIO.setup(self.BUT_LEFT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.BUT_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.EN_HUM, GPIO.OUT)
        GPIO.output(self.EN_HUM, GPIO.LOW)

    def init_data(self):
        self.data = {
            "ngrok-skip-browser-warning": "true",
            "status": self.status,  # current status
            "error_time": self.last_error[0],  # last registered error
            "error_message": str(self.last_error[1]),  # last registered error
            "growth": -1.0,  # plant's growth
            "weight": -1.0,  # plant's (measured) weight
            "weight_g": -1.0,  # plant's (measured) weight in grams (if calibrated)
            "standard_deviation": -1.0,  # measured weight standard deviation
            "picture": "",  # last picture as a base-64 string
            "humidity": -1.0, # soil humidity
            "light": -1.0
        }
        self.to_save = ["growth", "weight", "weight_g", "standard_deviation", "humidity", "light"]

    def calib_img_param(self, image_path: str, channel: str = 'k', sigma: float = 2, kernel: int = 20, calib_test_num: int = 1):
        """
        Automatically calibrate sigma and kernel_size for optimal segmentation, using multithreading.
        """
        best_params = None
        best_score = self.best_score
        spread = sigma / calib_test_num
        sigma_values = np.linspace(sigma - spread, sigma + spread, num=10)
        sigma_values = np.clip(sigma_values, 0.05, 5)
        sigma_values = np.unique(sigma_values)
        kernel_values = np.arange(kernel - 5, kernel + 5, step=1, dtype=int)
        kernel_values = np.clip(kernel_values, 10, None)
        kernel_values = np.unique(kernel_values)
        param_grid = list(product(sigma_values, kernel_values))
    
        print(f"Best score: {best_score}")
        print(f"Sigma: {sigma_values}, Kernel: {kernel_values}")
    
        # Préparer 4 copies de skeleton_ref.jpg
        for i in range(4):
            shutil.copyfile(self.image_path + "skeleton_ref.jpg", self.image_path + f"skeleton_ref_{i}.jpg")
            shutil.copyfile(self.image_path + "img_calib.jpg", self.image_path + f"img_calib_{i}.jpg")
    
        def worker(index, sig, ker):
            # Nom unique de squelette généré pour chaque thread
            skeleton_filename = f"skeleton_{index}.jpg"
            skeleton_path = os.path.join(self.image_path, skeleton_filename)
            ref_path = os.path.join(self.image_path, f"skeleton_ref_{index}.jpg")
            image_calib_path = os.path.join(self.image_path, f"img_calib_{index}.jpg")
    
            try:
                path_lengths = get_segment_list(image_calib_path, channel, ker, sig, skeleton_filename)
            except Exception as e:
                print(f"[Thread {index}] Erreur: get_segment_list a échoué ({e})")
                return (None, sig, ker)
    
            if path_lengths is None:
                return (None, sig, ker)
    
            try:
                dsc, _ = self.evaluate_skeleton(skeleton_path, ref_path)
            except Exception as e:
                print(f"[Thread {index}] Erreur: evaluate_skeleton a échoué ({e})")
                return (None, sig, ker)
    
            return (dsc, sig, ker)
    
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, (sig, ker) in enumerate(param_grid):
                thread_id = i % 4
                futures.append(executor.submit(worker, thread_id, sig, ker))
    
            for future in futures:
                result = future.result()
                if result is None:
                    continue
                score, sig, ker = result
                if score is not None and score > best_score:
                    best_score = score
                    best_params = (sig, ker)
                    print(f"Nouvelle meilleure combinaison: sigma={sig}, kernel={ker}, score={score}")
    
        if best_params is None:
            return (sigma, kernel)
        if best_params[0] > 4: best_params = (4, best_params[1])
        if best_params[1] < 12: best_params= (best_params[0], 12)   
    
        self.best_score = best_score
        self.parser["image_arg"]["best_score"] = str(best_score)
        return best_params

    def evaluate_skeleton(self, generated_skeleton_path: str, reference_skeleton_path: str) -> dict:
        """
        Compare un squelette généré avec un squelette de référence.
        
        :param generated_skeleton_path: Chemin de l'image du squelette généré
        :param reference_skeleton_path: Chemin de l'image du squelette de référence
        :return: Dictionnaire contenant le nombre de branches, intersections et DSC
        """
        # Charger les images en niveaux de gris
        gen_skel = cv2.imread(generated_skeleton_path)
        ref_skel = cv2.imread(reference_skeleton_path)

        # Crop image edges
        height, width = ref_skel.shape[0], ref_skel.shape[1]
        ref_skel = pcv.crop(ref_skel, 5, 5, height - 10, width - 10)
        
        # Convertir en format binaire (0 et 1)
        gen_skel_bin = gen_skel // 255
        ref_skel_bin = ref_skel // 255
        
        # Calcul du Dice Similarity Coefficient (DSC)
        intersection = np.sum(gen_skel_bin * ref_skel_bin)
        dsc = (2.0 * intersection) / (np.sum(gen_skel_bin) + np.sum(ref_skel_bin))
        
        # Comptage des branches avec l'opération de squelette
        """skeleton = pcv.morphology.skeletonize(mask=gen_skel_bin)
        num_branches = np.sum(skeleton)
        segmented_img, obj = pcv.morphology.segment_skeleton(skel_img=skeleton)
        _ = pcv.morphology.segment_path_length(segmented_img=segmented_img, objects=obj, label="default")"""

        num_branches = 4
        
        return (dsc, num_branches)

    def register_error(self, exception: Exception) -> None:
        """
        Register an exception by logging it, updating the station's status and sending it to the DB
        :param exception: The exception that occurred
        """
        LOGGER.error(f"{type(exception).__name__}: {exception}")
        timestamp = datetime.now().strftime(DATE_FORMAT)
        self.status = -1
        self.last_error = (timestamp, exception)
        self.data["status"] = self.status
        self.data["error_time"] = self.last_error[0]
        self.data["error_message"] = str(self.last_error[1])

    def send_to_db(self) -> bool:
        """
        Saves the measurements to the csv file, then sends it to InfluxDB (if connected)
        Uses `PhenoHiveStation.measurements` dictionary containing the measurements and their values.
        :return True if the data was sent to the DB, False otherwise
        """

        # Ping the DB
        self.connected = self.client.ping()
        now = datetime.now()
        timestamp = now.strftime(DATE_FORMAT)
    
        # Create CSV if it doesn't exist
        if not os.path.exists(self.csv_path):
            save_to_csv(["time"] + self.to_save, self.csv_path)
    
        # Saves the current measurement
        current_row = [timestamp] + [self.data[key] for key in self.to_save]
        save_to_csv(current_row, self.csv_path)

        if self.last_data_send_time == None:
            self.last_data_send_time = timestamp
    
        if not self.connected:
            return False

        # Sends the current data
        points = []
        for field, value in self.data.items():
            p = Point(f"station_{self.station_id}").field(field, value)
            points.append(p)
        self.write_api.write(bucket=self.bucket, org=self.org, record=points)

        if self.last_data_send_time == timestamp:
            LOGGER.debug(f"Sending data to the DB: {str(points)}")
            return True

        # Read the file and find the lines not yet sent
        with open(self.csv_path, "r", newline="") as f:
            reader = list(csv.reader(f))
            rows = reader[1:]

        found_last_sent = False
        
        for row in list(reversed(rows))[1:]: 
            row_time = row[0].replace('\x00', '').strip()
            if not found_last_sent:
                if row_time == self.last_data_send_time:
                    found_last_sent = True
                    break
                    
            # Adapting time to that of the database
            dt_input = datetime.strptime(row_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

            local_now = datetime.now(timezone.utc).astimezone()   # heure locale
            utc_now = datetime.now().replace(tzinfo=timezone.utc) # heure UTC
            time_offset = local_now.utcoffset() or timedelta(0)
            
            dt_corrected = dt_input - time_offset
            timestamp_ns = int(dt_corrected.timestamp())

            # Sending
            pts = []
            for i, field in enumerate(self.to_save):
                p = Point(f"station_{self.station_id}").field(field, float(row[i+1])).time(timestamp_ns, write_precision='s')
                pts.append(p)
            self.write_api.write(bucket=self.bucket, org=self.org, record=pts)
            LOGGER.debug(f"Sending {len(pts)} points to the DB")
            
        self.last_data_send_time = timestamp
        return True


    def get_weight(self, n: int = 15) -> tuple[float, float]:
        """
        Get the weight from the load cell (median of n measurements)
        :param n: the number of measurements to take (default = 15)
        :return: The median of the measurements (-1 in case of error) and the observed standard deviation
        """
        measurements = self.hx.get_raw_data(times=n)
        if not measurements:
            self.register_error(RuntimeError("Error while getting raw data (no data), check load cell connection"))
            return -1.0, -1.0
        return statistics.median(measurements), statistics.stdev(measurements)

    def capture_and_display(self) -> tuple[str, str]:
        """
        Take a photo, display it on the screen and return it in base64
        :return: a tuple with the photo in base64 and the path to the photo
        """
        # Take the photo
        try:
            GPIO.output(self.LED, GPIO.HIGH)
            time.sleep(1)
            path_img = self.save_photo(preview=False, time_to_wait=7)
            time.sleep(1)
            GPIO.output(self.LED, GPIO.LOW)
        except Exception as e:
            LOGGER.error(f"Error capturing photo: {e}")
            self.register_error(e)
            return "", ""
        
        if path_img != "":
            LOGGER.debug(f"Photo taken and saved at {path_img}")
            self.disp.show_image(path_img)
            # Convert image to base64
            with open(path_img, "rb") as image_file:
                pic = base64.b64encode(image_file.read()).decode('utf-8')
            time.sleep(2)
            return pic, path_img
        else:
            return "", ""

    def save_photo(self, preview: bool = False, time_to_wait: int = 8, img_name: str = None) -> str:
        """
        Take a photo and save it
        :param preview: if True the photo will be saved as "img.jpg" (used for the display)
        :param time_to_wait: time to wait before taking the photo (in seconds)
        :return: the path to the photo
        """        
        if img_name != None:
            name = img_name
        elif not preview:
            name = datetime.now().strftime(DATE_FORMAT_FILE)
        else:
            name = "preview"
        path_img = self.image_path + "/%s.jpg" % name

        try:
            self.cam.stop_preview()
        except Exception:
            pass
        try:
            self.cam.stop()
        except Exception:
            pass
        try:
            self.cam.close()
        except Exception:
            pass
            
        try:
            LOGGER.debug("[save_photo] Capturing file...")
            result = self.capture_with_timeout(path_img, time_to_wait=time_to_wait, timeout=16)
            if result != "":
                LOGGER.debug("Capturing done")
            else:
                self.register_error("Error while capturing the photo")
                path_img = ""
        except Exception as e:
            self.register_error(type(e)(f"Error while capturing the photo: {e}"))
            path_img = ""
            
        try:
            self.cam = Picamera2()
        except Exception as e:
            LOGGER.warning(f"Failed to re init camera: {e}")

        return path_img

    def capture_with_timeout(self, path_img, time_to_wait=8, timeout=16):
        """
        Try to capture an image with timeout and process isolation.
        """
        # Picamera2 is not serializable, so we avoid passing the live instance
        # You could alternatively recreate a new camera instance inside the worker
        manager = mp.Manager()
        return_dict = manager.dict()

        def capture_worker(path_img, time_to_wait, return_dict):
            cam = Picamera2()
            cam.start_preview(Preview.NULL)
            cam.start()
            time.sleep(time_to_wait)  # laisser un petit délai pour initialiser
            try:
                cam.capture_file(path_img)
                return_dict["success"] = True
            except Exception as e:
                return_dict["error"] = str(e)
            cam.stop_preview()
            cam.stop()
            cam.close()

    
        p = mp.Process(target=capture_worker, args=(path_img, time_to_wait, return_dict))
        p.start()
        p.join(timeout+time_to_wait)
    
        if p.is_alive():
            LOGGER.error("Capture timed out. Terminating process.")
            p.terminate()
            p.join()
            return ""
    
        if return_dict.get("success"):
            return path_img
        else:
            LOGGER.error(f"Capture failed: {return_dict.get('error')}")
            return ""

    def measurement_pipeline(self) -> tuple[int, float, int]:
        """
        Measurement pipeline
        :return: a tuple with the growth value and the weight
        """
        LOGGER.info("Starting measurement pipeline")
        self.status = 1
        self.disp.show_collecting_data("Starting measurement pipeline")
        time.sleep(1)

        # Take and process the photo
        try:
            self.disp.show_collecting_data("Taking photo")
            pic, growth_value = self.picture_pipeline()
            self.data["picture"] = pic
            self.data["growth"] = float(growth_value)
        except Exception as e:
            self.register_error(type(e)(f"Error while taking/processing the photo: {e}"))
            self.disp.show_collecting_data("Error while taking/processing the photo")
            time.sleep(5)
            growth_value = 0.0

        # Get weight
        try:
            weight, std_dev = self.weight_pipeline()
            self.data["weight"] = weight
            self.load_cell_cal = float(self.parser["cal_coef"]["load_cell_cal"])
            self.data["weight_g"] = weight * self.load_cell_cal
            self.data["standard_deviation"] = std_dev

            # Measurement finished, display the weight
            self.disp.show_collecting_data(f"Weight : {round(weight * self.load_cell_cal, 2)}")
            time.sleep(2)
        except Exception as e:
            self.register_error(type(e)(f"Error while getting the weight: {e}"))
            self.disp.show_collecting_data("Error while getting the weight")
            time.sleep(5)
            weight = 0

        # Get humidity
        try:
            humidity = self.humidity_pipeline()
            self.data["humidity"] = humidity

         # Measurement finished, display the humidity
            self.disp.show_collecting_data(f"Humidity : {humidity}")
            time.sleep(2)
            
        except Exception as e:
            self.register_error(type(e)(f"Error while getting the humidity: {e}"))
            self.disp.show_collecting_data("Error while getting the humidity")
            time.sleep(5)
            humidity = 0

        # Get light
        try:
            light = self.light_pipeline()
            self.data["light"] = light

         # Measurement finished, display the humidity
            self.disp.show_collecting_data(f"Light : {light}")
            time.sleep(2)
            
        except Exception as e:
            self.register_error(type(e)(f"Error while getting the light: {e}"))
            self.disp.show_collecting_data("Error while getting the light")
            time.sleep(5)
            light = 0

        # Send data to the DB
        try:
            self.disp.show_collecting_data("Sending data to the DB")
            if self.send_to_db():
                LOGGER.debug("Data sent to the DB")
                self.disp.show_collecting_data("Data sent to the DB")
            else:
                # Data could not be sent to the database but the measurements were still saved to the csv file
                LOGGER.warning("Could not send data to the DB, no connection")
                self.disp.show_collecting_data("Could not send data to the DB, no connection")
            time.sleep(2)
        except Exception as e:
            self.register_error(type(e)(f"Error while sending data to the DB: {e}"))
            self.disp.show_collecting_data("Error while sending data to the DB")
            time.sleep(5)
            return growth_value, weight, humidity, light

        if self.status == 1:
            self.status = 0
            self.data["status"] = self.status
            
        LOGGER.info("Measurement pipeline finished")
        self.disp.show_collecting_data("Measurement pipeline finished")
        time.sleep(1)
        return growth_value, weight, humidity, light

    def picture_pipeline(self) -> tuple[str, float]:
        """
        Picture processing pipeline
        :return: the picture and the growth value
        """
        # Take and display the photo
        try:
            pic, path_img = self.capture_and_display()
        except Exception as e:
            LOGGER.error(f"Error during capture: {e}")
            self.register_error(e)
            self.disp.show_collecting_data("Error capturing photo")
            return "", ""
        self.disp.show_collecting_data("Processing photo")
        time.sleep(1)
        # Process the segment lengths to get the growth value
        last_growth_value = get_values_from_csv(self.csv_path, "growth", last_n=1)[0] #takes the last measure in case of error
        if last_growth_value != None:
            growth_value = last_growth_value
        else:
            growth_value = -1
        if pic != "" and path_img != "":
            try:
                growth_value = get_total_length(image_path=path_img, channel=self.channel, kernel_size=self.kernel_size, sigma=self.sigma)
            except Exception as e:
                self.register_error(KeyError("Error while processing the photo, no segment found in the image."
                                             "Check that the plant is clearly visible."))
                self.disp.show_collecting_data("Error while processing the photo")
                time.sleep(5)
                return pic, 0
            LOGGER.debug(f"Growth value : {growth_value}")
            self.disp.show_collecting_data(f"Growth value : {round(growth_value, 2)}")
            time.sleep(2)
            
        # anti grandes valeurs
        last_date = get_values_from_csv(self.csv_path, "date", last_n=1)[0]
        last_date = datetime.strptime(last_date, DATE_FORMAT)
        now = now = datetime.now()
        if abs(growth_value - last_growth_value) > 50 and now - last_date < timedelta(minutes=3):
            print("anti-pic")
            growth_value = last_growth_value

        # moyenne pour lissage
        moy_value = 20
        x_last_values = [float(v) for v in get_values_from_csv(self.csv_path, "growth", last_n=moy_value)]
        x_last_dates = get_values_from_csv(self.csv_path, "date", last_n=moy_value)
        print(x_last_dates)
        delta_time = now - datetime.strptime(x_last_dates[0], DATE_FORMAT)
        limit_time = timedelta(minutes=1.5 * self.time_interval * moy_value)
        print(delta_time)
        print(limit_time)
        
        if delta_time < limit_time and len(x_last_values) == moy_value:
            moy = sum(x_last_values) / moy_value
            growth_value = growth_value * 0.5 + moy * 0.5
          
        return pic, growth_value

    def weight_pipeline(self, n=10) -> tuple[float, float]:
        """
        Weight collection pipeline
        :param n: The number of measurements to take (default = 10)
        :return: The median of the measurements (-1 in case of error) and the observed standard deviation
        """
        self.disp.show_collecting_data("Measuring weight")
        start = time.time()
        median_weight, std_dev = self.get_weight(n)
        median_weight = median_weight - self.tare
        if median_weight == -1.0:
            return -1.0, -1.0
        elapsed = time.time() - start
        LOGGER.debug(f"Weight: {median_weight} in {elapsed}s (with standard deviation: {std_dev}")
        return median_weight, std_dev

    def humidity_pipeline(self):
        self.disp.show_collecting_data("Measuring humidity")
        GPIO.output(self.EN_HUM, GPIO.HIGH)
        time.sleep(0.5)
        
        analog_voltage = self.mcp.read_adc(self.HUM) * (3.3 / 1023.0)
        GPIO.output(self.EN_HUM, GPIO.LOW)
        if analog_voltage >= 3.28: return 0.0
        hum = (np.tan((2.041-analog_voltage)/0.994) + 4.515) / 0.119 
        return hum


    def light_pipeline(self):
        self.disp.show_collecting_data("Measuring light")
        time.sleep(0.5)
        analog_voltage = self.mcp.read_adc(self.LIGHT) * (3.3 / 1023.0)
        lux = 294.5/(analog_voltage - 0.12) - 92.58
        return lux

class DebugHx711(hx711.HX711):
    """
    DebugHx711 class, inherits from hx711.HX711
    Modified to avoid the infinite loop in the _read function
    """

    def __init__(self, dout_pin, pd_sck_pin):
        super().__init__(dout_pin, pd_sck_pin)

    def _read(self, times: int = 10):
        # Custom read function to debug (times=10 to reduce the time of the measurement)
        return super()._read(times)

    def get_raw_data(self, times: int = 5):
        # Modified read function to debug (with a max of 1000 tries) to avoid infinite loops.
        # Furthermore, we check if the data is valid (not False or -1) before appending it to the list
        data_list = []
        count = 0
        while len(data_list) < times and count < 1000:
            data = self._read()
            if data not in [False, -1]:
                data_list.append(data)
            count += 1
        return data_list
