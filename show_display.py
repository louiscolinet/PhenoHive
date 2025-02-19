from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LOGO = "assets/logo_phenohive.jpg"
THICKNESS = 3  # Outline thickness for the status


class Display:
    def __init__(self, station) -> None:
        """
        Initialize the class variables
        :param station: PhenoStation instance
        """
        self.STATION = station
        self.SCREEN = self.STATION.st7735
        self.SCREEN.clear()
        self.SCREEN.begin()
        self.WIDTH = self.STATION.WIDTH
        self.HEIGHT = self.STATION.HEIGHT
        self.SIZE = (self.WIDTH, self.HEIGHT)
        self.LOGO = Image.open(LOGO).rotate(0).resize((128, 70))

    def get_status(self) -> str:
        """
        Return the color status of the station in function of its current status
        :return: the color corresponding to the current status of the station as a string
                green = OK
                blue = OK but not connected to the DB
                yellow = processing
                red = error
        :raises: ValueError: If the station's status incorrect (not -1, 0, or 1)
        """
        if self.STATION.status == -1:
            # Error
            return "red"
        elif self.STATION.status == 1:
            # Processing
            return "yellow"
        elif self.STATION.status == 0:
            if self.STATION.connected:
                # OK and connected to the DB
                return "green"
            else:
                # OK but not connected to the DB
                return "blue"
        else:
            # Station status is not valid
            raise ValueError(f'Station status is incorrect, should be -1, 0, or 1. Got: {self.STATION.status}')

    def create_image(self, logo: bool = False) -> tuple[Image, ImageDraw]:
        """
        Create a blank image with the outline
        :param logo: if True, the logo is added to the image (default: False)
        :return: the image and the draw object as a tuple
        """
        img = Image.new('RGB', self.SIZE, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        if logo:
            img.paste(self.LOGO, (0, 0))

        # Draw outline showing the status
        for i in range(THICKNESS):
            draw.rectangle((i, i, self.WIDTH-1-i, self.HEIGHT-1-i), outline=self.get_status())
        return img, draw

    def show_image(self, path_img: str) -> None:
        """
        Show an image on the display
        :param path_img: path of the image to show
        """
        image = Image.open(path_img)
        image = image.rotate(0).resize(self.SIZE)
        self.SCREEN.display(image)

    def show_measuring_menu(self, weight: float, growth: int, time_now: str, time_next_measure: str,
                            n_rounds: int) -> None:
        """
        Show the measuring menu
        :param weight: weight of the plant
        :param growth: growth value of the plant
        :param time_now: current time
        :param time_next_measure: time of the next measurement
        :param n_rounds: number of the current measurement
        """
        img, draw = self.create_image(logo=True)

        font = ImageFont.truetype(FONT, 10)
        draw.text((5, 70), str(time_now), font=font, fill=(0, 0, 0))
        draw.text((0, 90), "Next : " + str(time_next_measure), font=font, fill=(0, 0, 0))
        draw.text((0, 120), "Measurement n°" + str(n_rounds), font=font, fill=(0, 0, 0))
        draw.text((0, 100), "Weight : " + str(weight), font=font, fill=(0, 0, 0))
        draw.text((0, 110), "Growth : " + str(growth), font=font, fill=(0, 0, 0))
        draw.text((0, 130), "<-- Status", font=font, fill=(0, 0, 0))
        draw.text((80, 130), "Stop -->", font=font, fill=(0, 0, 0))

        self.SCREEN.display(img)

    def show_menu(self) -> None:
        """
        Show the main menu
        """
        # Initialize display.
        img, draw = self.create_image(logo=True)
        # Menu
        font = ImageFont.truetype(FONT, 13)
        draw.text((40, 80), "Menu", font=font, fill=(0, 0, 0))
        # Button
        font = ImageFont.truetype(FONT, 10)
        draw.text((0, 130), "<-- Config        Start -->", font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)

    def show_cal_prev_menu(self) -> None:
        """
        Show the preview menu
        """
        img, draw = self.create_image(logo=True)
        # Menu
        font = ImageFont.truetype(FONT, 13)
        draw.text((13, 80), "Configuration", font=font, fill=(0, 0, 0))
        # Button
        font = ImageFont.truetype(FONT, 10)
        draw.text((0, 130), "<-- Calib           Prev -->", font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)

    def show_cal_menu(self, weight, tare) -> None:
        """
        Show the calibration menu
        :param weight: current weight value
        :param tare: tare value
        :return:
        """
        img, draw = self.create_image(logo=True)
        # Menu
        font = ImageFont.truetype(FONT, 10)
        draw.text((0, 80), f"Tare value: {tare}", font=font, fill=(0, 0, 0))
        draw.text((0, 95), f"Current value: {weight}", font=font, fill=(0, 0, 0))
        draw.text((0, 110), f"Net value: {weight - tare}", font=font, fill=(0, 0, 0))
        # Button
        font = ImageFont.truetype(FONT, 10)
        draw.text((0, 130), "<-- Get Calib    Back -->", font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)

    def show_collecting_data(self, action):
        """
        Show the collecting data menu
        :param action: Current action performed by the station (ex: "Taking photo...")
        """
        img, draw = self.create_image(logo=True)
        # Menu
        font = ImageFont.truetype(FONT, 12)
        draw.text((5, 85), "Collecting data...", font=font, fill=(0, 0, 0))
        if action != "":
            font = ImageFont.truetype(FONT, 8)
            draw.text((5, 100), action, font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)

    def show_status(self) -> None:
        """
        Show the status menu
        """
        img, draw = self.create_image(logo=True)
        font = ImageFont.truetype(FONT, 13)
        draw.text((40, 80), "Status", font=font, fill=(0, 0, 0))
        # Status
        status = self.get_status()
        if status == "green":
            font = ImageFont.truetype(FONT, 8)
            draw.text((5, 95), "OK", font=font, fill=(0, 0, 0))
        elif status == "blue":
            font = ImageFont.truetype(FONT, 8)
            draw.text((5, 95), "Not connected to the DB", font=font, fill=(0, 0, 0))
        elif status == "red":
            font = ImageFont.truetype(FONT, 7)
            draw.text((3, 95), f"Error at {self.STATION.last_error[0]}", font=font, fill=(0, 0, 0))
            draw.text((3, 110), f"{self.STATION.last_error[1]}", font=font, fill=(0, 0, 0))

        # Button
        font = ImageFont.truetype(FONT, 10)
        draw.text((0, 130), "<-- Stop       Resume -->", font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)
