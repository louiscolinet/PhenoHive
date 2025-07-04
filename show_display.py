from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LOGO = "assets/logo_phenohive.jpg"
THICKNESS = 3  # Outline thickness for the status


class Display:
    def __init__(self, station) -> None:
        """
        Initialize the class variables
        :param station: PhenoHiveStation instance
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
        #print(self.STATION.status)
        if self.STATION.status == -1:
            # Error
            return (0, 0, 255)  # red
        elif self.STATION.status == 1:
            # Processing
            return (0, 255, 255)  # yellow
        elif self.STATION.status == 0:
            if self.STATION.connected:
                # OK and connected to the DB
                return (0, 128, 0)  # green
            else:
                # OK but not connected to the DB
                return (139, 0, 0)  # dark blue
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

    def show_measuring_menu(self, weight: float, growth: int, humidity: int, time_now: str, time_next_measure: str,
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
        draw.text((5, 60), str(time_now), font=font, fill=(0, 0, 0))
        draw.text((5, 70), "Next : " + str(time_next_measure), font=font, fill=(0, 0, 0))
        draw.text((5, 90), "Weight : " + str(weight), font=font, fill=(0, 0, 0))
        draw.text((5, 100), "Growth : " + str(growth), font=font, fill=(0, 0, 0))
        draw.text((5, 110), "Humidity : " + str(humidity), font=font, fill=(0, 0, 0))
        draw.text((5, 120), "Measurement n°" + str(n_rounds), font=font, fill=(0, 0, 0))
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
        draw.text((0, 130), "<-- Config      Start -->", font=font, fill=(0, 0, 0))
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
        draw.text((0, 130), "<-- Calib          Prev -->", font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)

    def show_calib_menu(self) -> None:
        """
        Show the preview menu
        """
        img, draw = self.create_image(logo=True)
        # Menu
        font = ImageFont.truetype(FONT, 13)
        draw.text((24, 80), "Calibration", font=font, fill=(0, 0, 0))
        # Button
        font = ImageFont.truetype(FONT, 10)
        draw.text((0, 130), "<-- Weight       Img -->", font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)

    def show_weight_cal_menu(self, raw_weight, weight_g, tare, calib_or_test) -> None:
        """
        Show the calibration menu
        :param raw_weight: measured weight before conversion
        :param weight_g: measured weight in grams
        :param tare: tare value
        :param calib_or_test: to know if calib or test have to be show
        :return:
        """
        img, draw = self.create_image(logo=True)
        # Menu
        font = ImageFont.truetype(FONT, 10)
        draw.text((5, 80), f"Tare value: {tare}", font=font, fill=(0, 0, 0))
        draw.text((5, 95), f"Raw value: {raw_weight}", font=font, fill=(0, 0, 0))
        draw.text((5, 110), f"Weight: {weight_g}", font=font, fill=(0, 0, 0))
        # Button
        font = ImageFont.truetype(FONT, 10)
        if calib_or_test < 3:
            draw.text((0, 130), "<-- Calib         Back -->", font=font, fill=(0, 0, 0))
        else:
            draw.text((0, 130), "<-- Test          Back -->", font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)

    def show_img_param_menu(self, sigma, kernel_size, inc) -> None:
        """
        Show the measuring menu
        :param sigma: sigma used in image processing
        :param kernel_size: kernel_size used in image processing for the bluring
        :param inc: to know how much test had already done
        """
        img, draw = self.create_image(logo=True)
        # Button
        font = ImageFont.truetype(FONT, 10)
        draw.text((5, 80), f"Sigma value: {round(sigma,3)}", font=font, fill=(0, 0, 0))
        draw.text((5, 95), f"Kernel size value: {kernel_size}", font=font, fill=(0, 0, 0))
        if inc%2 == 0:
            draw.text((0, 130), f"<-- Calib {inc // 2 + 1}      Back -->", font=font, fill=(0, 0, 0))
            self.SCREEN.display(img)
        else:
            draw.text((0, 130), f"<-- Photo        Back -->", font=font, fill=(0, 0, 0))
            self.SCREEN.display(img)

    def show_photo_taken(self, inc):
        """
        tell the user that a photo has just been taken
        :param inc: to know how much test had already done
        """
        img, draw = self.create_image(logo=True)
        # Button
        font = ImageFont.truetype(FONT, 10)
        draw.text((25, 80), "Photo taken", font=font, fill=(0, 0, 0))
        draw.text((0, 130), f"<-- Calib {inc // 2 + 1}      Back -->", font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)

    def wrap_text(self, text, font, max_width, draw):
        """
        Wrap text so it fits within the given width.
        """
        lines = []
        words = text.split()
        current_line = ""
    
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            width, _ = draw.textsize(test_line, font=font)
            if width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines


    def show_collecting_data(self, action):
        """
        Show the collecting data menu
        :param action: Current action performed by the station (ex: "Taking photo...")
        """
        img, draw = self.create_image(logo=True)
        font_title = ImageFont.truetype(FONT, 12)
        draw.text((5, 85), "Collecting data...", font=font_title, fill=(0, 0, 0))
    
        font_action = ImageFont.truetype(FONT, 8)
        if action:
            wrapped_lines = self.wrap_text(action, font_action, max_width=self.WIDTH - 10, draw=draw)
            y = 100
            for line in wrapped_lines:
                draw.text((5, y), line, font=font_action, fill=(0, 0, 0))
                y += 10
    
        self.SCREEN.display(img)

    def show_status(self) -> None:
        """
        Show the status menu
        """
        img, draw = self.create_image(logo=False)
        font = ImageFont.truetype(FONT, 13)
        draw.text((40, 15), "Status", font=font, fill=(0, 0, 0))
        # Status
        status = self.get_status()
        if self.STATION.status == 0:
            font = ImageFont.truetype(FONT, 8)
            draw.text((5, 40), "OK", font=font, fill=(0, 0, 0))
        if self.STATION.status == -1:
            if not self.STATION.connected:
                font = ImageFont.truetype(FONT, 8)
                draw.text((5, 40), "Not connected to the DB", font=font, fill=(0, 0, 0))
            else:
                font = ImageFont.truetype(FONT, 8)
                timestamp = self.STATION.last_error[0]
                dt = datetime.fromisoformat(timestamp.replace("Z", ""))
                formatted_time = dt.strftime("%d %b, %H:%M")
                draw.text((3, 40), f"Error at {formatted_time}", font=font, fill=(0, 0, 0))
            
                # Wrap error message
                error_text = str(self.STATION.last_error[1])
                wrapped_lines = self.wrap_text(error_text, font, max_width=self.WIDTH - 10, draw=draw)
                y = 50
                for line in wrapped_lines:
                    draw.text((3, y), line, font=font, fill=(0, 0, 0))
                    y += 10  # espacement vertical entre les lignes

        # Button
        font = ImageFont.truetype(FONT, 10)
        draw.text((0, 130), "<-- Stop     Resume -->", font=font, fill=(0, 0, 0))
        self.SCREEN.display(img)
