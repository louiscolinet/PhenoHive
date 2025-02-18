![PhenoHive Logo](assets/logo_phenohive.jpg)

-----

# PhenoHive

Low-cost Raspberry pi-based phenotyping station.
Based on a [prototype](https://github.com/marty12342000/PhenoHive) by M. Lallemand and L. Lemaire

## Table of contents

- [Project Description](#project-description)
- [System Operation](#system-operation)
  - [Configuration](#configuration)
  - [Initialisation](#initialisation)
  - [Configuration Menu](#configuration-menu)
  - [Measurement Mode](#measurement-mode)
    - [Measurement pipeline](#measurement-pipeline)
    - [Display and status](#display-and-status)
    - [Measurement format](#measurement-format)
  - [Logging and error handling](#logging-and-error-handling)
- [Installation](#installation)
  - [Operating System](#operating-system)
    - [Using the pre-built image](#using-the-pre-built-image)
    - [Automated installation](#automated-installation)
    - [Manual installation](#manual-installation)
  - [Project setup](#project-setup)
    - [Automated setup](#automated-setup)
    - [Manual setup](#manual-setup)
  - [SSH connection](#ssh-connection)

## Project Description

PhenoHive is a low-cost station for plant phenotyping.

The design is based on a Raspberry Pi Zero W running on DietPi OS with:
- A Tal226 load cell connected to a HX711 controller to measure plant weight.
- A Raspberry Pi Camera and a LED lightning strip connected to a KY-019 relay module to take pictures of the plant.
- A ST7735 LCD screen and two buttons to interact with the user.

The software is written in Python, with a [bash setup script](setup.sh) to set up the system.
The Python code is divided in five files:
- [main.py](main.py) is the main file, it initialises the system and handles the user interactions as well as the different pipelines.
- [PhenoHiveStation.py](PhenoHiveStation.py) contains a singleton class that handles the hardware interactions. It contains the different variables and methods to take pictures, measure weight, and communicate with the database.
- [image_processing.py](image_processing.py) contains the different functions to analyse the plant images and compute its growth.
- [show_display.py](show_display.py) contains the different functions to display the information on the LCD screen.
- [utils.py](utils.py) contains two functions, one to set up the logger used by the system, and one to compute the growth of the plant.

## System Operation

### Configuration

Each variable of the station, such as the different pins of each sensor,
the time interval between each measurement, etc. is set in [config.ini](config.ini).

### Initialisation

Once the system has been set up (see [Installation](#installation) for more details), [main.py](main.py) will be run at startup.
It will initialise the logger, instantiate the PhenoHiveStation class, and start the main loop.
At this point, it will display the main menu on the LCD screen and either:
- Wait for the user to press "Start" to enter measurement mode or "Config" to configure enter the configuration menu.
- Automatically start the measurement pipeline if the system unexpectedly shut down in measurement mode.

### Configuration Menu

The configuration menu allows the user to:
- Tare the load cell in the "Calib" menu.
- Ensure that the camera is correctly positioned in the "Prev" menu (this menu can be exited by pressing the right button).

### Measurement Mode

#### Measurement pipeline

The measurement mode is divided in several pipelines to improve modularity and ease of use:
- the picture pipeline takes a picture of the plant, saves it [data/images](data/images), and displays it on the LCD screen.
Then, the picture is analysed using plantcv to compute the growth of the plant (see [image_processing.py](image_processing.py)).
- the weight pipeline measures the weight of the plant, by taking the median of several measurements to avoid abnormal values.
- the database pipeline sends the different measurements to the InfluxDB database. The measurements are also saved in a CSV file in the [data](data) folder to avoid data loss in case of database failure.

#### Display and status

The different steps of the pipelines are displayed on the LCD screen to inform the user of the system status.
A status menu is also available when the system is not in a pipeline to display the station's status.
Furthermore, the display will show a colour outline at all times to inform the user of the system status:
- Green: the system is nominal.
- Blue: the system is nominal but not connected to the database.
- Yellow: the system is in a pipeline.
- Red: the system encountered an error. The error message and time will be displayed on the status menu.

#### Measurement format

The different data sent to the database and saved in the CSV file are:
- "weight": the weight of the plant (raw value without a conversion to grams).
- "weight_g": the weight of the plant (in grams if the calibration coefficient was set using [tools/calibration.py](tools/calibration.py)).
- "standard_deviation": the standard deviation of the weight measurements.
- "growth": the growth of the plant (in pixels).
- "picture": the picture of the plant (in base64 format).
- "status": the status of the station.
- "error_time": the time of the last error. 
- "error_message": the last error message.

### Logging and error handling

The system logs are saved in [logs](logs) folder, with the time and date of the log as the filename. If the logging level is not given as argument when starting the station, the default level is DEBUG.

When an error occurs, the system will register the error message and time using the `register_error` method of the PhenoHiveStation class.
The error message will be logged, displayed on the LCD screen, and the status will be set to red.
Critical steps, such as when collecting the weight or taking a picture, will be wrapped in a try/except block to catch any error and register it.
However, unexpected errors can still occur. In this case, the system will try to catch and register the error, but if more than 10 unexpected errors are encountered, the system will raise a RuntimeError and restart (if the [phenohive.service](tools/phenohive.service) is set to restart on failure).

## Installation

The system is designed to run on a Raspberry Pi Zero W with DietPi OS.
The system was tested with DietPi v9.6.1 (released in 2024-07-07).

To install the system, you can either:

- Use the provided pre-built image of the system with the system already installed and configured.
- Follow the steps below to install the system on a Raspberry Pi Zero W with DietPi OS (or other debian-based OS).

### Operating System

The steps below are for DietPi OS,
but the [automated setup script](setup.sh) can be used on other Debian-based OS (e.g. Raspberry Pi OS).
If you run into any trouble or if you are trying to install on another OS, the steps to set up the project manually are given in the [manual setup](#manual-setup) section.

#### Using the preconfigured image

1. Download the preconfigured image from [here](https://uclouvain-my.sharepoint.com/:u:/g/personal/dylan_goffinet_student_uclouvain_be/Eb_mK8L4GptGhppHbfrdEPoBmDWEppfjG-rZpERCGurvsw?e=2BCUSQ).
2. Flash the image on a microSD card of at least 8Gb using [Balena Etcher](https://www.balena.io/etcher/).
3. Insert the microSD card in the Raspberry Pi Zero W and power it on.
4. Connect to the Raspberry Pi Zero W using a keyboard and a screen or via SSH (default login: root, password: phenohive).
   - The Raspberry Pi Zero W will automatically connect to the internet using the Wi-Fi network configured in [tools/dietpi/dietpi-wifi.txt](tools/dietpi/dietpi-wifi.txt) file
   (default: SSID=PhenoHive, Password=Phenohive).
   - For a step-by-step guide on how to connect to the Raspberry Pi Zero W using SSH, see [below](#ssh-connection).
5. Navigate to the PhenoHive folder using `cd /root/PhenoHive`, and ensure that the system is up to date using `git pull`.
6. Modify the configuration file [config.ini](config.ini) according to your needs (see [Configuration](#configuration)).
7. Run the [calibration script](tools/calibration.py) using `python3 tools/calibration.py` to calibrate the load cell (warning: this will overwrite the previous calibration coefficient).
8. Enable the service using `systemctl enable phenohive.service`, and start the service using `systemctl start phenohive.service`.
9. The system is now running; you can check the status of the service using `systemctl status phenohive.service`.

#### Automated installation

1. Download the latest version of DietPi OS from the [official website](https://dietpi.com/#downloadinfo).
2. Flash the image on a microSD card using [Balena Etcher](https://www.balena.io/etcher/).
3. You should have two partitions on the microSD card, one named "bootfs" and the other named "rootfs".
4. Copy the files located in the [tools/dietpi](tools/dietpi) folder to the "bootfs" partition. A detailed breakdown of the files is given in [DietPi_files.md](tools/dietpi/DietPi_files.md).
5. Modify the [dietpi-wifi.txt](tools/dietpi/dietpi-wifi.txt) file to include your Wi-Fi network SSID and password (default: SSID=PhenoHive, Password=Phenohive).
6. Insert the microSD card in the Raspberry Pi Zero W and power it on.
7. Connect to the Raspberry Pi Zero W using a keyboard and a screen or via SSH (default login: root, password: phenohive).
   - The Raspberry Pi Zero W will automatically connect to the internet using the Wi-Fi network configured in [tools/dietpi/dietpi-wifi.txt](tools/dietpi/dietpi-wifi.txt) file
   (default: SSID=PhenoHive, Password=Phenohive).
   - For a step-by-step guide on how to connect to the Raspberry Pi Zero W using SSH, see [below](#ssh-connection).

#### Manual installation

1. Download the latest version of DietPi OS [from the official website](https://dietpi.com/#downloadinfo).
2. Flash the image on a microSD card using [Balena Etcher](https://www.balena.io/etcher/).
3. Insert the microSD card in the Raspberry Pi and power it on.
4. Connect to the Raspberry Pi using a keyboard and a screen and follow the DietPi setup.
   - A detailed guide on how to set up DietPi at the first boot is available [on the official website](https://dietpi.com/docs/install/#3-prepare-the-first-boot).

### Project setup

#### Automated setup

1. Ensure that the Raspberry Pi is connected to the internet; you can check the connection using `ping google.com`.
2. Clone the repository using `git clone https://github.com/Oldgram/PhenoHive.git PhenoHive`. If you don't have git installed, you can install it using `apt-get install git`.
3. Navigate to the PhenoHive folder using `cd PhenoHive`.
4. Run the setup script using `bash setup.sh`. This script will install the necessary packages, set up the system, and enable the service.
   - _Note:_ Due to the limited resources of the Raspberry Pi Zero W, the setup script can take up 2 to 3 hours to complete.
   - If the setup was successful, you should see a blue message saying "Setup complete. A reboot is required before running the service.".
   - If you encounter any error during the setup, follow the steps given in the [manual setup](#manual-setup) section below.
5. Modify the configuration file [config.ini](config.ini) according to your needs (see [Configuration](#configuration)).
6. Run the [calibration script](tools/calibration.py) using `python3 tools/calibration.py` to calibrate the load cell (warning: this will overwrite the previous calibration coefficient).
7. Reboot the Raspberry Pi using `reboot`, the PhenoHive service will start automatically at boot.
8. The system is now running; you can check the status of the service using `systemctl status phenohive.service`.

#### Manual setup

1. Ensure that the Raspberry Pi is connected to the internet; you can check the connection using `ping google.com`.
2. Clone the repository using `git clone https://github.com/Oldgram/PhenoHive.git PhenoHive`. If you don't have git installed, you can install it using `apt-get install git`.
3. Navigate to the PhenoHive folder using `cd PhenoHive`.
4. Install the necessary packages:
   ```bash
   sudo apt-get update
   sudo apt-get install \
    build-essential \
    cmake \
    gfortran \
    pkg-config \
    git \
    python3 \
    python3-pip \
    python-is-python3 \
    libopenblas-dev \
    libatlas-base-dev \
    patchelf \
    ninja-build \
    libcap-dev \
    ffmpeg \
    python3-dev \
    python3-smbus \
    python3-pil \
    python3-rpi.gpio \
    python3-av \
    python3-libcamera \
    python3-kms++ \
    python3-pyqt5 \
    python3-prctl \
    python3-numpy \
    python3-scipy \
    python3-sklearn \
    python3-skimage \
    python3-statsmodels
   ```
5. Install the necessary Python packages:
   ```bash
   pip3 install --break-system-packages --root-user-action=ignore --no-cache-dir \
    configparser==7.0.0 \
    influxdb-client==1.44.0 \
    hx711==1.1.2.3 \
    pandas==1.5.1 \
    adafruit-gpio==1.0.3 \
    opencv-python==4.7.0.72 \
    plantcv==3.14.3 \
    mizani==0.9.0 \
    plotnine==0.10.1
   ```
6. Download and install the ST7735 library:
   ```bash
   git clone https://github.com/degzero/Python_ST7735.git
   cd Python_ST7735
   python3 setup.py install
   ```
7. Enable the SPI interface:
   - On Raspberry Pi OS: `sudo raspi-config nonint do_spi 0`.
   - On DietPi OS or other: `echo "dtparam=spi=on" | sudo tee -a /boot/config.txt`.
8. Create and enable PhenoHive service:
   - Copy the [phenohive.service](tools/phenohive.service) file to the systemd folder using `cp tools/phenohive.service /lib/systemd/system/`.
   - Edit the service file using `nano /lib/systemd/system/phenohive.service` and replace:
     - `User=root` to your current user (on DietPi, you can ignore this step).
     - `PHENOHIVE_DIRECTORY` in the `WorkingDirectory` and `ExecStart` fields to the path of the PhenoHive folder.
     - The file should look like this (example for root user of DietPi):
       ```ini
       [Unit]
       Description=PhenoHive station service
       Documentation=https://github.com/Oldgram/PhenoHive
       After=multi-user.target
       
       [Service]
       User=root
       WorkingDirectory=/root/PhenoHive/
       ExecStart=/usr/bin/python /root/PhenoHive/main.py
       Restart=on-failure
       
       [Install]
       WantedBy=multi-user.target
       Alias=phenoHive.service
       ```
   - Enable the service using:
   ```bash
   sudo chmod 644 /lib/systemd/system/phenohive.service
   chmod +x main.py
   sudo systemctl daemon-reload
   sudo systemctl enable phenohive.service
   ```
9. Modify the configuration file [config.ini](config.ini) according to your needs (see [Configuration](#configuration)).
10. Run the [calibration script](tools/calibration.py) using `python3 tools/calibration.py` to calibrate the load cell (warning: this will overwrite the previous calibration coefficient).
11. Reboot the Raspberry Pi using `reboot`, the PhenoHive service will start automatically at boot.
12. The system is now running; you can check the status of the service using `systemctl status phenohive.service`.

### SSH connection

1. Make sure that you are connected to the same network as the Raspberry Pi.
2. Find your current IP using `ifconfig /all` (on Windows or macOS) or `ip a` (on Linux).
   - You should see an IP address in the form of `192.168.1.X`, `10.42.0.X` or similar.
   Example output of `ifconfig /all`:
   ```bash
   PS C:\Users\goffi> ipconfig /all
    [...]
    Carte réseau sans fil Wi-Fi :
    
        Suffixe DNS propre à la connexion. . . :
        Description. . . . . . . . . . . . . . : Killer(R) Wi-Fi 6 AX1650i 160MHz Wireless Network Adapter (201NGW)
        Adresse physique . . . . . . . . . . . : 70-A8-D3-1B-15-93
        DHCP activé. . . . . . . . . . . . . . : Oui
        Configuration automatique activée. . . : Oui
        Adresse IPv6 de liaison locale. . . . .: fe80::4101:1239:213e:be14%2(préféré)
        Adresse IPv4. . . . . . . . . . . . . .: 192.168.166.174(préféré)
        Masque de sous-réseau. . . . . . . . . : 255.255.255.0
        Bail obtenu. . . . . . . . . . . . . . : samedi 17 juillet 2024 09:40:54
        Bail expirant. . . . . . . . . . . . . : samedi 17 juillet 2024 10:40:52
        Passerelle par défaut. . . . . . . . . : 192.168.166.254
        Serveur DHCP . . . . . . . . . . . . . : 192.168.166.254
        IAID DHCPv6 . . . . . . . . . . . : 40937683
        DUID de client DHCPv6. . . . . . . . : 00-01-00-01-2A-30-89-94-08-8F-C3-64-23-E1
        Serveurs DNS. . .  . . . . . . . . . . : 192.168.166.254
        NetBIOS sur Tcpip. . . . . . . . . . . : Activé
   ```
   - If your IP is, for example `192.168.34.17`, than your IP range is `192.168.34.0/24`.
   In the above example, the IP is `192.168.166.254`, so the IP range is `192.168.166.0/24`.
3. Find the IP address of the Raspberry Pi:
   - Using [nmap](https://nmap.org/), scan the network using `nmap -sn <YOUR_IP_RANGE>` (replace `<YOUR_IP_RANGE>` with your IP range found at step 2).
   - The Raspberry Pi should appear in the list with "Raspberry Pi Foundation" next to the MAC Address, its IP will be just above.
   - Example:
   ```bash
    PS C:\Users\goffi> nmap -sn 192.168.166.0/24
    Starting Nmap 7.95 ( https://nmap.org ) at 2024-07-17 09:47 Paris, Madrid (heure dÆÚtÚ)
    Nmap scan report for 192.168.166.16
    Host is up (0.013s latency).
    MAC Address: B8:27:EB:BF:6D:1A (Raspberry Pi Foundation)
    Nmap scan report for 192.168.166.254
    Host is up (0.017s latency).
    MAC Address: 4E:A5:88:52:12:F3 (Unknown)
    Nmap scan report for 192.168.166.174
    Host is up.
    Nmap done: 256 IP addresses (3 hosts up) scanned in 2.92 seconds
   ```
   Here, the Raspberry Pi has the IP `192.168.166.16` (the first in the list).
4. Connect to the Raspberry Pi using SSH with `ssh root@<RASPBERRY_PI_IP>` (replace `<RASPBERRY_PI_IP>` with the IP found at step 3).
Enter the password when prompted (default: phenohive).
5. You can close the connection at any time using `exit`.

-----

![UCLouvain Logo](assets/logo_UCLouvain.jpg)
