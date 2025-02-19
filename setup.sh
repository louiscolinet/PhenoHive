#!/bin/bash
# Setup script for the PhenoHive project
# This script installs the necessary packages and enables the SPI interface on the Raspberry Pi
# It also sets up the PhenoHive service to run on boot
# It is intended to be run on a fresh install on a Debian Linux Distribution (Raspbian or DietPi)

CONFIG_FILE="tools/setup.config"
REQUIREMENTS_FILE="requirements.txt"

# Color codes for the outputs
ERROR='\033[0;31m'
INFO='\033[0;36m'
WHITE='\033[0m'

check_internet() {
    if ! ping -q -c 1 -W 1 google.com &> /dev/null; then
        echo -e "${ERROR}Please ensure that the Raspberry Pi is connected to the internet before running this script.${WHITE}"
        exit 1
    fi
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${ERROR}Please run as root: sudo bash $0${WHITE}"
        exit 1
    fi
}

check_directory() {
    if [ ! -f main.py ]; then
        echo -e "${ERROR}Please run the script from the PhenoHive directory. Current directory: $(pwd)${WHITE}"
        exit 1
    fi
}

disable_apt_compression() {
    # If running on DietPi, remove apt compression to speed up apt
    if [ -f /boot/dietpi/.dietpi ]; then
        echo 'Acquire::GzipIndexes "false";' > /etc/apt/apt.conf.d/98dietpi-uncompressed
        /boot/dietpi/func/dietpi-set_software apt-cache clean
        apt update
    fi
}

install_packages() {
    echo -e "${INFO}Installing necessary packages...${WHITE}"
    if ! apt-get update; then
        echo -e "${ERROR}Failed to update package list. Exiting.${WHITE}"
        exit 1
    fi
    if ! grep -vE '^\s*#' $CONFIG_FILE | xargs apt-get -y install; then
        echo -e "${ERROR}Failed to install packages. Exiting.${WHITE}"
        exit 1
    fi
}

install_python_packages() {
    echo -e "${INFO}Installing necessary Python packages...${WHITE}"
    if ! pip install -v -r $REQUIREMENTS_FILE --break-system-packages --root-user-action=ignore --no-cache-dir; then
        echo -e "${ERROR}Failed to install Python packages. Exiting.${WHITE}"
        exit 1
    fi
}

install_st7735() {
    echo -e "${INFO}Installing ST7735 library...${WHITE}"
    git clone https://github.com/degzero/Python_ST7735.git
    cd Python_ST7735 || echo -e "${ERROR}Python_ST335 could not be installed: Could not find directory.${WHITE}"
    python3 setup.py install
    cd ..
}

enable_spi() {
    echo -e "${INFO}Enabling SPI interface...${WHITE}"
    if [ -f /etc/os-release ]; then
        source /etc/os-release
        if [[ $ID != "raspbian" ]]; then
            # On a raspbian system, use raspi-config to enable SPI
            raspi-config nonint do_spi 0
        else
            # On a dietpi or other non-raspbian system, enable the SPI interface in the config file
            echo "dtparam=spi=on" | sudo tee -a /boot/config.txt
        fi
    fi
}

setup_service() {
    echo -e "${INFO}Setting up PhenoHive service...${WHITE}"
    # Modify the WorkingDirectory and ExecStart in the service file to point to the correct (current) directory
    PROJECT_DIR=$(pwd)
    sed -i "s|WorkingDirectory=.*|WorkingDirectory=${PROJECT_DIR}|" tools/phenohive.service
    sed -i "s|ExecStart=.*|ExecStart=/usr/bin/python ${PROJECT_DIR}/main.py|" tools/phenohive.service
    cp tools/phenohive.service /lib/systemd/system/
    chmod 644 /lib/systemd/system/phenohive.service
    chmod +x main.py
    systemctl daemon-reload
    systemctl enable phenohive.service
}

echo -e "${INFO}PhenoHive setup script.\n" \
    "\t This script installs the necessary packages and enables the SPI interface.\n" \
    "\t It also sets up the PhenoHive service to run on boot.\n" \
    "\t It is intended to be run on a fresh install of a Debian Linux Distribution (DietPi or Raspbian) running on a Raspberry Pi Zero W.${WHITE}"

echo -e "${INFO}Running pre-setup checks...${WHITE}"
check_internet
check_root
check_directory
disable_apt_compression

# Install required packages
install_packages
install_python_packages
install_st7735

# Enable spi interface (should be enabled by default)
enable_spi
# Setup PhenoHive as a service so PhenoHive/main.py is run on boot
setup_service

# Setup complete, reboot the Raspberry Pi
echo -e "${INFO}Setup complete. A reboot is required before running the service.${WHITE}"
