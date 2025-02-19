# DietPi configuration files

This document describes the different changes made to the DietPi configuration files to automate the setup.

## `dietpi.txt`

### Language/Regional options

#### Change keyboard layout to french

Line 14: `AUTO_SETUP_KEYBOARD_LAYOUT=gb` -> AUTO_SETUP_KEYBOARD_LAYOUT=fr`

#### Change timezone to Brussels

Line 17: `AUTO_SETUP_TIMEZONE=Europe/London` -> AUTO_SETUP_TIMEZONE=Europe/Brussels`

### Network options

#### Disable Ethernet and enable wifi

Line 23: `AUTO_SETUP_NET_ETHERNET_ENABLED=1` -> `AUTO_SETUP_NET_ETHERNET_ENABLED=0`

Line 24: `AUTO_SETUP_NET_WIFI_ENABLED=0` -> `AUTO_SETUP_NET_WIFI_ENABLED=1`

#### Change wifi country code to Belgium

Line 28: `AUTO_SETUP_NET_WIFI_COUNTRY_CODE=GB` -> `AUTO_SETUP_NET_WIFI_COUNTRY_CODE=BE`

#### (optional) Change hostname

Line 41: `AUTO_SETUP_NET_HOSTNAME=DietPi` -> `AUTO_SETUP_NET_HOSTNAME=PhenoHive`

#### Disable delay at boot until network connection is established

This will speed up each boot, as the stations might not always have an internet connection.

Line 48: `AUTO_SETUP_BOOT_WAIT_FOR_NETWORK=1` -> `AUTO_SETUP_BOOT_WAIT_FOR_NETWORK=0`

### Software options

#### Change SSH server to OpenSSH (instead of Dropbear)

Line 80: `AUTO_SETUP_SSH_SERVER_INDEX=-1` -> `AUTO_SETUP_SSH_SERVER_INDEX=-2`

### Non-interactive first run setup

#### On first login, run update, initial setup and software installs without any user input

Line 117: `AUTO_SETUP_AUTOMATED=0` -> `AUTO_SETUP_AUTOMATED=1`

#### Change global password to be applied for the system

line 125: `AUTO_SETUP_GLOBAL_PASSWORD=dietpi` -> `AUTO_SETUP_GLOBAL_PASSWORD=phenohive`

#### Software to automatically install (add each line after line 133)

```bash
AUTO_SETUP_INSTALL_SOFTWARE_ID=17   # Git
AUTO_SETUP_INSTALL_SOFTWARE_ID=69   # RPi.GPIO
AUTO_SETUP_INSTALL_SOFTWARE_ID=74   # InfluxDB
AUTO_SETUP_INSTALL_SOFTWARE_ID=77   # Grafana
AUTO_SETUP_INSTALL_SOFTWARE_ID=130  # Python 3 pip
```

### Misc DietPi program settings

#### (optional) Opt-out of DietPi-Survey

Line 140: `SURVEY_OPTED_IN=-1` -> `SURVEY_OPTED_IN=0`

### DietPi-Config settings

#### (optional) Disable Daily check for DietPi updates

Line 198: `CONFIG_CHECK_DIETPI_UPDATES=1` -> `CONFIG_CHECK_DIETPI_UPDATES=0`

#### (optional) Disable Daily check for APT package updates

Line 202: `CONFIG_CHECK_APT_UPDATES=1` -> `CONFIG_CHECK_APT_UPDATES=0`

## `dietpi-wifi.txt`

### Configure Wifi connection

This requires wifi to be enabled in `dietpi.txt` (see [above](#disable-ethernet-and-enable-wifi))

#### For each entry

One entry = one wifi, here the example is for entry 0. You do not need to change each field.

SSID: `aWIFI_SSID[0]='PhenoHive'`

Key (password): `aWIFI_SSID[0]='PhenoHive'`

## `config.txt`

### Enable picamera module and picamera detection

Warning: be sure to remove the leading '#'

Line 59: `#start_x=1` -> `start_x=1`

Add line 60: `camera_auto_detect=0`

### Enable SPI interface

Warning: be sure to remove the leading '#'

Line 78: `#dtparam=spi=off` -> `dtparam=spi=on`
