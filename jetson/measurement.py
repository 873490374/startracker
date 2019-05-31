import csv
import fcntl
import os
import time

VDD_SYS_GPU = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/rail_name_0',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_voltage0_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_current0_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_current0_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/crit_current_limit_0',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/warn_current_limit_0',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power0_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power0_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/crit_power_limit_0',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/ui_input_0',
}
VDD_SYS_SOC = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/rail_name_1',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_voltage1_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_current1_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_current1_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/crit_current_limit_1',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/warn_current_limit_1',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power1_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power1_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/crit_power_limit_1',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/ui_input_1',
}
VDD_4V0_WIFI = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/rail_name_2',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_voltage2_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_current2_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_current2_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/crit_current_limit_2',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/warn_current_limit_2',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power2_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power2_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/crit_power_limit_2',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/ui_input_2',
}
VDD_IN = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/rail_name_0',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_voltage0_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_current0_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_current0_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/crit_current_limit_0',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/warn_current_limit_0',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power0_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power0_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/crit_power_limit_0',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/ui_input_0',
}
VDD_SYS_CPU = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/rail_name_1',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_voltage1_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_current1_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_current1_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/crit_current_limit_1',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/warn_current_limit_1',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power1_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power1_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/crit_power_limit_1',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/ui_input_1',
}
VDD_SYS_DDR = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/rail_name_2',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_voltage2_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_current2_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_current2_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/crit_current_limit_2',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/warn_current_limit_2',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power2_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power2_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/crit_power_limit_2',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/ui_input_2',
}
VDD_MUX = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/rail_name_0',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_voltage0_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_current0_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_current0_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/crit_current_limit_0',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/warn_current_limit_0',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_power0_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_power0_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/crit_power_limit_0',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/ui_input_0',
}
VDD_5V0_IO_SYS = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/rail_name_1',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_voltage1_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_current1_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_current1_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/crit_current_limit_1',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/warn_current_limit_1',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_power1_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_power1_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/crit_power_limit_1',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/ui_input_1',
}
VDD_3V3_SYS = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/rail_name_2',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_voltage2_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_current2_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_current2_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/crit_current_limit_2',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/warn_current_limit_2',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_power2_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/in_power2_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/crit_power_limit_2',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0042/iio_device/ui_input_2',
}
VDD_3V3_IO_SLP = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/rail_name_0',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_voltage0_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_current0_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_current0_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/crit_current_limit_0',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/warn_current_limit_0',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_power0_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_power0_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/crit_power_limit_0',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/ui_input_0',
}
VDD_1V8_IO = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/rail_name_1',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_voltage1_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_current1_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_current1_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/crit_current_limit_1',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/warn_current_limit_1',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_power1_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_power1_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/crit_power_limit_1',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/ui_input_1',
}
VDD_3V3_SYS_M2 = {
    'rail_name': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/rail_name_2',
    'in_voltage_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_voltage2_input',
    'in_current_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_current2_input',
    'in_current_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_current2_trigger_input',
    'crit_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/crit_current_limit_2',
    'warn_current_limit': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/warn_current_limit_2',
    'in_power_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_power2_input',
    'in_power_trigger_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/in_power2_trigger_input',
    'crit_power_limit': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/crit_power_limit_2',
    'ui_input': '/sys/devices/3160000.i2c/i2c-0/0-0043/iio_device/ui_input_2',
}
devices = [
    VDD_SYS_GPU,
    VDD_SYS_SOC,
    # VDD_4V0_WIFI,
    # VDD_IN,
    VDD_SYS_CPU,
    # VDD_SYS_DDR,
    # VDD_MUX,
    # VDD_5V0_IO_SYS,
    # VDD_3V3_SYS,
    # VDD_3V3_IO_SLP,
    # VDD_1V8_IO,
    # VDD_3V3_SYS_M2,
]

if __name__ == '__main__':
    with open('measurements_power.csv', 'w') as csvfile_power, \
            open('measurements_voltage.csv', 'w') as csvfile_voltage, \
            open('measurements_current.csv', 'w') as csvfile_current:
        writer_power = csv.writer(csvfile_power, delimiter=' ',
                                  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer_voltage = csv.writer(csvfile_voltage, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer_current = csv.writer(csvfile_current, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        start = time.time()
        while time.time() - start < 300:
            for dev in devices:
                try:
                    f1 = os.open(dev['rail_name'], os.O_RDONLY | os.O_NONBLOCK)
                    name = os.read(f1, 100).decode("utf-8")[:-1]
                    os.close(f1)

                    f2 = os.open(
                        dev['in_voltage_input'], os.O_RDONLY | os.O_NONBLOCK)
                    voltage = int(str(os.read(f2, 100).decode("utf-8")))
                    writer_voltage.writerow([time.time() - start, name, voltage])
                    os.close(f2)

                    f3 = os.open(
                        dev['in_current_input'], os.O_RDONLY | os.O_NONBLOCK)
                    current = int(str(os.read(f3, 100).decode("utf-8")))
                    writer_current.writerow([time.time() - start, name, current])
                    os.close(f3)

                    f4 = os.open(
                        dev['in_power_input'], os.O_RDONLY | os.O_NONBLOCK)
                    power = int(str(os.read(f4, 100).decode("utf-8")))
                    writer_power.writerow([time.time() - start, name, power])
                    os.close(f4)

                    print(name, voltage, current, voltage, time.time() - start)
                    time.sleep(0.1)
                except:
                    pass
