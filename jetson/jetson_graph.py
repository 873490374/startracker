"""timestamp, name, value"""
import csv
import os

import matplotlib.pyplot as plt

from program.const import MAIN_PATH


def make_graph(filename, ylabel, power=1):
    with open(
            os.path.join(MAIN_PATH, filename), 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = list(csv_reader)
        names = []
        for row in data:
            names.append(row[0].split(' ')[1])
        names = list(set(names))
        res = {}
        for name in names:
            res[name] = {'t': [], 'v': []}
        for row in data:
            r = row[0].split(' ')
            if float(r[0]) > 45:
                break
            res[r[1]]['t'].append(float(r[0]))
            res[r[1]]['v'].append(float(r[2])*power)

        plt.plot(
            res[names[0]]['t'], res[names[0]]['v'], 'r-', label=names[0][4:])
        plt.plot(
            res[names[1]]['t'], res[names[1]]['v'], 'b-', label=names[1][4:])
        plt.plot(
            res[names[2]]['t'], res[names[2]]['v'], 'g-', label=names[2][4:])

        plt.xlabel('Time [s]')
        plt.ylabel(ylabel)
        plt.title(filename.split('_')[1].split('.')[0].capitalize())
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()


make_graph('measurements_current.csv', 'Current [A]', 10**-3)  # miliampers
make_graph('measurements_voltage.csv', 'Voltage [V]', 10**-3)  # milivolts
make_graph('measurements_power.csv', 'Power [W]', 10**-3)  # miliwats
