"""timestamp, name, value"""
import csv
import os

import matplotlib.pyplot as plt

from program.const import MAIN_PATH


def make_graph(filename):
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
            res[r[1]]['t'].append(float(r[0]))
            res[r[1]]['v'].append(float(r[2]))

        # red dashes, blue squares and green triangles
        plt.plot(res[names[0]]['t'], res[names[0]]['v'], 'r-', label=names[0])
        plt.plot(res[names[1]]['t'], res[names[1]]['v'], 'b-', label=names[1])
        plt.plot(res[names[2]]['t'], res[names[2]]['v'], 'g-', label=names[2])

        plt.xlabel('Time [s]')
        plt.ylabel('')
        plt.title(filename.split('_')[1].split('.')[0].capitalize())
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()


make_graph('measurements_current.csv')
make_graph('measurements_power.csv')
make_graph('measurements_voltage.csv')
