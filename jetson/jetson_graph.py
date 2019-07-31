import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from program.const import MAIN_PATH
from tests.cuda.expected_results_full_startracker import expected_full


def method_name(break_after, data, power, start_after):
    names = []
    for row in data:
        names.append(row[0].split(' ')[1])
    names = list(set(names))
    res = {}
    for name in names:
        res[name] = {
            't': [],
            'v': []
        }
    for row in data:
        r = row[0].split(' ')
        if float(r[0]) > break_after:
            break
        if float(r[0]) < start_after:
            continue
        res[r[1]]['t'].append(float(r[0]) - start_after)
        res[r[1]]['v'].append(float(r[2]) * power)
    # names.remove('VDD_SYS_SOC')
    labels = {
        'VDD_IN': {
            'name': 'Total',
            'color': 'k-'
        },
        'VDD_SYS_SOC': {
            'name': 'SOC',
            'color': 'g-'
        },
        'VDD_SYS_GPU': {
            'name': 'GPU',
            'color': 'r-'
        },
        'VDD_SYS_CPU': {
            'name': 'CPU',
            'color': 'b-'
        },
    }
    return labels, names, res


def make_measurement_graph(
        data, ylabel, power=1, start_after=3, break_after=33):
    labels, names, res = method_name(break_after, data, power, start_after)
    for name in names:
        plt.plot(
            res[name]['t'], res[name]['v'],
            labels[name]['color'], label=labels[name]['name'])

    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    plt.title(ylabel[:-3])
    plt.legend(loc=4, ncol=1, borderaxespad=0.)
    plt.show()


def make_time_graph():
    star_count = [len(s) for s in expected_full]
    N = len(time_data['ids'])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    fig = plt.figure()  # Create matplotlib figure

    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

    p1 = ax.bar(
        ind, time_data['centroid'], width)
    p2 = ax.bar(
        ind, time_data['ids'], width,
        bottom=time_data['centroid'])
    p3 = ax.bar(
        ind, time_data['attitude'], width,
        bottom=np.array(time_data['ids']) + np.array(time_data['centroid']))
    p4 = ax2.bar(
        ind, star_count,
        color='0.75', width=0.2, align='edge')

    plt.ylabel('Time [s]')
    plt.title('Scenes time computation and input stars number')
    plt.xticks(ind, np.arange(1, N + 1))

    plt.legend((p1[0], p2[0], p3[0], p4[0]), (
        'Star Recognition', 'Star Identification', 'Attitude Estimation',
        'Number of input stars (right)'
    ))

    ax.set_ylabel('Time [s]')
    ax2.set_ylabel('Number of input stars')

    plt.show()


def make_power_sectioned_graph(start_time=3):
    start_after = 3
    break_after = 33
    power = 10 ** -3

    labels, names, res = method_name(
        break_after, power_data, power, start_after)

    for name in names:
        plt.plot(
            res[name]['t'], res[name]['v'],
            labels[name]['color'], label=labels[name]['name'])

    end_time = 0
    for i in range(2):
        for j in range(len(time_data['start_time'])):
            if j % len(time_data['start_time']) == 0:
                color = 'red'
            else:
                color = 'black'
            tt = time_data['start_time'][j]
            if tt + start_time + end_time > 30:
                break
            plt.axvline(tt + start_time + end_time, color=color)
            if j % len(time_data['start_time']) == \
                    len(time_data['start_time']) - 1:
                end_time = tt + end_time

    plt.xlabel('Time [s]')
    plt.ylabel('Power')
    plt.title('Power divided by time scenes')
    plt.legend(loc=4, ncol=1, borderaxespad=0.)

    plt.show()


p = os.path.join(MAIN_PATH, 'tests/measurements')
with open(os.path.join(p, 'measurements_current.csv'), 'r') as current_file, \
     open(os.path.join(p, 'measurements_voltage.csv'), 'r') as voltage_file, \
     open(os.path.join(p, 'measurements_power.csv'), 'r') as power_file, \
     open(os.path.join(p, 'times'), 'r') as times_file:
    current_reader = csv.reader(current_file, delimiter=',')
    current_data = list(current_reader)
    voltage_reader = csv.reader(voltage_file, delimiter=',')
    voltage_data = list(voltage_reader)
    power_reader = csv.reader(power_file, delimiter=',')
    power_data = list(power_reader)
    times = times_file.readlines()

    time_data = {
        'centroid': [],
        'ids': [],
        'attitude': [],
        'start_time': [],
        'end_time': [],
    }

    for t in times:
        a = t.replace("\n", "")
        a = a.replace(" ", "")
        a = a.split(':')
        a[1] = float(a[1])

        time_data[a[0]].append(a[1])

    make_measurement_graph(
        current_data, 'Current [A]', 10 ** -3)  # milliamperes
    make_measurement_graph(
        voltage_data, 'Voltage [V]', 10 ** -3)  # millivolts
    make_measurement_graph(
        power_data, 'Power [W]', 10 ** -3)  # milliwatts
    make_time_graph()
    make_power_sectioned_graph()
