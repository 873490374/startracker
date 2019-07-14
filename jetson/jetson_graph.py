import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from program.const import MAIN_PATH


# TODO clean this mess


def make_measurement_graph(
        filename, ylabel, power=1, start_after=3, break_after=33):
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
            if float(r[0]) > break_after:
                break
            if float(r[0]) < start_after:
                continue
            res[r[1]]['t'].append(float(r[0])-start_after)
            res[r[1]]['v'].append(float(r[2])*power)
        names.remove('VDD_SYS_SOC')
        labels = {
            'VDD_IN': {'name': 'Total', 'color': 'k-'},
            'VDD_SYS_SOC': {'name': 'SOC', 'color': 'g-'},
            'VDD_SYS_GPU': {'name': 'GPU', 'color': 'r-'},
            'VDD_SYS_CPU': {'name': 'CPU', 'color': 'b-'},
        }
        for name in names:
            plt.plot(
                res[name]['t'], res[name]['v'],
                labels[name]['color'], label=labels[name]['name'])

        plt.xlabel('Time [s]')
        plt.ylabel(ylabel)
        plt.title(filename.split('_')[1].split('.')[0].capitalize())
        plt.legend(loc=4, ncol=1, borderaxespad=0.)
        plt.show()


def make_time_graph():
    d = {
        'centroid': [],
        'ids': [],
        'attitude': [],
        'start_time': [],
        'end_time': [],
    }
    with open(
            os.path.join(MAIN_PATH, 'times'), 'r') as times_file:
        lines = times_file.readlines()
        for l in lines:
            a = l.replace("\n", "")
            a = a.replace(" ", "")
            a = a.split(':')
            a[1] = float(a[1])

            d[a[0]].append(a[1])

    N = len(d['ids'])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, d['centroid'], width)
    p2 = plt.bar(ind, d['ids'], width, bottom=d['centroid'])
    p3 = plt.bar(ind, d['attitude'], width, bottom=np.array(d['ids'])+np.array(d['centroid']))

    plt.ylabel('Time [s]')
    # plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
    plt.title('Scores by group and gender')
    plt.xticks(ind, np.arange(1, N+1))

    plt.legend((p1[0], p2[0], p3[0]), ('Centroid', 'IDs', 'Attitude'))

    plt.show()


# make_measurement_graph(
#     'measurements_current.csv', 'Current [A]', 10 ** -3)  # milliamperes
# make_measurement_graph(
#     'measurements_voltage.csv', 'Voltage [V]', 10 ** -3)  # millivolts
# make_measurement_graph(
#     'measurements_power.csv', 'Power [W]', 10 ** -3)  # milliwatts
# make_time_graph()

def find_contiguous_colors(colors):
    # finds the continuous segments of colors and returns those segments
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg)  # the final one
    return segs


def plot_multicolored_lines(x, y, colors):
    segments = find_contiguous_colors(colors)
    plt.figure()
    start = 0
    for seg in segments:
        end = start + len(seg)
        l, = plt.gca().plot(
            x[start:end], y[start:end], lw=2, c=seg[0])
        start = end


def make_power_sectioned_graph(start_time=3):
    start_after = 3
    break_after = 33
    power = 10 ** -3
    with open(
            os.path.join(MAIN_PATH, 'measurements_power.csv'), 'r'
    ) as csv_file:
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
            if float(r[0]) > break_after:
                break
            if float(r[0]) < start_after:
                continue
            res[r[1]]['t'].append(float(r[0]) - start_after)
            res[r[1]]['v'].append(float(r[2]) * power)
        names.remove('VDD_SYS_SOC')
        labels = {
            'VDD_IN': {'name': 'Total', 'color': 'k-'},
            'VDD_SYS_SOC': {'name': 'SOC', 'color': 'g-'},
            'VDD_SYS_GPU': {'name': 'GPU', 'color': 'r-'},
            'VDD_SYS_CPU': {'name': 'CPU', 'color': 'b-'},
        }

    d = {
        'centroid': [],
        'ids': [],
        'attitude': [],
        'start_time': [],
    }
    with open(
            os.path.join(MAIN_PATH, 'times'), 'r') as times_file:
        lines = times_file.readlines()
        for l in lines:
            a = l.replace("\n", "")
            a = a.replace(" ", "")
            a = a.split(':')
            a[1] = float(a[1])

            d[a[0]].append(a[1])

    for name in names:
        plt.plot(
            res[name]['t'], res[name]['v'],
            labels[name]['color'], label=labels[name]['name'])

    end_time = 0
    for i in range(2):
        for j in range(len(d['start_time'])):
            if j % len(d['start_time']) == 0:
                color = 'red'
            else:
                color = 'black'
            tt = d['start_time'][j]
            if tt + start_time + end_time > 30:
                break
            plt.axvline(tt + start_time + end_time, color=color)
            if j % len(d['start_time']) == len(d['start_time'])-1:
                end_time = tt + end_time

    plt.xlabel('Time [s]')
    plt.ylabel('Power')
    plt.title('Power divided by time scenes')
    plt.legend(loc=4, ncol=1, borderaxespad=0.)

    plt.show()


make_power_sectioned_graph()
