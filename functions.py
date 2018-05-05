import rnn

import csv, random

def get_price_volume_data(file, price_column, volume_column):
    data = []
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            v = {}
            v['price'] = float(line[price_column])
            v['volume'] = float(line[volume_column])
            data.append(v)
    return data

def scale_input(input_vector, s = 1):
    values = {}
    min = {}
    max = {}
    for day in input_vector:
        for name, value in day.items():
            if name not in values:
                values[name] = [value]
                min[name] = value
                max[name] = value
            else:
                values[name].append(value)
                if min[name] > value:
                    min[name] = value
                if max[name] < value:
                    max[name] = value

    #scale the data
    v = {}
    for name, data in values.items():
        d = []
        for i, day in enumerate(data):
            d.append((day - min[name]) / (s * (max[name] - min[name])) + (1.0 - 1.0 / s) / 2)
        v[name] = d

    values = v
    #reshape the data
    series = []
    titles = []
    for name, data in values.items():
        series.append(data)
        titles.append(name)



    values = []

    for i in range(len(series[0])):
        day = {}
        for j in range(len(titles)):
            day[titles[j]] = series[j][i]
        values.append(day)
    return values, min, max


def scale_output(output, min, max, s = 1):
    v = []
    #print(output)
    for i in range(len(output)):
        day = {}
        for name, value in output[i].items():
            day[name] = (value - min[name]) / (s * (max[name] - min[name])) + (1.0 - 1.0 / s) / 2
        v.append(day)
    return v

def unscale(data, min, max, s = 1):
    v = []
    for d in data:
        v.append((d - (1.0 - 1.0 / s) / 2) * s * (max - min) + min)
    return v


def parse_input(input):
    input_tensor = []
    for i in range(len(input)):
        week = []
        for j in range(len(input[i])):
            day = []
            for key, value in input[i][j].items():
                day.append(value)
            week.append(day)
        input_tensor.append(week)
    return input_tensor

def parse_output(output):
    output_tensor = []
    for i in range(len(output)):
        day = []
        for key, value in output[i].items():
            day.append(value)
        output_tensor.append(day)
    return output_tensor
