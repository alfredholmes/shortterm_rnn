import rnn

import csv, random

from functions import *

def main():
    scale = 3
    traning = False
    inputs = 168
    forcaster = rnn.TimeSeriesForcaster('forcaster', inputs, 2, 2, 0.001)
    forcaster.restore()

    daily = get_price_volume_data('hourly_btc.csv', 1, 2)
    #split daily data up into inputs day chunks and collect the outputs
    chunks = []
    outputs = []
    for i in range(len(daily) - inputs):
        chunks.append(daily[i:i+inputs])
        outputs.append(daily[i+inputs])

    #scale the data
    min_max = []
    for i in range(len(chunks)):
        chunks[i], min, max = scale_input(chunks[i], scale)
        outputs[i] = scale_output([outputs[i]], min, max, scale)[0]
        min_max.append([min, max])

    input = parse_input(chunks)
    output = parse_output(outputs)

    #print(outputs)

    total_days = len(input)
    train_cutoff = int(len(input) * 8 / 10)
    #print(output[:train_cutoff])
    if traning:
        print('Traning:')
        #for i in range(1000):
        #    print(forcaster.basic_train(input[:train_cutoff], output[:train_cutoff], input[train_cutoff:], output[train_cutoff:]))
        forcaster.train_to_minimum(1, input[:train_cutoff], output[:train_cutoff], input[train_cutoff:], output[train_cutoff:])
        forcaster.save()

    #calculate errors for price
    print('Calculating errors')
    price_error_paramaters = forcaster.get_error_for_metric(0, input[train_cutoff:], output[train_cutoff:])
    volume_error_paramaters = forcaster.get_error_for_metric(1, input[train_cutoff:], output[train_cutoff:])



    #forecast from the train train_cutoff
    price_store = []
    volume_store = []
    actual_price = []
    actual_volume = []
    predictions = []
    min_prices = []
    max_prices = []
    price_error_paramaters = {'mean': price_error_paramaters[0], 'variance': price_error_paramaters[1] - price_error_paramaters[1]**2 }
    volume_error_paramaters = {'mean': volume_error_paramaters[0], 'variance': volume_error_paramaters[1] - volume_error_paramaters[1]**2 }
    #mean = variance = 0
    start_day = -1100
    with open('predictions.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        for j in range(50):

            price_volume = daily[-inputs+start_day:start_day]
            #random.seed(_)
            price_store = []
            volume_store = []
            print(j)
            random.seed(j)
            for i in range(24):
            #    print('\t' + str(i))
                scaled, min, max = scale_input(price_volume, scale)

                scaled = parse_input([scaled])
                output = forcaster.predict(scaled)[0]
                #print(output)
                price = unscale([output[0] + random.gauss(-price_error_paramaters['mean'], price_error_paramaters['variance']**0.5)], min['price'], max['price'], scale)[0]
                volume = unscale([output[1] + random.gauss(-volume_error_paramaters['mean'], volume_error_paramaters['variance']**0.5)], min['volume'], max['volume'], scale)[0]
                if i + start_day < 0 and j == 0:
                    actual_price.append(daily[start_day+i]['price'])
                    actual_volume.append(daily[start_day+i]['volume'])

                #price_store.append(price)
                #volume_store.append(volume)
                #new_average = current_average * i + price

                price_store.append(price)

                output = {'price': price, 'volume': volume}
                price_volume.append(output)
                price_volume = price_volume[1:]

            predictions.append(price_store)

            #writer.writerow(price_store)
            #writer.writerow(volume_store)
        writer.writerow(actual_price)
    #    writer.writerow(average_price)
        #writer.writerow(min_prices)
        #writer.writerow(max_prices)
        #writer.writerow(actual_volume)

        #reshape the prediction array
        days = []
        for prediction in predictions:
            for i, day in enumerate(prediction):
                if len(days) < i + 1:
                    days.append([day])
                else:
                    days[i].append(day)

        #print(days)
        #calculate the average price
        average_prices = []
        for day in days:
            sum = 0
            for p in day:
                sum += p
            average_prices.append(p / len(day))

        #lower, median and upper quartile
        lower_quartile = []
        upper_quartile = []
        median = []
        for day in days:
            day.sort()
            lower_quartile.append(day[int(len(day) / 4)])
            upper_quartile.append(day[int(3 * len(day) / 4)])
            median.append(day[int(len(day) / 2)])

        writer.writerow(median)
        writer.writerow(lower_quartile)
        writer.writerow(upper_quartile)

        #graph.plot_graph([lower_quartile, upper_quartile, median])



if __name__ == '__main__':
    main()
