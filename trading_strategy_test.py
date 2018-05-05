import rnn

import csv, random

from functions import *

#get the price and volume data from file

def main():
    scale = 3
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

    #calculate errors for price
    print('Calculating errors')
    price_error_paramaters = forcaster.get_error_for_metric(0, input[train_cutoff:], output[train_cutoff:])
    volume_error_paramaters = forcaster.get_error_for_metric(1, input[train_cutoff:], output[train_cutoff:])



    #forecast from the train train_cutoff
    price_error_paramaters = {'mean': price_error_paramaters[0], 'variance': price_error_paramaters[1] - price_error_paramaters[1]**2 }
    volume_error_paramaters = {'mean': volume_error_paramaters[0], 'variance': volume_error_paramaters[1] - volume_error_paramaters[1]**2 }
    #mean = variance = 0

    USD = 1000
    BTC = 0

    currency = 0 # 0 for USD 1 for BTC
    hold_btc = USD / daily[-train_cutoff]['price']

    for i in range(-train_cutoff, -inputs, 24):
        print(i)
        price_volume = daily[i:i+inputs]
        start_price = price_volume[0]['price']
        higher = 0
        for j in range(20):
            for k in range(24):
                scaled, min, max = scale_input(price_volume, scale)
                scaled = parse_input([scaled])
                output = forcaster.predict(scaled)[0]

                price = unscale([output[0] + random.gauss(-price_error_paramaters['mean'], price_error_paramaters['variance']**0.5)], min['price'], max['price'], scale)[0]
                volume = unscale([output[1] + random.gauss(-volume_error_paramaters['mean'], volume_error_paramaters['variance']**0.5)], min['volume'], max['volume'], scale)[0]

                output = {'price': price, 'volume': volume}

                #price_volume.append(price)
                price_volume.append(output)
                price_volume = price_volume[1:]
            if price_volume[-1]['price'] > start_price:
                higher = higher + 1
        if higher > 10 and BTC == 0:
            BTC = USD / start_price
            USD = 0
        elif higher < 10 and USD == 0:
            USD = BTC * start_price
            BTC = 0
        total = USD + BTC * daily[i]['price']
        print('Total Value: ' + str(total) + 'Hold btc value: ' +str(hold_btc * daily[i]['price']))

if __name__ == '__main__':
    main()
