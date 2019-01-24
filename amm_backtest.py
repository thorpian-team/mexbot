# -*- coding: utf-8 -*-
import pandas as pd
from backtest import Backtest, BacktestReport, BacktestIteration
from hyperopt import hp
from indicator import *
from math import tanh, floor, ceil, log

def advanced_market_make_backtest(ohlcv):

    buy_last = 0
    sell_last = 0

    dev = stdev(ohlcv.close, 17)
    volma1 = sma(ohlcv.volume, 6)
    volma2 = sma(ohlcv.volume, 12)
    volimb = sma(ohlcv.volume_imbalance, 6)

    def flooring(n,q=100):
        return int(floor(n/q)*q)

    def ceilling(n,q=100):
        return int(ceil(n/q)*q)

    def yourlogic(O,H,L,C,n,position_size,position_avg_price,**others):
        nonlocal buy_last
        nonlocal sell_last
        orders = []

        if n<17:
            return []

        maxsize = 0.05
        buysize = sellsize = 0.05

        if volma1[n]>100 or volma2[n]>100 or ohlcv.volume[n]>150:
            orders.append((0, 0, 0, 'S'))
            orders.append((0, 0, 0, 'L'))
            if volimb[n] > 1:
                lot = min(maxsize - position_size, maxsize)
                if lot > 0:
                    orders.append((+1, 0, lot, 'Lf'))
            elif volimb[n] < -1:
                lot = min(maxsize + position_size, maxsize)
                if lot > 0:
                    orders.append((-1, 0, lot, 'Sf'))
            return orders

        spr = max(dev[n],110)
        buy = C-spr
        sell = C+spr
        buy = flooring(buy)
        sell = ceilling(sell)

        risk = 0.333
        if position_size>0:
            buy = buy - spr*risk
            sell = sell - spr*risk
            buy = flooring(buy)
            sell = ceilling(sell)
            if (C-position_avg_price)>spr*(1-risk):
                sell = C
        elif position_size<0:
            buy = buy + spr*risk
            sell = sell + spr*risk
            buy = flooring(buy)
            sell = ceilling(sell)
            if (position_avg_price-C)>spr*(1-risk):
                buy = C

        maxlots=1
        if position_size < maxsize and abs(buy_last-buy)>spr*0.111:
            orders.append((+1, buy, buysize/maxlots, 'L'+str(n%maxlots)))
            buy_last = buy

        if position_size > -maxsize and abs(sell_last-sell)>spr*0.111:
            orders.append((-1, sell, sellsize/maxlots, 'S'+str(n%maxlots)))
            sell_last = sell

        return orders


    return Backtest(**locals())

if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bffx_2019-01-23_5s.csv', index_col="exec_date", parse_dates=True)

    default_parameters = {
        'ohlcv': ohlcv,
    }

    hyperopt_parameters = {
    }

    best, report = BacktestIteration(advanced_market_make_backtest, default_parameters, hyperopt_parameters, 0)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
