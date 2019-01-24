# -*- coding: utf-8 -*-
import pandas as pd
from backtest import Backtest, BacktestReport, BacktestIteration
from hyperopt import hp
from indicator import *

def simple_market_make_backtest(ohlcv):

    def yourlogic(O, H, L, C, n, position_size, **others):
        orders = []

        maxsize = 0.05
        buysize = sellsize = 0.025
        spr = C * 0.00225
        buy = C - spr/2
        sell = C + spr/2

        if position_size < maxsize:
            orders.append((+1, buy, buysize, 'L'))
        else:
            orders.append((0, 0, 0, 'L'))

        if position_size > -maxsize:
            orders.append((-1, sell, sellsize, 'S'))
        else:
            orders.append((0, 0, 0, 'S'))

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

    best, report = BacktestIteration(simple_market_make_backtest, default_parameters, hyperopt_parameters, 0)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
