# -*- coding: utf-8 -*-
import pandas as pd
from backtest import Backtest, BacktestReport, BacktestIteration
from hyperopt import hp
from indicator import *

def simple_market_make_backtest(ohlcv):

    def smm_logic1(O, H, L, C, n, position_size, **others):
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

    def smm_logic2(O, H, L, C, n, position_size, **others):
        orders = []
        pairs = [(0.03, 200), (0.02, 100), (0.01, 50)]
        maxsize = sum(p[0] for p in pairs)
        buymax = sellmax = position_size
        for pair in pairs:
            suffix = str(pair[1])
            buymax += pair[0]
            sellmax -= pair[0]
            if buymax < maxsize:
                orders.append((+1, C-pair[1], pair[0], 'L'+suffix))
            else:
                orders.append((0, 0, 0, 'L'+suffix))
            if sellmax > -maxsize:
                orders.append((-1, C+pair[1], pair[0], 'S'+suffix))
            else:
                orders.append((0, 0, 0, 'S'+suffix))
        return orders

    # yourlogic = smm_logic1
    yourlogic = smm_logic2

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
