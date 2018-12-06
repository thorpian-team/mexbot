# -*- coding: utf-8 -*-
import pandas as pd
from backtest import Backtest, BacktestReport, BacktestIteration
from hyperopt import hp
from indicator import *

def logic_backtest(ohlcv,fastlen,slowlen,margin):

    # ma1 = sma(ohlcv.close,fastlen)
    # ma2 = sma(ohlcv.close,slowlen)
    # buy = crossover(ma1,ma2)
    # sell = crossunder(ma1,ma2)

    buy_exp_n = 0
    sell_exp_n = 0

    def yourlogic(O,H,L,C,n,PositionSize):
        nonlocal buy_exp_n
        nonlocal sell_exp_n
        orders = []

        if buy_exp_n:
            buy_exp_n -= 1

        if buy_exp_n == 0:
            if PositionSize <= 0.01:
                orders.append((1, C-margin, 0.01))
                buy_exp_n = fastlen

        if sell_exp_n:
            sell_exp_n -= 1

        if sell_exp_n == 0:
            if PositionSize >= 0.01:
                orders.append((-1, C+margin, 0.01))
                sell_exp_n = fastlen

        return orders

        # orders = []
        # lot = 1 if PositionSize == 0 else 2
        # if buy[n]:
        #     orders.append((1, 0, lot))
        # if sell[n]:
        #     orders.append((-1, 0, lot))
        # return orders

    return Backtest(**locals())

if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bffx_2018-12-3_1m.csv', index_col="exec_date", parse_dates=True)

    default_parameters = {
        'ohlcv': ohlcv,
        'fastlen':0,
        'slowlen':26,
        'margin':600,
    }

    hyperopt_parameters = {
        'fastlen': hp.quniform('fastlen', 1, 100, 1),
        #'slowlen': hp.quniform('slowlen', 1, 100, 1),
        'margin': hp.quniform('margin', 50, 2000, 10),
    }

    best, report = BacktestIteration(logic_backtest, default_parameters, hyperopt_parameters, 0)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
