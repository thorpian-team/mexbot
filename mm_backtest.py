# -*- coding: utf-8 -*-
import pandas as pd
from backtest import Backtest, BacktestReport, BacktestIteration
from hyperopt import hp
from indicator import *

def mm_backtest(ohlcv,margin):

    ratio = ((rsi(ohlcv.imbalance, 21)/100) + (rsi(ohlcv.average, 21)/100))/2

    buy_exp_n = 0
    sell_exp_n = 0

    def yourlogic(O,H,L,C,n,position_size,position_avg_price):
        nonlocal buy_exp_n
        nonlocal sell_exp_n
        orders = []

        buy = C-margin#+offset[n]
        sell = C+margin#+offset[n]

        # if position_size > 0:
        #     if buy > (position_avg_price+margin*2):
        #         buy = (position_avg_price+margin*2)
        # if position_size < 0:
        #     if sell < (position_avg_price-margin*2):
        #         sell = (position_avg_price-margin*2)

        lot = 0.03
        buy_size = lot * ratio[n]
        sell_size = lot - buy_size

        if position_size < buy_size:
            orders.append((1, buy, buy_size-max(position_size,0)))
        if position_size > -sell_size:
            orders.append((-1, sell, sell_size+min(position_size,0)))

        # if buy_exp_n:
        #     buy_exp_n -= 1

        # if buy_exp_n == 0:
        #     if position_size <= 0.01:
        #         orders.append((1, C-margin, 0.01))
        #         buy_exp_n = 0

        # if sell_exp_n:
        #     sell_exp_n -= 1

        # if sell_exp_n == 0:
        #     if position_size >= 0.01:
        #         orders.append((-1, C+margin, 0.01))
        #         sell_exp_n = 0

        return orders

    return Backtest(**locals())

if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bffx_2018-12-6_1m.csv', index_col="exec_date", parse_dates=True)

    default_parameters = {
        'ohlcv': ohlcv,
        'margin':600,
    }

    hyperopt_parameters = {
        'margin': hp.quniform('margin', 50, 2000, 10),
    }

    best, report = BacktestIteration(mm_backtest, default_parameters, hyperopt_parameters, 0)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
