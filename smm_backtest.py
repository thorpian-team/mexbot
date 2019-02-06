# -*- coding: utf-8 -*-
import pandas as pd
from backtest import Backtest, BacktestReport, BacktestIteration
from hyperopt import hp
from indicator import *

def simple_market_make_backtest(ohlcv):

    def smm_logic1(O, H, L, C, n, strategy):
        maxsize = 0.1
        buysize = sellsize = 0.1
        spr = ohlcv.stdev[n]*2.5
        mid = (C+H+L)/3
        buy = mid - spr/2
        sell = mid + spr/2
        if strategy.position_size < maxsize:
            strategy.order('L', 'buy', qty=buysize, limit=buy)
        else:
            strategy.cancel('L')
        if strategy.position_size > -maxsize:
            strategy.order('S', 'sell', qty=sellsize, limit=sell)
        else:
            strategy.cancel('S')

    def smm_logic2(O, H, L, C, n, strategy):
        orders = []
        pairs = [(0.04, 400, 3), (0.03, 200, 2), (0.02, 100, 1), (0.01, 50, 0)]
        maxsize = sum(p[0] for p in pairs)
        buymax = sellmax = strategy.position_size
        mid = (C+H+L)/3
        for pair in pairs:
            suffix = str(pair[2])
            if buymax+pair[0] <= maxsize:
                buymax += pair[0]
                strategy.order('L'+suffix, 'buy', qty=pair[0], limit=mid-pair[1])
            else:
                strategy.cancel('L'+suffix)
            if sellmax-pair[0] >= -maxsize:
                sellmax -= pair[0]
                strategy.order('S'+suffix, 'sell', qty=pair[0], limit=mid+pair[1])
            else:
                strategy.cancel('S'+suffix)
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
