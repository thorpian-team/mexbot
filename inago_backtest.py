# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
from backtest import Backtest, BacktestReport, BacktestIteration
from hyperopt import hp
from indicator import *
from functools import lru_cache

def inago_backtest(ohlcv, buyth, sellth, eventh):

    # 売り買い優劣でエントリー
    if 0:
        buy_entry = (ohlcv.buy_volume > ohlcv.sell_volume) & ((ohlcv.buy_volume - ohlcv.sell_volume).abs() > eventh)
        sell_entry = (ohlcv.buy_volume < ohlcv.sell_volume) & ((ohlcv.buy_volume - ohlcv.sell_volume).abs() > eventh)
        buy_exit = sell_entry
        sell_exit = buy_entry

    # 売り買いサイズがN期間ATRを超えたらエントリー
    if 1:
        buy_atr = atr(ohlcv.buy_volume, ohlcv.buy_volume, ohlcv.buy_volume, buyth).shift(1)
        sell_atr = atr(ohlcv.buy_volume, ohlcv.buy_volume, ohlcv.buy_volume, sellth).shift(1)
        buy_entry = change(ohlcv.buy_volume) > buy_atr
        sell_entry = change(ohlcv.sell_volume) > sell_atr
        buy_exit = sell_entry
        sell_exit = buy_entry

    # entry_exit = pd.DataFrame({'close':ohlcv.close,
    #     'buy_entry':buy_entry, 'buy_exit':buy_exit, 'sell_entry':sell_entry, 'sell_exit':sell_exit})
    # entry_exit.to_csv('entry_exit.csv')

    return Backtest(**locals())


if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bf28sep2018_1s.csv', index_col='timestamp', parse_dates=True, dtype = {'open':'float', 'close':'float', 'high':'float', 'low':'float'})

    default_parameters = {
        'ohlcv':ohlcv,
        'buyth':15,
        'sellth':5,
        'eventh':1,
    }

    hyperopt_parameters = {
        'buyth': hp.quniform('buyth', 1, 100, 1),
        'sellth': hp.quniform('sellth', 1, 100, 1),
        'eventh': hp.quniform('eventh', 1, 100, 1),
    }

    def maximize(r):
        return ((r.All.WinRatio * r.All.WinPct) + ((1 - r.All.WinRatio) * r.All.LossPct)) * r.All.Trades
        # return r.All.WinPct * r.All.WinRatio * r.All.WinTrades
        # return r.All.Profit

    best, report = BacktestIteration(inago_backtest, default_parameters, hyperopt_parameters, 0, maximize=maximize)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
