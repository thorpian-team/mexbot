import pandas as pd
import numpy as np
from backtest import Backtest, BacktestReport, BacktestIteration
from hyperopt import hp
from indicator import *

def eth_macross_backtest(ohlcv, fastperiod, slowperiod):

    fst = sma(change(ohlcv.eth_close), fastperiod)
    slw = sma(change(ohlcv.eth_close), slowperiod)
    # fst = sma(ohlcv.eth_close, fastperiod)
    # slw = sma(ohlcv.eth_close, slowperiod)

    up = (fst>slw)
    down = (fst<slw)

    limit_buy_entry = ohlcv.close.copy()
    limit_buy_entry[down] = 0
    limit_buy_exit = ohlcv.close.copy()
    limit_buy_exit[up] = 0

    limit_sell_entry = ohlcv.close.copy()
    limit_sell_entry[up] = 0
    limit_sell_exit = ohlcv.close.copy()
    limit_sell_exit[down] = 0

    return Backtest(**locals())

if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bitmex_2019_5m.csv', index_col='timestamp', parse_dates=True)

    # 指標データ読み込み
    ethohlcv = pd.read_csv('csv/bitmex_2019_ethusd_5m.csv', index_col='timestamp', parse_dates=True)
    ohlcv = ohlcv.assign(eth_close=ethohlcv.close)

    default_parameters = {
        'ohlcv':ohlcv,
        'fastperiod':9,
        'slowperiod':23,
    }

    hyperopt_parameters = {
        'fastperiod': hp.quniform('fastperiod', 1, 200, 1),
        'slowperiod': hp.quniform('slowperiod', 1, 200, 1),
    }

    best, report = BacktestIteration(eth_macross_backtest, default_parameters, hyperopt_parameters, 300)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
