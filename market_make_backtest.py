# -*- coding: utf-8 -*-
import pandas as pd
from backtest import Backtest, BacktestWithTickData, BacktestReport, BacktestIteration
from hyperopt import hp
from indicator import *
from functools import lru_cache


def market_make_backtest(ohlcv, period, margin, multi):

    buy_entry = None
    sell_entry = None
    buy_exit = None
    sell_exit = None

    limit_buy_entry = None
    limit_sell_entry = None
    limit_buy_exit = None
    limit_sell_exit = None

    # シンプルハイロー
    # range_high = ohlcv.high
    # range_low = ohlcv.low
    # limit_buy_entry = range_low
    # limit_buy_exit = range_high
    # limit_sell_entry = range_high
    # limit_sell_exit = range_low

    # 売り買い優劣でエントリー価格を調整
    # buy_volume = sma(ohlcv.buy_volume, 3)
    # sell_volume = sma(ohlcv.sell_volume, 3)
    # limit_buy_entry = limit_buy_entry - (buy_volume < sell_volume) * 50
    # limit_buy_exit = limit_buy_exit - (buy_volume < sell_volume) * 50
    # limit_sell_entry = limit_sell_entry + (buy_volume > sell_volume) * 50
    # limit_sell_exit = limit_sell_exit + (buy_volume > sell_volume) * 50

    # 売り買い優劣でエントリー1
    # buy_volume = sma(ohlcv.buy_volume, period)
    # sell_volume = sma(ohlcv.sell_volume, period)
    # power = buy_volume - sell_volume
    # buy_entry = power > 0
    # sell_entry = power < 0
    # buy_exit = sell_entry
    # sell_exit = buy_entry

    # 売り買い優劣でエントリー2
    # buy_volume = rci(ohlcv.buy_volume, 20)
    # sell_volume = rci(ohlcv.sell_volume, 20)
    # buy_entry = (buy_volume > 0) & (sell_volume < 0)
    # sell_entry = (buy_volume < 0) & (sell_volume > 0)
    # buy_exit = sell_entry
    # sell_exit = buy_entry

    # 売り買い優劣でエントリー3
    # buy_volume = rsi(ohlcv.buy_volume, period)
    # sell_volume = rsi(ohlcv.sell_volume, period)
    # power = buy_volume - sell_volume
    # buy_entry = power > 0
    # sell_entry = power < 0
    # buy_exit = sell_entry
    # sell_exit = buy_entry

    # 売り買い優劣でエントリー4
    # plus_minus_sum = ohlcv.plus_minus.rolling(int(period)).sum()
    # plus_minus_sum = sma(plus_minus_sum, 3)
    # buy_entry = change(plus_minus_sum) > 0
    # sell_entry = change(plus_minus_sum) < 0
    # buy_exit = sell_entry
    # sell_exit = buy_entry

    # 売り買い優劣でエントリー5
    # pm = ohlcv.plus_minus.cumsum()
    # buy_entry = change(pm) > 0
    # sell_entry = change(pm) < 0
    # buy_exit = change(pm) < 0
    # sell_exit = change(pm) > 0

    # 売り買い優劣でエントリー6
    # sv = ohlcv.signed_volume.cumsum()
    # buy_entry = change(sv) > 0
    # sell_entry = change(sv) < -0
    # buy_exit = sell_entry
    # sell_exit = buy_entry

    # 売り買い優劣でエントリー7
    # sv = ohlcv.signed_volume
    # buy_entry = sv > 0
    # sell_entry = sv < 0
    # buy_exit = sell_entry
    # sell_exit = buy_entry

    # 売り買い優劣でエントリー8
    # buy_atr = atr(ohlcv.buy_volume, ohlcv.buy_volume, ohlcv.buy_volume, 20).shift(1)
    # sell_atr = atr(ohlcv.buy_volume, ohlcv.buy_volume, ohlcv.buy_volume, 5).shift(1)
    # buy_entry = change(ohlcv.buy_volume) > buy_atr
    # sell_entry = change(ohlcv.sell_volume) > sell_atr
    # buy_exit = sell_entry
    # sell_exit = buy_entry

    # 売り買い優劣でエントリー価格を調整2
    # buy_volume = rsi(ohlcv.buy_volume, 10)
    # sell_volume = rsi(ohlcv.sell_volume, 10)
    # limit_buy_entry = limit_buy_entry - (buy_volume < sell_volume) * 50
    # limit_buy_exit = limit_buy_exit - (buy_volume < sell_volume) * 50
    # limit_sell_entry = limit_sell_entry + (buy_volume > sell_volume) * 50
    # limit_sell_exit = limit_sell_exit + (buy_volume > sell_volume) * 50

    # ボリュームでエントリー価格を調整
    # vol_sum = ohlcv.volume.rolling(int(period)).sum()
    # vol_avg = sma(ohlcv.volume, period)
    # price_range = ((vol_sum + vol_avg) * 1.5) + 20
    # limit_buy_entry = ohlcv.close - (price_range / 2)
    # limit_sell_entry = ohlcv.close + (price_range / 2)
    # limit_buy_exit = ohlcv.high
    # limit_sell_exit = ohlcv.low

    # 高値安値でエントリー
    # range_high = highest(ohlcv.high, period)
    # range_low = lowest(ohlcv.low, period)
    # limit_buy_entry = range_low
    # limit_buy_exit = range_high
    # limit_sell_entry = range_high
    # limit_sell_exit = range_low

    # ATRでエントリー
    # vatr = atr(ohlcv.close, ohlcv.high, ohlcv.low, period) * multi
    # limit_buy_entry = ohlcv.close - vatr
    # limit_sell_entry = ohlcv.close + vatr
    # limit_buy_exit = limit_sell_entry
    # limit_sell_exit = limit_buy_entry

    # limit_buy_entry[(high - low)<100] = 0
    # limit_sell_entry[(high - low)<100] = 0

    # 安く買って高く売る
    limit_buy_entry = ohlcv.low - margin
    limit_buy_exit = ohlcv.high + margin
    limit_sell_entry = ohlcv.high + margin
    limit_sell_exit = ohlcv.low - margin

    # バックテスト実施
    # entry_exit = pd.DataFrame({'close':ohlcv['close'], 'high':ohlcv['high'], 'low':ohlcv['low'],# 'plus_minus_sum':plus_minus_sum, "corr_plus_minus_sum":corr_plus_minus_sum,
    #     'limit_buy_entry':limit_buy_entry, 'limit_buy_exit':limit_buy_exit, 'limit_sell_entry':limit_sell_entry, 'limit_sell_exit':limit_sell_exit,
    #     'buy_entry':buy_entry, 'buy_exit':buy_exit, 'sell_entry':sell_entry, 'sell_exit':limit_sell_exit,
    #     })
    # entry_exit.to_csv('entry_exit.csv')

    # buy_size = sell_size = 0.001
    # max_buy_size = max_sell_size = 0.001

    return Backtest(**locals())
    # return BacktestWithTickData(**locals())


if __name__ == '__main__':

    # テストデータ読み込み
    ohlcv = pd.read_csv('csv/bf28sep2018_1s.csv', index_col="timestamp", parse_dates=True)
    ohlcv = ohlcv['2018-7-29':]
    # ohlcv = ohlcv[:3000]
    # ticks = pd.read_csv('csv/bffx_20180628_executions.csv', names=['id', 'side', 'price', 'size', 'exec_date'], index_col="exec_date", parse_dates=True)
    #ticks = ticks[:100000]

    default_parameters = {
        # 'ticks': ticks,
        'ohlcv': ohlcv,
        'period': 200,
        'margin': 800,
        'multi': 3,
    }

    hyperopt_parameters = {
        'period': hp.quniform('period', 1, 1000, 1),
        'margin': hp.uniform('margin', 10, 10000),
        'multi': hp.uniform('multi', 0.1, 5.0),
    }

    def maximize(r):
        # return ((r.All.WinRatio * r.All.WinPct) + ((1 - r.All.WinRatio) * r.All.LossPct)) * r.All.Trades
        # return r.All.WinPct * r.All.WinRatio * r.All.WinTrades
        return r.All.Profit

    best, report = BacktestIteration(market_make_backtest, default_parameters, hyperopt_parameters, 0, maximize=maximize)
    report.DataFrame.to_csv('TradeData.csv')
    report.Equity.to_csv('Equity.csv')
