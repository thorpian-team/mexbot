# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from numba import jit, b1, f8, i8, void
from utils import dotdict
from hyperopt import hp, tpe, Trials, fmin, rand, anneal
from collections import deque

# PythonでFXシストレのバックテスト(1)
# https://qiita.com/toyolab/items/e8292d2f051a88517cb2 より

@jit(f8(f8,f8,f8,f8,f8,f8,f8),nopython=True)
def buy_order(market, limit, stop, O, H, L, C):
    exec_price = 0
    # STOP注文
    if stop > 0 and H >= stop:
        if stop >= O:
            exec_price = stop
        else:
            exec_price = O
    # 指値注文
    elif limit > 0 and L <= limit:
        if limit > O:
            exec_price = H
        else:
            exec_price = limit
    # 成行注文
    elif market:
        exec_price = O
    # 注文執行
    return exec_price

@jit(f8(f8,f8,f8,f8,f8,f8,f8),nopython=True)
def buy_close(profit, stop, exec_price, O, H, L, C):
    close_price = 0
    if stop > 0:
        # 損切判定
        stop_price = exec_price - stop
        if L <= stop_price:
            close_price = stop_price
    if profit > 0:
        # 利確判定
        profit_price = exec_price + profit
        if H >= profit_price:
            close_price = profit_price
    return close_price

@jit(f8(f8,f8,f8,f8,f8,f8,f8),nopython=True)
def sell_order(market, limit, stop, O, H, L, C):
    exec_price = 0
    # STOP注文
    if stop > 0 and L <= stop:
        if stop <= O:
            exec_price = stop
        else:
            exec_price = O
    # 指値注文
    elif limit > 0 and H >= limit:
        if limit < O:
            exec_price = L
        else:
            exec_price = limit
    # 成行注文
    elif market:
        exec_price = O
    # 注文執行
    return exec_price

@jit(f8(f8,f8,f8,f8,f8,f8,f8),nopython=True)
def sell_close(profit, stop, exec_price, O, H, L, C):
    close_price = 0
    if stop > 0:
        # 損切判定
        stop_price = exec_price + stop
        if H >= stop_price:
            close_price = stop_price
    if profit > 0:
        # 利確判定
        profit_price = exec_price - profit
        if L <= profit_price:
            close_price = profit_price
    return close_price

@jit(f8(f8,f8,f8,f8),nopython=True)
def calclots(capital, price, percent, lot):
    if percent > 0:
        if capital > 0:
            return ((capital * percent) / price)
        else:
            return 0
    else:
        return lot

@jit(void(f8[:],f8[:],f8[:],f8[:],i8[:],i8,
    b1[:],b1[:],b1[:],b1[:],
    f8[:],f8[:],f8[:],f8[:],
    f8[:],f8[:],f8[:],f8[:],
    f8[:],f8[:],f8,f8,
    f8,f8,f8,f8,f8,f8,f8,i8,i8,f8,i8,
    f8[:],f8[:],f8[:],f8[:],f8[:],f8[:]), nopython=True)
def BacktestCore(Open, High, Low, Close, Trades, N,
    buy_entry, sell_entry, buy_exit, sell_exit,
    stop_buy_entry, stop_sell_entry, stop_buy_exit, stop_sell_exit,
    limit_buy_entry, limit_sell_entry, limit_buy_exit, limit_sell_exit,
    buy_size, sell_size, max_buy_size, max_sell_size,
    spread, take_profit, stop_loss, trailing_stop, slippage, percent, capital, trades_per_n, delay_n, max_drawdown, wait_n_for_mdd,
    LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct):

    buyExecPrice = sellExecPrice = 0.0 # 売買価格
    buyMarketEntry = buyMarketExit = sellMarketEntry = sellMarketExit = 0
    buyStopEntry = buyStopExit = sellStopEntry = sellStopExit = 0
    buyLimitEntry = buyLimitExit = sellLimitEntry = sellLimitExit = 0
    buyExecLot = sellExecLot = 0
    dd = max_profit = dd_wait = 0

    for i in range(delay_n, N):
        # O, H, L, C = Open[i], High[i], Low[i], Close[i]
        BuyNow = SellNow = False

        # ドローダウンが最大値を超えていたら一定時間取引停止
        EntryReject = dd_wait > 0
        if dd_wait > 0:
            dd_wait = dd_wait - 1

        # 約定数が規定値を超えていたら注文拒否
        OrderReject = Trades[i] > trades_per_n

        # 新規注文受付
        if not OrderReject and not EntryReject:
            # 買い
            buyMarketEntry = buy_entry[i-delay_n]
            buyLimitEntry = limit_buy_entry[i-delay_n]
            buyStopEntry = stop_buy_entry[i-delay_n]
            buyOpenSize = buy_size[i-delay_n]
            # 売り
            sellMarketEntry = sell_entry[i-delay_n]
            sellLimitEntry = limit_sell_entry[i-delay_n]
            sellStopEntry = stop_sell_entry[i-delay_n]
            sellOpenSize = sell_size[i-delay_n]

        # 決済注文受付
        if not OrderReject:
            # 買い決済
            buyMarketExit = buy_exit[i-delay_n]
            buyLimitExit = limit_buy_exit[i-delay_n]
            buyStopExit = stop_buy_exit[i-delay_n]
            buyCloseSize = buy_size[i-delay_n]
            # 売り決済
            sellMarketExit = sell_exit[i-delay_n]
            sellLimitExit = limit_sell_exit[i-delay_n]
            sellStopExit = stop_sell_exit[i-delay_n]
            sellCloseSize = sell_size[i-delay_n]

        # 買い注文処理
        if buyExecLot < max_buy_size:
            #OpenPrice = buy_order(buy_entry[i-delay_n],limit_buy_entry[i-delay_n],stop_buy_entry[i-delay_n],O,H,L,C)
            OpenPrice = 0
            # 指値注文
            if buyLimitEntry > 0 and Low[i] <= buyLimitEntry:
                OpenPrice = buyLimitEntry
                buyLimitEntry = 0
            # STOP注文
            if buyStopEntry > 0 and High[i] >= buyStopEntry:
                if Open[i] <= buyStopEntry:
                    OpenPrice = buyStopEntry
                else:
                    OpenPrice = Open[i]
                buyStopEntry = 0
            # 成行注文
            if buyMarketEntry > 0:
                OpenPrice = Open[i]
                buyMarketEntry = 0
            # 注文執行
            if OpenPrice > 0:
                execPrice = OpenPrice + spread + slippage
                LongTrade[i] = execPrice #買いポジションオープン
                execLot =  calclots(capital, OpenPrice, percent, buyOpenSize)
                buyExecPrice = ((execPrice*execLot)+(buyExecPrice*buyExecLot))/(buyExecLot+execLot)
                buyExecLot = buyExecLot + execLot
                BuyNow = True

        # 買い手仕舞い
        if buyExecLot > 0 and not BuyNow:
            # ClosePrice = sell_order(buy_exit[i-delay_n],limit_buy_exit[i-delay_n],stop_buy_exit[i-delay_n],O,H,L,C)
            ClosePrice = 0
            # 指値注文
            if buyLimitExit > 0 and High[i] >= buyLimitExit:
                ClosePrice = buyLimitExit
                buyLimitExit = 0
            # STOP注文
            if buyStopExit > 0 and Low[i] <= buyStopExit:
                if Open[i] >= buyStopExit:
                    ClosePrice = buyStopExit
                else:
                    ClosePrice = Open[i]
                buyStopExit = 0
            # 成行注文
            if buyMarketExit > 0:
                ClosePrice = Open[i]
                buyMarketExit = 0
            # 注文執行
            if ClosePrice > 0:
                if buyExecLot > buyCloseSize:
                    buy_exit_lot = buyCloseSize
                    buy_exec_price = buyExecPrice
                    buyExecLot = buyExecLot - buy_exit_lot
                else:
                    buy_exit_lot = buyExecLot
                    buy_exec_price = buyExecPrice
                    buyExecPrice = buyExecLot = 0
                ClosePrice = ClosePrice - slippage
                LongTrade[i] = ClosePrice #買いポジションクローズ
                LongPL[i] = (ClosePrice - buy_exec_price) * buy_exit_lot #損益確定
                LongPct[i] = LongPL[i] / buy_exec_price

        # 売り注文処理
        if sellExecLot < max_sell_size:
            #OpenPrice = sell_order(sell_entry[i-delay_n],limit_sell_entry[i-delay_n],stop_sell_entry[i-delay_n],O,H,L,C)
            OpenPrice = 0
            # 指値注文
            if sellLimitEntry > 0 and High[i] >= sellLimitEntry:
                OpenPrice = sellLimitEntry
                sellLimitEntry = 0
            # STOP注文
            if sellStopEntry > 0 and Low[i] <= sellStopEntry:
                if Open[i] >= sellStopEntry:
                    OpenPrice = sellStopEntry
                else:
                    OpenPrice = Open[i]
                sellStopEntry = 0
            # 成行注文
            if sellMarketEntry > 0:
                OpenPrice = Open[i]
                sellMarketEntry = 0
            # 注文執行
            if OpenPrice:
                execPrice = OpenPrice - slippage
                ShortTrade[i] = execPrice #売りポジションオープン
                execLot = calclots(capital,OpenPrice,percent,sellOpenSize)
                sellExecPrice = ((execPrice*execLot)+(sellExecPrice*sellExecLot))/(sellExecLot+execLot)
                sellExecLot = sellExecLot + execLot
                SellNow = True

        # 売り手仕舞い
        if sellExecLot > 0 and not SellNow:
            #ClosePrice = buy_order(sell_exit[i-delay_n],limit_sell_exit[i-delay_n],stop_sell_exit[i-delay_n],O,H,L,C)
            ClosePrice = 0
            # 指値注文
            if sellLimitExit > 0 and Low[i] <= sellLimitExit:
                ClosePrice = sellLimitExit
                sellLimitExit = 0
            # STOP注文
            if sellStopExit > 0 and High[i] >= sellStopExit:
                if Open[i] <= sellStopExit:
                    ClosePrice = sellStopExit
                else:
                    ClosePrice = Open[i]
                sellStopExit = 0
            # 成行注文
            if sellMarketExit > 0:
                ClosePrice = Open[i]
                sellMarketExit = 0
            # 注文執行
            if ClosePrice > 0:
                if sellExecLot > sellCloseSize:
                    sell_exit_lot = sellCloseSize
                    sell_exec_price = sellExecPrice
                    sellExecLot = sellExecLot - sell_exit_lot
                else:
                    sell_exit_lot = sellExecLot
                    sell_exec_price = sellExecPrice
                    sellExecPrice = sellExecLot = 0
                ClosePrice = ClosePrice + spread + slippage
                ShortTrade[i] = ClosePrice #売りポジションクローズ
                ShortPL[i] = (sell_exec_price - ClosePrice) * sell_exit_lot #損益確定
                ShortPct[i] = ShortPL[i] / sell_exec_price

        # 利確 or 損切によるポジションの決済(エントリーと同じ足で決済しない)
        if buyExecPrice > 0 and not BuyNow and not OrderReject:
            # ClosePrice = buy_close(take_profit,stop_loss,O,H,L,C)
            ClosePrice = 0
            if stop_loss > 0:
                # 損切判定
                StopPrice = buyExecPrice - stop_loss
                if Low[i] <= StopPrice:
                    ClosePrice = Close[i]
            if take_profit > 0:
                # 利確判定
                LimitPrice = buyExecPrice + take_profit
                if High[i] >= LimitPrice:
                    ClosePrice = Close[i]
            if ClosePrice > 0:
                ClosePrice = ClosePrice - slippage
                LongTrade[i] = ClosePrice #買いポジションクローズ
                LongPL[i] = (ClosePrice - buyExecPrice) * buyExecLot #損益確定
                LongPct[i] = LongPL[i] / buyExecPrice
                buyExecPrice = buyExecLot = 0

        if sellExecPrice > 0 and not SellNow and not OrderReject:
            # ClosePrice = sell_close(take_profit,stop_loss,O,H,L,C)
            ClosePrice = 0
            if stop_loss > 0:
                # 損切判定
                StopPrice = sellExecPrice + stop_loss
                if High[i] >= StopPrice:
                    ClosePrice = Close[i]
            if take_profit > 0:
                # 利確判定
                LimitPrice = sellExecPrice - take_profit
                if Low[i] <= LimitPrice:
                    ClosePrice = Close[i]
            if ClosePrice > 0:
                ClosePrice = ClosePrice + slippage
                ShortTrade[i] = ClosePrice #売りポジションクローズ
                ShortPL[i] = (sellExecPrice - ClosePrice) * sellExecLot #損益確定
                ShortPct[i] = ShortPL[i] / sellExecPrice
                sellExecPrice = sellExecLot = 0

        capital = capital + ShortPL[i] + LongPL[i]
        max_profit = max(capital, max_profit)
        dd = max_profit - capital
        if max_drawdown>0 and dd>max_drawdown:
            dd_wait = wait_n_for_mdd
            max_profit = capital

    # ポジションクローズ
    if buyExecPrice > 0:
        ClosePrice = Close[N-1]
        LongTrade[N-1] = ClosePrice #買いポジションクローズ
        LongPL[N-1] = (ClosePrice - buyExecPrice) * buyExecLot #損益確定
        LongPct[N-1] = LongPL[N-1] / buyExecPrice
    if sellExecPrice > 0:
        ClosePrice = Close[N-1]
        ShortTrade[N-1] = ClosePrice #売りポジションクローズ
        ShortPL[N-1] = (sellExecPrice - ClosePrice) * sellExecLot #損益確定
        ShortPct[N-1] = ShortPL[N-1] / sellExecPrice


def BacktestCore2(Open, High, Low, Close, Trades, N, YourLogic,
                  LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct, PositionSize,
                  delay_n, trades_per_n):

    positions = deque()
    position_size = 0
    position_avg_price = 0
    netprofit = 0
    remaining_orders = {}

    for i in range(delay_n, N):

        # 約定数が規定値を超えていたら注文拒否
        order_reject = Trades[i] > trades_per_n

        # 1つ前の足で注文作成
        if not order_reject:
            n = i-delay_n
            O, H, L, C = Open[n], High[n], Low[n], Close[n]
            orders = YourLogic(O,H,L,C,n,position_size=position_size,position_avg_price=position_avg_price,netprofit=netprofit)

            # 注文受付
            for o in orders:
                o_side, o_price, o_size, o_id = o
                if o_size>0:
                    remaining_orders[o_id] = o
                    # print(i, 'Open', o_id, o_side, o_price, o_size)
                else:
                    if o_id in remaining_orders:
                        # print(i, 'Cancel', o_id)
                        del remaining_orders[o_id]

        # 現在の足で約定
        O, H, L, C = Open[i], High[i], Low[i], Close[i]

        # 約定
        remain = {}
        for k,o in remaining_orders.items():
            o_side, o_price, o_size, o_id = o
            exec_price = 0

            if o_side > 0:
                exec_price = buy_order(o_price==0, o_price, 0, O, H, L, C)
            elif o_side < 0:
                exec_price = sell_order(o_price==0, o_price, 0, O, H, L, C)

            if exec_price > 0:
                positions.append([o_side, exec_price, o_size])
                # print(i, 'Exec', o_id, o_side, exec_price, o_size)
                if o_side > 0:
                    LongTrade[i] = exec_price
                else:
                    ShortTrade[i] = exec_price

                # 決済
                while len(positions)>=2:
                    l_side, l_price, l_size = positions.popleft()
                    r_side, r_price, r_size = positions.pop()
                    if l_side != r_side:
                        if l_size >= r_size:
                            pnl = (r_price - l_price) * (r_size * l_side)
                            c_size = r_size
                            l_size = round(l_size-r_size,8)
                            if l_size > 0:
                                positions.appendleft((l_side,l_price,l_size))
                        else:
                            pnl = (r_price - l_price) * (l_size * l_side)
                            c_size = l_size
                            r_size = round(r_size-l_size,8)
                            if r_size > 0:
                                positions.append((r_side,r_price,r_size))
                        # print(i, 'Close', l_side, l_price, c_size, r_price, pnl)
                        if l_side > 0:
                            LongPL[i] = LongPL[i] + pnl
                            # LongTrade[i] = r_price
                            LongPct[i] = LongPL[i] / r_price
                        else:
                            ShortPL[i] = ShortPL[i] + pnl
                            # ShortTrade[i] = r_price
                            ShortPct[i] = ShortPL[i] / r_price
                    else:
                        positions.appendleft((l_side,l_price,l_size))
                        positions.append((r_side,r_price,r_size))
                        break

                # ポジションサイズ計算
                pos = len(positions)
                if pos:
                    position_size = math.fsum(p[2]*p[0] for p in positions)
                    position_avg_price = math.fsum(p[1] for p in positions) / pos
                else:
                    position_size = position_avg_price = 0
                # print(i,'Pos',position_avg_price,position_size)
            else:
                remain[o_id] = o

        # ポジション情報保存
        PositionSize[i] = position_size

        # 残りの注文
        remaining_orders = remain

        # 合計損益
        netprofit = netprofit + LongPL[i] + ShortPL[i]

    # 残ポジションクローズ
    if len(positions):
        position_size = sum(p[2]*p[0] for p in positions)
        position_avg_price = sum(p[1] for p in positions)/len(positions)
        price = Close[N-1]
        pnl = (position_avg_price - price) * position_size * -1
        if position_size > 0:
            # print(N-1, 'Close', 1, position_avg_price, position_size, price, pnl)
            LongPL[i] = pnl
            LongTrade[i] = price
        elif position_size < 0:
            # print(N-1, 'Close', -1, position_avg_price, position_size, price, pnl)
            ShortPL[i] = pnl
            ShortTrade[i] = price


def Backtest(ohlcv,
    buy_entry=None, sell_entry=None, buy_exit=None, sell_exit=None,
    stop_buy_entry=None, stop_sell_entry=None, stop_buy_exit=None, stop_sell_exit=None,
    limit_buy_entry=None, limit_sell_entry=None, limit_buy_exit=None, limit_sell_exit=None,
    buy_size=1.0, sell_size=1.0, max_buy_size=1.0, max_sell_size=1.0,
    spread=0, take_profit=0, stop_loss=0, trailing_stop=0, slippage=0, percent_of_equity=0.0, initial_capital=0.0, trades_per_second = 0, delay_n = 0,
    max_drawdown=0, wait_seconds_for_mdd=0, yourlogic=None,
    **kwargs):
    Open = ohlcv.open.values #始値
    Low = ohlcv.low.values #安値
    High = ohlcv.high.values #高値
    Close = ohlcv.close.values #始値

    N = len(ohlcv) #データサイズ
    buyExecPrice = sellExecPrice = 0.0 # 売買価格
    buyStopEntry = buyStopExit = sellStopEntry = sellStopExit = 0
    buyExecLot = sellExecLot = 0

    LongTrade = np.zeros(N) # 買いトレード情報
    ShortTrade = np.zeros(N) # 売りトレード情報

    LongPL = np.zeros(N) # 買いポジションの損益
    ShortPL = np.zeros(N) # 売りポジションの損益

    LongPct = np.zeros(N) # 買いポジションの損益率
    ShortPct = np.zeros(N) # 売りポジションの損益率

    PositionSize = np.zeros(N) # ポジション情報

    place_holder = np.zeros(N) # プレースホルダ
    bool_place_holder = np.zeros(N, dtype=np.bool) # プレースホルダ
    if isinstance(buy_size, pd.Series):
        buy_size = buy_size.values
    else:
        buy_size = np.full(shape=(N), fill_value=float(buy_size))
    if isinstance(sell_size, pd.Series):
        sell_size = sell_size.values
    else:
        sell_size = np.full(shape=(N), fill_value=float(sell_size))

    buy_entry = bool_place_holder if buy_entry is None else buy_entry.values
    sell_entry = bool_place_holder if sell_entry is None else sell_entry.values
    buy_exit = bool_place_holder if buy_exit is None else buy_exit.values
    sell_exit = bool_place_holder if sell_exit is None else sell_exit.values

    # トレーリングストップ価格を設定(STOP注文として処理する)
    if trailing_stop > 0:
        stop_buy_exit = ohlcv.high - trailing_stop
        stop_sell_exit = ohlcv.low + trailing_stop

    stop_buy_entry = place_holder if stop_buy_entry is None else stop_buy_entry.values
    stop_sell_entry = place_holder if stop_sell_entry is None else stop_sell_entry.values
    stop_buy_exit = place_holder if stop_buy_exit is None else stop_buy_exit.values
    stop_sell_exit = place_holder if stop_sell_exit is None else stop_sell_exit.values

    limit_buy_entry = place_holder if limit_buy_entry is None else limit_buy_entry.values
    limit_sell_entry = place_holder if limit_sell_entry is None else limit_sell_entry.values
    limit_buy_exit = place_holder if limit_buy_exit is None else limit_buy_exit.values
    limit_sell_exit = place_holder if limit_sell_exit is None else limit_sell_exit.values

    # 約定数
    Trades = place_holder
    trades_per_n = trades_per_second * (ohlcv.index[1] - ohlcv.index[0]).total_seconds()
    if trades_per_n:
        if 'trades' in ohlcv:
            Trades = ohlcv.trades.values

    # ドローダウン時の待ち時間
    wait_n_for_mdd = math.ceil(wait_seconds_for_mdd / (ohlcv.index[1] - ohlcv.index[0]).total_seconds())

    percent = percent_of_equity
    capital = initial_capital

    if yourlogic:
        BacktestCore2(Open.astype(float), High.astype(float), Low.astype(float), Close.astype(float), Trades.astype(int), N, yourlogic,
        LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct, PositionSize,
        int(delay_n+1), int(trades_per_n))
    else:
        BacktestCore(Open.astype(float), High.astype(float), Low.astype(float), Close.astype(float), Trades.astype(int), N,
            buy_entry, sell_entry, buy_exit, sell_exit,
            stop_buy_entry, stop_sell_entry, stop_buy_exit, stop_sell_exit,
            limit_buy_entry, limit_sell_entry, limit_buy_exit, limit_sell_exit,
            buy_size, sell_size, max_buy_size, max_sell_size,
            float(spread), float(take_profit), float(stop_loss), float(trailing_stop), float(slippage), float(percent), float(capital), int(trades_per_n), int(delay_n+1),
            float(max_drawdown), int(wait_n_for_mdd),
            LongTrade, LongPL, LongPct, ShortTrade, ShortPL, ShortPct)

    return BacktestReport(pd.DataFrame({
        'LongTrade':LongTrade, 'ShortTrade':ShortTrade,
        'LongPL':LongPL, 'ShortPL':ShortPL,
        'LongPct':LongPct, 'ShortPct':ShortPct,
        'PositionSize':PositionSize,
        }, index=ohlcv.index))


class BacktestReport:
    def __init__(self, DataFrame):
        self.DataFrame = DataFrame

        # ロング統計
        LongPL = DataFrame['LongPL']
        self.Long = dotdict()
        self.Long.PL = LongPL
        self.Long.Pct = DataFrame['LongPct']
        self.Long.Trades = np.count_nonzero(LongPL)
        if self.Long.Trades > 0:
            self.Long.GrossProfit = LongPL.clip_lower(0).sum()
            self.Long.GrossLoss =  LongPL.clip_upper(0).sum()
            self.Long.Profit = self.Long.GrossProfit + self.Long.GrossLoss
            self.Long.AvgReturn = self.Long.Pct[self.Long.Pct!=0].mean()
        else:
            self.Long.GrossProfit = 0.0
            self.Long.GrossLoss = 0.0
            self.Long.Profit = 0.0
            self.Long.AvgReturn = 0.0
        self.Long.WinTrades = np.count_nonzero(LongPL.clip_lower(0))
        if self.Long.WinTrades > 0:
            self.Long.WinMax = LongPL.max()
            self.Long.WinAverage = self.Long.GrossProfit / self.Long.WinTrades
            self.Long.WinPct = self.Long.Pct[self.Long.Pct > 0].mean()
            self.Long.WinRatio = self.Long.WinTrades / self.Long.Trades
        else:
            self.Long.WinMax = 0.0
            self.Long.WinAverage = 0.0
            self.Long.WinPct = 0.0
            self.Long.WinRatio = 0.0
        self.Long.LossTrades = np.count_nonzero(LongPL.clip_upper(0))
        if self.Long.LossTrades > 0:
            self.Long.LossMax = LongPL.min()
            self.Long.LossAverage = self.Long.GrossLoss / self.Long.LossTrades
            self.Long.LossPct = self.Long.Pct[self.Long.Pct < 0].mean()
        else:
            self.Long.LossMax = 0.0
            self.Long.LossAverage = 0.0
            self.Long.LossPct = 0.0

        # ショート統計
        ShortPL = DataFrame['ShortPL']
        self.Short = dotdict()
        self.Short.PL = ShortPL
        self.Short.Pct = DataFrame['ShortPct']
        self.Short.Trades = np.count_nonzero(ShortPL)
        if self.Short.Trades > 0:
            self.Short.GrossProfit = ShortPL.clip_lower(0).sum()
            self.Short.GrossLoss = ShortPL.clip_upper(0).sum()
            self.Short.Profit = self.Short.GrossProfit + self.Short.GrossLoss
            self.Short.AvgReturn = self.Short.Pct[self.Short.Pct!=0].mean()
        else:
            self.Short.GrossProfit = 0.0
            self.Short.GrossLoss = 0.0
            self.Short.Profit = 0.0
            self.Short.AvgReturn = 0.0
        self.Short.WinTrades = np.count_nonzero(ShortPL.clip_lower(0))
        if self.Short.WinTrades > 0:
            self.Short.WinMax = ShortPL.max()
            self.Short.WinAverage = self.Short.GrossProfit / self.Short.WinTrades
            self.Short.WinPct = self.Short.Pct[self.Short.Pct > 0].mean()
            self.Short.WinRatio = self.Short.WinTrades / self.Short.Trades
        else:
            self.Short.WinMax = 0.0
            self.Short.WinAverage = 0.0
            self.Short.WinPct = 0.0
            self.Short.WinRatio = 0.0
        self.Short.LossTrades = np.count_nonzero(ShortPL.clip_upper(0))
        if self.Short.LossTrades > 0:
            self.Short.LossMax = ShortPL.min()
            self.Short.LossAverage = self.Short.GrossLoss / self.Short.LossTrades
            self.Short.LossPct = self.Short.Pct[self.Short.Pct < 0].mean()
        else:
            self.Short.LossMax = 0.0
            self.Short.LossTrades = 0.0
            self.Short.LossPct = 0.0

        # 資産
        self.Equity = (LongPL + ShortPL).cumsum()

        # 全体統計
        self.All = dotdict()
        self.All.Trades = self.Long.Trades + self.Short.Trades
        self.All.WinTrades = self.Long.WinTrades + self.Short.WinTrades
        self.All.WinPct = (self.Long.WinPct + self.Short.WinPct) / 2
        self.All.WinRatio = self.All.WinTrades / self.All.Trades if self.All.Trades > 0 else 0.0
        self.All.LossTrades = self.Long.LossTrades + self.Short.LossTrades
        self.All.GrossProfit = self.Long.GrossProfit + self.Short.GrossProfit
        self.All.GrossLoss = self.Long.GrossLoss + self.Short.GrossLoss
        self.All.WinAverage = self.All.GrossProfit / self.All.WinTrades if self.All.WinTrades > 0 else 0
        self.All.LossPct = (self.Long.LossPct + self.Short.LossPct) / 2
        self.All.LossAverage = self.All.GrossLoss / self.All.LossTrades if self.All.LossTrades > 0 else 0
        self.All.Profit = self.All.GrossProfit + self.All.GrossLoss
        self.All.AvgReturn = (self.Long.AvgReturn + self.Short.AvgReturn) / 2
        self.All.DrawDown = (self.Equity.cummax() - self.Equity).max()
        self.All.ProfitFactor = self.All.GrossProfit / -self.All.GrossLoss if -self.All.GrossLoss > 0 else 0
        if self.All.Trades > 1:
            pct = pd.concat([self.Long.Pct, self.Short.Pct])
            pct = pct[pct > 0]
            self.All.SharpeRatio = pct.mean() / pct.std()
        else:
            self.All.SharpeRatio = 1.0
        self.All.RecoveryFactor = self.All.ProfitFactor / self.All.DrawDown if self.All.DrawDown > 0 else 0
        self.All.ExpectedProfit = (self.All.WinAverage * self.All.WinRatio) + ((1 - self.All.WinRatio) * self.All.LossAverage)
        self.All.ExpectedValue = (self.All.WinRatio * (self.All.WinAverage / abs(self.All.LossAverage))) - (1 - self.All.WinRatio) if self.All.LossAverage < 0 else 1


    def __str__(self):
        return 'Long\n' \
        '  Trades :' + str(self.Long.Trades) + '\n' \
        '  WinTrades :' + str(self.Long.WinTrades) + '\n' \
        '  WinMax :' + str(self.Long.WinMax) + '\n' \
        '  WinAverage :' + str(self.Long.WinAverage) + '\n' \
        '  WinPct :' + str(self.Long.WinPct) + '\n' \
        '  WinRatio :' + str(self.Long.WinRatio) + '\n' \
        '  LossTrades :' + str(self.Long.LossTrades) + '\n' \
        '  LossMax :' + str(self.Long.LossMax) + '\n' \
        '  LossAverage :' + str(self.Long.LossAverage) + '\n' \
        '  LossPct :' + str(self.Long.LossPct) + '\n' \
        '  GrossProfit :' + str(self.Long.GrossProfit) + '\n' \
        '  GrossLoss :' + str(self.Long.GrossLoss) + '\n' \
        '  Profit :' + str(self.Long.Profit) + '\n' \
        '  AvgReturn :' + str(self.Long.AvgReturn) + '\n' \
        '\nShort\n' \
        '  Trades :' + str(self.Short.Trades) + '\n' \
        '  WinTrades :' + str(self.Short.WinTrades) + '\n' \
        '  WinMax :' + str(self.Short.WinMax) + '\n' \
        '  WinAverage :' + str(self.Short.WinAverage) + '\n' \
        '  WinPct :' + str(self.Short.WinPct) + '\n' \
        '  WinRatio :' + str(self.Short.WinRatio) + '\n' \
        '  LossTrades :' + str(self.Short.LossTrades) + '\n' \
        '  LossMax :' + str(self.Short.LossMax) + '\n' \
        '  LossAverage :' + str(self.Short.LossAverage) + '\n' \
        '  LossPct :' + str(self.Short.LossPct) + '\n' \
        '  GrossProfit :' + str(self.Short.GrossProfit) + '\n' \
        '  GrossLoss :' + str(self.Short.GrossLoss) + '\n' \
        '  Profit :' + str(self.Short.Profit) + '\n' \
        '  AvgReturn :' + str(self.Short.AvgReturn) + '\n' \
        '\nAll\n' \
        '  Trades :' + str(self.All.Trades) + '\n' \
        '  WinTrades :' + str(self.All.WinTrades) + '\n' \
        '  WinAverage :' + str(self.All.WinAverage) + '\n' \
        '  WinPct :' + str(self.All.WinPct) + '\n' \
        '  WinRatio :' + str(self.All.WinRatio) + '\n' \
        '  LossTrades :' + str(self.All.LossTrades) + '\n' \
        '  LossAverage :' + str(self.All.LossAverage) + '\n' \
        '  LossPct :' + str(self.All.LossPct) + '\n' \
        '  GrossProfit :' + str(self.All.GrossProfit) + '\n' \
        '  GrossLoss :' + str(self.All.GrossLoss) + '\n' \
        '  Profit :' + str(self.All.Profit) + '\n' \
        '  AvgReturn :' + str(self.All.AvgReturn) + '\n' \
        '  DrawDown :' + str(self.All.DrawDown) + '\n' \
        '  ProfitFactor :' + str(self.All.ProfitFactor) + '\n' \
        '  SharpeRatio :' + str(self.All.SharpeRatio) + '\n'

# 参考
# https://qiita.com/kenchin110100/items/ac3edb480d789481f134

def BacktestIteration(testfunc, default_parameters, hyperopt_parameters, max_evals, maximize=lambda r:r.All.Profit):

    needs_header = [True]

    def go(args):
        params = default_parameters.copy()
        params.update(args)
        report = testfunc(**params)
        if 'ohlcv' in params:
            del params['ohlcv']
        if 'ticks' in params:
            del params['ticks']
        params.update(report.All)
        if needs_header[0]:
            print(','.join(params.keys()))
        print(','.join([str(x) for x in params.values()]))
        needs_header[0] = False
        return report

    if max_evals > 0:
        # 試行の過程を記録するインスタンス
        trials = Trials()

        best = fmin(
            # 最小化する値を定義した関数
            lambda args: -1 * maximize(go(args)),
            # 探索するパラメータのdictもしくはlist
            hyperopt_parameters,
            # どのロジックを利用するか、基本的にはtpe.suggestでok
            # rand.suggest ランダム・サーチ？
            # anneal.suggest 焼きなましっぽい
            algo=tpe.suggest,
            #algo=rand.suggest,
            #algo=anneal.suggest,
            max_evals=max_evals,
            trials=trials,
            # 試行の過程を出力
            verbose=0
        )
    else:
        best = default_parameters

    params = default_parameters.copy()
    params.update(best)
    report = go(params)
    print(report)
    return (params, report)


if __name__ == '__main__':

    from utils import stop_watch

    ohlcv = pd.read_csv('csv/bitmex_2018_1m.csv', index_col='timestamp', parse_dates=True)
    buy_entry = ohlcv.close > ohlcv.close.shift(1)
    sell_entry = ohlcv.close < ohlcv.close.shift(1)
    buy_exit = sell_entry
    sell_exit = buy_entry
    Backtest = stop_watch(Backtest)

    Backtest(**locals())
    Backtest(**locals())
    Backtest(**locals())
    Backtest(**locals())
