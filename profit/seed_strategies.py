"""
Seed strategies for ProFiT initialization.
These are valid Python code strings defining strategies compatible with backtesting.py.
"""

BOLLINGER_STRATEGY = """
from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np

class BollingerMeanReversion(Strategy):
    n = 20
    n_std = 2

    def init(self):
        self.sma = self.I(lambda x: pd.Series(x).rolling(self.n).mean(), self.data.Close)
        self.std = self.I(lambda x: pd.Series(x).rolling(self.n).std(), self.data.Close)
        self.upper = self.I(lambda s, d: s + self.n_std * d, self.sma, self.std)
        self.lower = self.I(lambda s, d: s - self.n_std * d, self.sma, self.std)

    def next(self):
        if self.data.Close[-1] < self.lower[-1]:
            if not self.position.is_long:
                if self.position.is_short:
                    self.position.close()
                self.buy()
        elif self.data.Close[-1] > self.upper[-1]:
            if not self.position.is_short:
                if self.position.is_long:
                    self.position.close()
                self.sell()
"""

CCI_STRATEGY = """
from backtesting import Strategy
import pandas as pd
import numpy as np

def CCI(high, low, close, n=20, c=0.015):
    tp = (high + low + close) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - sma) / (c * mad)

class CCIStrategy(Strategy):
    n = 20
    c = 0.015
    limit = 100

    def init(self):
        self.cci = self.I(CCI, 
                          pd.Series(self.data.High), 
                          pd.Series(self.data.Low), 
                          pd.Series(self.data.Close), 
                          self.n, self.c)

    def next(self):
        if self.cci[-1] > self.limit:
            if not self.position.is_long:
                if self.position.is_short:
                    self.position.close()
                self.buy()
        elif self.cci[-1] < -self.limit:
            if not self.position.is_short:
                if self.position.is_long:
                    self.position.close()
                self.sell()
"""

EMA_CROSSOVER_STRATEGY = """
from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd

class EMACrossover(Strategy):
    fast = 10
    slow = 20

    def init(self):
        self.ema_fast = self.I(lambda x: pd.Series(x).ewm(span=self.fast).mean(), self.data.Close)
        self.ema_slow = self.I(lambda x: pd.Series(x).ewm(span=self.slow).mean(), self.data.Close)

    def next(self):
        if crossover(self.ema_fast, self.ema_slow):
            if not self.position.is_long:
                if self.position.is_short:
                    self.position.close()
                self.buy()
        elif crossover(self.ema_slow, self.ema_fast):
            if not self.position.is_short:
                if self.position.is_long:
                    self.position.close()
                self.sell()
"""

MACD_STRATEGY = """
from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd

class MACDStrategy(Strategy):
    fast = 12
    slow = 26
    signal = 9

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema_fast = self.I(lambda x: x.ewm(span=self.fast).mean(), close)
        self.ema_slow = self.I(lambda x: x.ewm(span=self.slow).mean(), close)
        self.macd = self.I(lambda a, b: a - b, self.ema_fast, self.ema_slow)
        self.signal_line = self.I(lambda x: x.ewm(span=self.signal).mean(), pd.Series(self.macd))

    def next(self):
        if crossover(self.macd, self.signal_line):
            if not self.position.is_long:
                if self.position.is_short:
                    self.position.close()
                self.buy()
        elif crossover(self.signal_line, self.macd):
            if not self.position.is_short:
                if self.position.is_long:
                    self.position.close()
                self.sell()
"""

WILLIAMS_R_STRATEGY = """
from backtesting import Strategy
import pandas as pd
import numpy as np

def WilliamsR(high, low, close, n=14):
    highest_high = high.rolling(n).max()
    lowest_low = low.rolling(n).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

class WilliamsRStrategy(Strategy):
    n = 14
    upper = -20
    lower = -80

    def init(self):
        self.williams_r = self.I(WilliamsR, 
                                 pd.Series(self.data.High), 
                                 pd.Series(self.data.Low), 
                                 pd.Series(self.data.Close), 
                                 self.n)

    def next(self):
        if self.williams_r[-1] < self.lower:
            if not self.position.is_long:
                if self.position.is_short:
                    self.position.close()
                self.buy()
        elif self.williams_r[-1] > self.upper:
            if not self.position.is_short:
                if self.position.is_long:
                    self.position.close()
                self.sell()
"""

ALL_SEEDS = [
    BOLLINGER_STRATEGY,
    CCI_STRATEGY,
    EMA_CROSSOVER_STRATEGY,
    MACD_STRATEGY,
    WILLIAMS_R_STRATEGY
]

RANDOM_STRATEGY = """
from backtesting import Strategy
import random

class RandomStrategy(Strategy):
    def next(self):
        # 0.5 probability to exit if in position
        if self.position:
            if random.random() < 0.5:
                self.position.close()
        
        # If flat, choose uniformly: enter long, enter short, do nothing
        else:
            r = random.random()
            if r < 0.33:
                self.buy()
            elif r < 0.66:
                self.sell()
"""

BUY_AND_HOLD_STRATEGY = """
from backtesting import Strategy

class BuyAndHold(Strategy):
    def next(self):
        if not self.position:
            self.buy()
"""
