import pandas as pd
from typing import Tuple
import numpy as np

class Strategy():
    # The __init__ method is a constructor that runs when you create a new Strategy object.
    # We will pass the correct file path to it from main.py.
    def __init__(self, weights_filepath: str):
        """
        Initializes the Strategy by loading signals data from the provided filepath.
        """
        # 1. Load the data using the path we provide.
        #    The data is now stored in 'self.signalsData', making it unique to this object.
        self.signalsData = pd.read_csv(
            weights_filepath,
            na_values=['nan', 'NaN', ''],
            keep_default_na=True
        )
        # 2. Set the index, just like before.
        self.signalsData.set_index(self.signalsData.columns[0], inplace=True)
    
    def process_data(self, data) -> pd.DataFrame:
        return data

    def get_signals(self, tradingState: dict) -> Tuple[pd.Series, int]:
        """
        Gets the trading signals for the current state.
        """
        # 3. IMPORTANT: Change 'Strategy.signalsData' to 'self.signalsData'.
        #    This now refers to the data loaded specifically for this object.
        signal_row = self.signalsData.iloc[tradingState['traderData']]
        
        # Ensure it's a Series with correct index
        tickers = signal_row.index.tolist()
        signal = pd.Series(signal_row.values, index=tickers)
        
        traderData = tradingState['traderData'] + 1
    
        return signal, traderData