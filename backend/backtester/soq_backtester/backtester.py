import pandas as pd
import numpy as np
import vectorbt as vbt
import tqdm
from tabulate import tabulate
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import script
import importlib
importlib.reload(script)
import scipy.stats as stats
import json

from script import Strategy 
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime


import re
import backtrader as bt
import google.generativeai as genai

# =============================================================================
# --- 1. CONFIGURATION & SETTINGS ---
# =============================================================================

# --- API Configuration ---
# IMPORTANT: It's recommended to use environment variables for API keys
# For example: genai.configure(api_key=os.environ["GEMINI_API_KEY"])
#AIzaSyBOTxcuKPVGhSGqcJnhnkmny_k6K0f8YoI
try:
    genai.configure(api_key="AIzaSyDwdQgVzt9GXc5i3Og9cdzJkTxa2a7DjXI") # <-- PASTE YOUR API KEY HERE
except Exception as e:
    print(f"⚠️ Gemini API Key not configured. Please set it. Error: {e}")


# =============================================================================
# --- START OF USER-CONFIGURABLE SECTION ---
# NOTE: The code between these START/END markers will be replaced by the web editor.

TICKERS = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'HINDUNILVR',
           'INFY', 'ITC', 'BHARTIARTL', 'SBIN', 'BAJFINANCE']

START_DATE = '2018-01-25'
END_DATE = '2025-04-20'
TARGET = 0.06
STOP_LOSS = 0.03
MAX_POSITIONS = 40

# --- END OF USER-CONFIGURABLE SECTION ---
# =============================================================================


# --- Dynamic Path Configuration ---

# 1. Get the absolute path to the directory where this script is located.
#    os.path.abspath(__file__) gets the full path to the current script.
#    os.path.dirname(...) gets the directory part of that path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define the path to the data directory relative to the script's location.
#    os.path.join() is used to safely combine path parts (works on Windows, Mac, and Linux).
DATA_DIR = SCRIPT_DIR

# 3. Build the full paths to your files using the DATA_DIR.
RAW_DATA_PATH = {
    'open': os.path.join(DATA_DIR, "cleaned_daily_open.csv"),
    'close': os.path.join(DATA_DIR, "cleaned_daily_close.csv"),
    'high': os.path.join(DATA_DIR, "cleaned_daily_high.csv"),
    'low': os.path.join(DATA_DIR, "cleaned_daily_low.csv"),
    'volume': os.path.join(DATA_DIR, "cleaned_daily_volume.csv"),
    'multi_level': os.path.join(DATA_DIR, 'final_multi_level_stocks.csv')
}

# 4. For output files, you can decide where they should go.
#    Saving them in the same directory as the script is a common choice.
OUTPUT_FILES = {
    'clipped_data': os.path.join(SCRIPT_DIR, 'clipped_data.csv'),
    'trades': os.path.join(SCRIPT_DIR, 'trades.csv'),
    'signals': os.path.join(SCRIPT_DIR, 'signals.csv'),
    'weights': os.path.join(SCRIPT_DIR, 'weights.csv')
}

# =============================================================================
# --- 2. INITIALIZE STRATEGY ---
# =============================================================================

# This is the new, crucial step.
# 1. Get the correct weights file path from your dictionary.
weights_path = OUTPUT_FILES['weights']

# 2. Create an instance of your Strategy, passing the correct path to its constructor.
#    This triggers the __init__ method in your Strategy class to run and load the data.
strategy_instance = Strategy(weights_filepath=weights_path)

# NOW, you can use `strategy_instance` to run your backtest.
# For example, if you need to call get_signals in a loop, you would use:
#
# trading_state = {'traderData': 0} 
# signals, new_trader_data = strategy_instance.get_signals(trading_state)


# =============================================================================
# --- 2. BACKTRADER STRATEGY & INDICATOR CLASSES ---
# =============================================================================

class DynamicStrategy(bt.Strategy):
    params = {
        'target': 0.0,
        'stop_loss': 0.0,
        'max_positions': 10,
        'entry_rules': [],
        'exit_rules': []
    }

    def __init__(self):
        self.data_map = {}
        self.indicators = {}
        self.rule_env = {}
        self.order_log = []

        # Map feeds by symbol/timeframe from names like "RELIANCE_D"
        for d in self.datas:
            symbol, tf = d._name.split('_')
            self.data_map.setdefault(symbol, {})[tf] = d

        # Build env per symbol and pre-create indicators
        for sym, feeds in self.data_map.items():
            self.indicators[sym] = {}
            self.rule_env[sym] = {}
            for tf_key, feed in feeds.items():
                self.rule_env[sym][f'close_{tf_key}'] = feed.close

            all_rules = self.p.entry_rules + self.p.exit_rules
            for rule in all_rules:
                tokens = re.findall(r'(sma_[DWM]_\d+|ema_[DWM]_\d+)', rule)
                for token in tokens:
                    if token not in self.rule_env[sym]:
                        self.build_indicator(sym, token.split('_')[1], token)

        self.position_state = {sym: {"in_position": False, "entry_price": None}
                               for sym in self.data_map.keys()}

    def build_indicator(self, sym, tf_key, ind_str):
        match = re.match(r'(sma|ema)_([DWM])_(\d+)', ind_str)
        if not match:
            raise ValueError(f"Unsupported indicator: {ind_str}")

        ind_type, _, period = match.groups()
        period = int(period)

        feed = self.data_map[sym][tf_key]
        ind = bt.ind.SMA(feed.close, period=period) if ind_type == "sma" else bt.ind.EMA(feed.close, period=period)

        self.indicators[sym][ind_str] = ind
        self.rule_env[sym][ind_str] = ind
        return ind

    def is_long_entry(self, sym):
        try:
            return all(eval(rule, {}, self.rule_env[sym]) for rule in self.p.entry_rules)
        except Exception as e:
            # print(f"Entry eval error {sym}: {e}")
            return False

    def is_long_exit(self, sym):
        try:
            state = self.position_state[sym]
            if state["in_position"]:
                current_price = self.data_map[sym]['D'].close[0]
                entry_price = state["entry_price"]
                pnl = (current_price - entry_price) / entry_price
                if pnl <= -self.p.stop_loss or pnl >= self.p.target:
                    return True

            return any(eval(rule, {}, self.rule_env[sym]) for rule in self.p.exit_rules)
        except Exception as e:
            # print(f"Exit eval error {sym}: {e}")
            return False

    def next(self):
        current_positions = sum(1 for st in self.position_state.values() if st["in_position"])

        for sym, feeds in self.data_map.items():
            d = feeds['D']
            state = self.position_state[sym]

            if state["in_position"] and self.is_long_exit(sym):
                self.close(data=d)
                state["in_position"] = False
                state["entry_price"] = None
                self.log_trade('SELL', sym, d.close[0])

            elif (not state["in_position"] and
                  current_positions < self.p.max_positions and
                  self.is_long_entry(sym)):
                self.buy(data=d)
                state["in_position"] = True
                state["entry_price"] = d.close[0]
                current_positions += 1
                self.log_trade('BUY', sym, d.close[0])

    def log_trade(self, action, ticker, price):
        self.order_log.append({
            'date': self.datetime.date(0),
            'ticker': ticker,
            'action': action,
            'price': price
        })

    def stop(self):
        if self.order_log:
            pd.DataFrame(self.order_log).to_csv(OUTPUT_FILES['trades'], index=False)
            print(f"✅ Trades log saved to '{OUTPUT_FILES['trades']}'")
        else:
            print("⚠️ No trades were executed during the backtest.")

# =============================================================================
# --- 3. HELPER FUNCTIONS ---
# =============================================================================

def prepare_data_and_clip_csv(tickers, start_date_str, end_date_str):
    """
    Loads raw OHLCV data, clips the multi-level master CSV to the specified
    date range, and prepares the data dictionary for Backtrader.
    """
    print("\n---  Preparing Data ---")
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # --- Clip the master multi-level CSV ---
    try:
        df_multi = pd.read_csv(
            RAW_DATA_PATH['multi_level'],
            header=[0, 1],
            index_col=0,
            skiprows=[2]
        )
        df_multi.index = pd.to_datetime(df_multi.index)
        clipped_df = df_multi.loc[start_date:end_date]
        clipped_df.to_csv(OUTPUT_FILES['clipped_data'])
        print(f"✅ Successfully clipped data and saved to '{OUTPUT_FILES['clipped_data']}'")
    except FileNotFoundError:
        print(f"❌ ERROR: Master file '{RAW_DATA_PATH['multi_level']}' not found.")
        return None
    except Exception as e:
        print(f"❌ An error occurred while clipping the CSV: {e}")
        return None

    # --- Load individual OHLCV data for backtesting ---
    try:
        data_frames = {k: pd.read_csv(v, index_col=0, parse_dates=True) for k, v in RAW_DATA_PATH.items() if k != 'multi_level'}
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find raw data file: {e.filename}")
        return None

    # --- Build data_dict for Backtrader ---
    data_dict = {}
    print("📥 Building data feeds for backtrader...")
    for sym in tickers:
        try:
            df = pd.DataFrame({
                'open':   data_frames['open'][sym],
                'high':   data_frames['high'][sym],
                'low':    data_frames['low'][sym],
                'close':  data_frames['close'][sym],
                'volume': data_frames['volume'][sym]
            })
            df = df.loc[start_date:end_date].dropna()

            if not df.empty:
                data_dict[sym] = df
                # print(f"  - Loaded {sym} with {len(df)} rows.")
            else:
                print(f"  - WARNING: No data for {sym} in the specified date range.")
        except KeyError:
            print(f"  - WARNING: Ticker '{sym}' not found in source CSVs. Skipping.")
        except Exception as e:
            print(f"  - ERROR loading data for {sym}: {e}")

    print(f"✅ Data preparation complete. Loaded {len(data_dict)} tickers.")
    return data_dict

def generate_strategy_from_prompt(prompt):
    """
    Calls the Gemini API to convert a natural language prompt into a
    Backtrader-compatible strategy dictionary.
    """
    print("\n--- Generating Strategy with Gemini ---")
    system_prompt = """
    You are a trading strategy generator. Return ONLY a Python dictionary named USER_STRATEGY.
    - Supported indicators: SMA, EMA (daily only, use '_D').
    - Format variable names as 'sma_D_20', 'ema_D_50', 'close_D'.
    - Entry and exit rules must be valid Python expressions using these variables.

    Example output format:
    {'baseline_tf': bt.TimeFrame.Days, 'entry': ["sma_D_20[0] > sma_D_100[0]"], 'exit': ["sma_D_20[0] < sma_D_100[0]"]}

    Now, generate USER_STRATEGY for the following prompt:
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest") # Using latest model
        full_prompt = f"{system_prompt}\nUser:\n{prompt}"
        response = model.generate_content(full_prompt)
        
        # Clean up response to extract only the dictionary
        raw_text = response.text.strip().replace("python", "").replace("```", "")
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if not match:
            raise ValueError("Could not extract a valid dictionary from Gemini's response.")
            
        strategy_str = match.group(0)
        
        # Using eval in a controlled way to parse the dictionary string
        local_scope = {'bt': bt}
        strategy_dict = eval(strategy_str, {"__builtins__": {}}, local_scope)
        
        print("✅ Strategy generated successfully.")
        print(strategy_dict)
        return strategy_dict
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        return None

def run_backtest(data_dict, strategy_rules, target, stop_loss, max_positions):
    """
    Initializes and runs the Backtrader engine.
    """
    print("\n--- Running Backtest ---")
    if not data_dict:
        print("❌ Cannot run backtest, data dictionary is empty.")
        return

    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True) # Cheat-on-close for realistic fills

    cerebro.addstrategy(
        DynamicStrategy,
        entry_rules=strategy_rules.get('entry', []),
        exit_rules=strategy_rules.get('exit', []),
        target=target,
        stop_loss=stop_loss,
        max_positions=max_positions
    )

    for sym, df in data_dict.items():
        d = bt.feeds.PandasData(dataname=df, name=f"{sym}_D")
        cerebro.adddata(d)
        # Resampling can be added here if needed (e.g., for weekly/monthly rules)
        # cerebro.resampledata(d, timeframe=bt.TimeFrame.Weeks, name=f"{sym}_W")
        # cerebro.resampledata(d, timeframe=bt.TimeFrame.Months, name=f"{sym}_M")

    print("🚀 Backtest engine starting...")
    cerebro.run()
    print("✅ Backtest complete.")

def generate_signals_file():
    """
    Converts the trades.csv log into a signals.csv file, aligned with the
    master date index and including ALL tickers from clipped_data.csv.
    """
    print("\n--- Generating Signals File ---")
    try:
        # Check if trades file exists and is not empty
        if not (os.path.exists(OUTPUT_FILES['trades']) and os.path.getsize(OUTPUT_FILES['trades']) > 0):
            print("⚠️ 'trades.csv' not found or is empty. Cannot generate signals.")
            return

        # 1. Load the master data to get both index (dates) and columns (all tickers)
        master_df = pd.read_csv(
            OUTPUT_FILES['clipped_data'],
            header=[0, 1],      # Reads the multi-level header
            index_col=0,
            skiprows=[2]
        )
        master_df.index = pd.to_datetime(master_df.index).normalize()
        #master_index = master_df.index
        # Get a unique list of all ticker names from the first level of the columns
        all_master_tickers = master_df.columns.get_level_values(0).unique().tolist()
        print(f"Found {len(all_master_tickers)} unique tickers in the master data file.")

        # 2. Process the trades log as before
        trades = pd.read_csv(OUTPUT_FILES['trades'], parse_dates=['date'])
        trades['date'] = trades['date'].dt.normalize()
        trades['pos'] = trades['action'].map({'BUY': 1, 'SELL': -1})

        trade_signals = trades.pivot_table(
            index='date', columns='ticker', values='pos', aggfunc='sum', fill_value=0
        )

        # 3. Reindex using the master date index AND the full list of master tickers
        #    This is the key change that adds all the missing ticker columns with 0s.
        final_signals = trade_signals.reindex(
            index=master_df.index, 
            columns=all_master_tickers, 
            fill_value=0
        ).astype(int)
        
        # 4. Trim the signals to start from the date of the very first trade
        if not trades.empty:
            first_trade_date = trades['date'].min()
            final_signals = final_signals.loc[first_trade_date:]

        # 5. Save the final, complete signals file
        final_signals.index.name = 'Date'
        final_signals.to_csv(OUTPUT_FILES['signals'])
        print(f"✅ Signals file saved to '{OUTPUT_FILES['signals']}' with all master tickers.")

         # 5. (NEW) Re-clip the master data file to match the signals file
        print(f"Ensuring consistency by re-clipping '{OUTPUT_FILES['clipped_data']}'...")
        final_clipped_df = master_df.loc[final_signals.index]
        final_clipped_df.to_csv(OUTPUT_FILES['clipped_data'])
        print(f"✅ '{OUTPUT_FILES['clipped_data']}' is now perfectly aligned with signals.")


    except FileNotFoundError as e:
        print(f"❌ Error finding a required file: {e.filename}")
    except Exception as e:
        print(f"❌ An unexpected error occurred while generating signals: {e}")

def calculate_weights_original(signals_df: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    """
    Original behavior:
      - Start each day with NaN for all tickers
      - Set 0 for tickers with -1 (exits)
      - Assign 1/slots_to_use only to *new* entries (signal == 1), up to max_positions
      - Do NOT auto-assign weights to already-held positions (they remain NaN)
    """
    current_positions = set()
    weights_history = []

    first_date = signals_df.index[0]

    for date, signals in signals_df.iterrows():
        weights = pd.Series(np.nan, index=signals.index, dtype=float)

        # Error check: -1 on first date
        if date == first_date and (signals == -1).any():
            tickers = signals.index[signals == -1].tolist()
            raise ValueError(f"-1 (exit) signal on first date for: {tickers}")

        # Handle exits
        for ticker in list(current_positions):
            if signals.get(ticker, 0) == -1:
                weights[ticker] = 0.0
                current_positions.remove(ticker)

        # New entries up to available slots
        new_entries = [t for t in signals.index if signals[t] == 1 and t not in current_positions]
        available_slots = max_positions - len(current_positions)
        slots_to_use = min(len(new_entries), max(0, available_slots))

        if slots_to_use > 0:
            dynamic_weight = 1.0 / slots_to_use
            for t in new_entries[:slots_to_use]:
                weights[t] = dynamic_weight
                current_positions.add(t)
            #print(f"{slots_to_use} slots used on {date.date()}")
        #else:
            #print(f"no slots available on {date.date()}")

        # Also set 0 for -1 on non-held (defensive)
        for t in signals.index:
            if signals[t] == -1 and t not in current_positions:
                weights[t] = 0.0

        weights_history.append(weights)

    weights_df = pd.DataFrame(weights_history, index=signals_df.index)
    return weights_df


def calculate_and_save_weights(max_positions: int):
    """
    Reads OUTPUT_FILES['signals'], applies the ORIGINAL weighting logic (default NaN),
    and saves to OUTPUT_FILES['weights'].
    """
    print("\n--- Calculating Portfolio Weights (original semantics: default=NaN) ---")
    try:
        signals_df = pd.read_csv(OUTPUT_FILES['signals'], index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"⚠️ '{OUTPUT_FILES['signals']}' not found. Cannot calculate weights.")
        return

    weights_df = calculate_weights_original(signals_df, max_positions)
    weights_df.to_csv(OUTPUT_FILES['weights'])
    print(f"✅ Portfolio weights calculated and saved to '{OUTPUT_FILES['weights']}'")
# =============================================================================
# --- 4. MAIN EXECUTION ---
# =============================================================================

def main():
    """
    Main function to run the entire workflow.
    """
    # Step 1: Prepare data and create the clipped_data.csv
    data_dict = prepare_data_and_clip_csv(TICKERS, START_DATE, END_DATE)

    # Step 2: Generate strategy from the prompt
    strategy_rules = generate_strategy_from_prompt(USER_PROMPT)

    # Proceed only if data and strategy are ready
    if data_dict and strategy_rules:
        # Step 3: Run the backtest to generate trades.csv
        run_backtest(data_dict, strategy_rules, TARGET, STOP_LOSS, MAX_POSITIONS)

        # Step 4: Convert trades to signals, creating signals.csv
        generate_signals_file()

        # Step 5: Calculate portfolio weights, creating weights.csv
        calculate_and_save_weights(MAX_POSITIONS)

        print("\n🎉 Workflow finished successfully!")
    else:
        print("\n❌ Workflow aborted due to errors in data preparation or strategy generation.")

class Backtester:
    def __init__(self, data: pd.DataFrame, initial_value: float):
        self.data = data
        self.portfolio_value = initial_value
        self.cash = initial_value
        self.investment = 0.0
        self.current_index = 1
        tickers = data.columns.get_level_values(0).unique()
        self.positions = pd.Series(0, index=tickers)
        self.all_positions = pd.DataFrame(columns=tickers)
        self.tradingState = {}
        self.dates = []
        self.portfolio_history = []
        self.cash_history = []
        self.investment_history = []
        self.all_signals = pd.DataFrame(columns=tickers)

    def calculate_positions(self, signal: pd.Series, value, open=True) -> pd.Series:
        if (signal < 0).any():
            raise ValueError(f'For timestamp {self.data.index[self.current_index]}, signal contains negative values: {signal[signal < 0]}')
        if not isinstance(signal, pd.Series):
            raise TypeError(f'For timestamp {self.data.index[self.current_index]}, signal must be a pandas Series, got {type(signal)}')
        if abs(signal).sum() - 1 > 1e-6:
            raise ValueError(f'For timestamp {self.data.index[self.current_index]} the sum of the abs(signals) must not be greater than 1, got {abs(signal).sum()}')

        prices = (
            self.data.xs('open', level=1, axis=1).iloc[self.current_index]
            if open
            else self.data.xs('close', level=1, axis=1).iloc[self.current_index]
        )
        prices = prices.reindex(signal.index)
        
        nan_index = signal.isna()
        value -= (self.positions[nan_index]*prices[nan_index]).sum()

        float_shares = (signal.replace(0,np.nan) * value) / prices.replace(0, np.nan)

        float_shares = (
            float_shares
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        new_positions = pd.Series(0, index=float_shares.index, dtype=int)
        longs  = float_shares > 0
        shorts = float_shares < 0

        new_positions[longs]  = np.floor(float_shares[longs]).astype(int)
        new_positions[shorts] = np.ceil (float_shares[shorts]).astype(int)
        
        new_positions[nan_index] = self.positions[nan_index]

        return new_positions

    def calculate_cash(self, positions: pd.Series, open=True) -> float:
        index = self.current_index
        price = self.data.xs('open',level=1,axis=1).iloc[index] if open else self.data.xs('close',level=1,axis=1).iloc[index]
        return self.portfolio_value - (abs(positions) * price).sum()

    def update_investment(self, positions: pd.Series, new_day=False) -> float:
        index = self.current_index
        price1 = self.data.xs('close',level=1,axis=1).iloc[index-1] if new_day else self.data.xs('open',level=1,axis=1).iloc[index]
        price2 = self.data.xs('open',level=1,axis=1).iloc[index] if new_day else self.data.xs('close',level=1,axis=1).iloc[index]
        return (positions * (price2 - price1)).sum() + self.investment

    def run(self,strategy_object):
        processed_data = strategy_object.process_data(self.data)
        self.all_positions.loc[self.data.index[0]] = self.positions
        traderData = 1
        for i in tqdm.tqdm(range(1, len(self.data))):
            self.tradingState = {
                'processed_data': processed_data[:i],
                'investment': self.investment,
                'cash': self.cash,
                'current_timestamp': self.data.index[self.current_index],
                'traderData': traderData,
                'positions': self.positions,
            }
            signal, traderData = strategy_object.get_signals(self.tradingState)
            if signal is None:
                raise ValueError(f'For timestamp {self.data.index[self.current_index]}, signal is None')
            self.investment = self.update_investment(self.positions, new_day=True)
            self.portfolio_value = self.investment + self.cash
            self.positions = self.calculate_positions(signal, self.portfolio_value)
            self.cash = self.calculate_cash(self.positions)
            self.investment = self.portfolio_value - self.cash
            self.investment = self.update_investment(self.positions, new_day=False)
            self.portfolio_value = self.investment + self.cash
            self.all_positions.loc[self.data.index[i]] = self.positions
            self.all_signals.loc[self.data.index[i-1]] = signal
            self.current_index += 1

    def vectorbt_run(self):
        open_prices = self.data.xs('open', level=1, axis=1).loc[self.all_positions.index, self.all_positions.columns]
        close_prices = self.data.xs('close', level=1, axis=1).loc[self.all_positions.index, self.all_positions.columns]
        
        order_size = self.all_positions.diff().fillna(0).astype(int)
        order_size = order_size.mask(order_size == 0)
        
        portfolio = vbt.Portfolio.from_orders(
            close=close_prices,
            size=order_size,
            price=open_prices,
            init_cash=initial_value,
            freq='1D',
            cash_sharing=True,
            ffill_val_price=True,
            call_seq='auto',
            log=True,  
        )
        return portfolio
    
    def export_results(self, portfolio, save_path="frontend_data"):
        os.makedirs(save_path, exist_ok=True)
        charts_dir = os.path.join(save_path, "charts")
        plots_dir = os.path.join(save_path, "plots")
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Portfolio summary data
        equity = portfolio.value()
        returns = equity.pct_change().fillna(0)
        cum_max = equity.cummax()
        drawdown = (equity - cum_max) / cum_max

        portfolio_summary = pd.DataFrame({
            'date': equity.index,
            'equity': equity.values,
            'returns': returns.values,
            'drawdown': drawdown.values
        })
        portfolio_summary.to_csv(os.path.join(save_path, "portfolio_summary.csv"), index=False)
        
        # Save as JSON for frontend
        portfolio_json = portfolio_summary.to_dict(orient='records')
        with open(os.path.join(save_path, "portfolio_summary.json"), 'w') as f:
            json.dump(portfolio_json, f, default=str)  # default=str handles datetime conversion

        # 2. Export portfolio value breakdown
        df = pd.concat([portfolio.value(), portfolio.asset_value(), portfolio.cash()], axis=1)
        df.columns = ['portfolio', 'investment', 'cash']
        df.to_csv(os.path.join(save_path, "portfolio.csv"))
        
        # 3. Save signals
        self.all_signals.to_csv(os.path.join(save_path, "signals.csv"))
        
        # 4. Generate candlestick data for each ticker
        tickers = self.data.columns.get_level_values(0).unique()
        for ticker in tqdm.tqdm(tickers, desc="Exporting ticker charts"):
            ticker_data = self.get_candlestick_data(ticker)
            with open(os.path.join(charts_dir, f"{ticker}.json"), 'w') as f:
                json.dump(self.clean_for_json(ticker_data), f, default=str)


        # NEW: Generate returns histogram
        returns_histogram = self.generate_returns_histogram(portfolio_summary['returns'])
        with open(os.path.join(save_path, "returns_histogram.json"), 'w') as f:
            json.dump(returns_histogram, f)
        
        # Also save as CSV
        pd.DataFrame(returns_histogram).to_csv(os.path.join(save_path, "returns_histogram.csv"), index=False)
        
        # NEW: Generate performance metrics
        performance_metrics = self.calculate_performance_metrics(portfolio, portfolio_summary)
        safe_metrics = self.clean_for_json(performance_metrics)
        with open(os.path.join(save_path, "performance_metrics.json"), 'w') as f:
            json.dump(safe_metrics, f, indent=2, allow_nan=False)


        print(f"📁 Results exported to `{save_path}/`.")
        return portfolio_summary

    def generate_returns_histogram(self, returns_series: pd.Series, bins: int = 30) -> list:
        returns = returns_series.dropna()
        counts, bin_edges = np.histogram(returns, bins=bins)

        histogram = []
        for i in range(len(counts)):
            histogram.append({
                "bin_start": round(float(bin_edges[i]), 6),
                "bin_end": round(float(bin_edges[i + 1]), 6),
                "frequency": int(counts[i])
            })

        return histogram


    def get_candlestick_data(self, ticker: str) -> dict:
        ohlcv = self.data[ticker][["open", "high", "low", "close"]].copy()
        ohlcv.index = pd.to_datetime(ohlcv.index)
        pos = self.all_positions[ticker].reindex(ohlcv.index).ffill().fillna(0)
        
        # Remove weekends
        mask = ohlcv.index.weekday < 5
        ohlcv = ohlcv.loc[mask]
        pos = pos.loc[mask]
        
        # Create dataframe with all data
        df = ohlcv.assign(position=pos)
        df["action"] = "hold"
        df["dpos"] = df["position"].diff().fillna(df["position"])
        
        # Identify buy/sell actions
        df.loc[df["dpos"] > 0, "action"] = "buy"
        df.loc[df["dpos"] < 0, "action"] = "sell"
        
        # Identify holding periods
        df["holding"] = df["position"] != 0
        df["grp"] = (df["holding"] != df["holding"].shift(fill_value=False)).cumsum()
        
        holdings = []
        for _, sub in df.groupby("grp"):
            if sub["holding"].iat[0]:
                holdings.append({
                    "start": sub.index[0].strftime("%Y-%m-%d"),
                    "end": sub.index[-1].strftime("%Y-%m-%d"),
                    "position": sub["position"].mean()
                })
        
        # Convert to frontend-friendly format
        data = []
        for date, row in df.iterrows():
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "position": row["position"],
                "action": row["action"]
            })
        
        return {
            "ticker": ticker,
            "data": data,
            "holdings": holdings
        }
    
    def calculate_performance_metrics(self, portfolio, portfolio_summary):
        # Get portfolio statistics
        stats_df = portfolio.stats().to_frame(name='Value').reset_index()
        stats_df.columns = ['Metric', 'Value']
        
        # Convert stats to a dictionary for easier access
        stats_dict = dict(zip(stats_df['Metric'], stats_df['Value']))
        
        # Extract key metrics from portfolio stats
        metrics = {
            'initial_value': portfolio.init_cash,
            'final_value': portfolio.value().iloc[-1],
            'total_return_pct': portfolio.total_return() * 100,
            'cagr': portfolio.annualized_return() * 100,
            'volatility_pct': portfolio.annualized_volatility() * 100,
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'sortino_ratio': portfolio.sortino_ratio(),
            'max_drawdown_pct': portfolio.max_drawdown() * 100,
            'calmar_ratio': portfolio.calmar_ratio(),
            'total_trades': stats_dict.get('Total Trades', 0),
            'win_rate_pct': stats_dict.get('Win Rate [%]', 0),
            'profit_factor': stats_dict.get('Profit Factor', 0)
        }
        
        # Additional metrics from returns
        returns = portfolio_summary['returns'].dropna()
        if len(returns) > 0:
            metrics['skewness'] = stats.skew(returns)
            metrics['kurtosis'] = stats.kurtosis(returns, fisher=False)
            metrics['var_95'] = np.percentile(returns, 5) * 100
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean()
            metrics['cvar_95'] = cvar_95 * 100 if not pd.isna(cvar_95) else 0
        else:
            metrics['skewness'] = 0
            metrics['kurtosis'] = 0
            metrics['var_95'] = 0
            metrics['cvar_95'] = 0

        # Format metrics for frontend
        formatted_metrics = [
            {
                "metric": "Total Return",
                "value": f"{metrics['total_return_pct']:.2f}%",
                "description": "Total return over the period"
            },
            {
                "metric": "CAGR",
                "value": f"{metrics['cagr']:.2f}%",
                "description": "Compound Annual Growth Rate"
            },
            {
                "metric": "Volatility",
                "value": f"{metrics['volatility_pct']:.2f}%",
                "description": "Annualized standard deviation of returns"
            },
            {
                "metric": "Sharpe Ratio",
                "value": f"{metrics['sharpe_ratio']:.2f}",
                "description": "Risk-adjusted return (risk-free rate=0)"
            },
            {
                "metric": "Sortino Ratio",
                "value": f"{metrics['sortino_ratio']:.2f}",
                "description": "Adjusted for downside volatility"
            },
            {
                "metric": "Max Drawdown",
                "value": f"{metrics['max_drawdown_pct']:.2f}%",
                "description": "Maximum peak-to-trough decline"
            },
            {
                "metric": "Calmar Ratio",
                "value": f"{metrics['calmar_ratio']:.2f}",
                "description": "CAGR divided by max drawdown"
            },
            {
                "metric": "Win Rate",
                "value": f"{metrics['win_rate_pct']:.2f}%",
                "description": "Percentage of profitable trades"
            },
            {
                "metric": "Profit Factor",
                "value": f"{metrics['profit_factor']:.2f}",
                "description": "Gross profit divided by gross loss"
            },
            {
                "metric": "Total Trades",
                "value": f"{int(metrics['total_trades'])}",
                "description": "Number of completed trades"
            },
            {
                "metric": "Skewness",
                "value": f"{metrics['skewness']:.2f}",
                "description": "Measure of return distribution asymmetry"
            },
            {
                "metric": "Kurtosis",
                "value": f"{metrics['kurtosis']:.2f}",
                "description": "Measure of tail risk in return distribution"
            },
            {
                "metric": "VaR (95%)",
                "value": f"{metrics['var_95']:.2f}%",
                "description": "Worst expected loss at 95% confidence"
            },
            {
                "metric": "CVaR (95%)",
                "value": f"{metrics['cvar_95']:.2f}%",
                "description": "Average loss beyond VaR at 95% confidence"
            }
        ]
        
        return formatted_metrics

    def clean_for_json(self, obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self.clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.clean_for_json(v) for v in obj]
        return obj


# ... (keep all imports and class definition the same) ...

if __name__ == "__main__":
    main()
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the data file
    data_path = os.path.join(script_dir, 'clipped_data.csv')
    
    # Verify the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    # Load the data
    data = pd.read_csv(
        data_path,
        index_col=0, header=[0,1], parse_dates=True
    )

    # tickers = data.columns.get_level_values(0).unique()[100:200]
    # data = data.loc[:, data.columns.get_level_values(0).isin(tickers)]

    initial_value = 100000.0
    backtester = Backtester(data, initial_value)
    backtester.run(strategy_object=strategy_instance)
    pf = backtester.vectorbt_run()
    
    # FIX: Change save path to be outside the backtester directory
    save_path = os.path.join(script_dir, "..", "..", "data", "frontend_data")
    os.makedirs(save_path, exist_ok=True)
    backtester.export_results(pf, save_path=save_path)