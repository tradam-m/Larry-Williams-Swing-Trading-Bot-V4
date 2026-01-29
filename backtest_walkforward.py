
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add current directory to path to import modules
sys.path.append(os.getcwd())

# Import components from the main bot
# We assume the user has the dependencies installed or we will handle import errors gracefully
try:
    from kraken_bot_v4_advanced import (
        TradingBotV4, Config, TradingPair, SwingDetectorV3, 
        RegimeDetector, StrategyType, PositionManagerV3
    )
    from ensemble_strategies import StrategySignal, EnsembleDecision
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ==============================================================================
# MOCK CLASSES FOR BACKTESTING
# ==============================================================================

class MockTelegram:
    def send(self, message: str):
        # In backtest we might want to suppress or log these
        pass

class MockKrakenClient:
    def __init__(self):
        self.balance = 10000.0  # Initial Balance USD
        self.positions = {}
        self.orders = []
        self.commission_rate = 0.0026  # Taker fee 0.26%

    def get_balance(self):
        return self.balance, 'USD'

    def get_available_margin(self):
        # Simplified margin calc
        used_margin = sum(p['margin'] for p in self.positions.values())
        return self.balance - used_margin

    def get_open_positions(self):
        return self.positions

    def place_order(self, pair, order_type, volume, leverage=1, reduce_only=False):
        price = self.current_prices.get(pair)
        if not price:
            return {'error': 'No price'}

        cost = volume * price
        
        # Calculate fee
        fee = cost * self.commission_rate
        self.balance -= fee
        
        pos_id = f"pos_{len(self.orders)}"
        
        if reduce_only:
            # Logic to close positions would go here
            # For simplicity in mock, we don't handle partial closes accurately 
            # without more complex logic, but PositionManager handles the closing logic
            # by calling close_position.
            pass
        else:
            if order_type == 'buy':
                margin_req = cost / leverage
                self.positions[pos_id] = {
                    'ordertxid': pos_id,
                    'pair': pair,
                    'type': 'long',
                    'vol': str(volume),
                    'cost': str(cost),
                    'fee': str(fee),
                    'margin': margin_req,
                    'leverage': str(leverage),
                    'time': datetime.now().timestamp()
                }
            elif order_type == 'sell':
                margin_req = cost / leverage
                self.positions[pos_id] = {
                    'ordertxid': pos_id,
                    'pair': pair,
                    'type': 'short',
                    'vol': str(volume),
                    'cost': str(cost), # keeping it positive for calc
                    'fee': str(fee),
                    'margin': margin_req,
                    'leverage': str(leverage),
                    'time': datetime.now().timestamp()
                }
        
        self.orders.append({
            'id': pos_id,
            'pair': pair,
            'type': order_type,
            'price': price,
            'volume': volume,
            'leverage': leverage,
            'time': datetime.now()
        })
        
        return {'result': {'txid': [pos_id]}}

    def close_position(self, pair, pos_type, volume, leverage):
        # Find position to close
        # In this mock, we close the first matching position for simplicity
        # real Kraken API closes by volume
        
        to_remove = []
        
        # Current price needed for PnL
        current_price = self.current_prices.get(pair)
        
        for pid, pos in self.positions.items():
            if pos['pair'] == pair and pos['type'] == pos_type:
                # Calculate PnL
                entry_price = float(pos['cost']) / float(pos['vol'])
                vol = float(pos['vol'])
                lev = float(pos['leverage'])
                
                if pos_type == 'long':
                    pnl = (current_price - entry_price) * vol
                else:
                    pnl = (entry_price - current_price) * vol
                
                # Update balance
                self.balance += pnl
                
                to_remove.append(pid)
                break # Close one per call
        
        for pid in to_remove:
            del self.positions[pid]
            
        return {'result': {'closed': to_remove}}

    # Helper to update current prices for the mock
    def set_current_prices(self, prices):
        self.current_prices = prices


class BacktestConfig(Config):
    # Override config for backtest
    DRY_RUN = True
    USE_SENTIMENT_ANALYSIS = False # Cannot backtest without historical data
    USE_ONCHAIN_ANALYSIS = False   # Cannot backtest without historical data
    USE_ENSEMBLE_SYSTEM = True
    USE_RL_POSITION_SIZING = True
    
# ==============================================================================
# WALK FORWARD BACKTESTER
# ==============================================================================

class BacktesterV4(TradingBotV4):
    def __init__(self):
        self.config = BacktestConfig()
        
        # Override components with Mocks
        self.kraken = MockKrakenClient()
        self.telegram = MockTelegram()
        self.position_mgr = PositionManagerV3(self.config, self.kraken, self.telegram)
        
        # Initialize V4 Components
        # Sentiment/OnChain disabled for backtest as we lack history
        self.sentiment_analyzer = None 
        self.onchain_analyzer = None
        
        # Initialize RL (we can train it during backtest!)
        if self.config.USE_RL_POSITION_SIZING:
            try:
                from rl_position_sizing import RLPositionSizer, PositionSizeCalculator
                self.rl_sizer = RLPositionSizer(
                    learning_rate=0.1,
                    discount_factor=0.95,
                    epsilon=0.1
                )
                self.rl_calculator = PositionSizeCalculator(self.rl_sizer)
            except ImportError:
                self.rl_sizer = None
                self.rl_calculator = None
        
        # Initialize Ensemble
        if self.config.USE_ENSEMBLE_SYSTEM:
            try:
                from ensemble_strategies import EnsembleSystem
                weights = {
                    StrategyType.SWING: 0.30,
                    StrategyType.MOMENTUM: 0.25,
                    StrategyType.MEAN_REVERSION: 0.25,
                    StrategyType.TREND_FOLLOWING: 0.20
                }
                self.ensemble = EnsembleSystem(weights=weights)
            except ImportError:
                self.ensemble = None
                
        self.trades_history = []
        self.data_cache = {}

    def load_data(self, days=365):
        print(f"Loading {days} days of historical data...")
        for pair in self.config.TRADING_PAIRS:
            try:
                print(f"Fetching {pair.yf_symbol}...")
                # Download more data to cover the lookback
                ticker = yf.Ticker(pair.yf_symbol)
                start_date = datetime.now() - timedelta(days=days+int(self.config.LOOKBACK_PERIOD[:-1])*2) # simplified
                df = ticker.history(start=start_date, interval="1h")
                self.data_cache[pair.yf_symbol] = df
                print(f"Loaded {len(df)} candles for {pair.yf_symbol}")
            except Exception as e:
                print(f"Error loading {pair.yf_symbol}: {e}")

    def run_walk_forward(self, training_window_days=90, test_window_days=30):
        print("\nStarting Walk Forward Analysis...")
        
        # Determine simulation range
        # We need a common index or we just iterate by time
        # For simplicity, we assume all pairs have similar timestamps
        if not self.data_cache:
            print("No data loaded.")
            return

        ref_symbol = self.config.TRADING_PAIRS[0].yf_symbol
        ref_data = self.data_cache[ref_symbol]
        
        start_time = ref_data.index[0] + timedelta(days=60) # Warmup
        end_time = ref_data.index[-1]
        
        current_time = start_time
        window_size = timedelta(days=test_window_days)
        
        total_pnl = 0
        
        while current_time < end_time:
            window_end = min(current_time + window_size, end_time)
            print(f"\nProcessing Window: {current_time} to {window_end}")
            
            # Walk through this window hour by hour
            # We iterate through the reference dataframe's index within this window
            window_slice = ref_data[current_time:window_end]
            
            for timestamp in window_slice.index:
                
                # Update Mock Prices
                current_prices = {}
                for pair in self.config.TRADING_PAIRS:
                    if pair.yf_symbol in self.data_cache:
                        try:
                            # Get price at this timestamp (or previous if missing)
                            val = self.data_cache[pair.yf_symbol].asof(timestamp)
                            current_prices[pair.kraken_pair] = val['Close']
                            current_prices[pair.yf_symbol] = val['Close']
                        except:
                            pass
                
                self.kraken.set_current_prices(current_prices)
                
                # Check open positions logic (Stop Loss / Take Profit)
                self.check_open_positions(timestamp)
                
                # Strategy Logic
                for pair in self.config.TRADING_PAIRS:
                    if pair.yf_symbol not in self.data_cache:
                        continue
                        
                    # Get data up to CURRENT execution time (looking back)
                    full_df = self.data_cache[pair.yf_symbol]
                    # We need a slice ending at 'timestamp'
                    # Performance note: slicing potentially slow in loop
                    historical_slice = full_df.loc[:timestamp].tail(200) # Lookback for indicators
                    
                    if len(historical_slice) < 50:
                        continue
                        
                    # Now we call the existing logic methods
                    # 1. Detect Swing
                    detector = SwingDetectorV3(
                        historical_slice, 
                        volume_filter=self.config.USE_VOLUME_FILTER,
                        use_ml=False # Disable ML validation for speed/Simplicity in WF
                    )
                    swing_signal = detector.get_signal() # returns (signal, price, confidence)
                    
                    if swing_signal[0]:
                        # 2. Analyze Opportunity (Ensemble etc.)
                        # Modifying analyze_trading_opportunity to accept data slice
                        # But wait, original method takes 'data' as argument.
                        # self.analyze_trading_opportunity uses self.sentiment_analyzer which is None
                        
                        analysis = self.analyze_trading_opportunity(
                            pair, historical_slice, swing_signal
                        )
                        
                        if analysis['can_trade'] and analysis['final_signal']:
                            # 3. Execute
                            cap, lev = self.calculate_position_size(
                                pair, historical_slice, analysis, self.kraken.get_available_margin()
                            )
                            
                            self.open_position(
                                pair, analysis, historical_slice, 
                                current_prices[pair.yf_symbol], cap, lev
                            )
                
            # End of Window
            # Walk Forward: Move start time
            current_time = window_end
            
            balance, _ = self.kraken.get_balance()
            print(f"End of Window Balance: ${balance:.2f}")

    def check_open_positions(self, current_timestamp):
        # Re-implementation of the loop in run() to check positions
        positions = self.kraken.get_open_positions().copy()
        
        for pid, pos_data in positions.items():
            pair_key = pos_data['pair']
            # Find trading pair info
            trading_pair = next((tp for tp in self.config.TRADING_PAIRS if tp.kraken_pair == pair_key), None)
            if not trading_pair: continue
            
            current_price = self.kraken.current_prices.get(pair_key)
            if not current_price: continue
            
            # Detect regime based on available history upto now
            df = self.data_cache[trading_pair.yf_symbol].loc[:current_timestamp].tail(100)
            regime = RegimeDetector.detect(df)
            regime_params = RegimeDetector.get_adapted_params(
                regime, 
                self.config.BASE_STOP_LOSS,
                self.config.BASE_TAKE_PROFIT,
                self.config.BASE_TRAILING_STOP
            )
            
            should_close, reason = self.position_mgr.check_position(
                pid, pos_data, current_price, regime_params
            )
            
            if should_close:
                pos_type = pos_data.get('type')
                vol = float(pos_data.get('vol'))
                self.position_mgr.close_position(
                    pair_key, pos_type, vol, reason, pos_data, current_price
                )


if __name__ == "__main__":
    print("Initializing V4 Walk Forward Backtester...")
    backtester = BacktesterV4()
    
    # Load 60 days for a quick demo
    backtester.load_data(days=60)
    
    # Run WF
    backtester.run_walk_forward(training_window_days=30, test_window_days=7)
