"""
KRAKEN SWING BOT V4 - ADVANCED AI SYSTEM
Integración completa de:
- V3: Multi-Asset + ML + Adaptive Regime + Correlation
- V4: Sentiment Analysis + On-Chain + Ensemble + RL Position Sizing
"""

import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

import time
import hmac
import hashlib
import base64
import urllib.parse
from datetime import datetime, timedelta
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import json
import traceback

# ═══════════════════════════════════════════════════════════════════════════
#                    IMPORTAR MÓDULOS V4
# ═══════════════════════════════════════════════════════════════════════════

try:
    from sentiment_analyzer import (
        SentimentAnalyzer, 
        should_trade_based_on_sentiment
    )
    SENTIMENT_AVAILABLE = True
except ImportError:
    print("⚠️ sentiment_analyzer.py no encontrado")
    SENTIMENT_AVAILABLE = False

try:
    from onchain_metrics import (
        OnChainAnalyzer,
        should_trade_based_on_onchain
    )
    ONCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ onchain_metrics.py no encontrado")
    ONCHAIN_AVAILABLE = False

try:
    from ensemble_strategies import (
        EnsembleSystem,
        StrategyType
    )
    ENSEMBLE_AVAILABLE = True
except ImportError:
    print("⚠️ ensemble_strategies.py no encontrado")
    ENSEMBLE_AVAILABLE = False

try:
    from rl_position_sizing import (
        RLPositionSizer,
        PositionSizeCalculator,
        MarketState
    )
    RL_AVAILABLE = True
except ImportError:
    print("⚠️ rl_position_sizing.py no encontrado")
    RL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#                          CONFIGURAZIONE V4
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradingPair:
    yf_symbol: str
    kraken_pair: str
    min_volume: float
    allocation: float

class Config:
    # ══════════════════ APIs ══════════════════
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
    KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET', '')
    KRAKEN_API_URL = 'https://api.kraken.com'
    
    CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY', '')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # ══════════════════ V4 Features ══════════════════
    USE_SENTIMENT_ANALYSIS = os.getenv('USE_SENTIMENT_ANALYSIS', 'false').lower() == 'true'
    MIN_SENTIMENT_CONFIDENCE = float(os.getenv('MIN_SENTIMENT_CONFIDENCE', '0.5'))
    
    USE_ONCHAIN_ANALYSIS = os.getenv('USE_ONCHAIN_ANALYSIS', 'false').lower() == 'true'
    MIN_ONCHAIN_STRENGTH = float(os.getenv('MIN_ONCHAIN_STRENGTH', '0.5'))
    
    USE_ENSEMBLE_SYSTEM = os.getenv('USE_ENSEMBLE_SYSTEM', 'false').lower() == 'true'
    MIN_ENSEMBLE_CONSENSUS = float(os.getenv('MIN_ENSEMBLE_CONSENSUS', '0.6'))
    MIN_ENSEMBLE_CONFIDENCE = float(os.getenv('MIN_ENSEMBLE_CONFIDENCE', '0.6'))
    
    USE_RL_POSITION_SIZING = os.getenv('USE_RL_POSITION_SIZING', 'false').lower() == 'true'
    RL_LEARNING_RATE = float(os.getenv('RL_LEARNING_RATE', '0.1'))
    RL_DISCOUNT_FACTOR = float(os.getenv('RL_DISCOUNT_FACTOR', '0.95'))
    RL_EPSILON = float(os.getenv('RL_EPSILON', '0.1'))
    RL_STATE_FILE = os.getenv('RL_STATE_FILE', 'rl_state.json')
    
    # Ensemble weights
    WEIGHT_SWING = float(os.getenv('WEIGHT_SWING', '0.30'))
    WEIGHT_MOMENTUM = float(os.getenv('WEIGHT_MOMENTUM', '0.25'))
    WEIGHT_MEAN_REVERSION = float(os.getenv('WEIGHT_MEAN_REVERSION', '0.25'))
    WEIGHT_TREND_FOLLOWING = float(os.getenv('WEIGHT_TREND_FOLLOWING', '0.20'))
    
    # ══════════════════ Multi-Asset ══════════════════
    _ALL_PAIRS = [
        ('BTC-USD', 'XBTUSD', 0.0001, 0.30, os.getenv('PAIR_BTC', 'true')),
        ('ETH-USD', 'ETHUSD', 0.001,  0.25, os.getenv('PAIR_ETH', 'true')),
        ('ADA-USD', 'ADAUSD', 1.0,    0.25, os.getenv('PAIR_ADA', 'true')),
        ('SOL-USD', 'SOLUSD', 0.01,   0.20, os.getenv('PAIR_SOL', 'true')),
        ('XRP-USD', 'XRPUSD', 1.0,    0.25, os.getenv('PAIR_XRP', 'true')),
    ]
    TRADING_PAIRS = [
        TradingPair(yf, kr, mv, al)
        for yf, kr, mv, al, enabled in _ALL_PAIRS
        if enabled.lower() == 'true'
    ]
    
    MAX_CORRELATION = float(os.getenv('MAX_CORRELATION', '0.7'))  # ← Debe tener 4 espacios
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '1'))  # ← Debe tener 4 espacios
    
    # ══════════════════ Trading ══════════════════
    LEVERAGE = int(os.getenv('LEVERAGE', '3'))
    MIN_BALANCE = float(os.getenv('MIN_BALANCE', '1.0'))
    MARGIN_SAFETY_FACTOR = 1.5
    
    # ══════════════════ Risk ══════════════════
    BASE_STOP_LOSS = float(os.getenv('STOP_LOSS_PCT', '2.0'))
    BASE_TAKE_PROFIT = float(os.getenv('TAKE_PROFIT_PCT', '3.5'))
    BASE_TRAILING_STOP = float(os.getenv('TRAILING_STOP_PCT', '2.5'))
    MIN_PROFIT_FOR_TRAILING = float(os.getenv('MIN_PROFIT_FOR_TRAILING', '1.8'))
    
    # ══════════════════ Strategy ══════════════════
    LOOKBACK_PERIOD = os.getenv('LOOKBACK_PERIOD', '180d')
    CANDLE_INTERVAL = os.getenv('CANDLE_INTERVAL', '1h')
    USE_VOLUME_FILTER = os.getenv('USE_VOLUME_FILTER', 'true').lower() == 'true'
    REGIME_LOOKBACK = int(os.getenv('REGIME_LOOKBACK', '30'))
    
    USE_ML_VALIDATION = os.getenv('USE_ML_VALIDATION', 'true').lower() == 'true'
    ML_CONFIDENCE_THRESHOLD = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.6'))
    
    # ══════════════════ Mode ══════════════════
    DRY_RUN = os.getenv('DRY_RUN', 'true').lower() == 'true'


# ═══════════════════════════════════════════════════════════════════════════
#                    KRAKEN CLIENT (del V3)
# ═══════════════════════════════════════════════════════════════════════════

class KrakenClient:
    def __init__(self, api_key: str, api_secret: str, api_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = api_url
        self.session = requests.Session()
    
    def _sign(self, urlpath: str, data: dict) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()
    
    def _request(self, endpoint: str, data: dict = None, private: bool = False) -> dict:
        url = self.api_url + endpoint
        
        if private:
            data = data or {}
            data['nonce'] = int(time.time() * 1000)
            headers = {
                'API-Key': self.api_key,
                'API-Sign': self._sign(endpoint, data)
            }
            response = self.session.post(url, data=data, headers=headers, timeout=30)
        else:
            response = self.session.get(url, params=data, timeout=30)
        
        response.raise_for_status()
        result = response.json()
        
        if result.get('error') and len(result['error']) > 0:
            raise Exception(f"Kraken error: {result['error']}")
        
        return result.get('result', {})
    
    def get_balance(self) -> Tuple[float, str]:
        result = self._request('/0/private/Balance', private=True)
        balances = {k: float(v) for k, v in result.items()}
        
        fiat = {'ZUSD': 'USD', 'USD': 'USD', 'ZEUR': 'EUR', 'EUR': 'EUR'}
        
        for key, currency in fiat.items():
            if key in balances and balances[key] > 0:
                return balances[key], currency
        
        return 0.0, 'EUR'
    
    def get_available_margin(self) -> float:
        try:
            result = self._request('/0/private/TradeBalance', private=True)
            margin_free = float(result.get('mf', 0))
            print(f"   💰 Margeine disponibile: {margin_free:.2f} EUR")
            return margin_free
        except Exception as e:
            print(f"   ⚠️ Errore ottenendo margine: {e}")
            balance, _ = self.get_balance()
            return balance * 0.5
    
    def get_open_positions(self) -> Dict:
        try:
            result = self._request('/0/private/OpenPositions', private=True)
            
            if not result:
                return {}
            
            consolidated = {}
            
            for pos_id, pos_data in result.items():
                vol = float(pos_data.get('vol', 0))
                vol_closed = float(pos_data.get('vol_closed', 0))
                open_vol = vol - vol_closed
                
                if open_vol <= 0:
                    continue
                
                pair = pos_data.get('pair', 'UNKNOWN')
                
                if pair in consolidated:
                    existing_vol = float(consolidated[pair].get('vol', 0))
                    consolidated[pair]['vol'] = str(existing_vol + open_vol)
                    
                    existing_cost = float(consolidated[pair].get('cost', 0))
                    new_cost = float(pos_data.get('cost', 0))
                    consolidated[pair]['cost'] = str(existing_cost + new_cost)
                else:
                    pos_data['vol'] = str(open_vol)
                    consolidated[pair] = pos_data
            
            return consolidated
            
        except Exception as e:
            if "No open positions" in str(e):
                return {}
            raise
    
    def place_order(self, pair: str, order_type: str, volume: float, 
                leverage: int = None, reduce_only: bool = False) -> dict:
        data = {
            'pair': pair,
            'type': order_type,
            'ordertype': 'market',
            'volume': str(round(volume, 8))
        }
        
        if leverage and leverage > 1:
            data['leverage'] = str(leverage)
            if reduce_only:
                data['reduce_only'] = 'true'
        
        return self._request('/0/private/AddOrder', data=data, private=True)
    
    def close_position(self, pair: str, position_type: str, volume: float, 
                    leverage: int = None) -> dict:
        opposite_type = 'sell' if position_type == 'long' else 'buy'
        is_margin_position = leverage and leverage > 1
        
        if is_margin_position:
            return self.place_order(
                pair=pair,
                order_type=opposite_type,
                volume=volume,
                leverage=leverage,
                reduce_only=True
            )
        else:
            return self.place_order(
                pair=pair,
                order_type=opposite_type,
                volume=volume,
                leverage=None,
                reduce_only=False
            )


# ═══════════════════════════════════════════════════════════════════════════
#                    COMPONENTI V3 (Regime, ML, Correlation, Swing)
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    @staticmethod
    def detect(data: pd.DataFrame, lookback: int = 30) -> str:
        if len(data) < lookback:
            return 'RANGING'
        
        recent = data.tail(lookback)
        returns = recent['Close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_volatility = data['Close'].pct_change().dropna().std()
        
        high_low = (recent['High'] - recent['Low']).mean()
        close_change = abs(recent['Close'].iloc[-1] - recent['Close'].iloc[0])
        trend_strength = close_change / (high_low * lookback) if high_low > 0 else 0
        
        if volatility > avg_volatility * 1.5:
            return 'VOLATILE'
        elif trend_strength > 0.5:
            return 'TRENDING'
        else:
            return 'RANGING'
    
    @staticmethod
    def get_adapted_params(regime: str, base_sl: float, base_tp: float, 
                        base_trail: float) -> Dict[str, float]:
        adaptations = {
            'TRENDING': {
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 1.5,
                'trailing_stop_multiplier': 1.0,
            },
            'RANGING': {
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 0.7,
                'trailing_stop_multiplier': 0.8,
            },
            'VOLATILE': {
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.0,
                'trailing_stop_multiplier': 1.3,
            }
        }
        
        mult = adaptations.get(regime, adaptations['RANGING'])
        
        return {
            'stop_loss': base_sl * mult['stop_loss_multiplier'],
            'take_profit': base_tp * mult['take_profit_multiplier'],
            'trailing_stop': base_trail * mult['trailing_stop_multiplier']
        }


class MLSwingValidator:
    @staticmethod
    def calculate_features(data: pd.DataFrame, swing_idx: int) -> Dict[str, float]:
        if swing_idx < 20 or swing_idx >= len(data) - 5:
            return None
        
        window = data.iloc[swing_idx-20:swing_idx+5]
        features = {}
        
        avg_vol = window['Volume'].mean()
        swing_vol = data['Volume'].iloc[swing_idx]
        features['volume_ratio'] = swing_vol / avg_vol if avg_vol > 0 else 1.0
        
        returns = window['Close'].pct_change()
        features['momentum'] = returns.mean()
        features['volatility'] = returns.std()
        
        swing_price = data['Close'].iloc[swing_idx]
        recent_high = window['High'].max()
        recent_low = window['Low'].min()
        price_range = recent_high - recent_low
        features['price_position'] = (swing_price - recent_low) / price_range if price_range > 0 else 0.5
        
        sma_20 = window['Close'].mean()
        features['distance_from_sma'] = abs(swing_price - sma_20) / sma_20 if sma_20 > 0 else 0
        
        return features
    
    @staticmethod
    def validate_swing(data: pd.DataFrame, swing_idx: int, 
                    swing_type: str, threshold: float = 0.6) -> Tuple[bool, float]:
        features = MLSwingValidator.calculate_features(data, swing_idx)
        
        if features is None:
            return False, 0.0
        
        score = 0.0
        weights = 0.0
        
        if features['volume_ratio'] > 1.2:
            score += 0.3
        elif features['volume_ratio'] > 0.8:
            score += 0.15
        weights += 0.3
        
        if swing_type == 'LOW' and features['momentum'] < -0.001:
            score += 0.25
        elif swing_type == 'HIGH' and features['momentum'] > 0.001:
            score += 0.25
        elif abs(features['momentum']) < 0.0005:
            score += 0.125
        weights += 0.25
        
        if swing_type == 'LOW' and features['price_position'] < 0.3:
            score += 0.2
        elif swing_type == 'HIGH' and features['price_position'] > 0.7:
            score += 0.2
        weights += 0.2
        
        if features['distance_from_sma'] > 0.02:
            score += 0.15
        weights += 0.15
        
        if features['volatility'] > 0.01:
            score += 0.1
        weights += 0.1
        
        confidence = score / weights if weights > 0 else 0.0
        is_valid = confidence >= threshold
        
        return is_valid, confidence


class CorrelationManager:
    @staticmethod
    def calculate_correlation_matrix(data_dict: Dict[str, pd.DataFrame], 
                                    lookback: int = 30) -> pd.DataFrame:
        returns_dict = {}
        for symbol, data in data_dict.items():
            if len(data) >= lookback:
                returns = data['Close'].tail(lookback).pct_change().dropna()
                returns_dict[symbol] = returns
        
        if len(returns_dict) < 2:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_dict)
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    @staticmethod
    def check_position_correlation(open_positions: List[str], new_symbol: str,
                                corr_matrix: pd.DataFrame, 
                                max_corr: float = 0.7) -> Tuple[bool, float]:
        if corr_matrix.empty or new_symbol not in corr_matrix.columns:
            return True, 0.0
        
        max_correlation = 0.0
        
        for pos_symbol in open_positions:
            if pos_symbol in corr_matrix.columns and pos_symbol != new_symbol:
                corr = abs(corr_matrix.loc[new_symbol, pos_symbol])
                max_correlation = max(max_correlation, corr)
        
        can_open = max_correlation < max_corr
        
        return can_open, max_correlation


def calculate_volume_ma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    return data['Volume'].rolling(window=period).mean()

class SwingDetectorV3:
    def __init__(self, data: pd.DataFrame, volume_filter: bool = True, 
                use_ml: bool = True, ml_threshold: float = 0.6):
        self.data = data.copy()
        self.volume_filter = volume_filter
        self.use_ml = use_ml
        self.ml_threshold = ml_threshold
        self.volume_ma = calculate_volume_ma(data) if volume_filter else None
        
        self.st_highs = pd.Series(index=data.index, dtype=float)
        self.st_lows = pd.Series(index=data.index, dtype=float)
        self.int_highs = pd.Series(index=data.index, dtype=float)
        self.int_lows = pd.Series(index=data.index, dtype=float)
        self.ml_confidence = {}
    
    def _check_volume(self, i: int) -> bool:
        if not self.volume_filter or self.volume_ma is None:
            return True
        
        if pd.isna(self.volume_ma.iloc[i]):
            return True
        
        return self.data['Volume'].iloc[i] > self.volume_ma.iloc[i]
    
    def detect(self):
        highs = self.data['High'].values
        lows = self.data['Low'].values
        
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                if self._check_volume(i):
                    self.st_lows.iloc[i] = lows[i]
        
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                if self._check_volume(i):
                    self.st_highs.iloc[i] = highs[i]
        
        st_high_idx = self.st_highs.dropna().index.tolist()
        for i in range(1, len(st_high_idx) - 1):
            p, c, n = st_high_idx[i-1], st_high_idx[i], st_high_idx[i+1]
            
            if self.st_highs[c] > self.st_highs[p] and self.st_highs[c] > self.st_highs[n]:
                if self.use_ml:
                    idx = self.data.index.get_loc(c)
                    is_valid, confidence = MLSwingValidator.validate_swing(
                        self.data, idx, 'HIGH', self.ml_threshold
                    )
                    if is_valid:
                        self.int_highs[c] = self.st_highs[c]
                        self.ml_confidence[c] = confidence
                else:
                    self.int_highs[c] = self.st_highs[c]
        
        st_low_idx = self.st_lows.dropna().index.tolist()
        for i in range(1, len(st_low_idx) - 1):
            p, c, n = st_low_idx[i-1], st_low_idx[i], st_low_idx[i+1]
            
            if self.st_lows[c] < self.st_lows[p] and self.st_lows[c] < self.st_lows[n]:
                if self.use_ml:
                    idx = self.data.index.get_loc(c)
                    is_valid, confidence = MLSwingValidator.validate_swing(
                        self.data, idx, 'LOW', self.ml_threshold
                    )
                    if is_valid:
                        self.int_lows[c] = self.st_lows[c]
                        self.ml_confidence[c] = confidence
                else:
                    self.int_lows[c] = self.st_lows[c]
    
    def get_signal(self) -> Tuple[Optional[str], Optional[float], float]:
        self.detect()
        
        highs = self.int_highs.dropna()
        lows = self.int_lows.dropna()
        
        if len(highs) == 0 and len(lows) == 0:
            return None, None, 0.0
        
        # Ensure timezone compatibility
        tz = highs.index.tz if len(highs) > 0 else (lows.index.tz if len(lows) > 0 else None)
        min_ts = pd.Timestamp.min
        if tz:
            min_ts = min_ts.replace(tzinfo=tz)

        last_high = highs.index[-1] if len(highs) > 0 else min_ts
        last_low = lows.index[-1] if len(lows) > 0 else min_ts
        
        if last_low > last_high:
            confidence = self.ml_confidence.get(last_low, 1.0)
            return 'BUY', lows.iloc[-1], confidence
        elif last_high > last_low:
            confidence = self.ml_confidence.get(last_high, 1.0)
            return 'SELL', highs.iloc[-1], confidence
        else:
            return None, None, 0.0


class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}" if token else None
    
    def send(self, message: str) -> bool:
        if not self.api_url or not self.chat_id:
            print(f"📱 {message}")
            return False
        
        try:
            if len(message) > 4000:
                message = message[:3900] + "\n..."
            
            data = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}
            response = requests.post(f"{self.api_url}/sendMessage", data=data, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False


class PositionManagerV3:
    def __init__(self, config: Config, kraken: KrakenClient, telegram: Telegram):
        self.config = config
        self.kraken = kraken
        self.telegram = telegram
        self.peak_prices = {}
        self.position_regimes = {}
    
    def check_position(self, pos_id: str, pos_data: dict, current_price: float,
                    regime_params: Dict[str, float]) -> Tuple[bool, str]:
        pos_type = pos_data.get('type', 'long')
        entry_price = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        leverage = float(pos_data.get('leverage', 1))
        
        if pos_type == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
        
        stop_loss = regime_params['stop_loss']
        take_profit = regime_params['take_profit']
        trailing_stop = regime_params['trailing_stop']
        
        if pnl_pct <= -stop_loss:
            return True, f"🛑 Stop Loss: {pnl_pct:.2f}%"
        
        if pnl_pct >= take_profit:
            return True, f"🎯 Take Profit: {pnl_pct:.2f}%"
        
        if pnl_pct >= self.config.MIN_PROFIT_FOR_TRAILING:
            if pos_id not in self.peak_prices:
                self.peak_prices[pos_id] = current_price
            
            if pos_type == 'long' and current_price > self.peak_prices[pos_id]:
                self.peak_prices[pos_id] = current_price
            elif pos_type == 'short' and current_price < self.peak_prices[pos_id]:
                self.peak_prices[pos_id] = current_price
            
            peak = self.peak_prices[pos_id]
            if pos_type == 'long':
                peak_pnl = ((peak - entry_price) / entry_price) * 100 * leverage
            else:
                peak_pnl = ((entry_price - peak) / entry_price) * 100 * leverage
            
            drawdown = peak_pnl - pnl_pct
            
            if drawdown >= trailing_stop:
                return True, f"📉 Trailing: peak {peak_pnl:.2f}%, actual {pnl_pct:.2f}%"
        
        return False, ""
    
    def close_position(self, pair: str, pos_type: str, volume: float, 
                    reason: str, pos_data: dict, current_price: float):
        print(f"\n🔴 Chiudendo {pair} ({pos_type})")
        print(f"   Motivo: {reason}")
        
        leverage = int(float(pos_data.get('leverage', 1)))
        
        if not self.config.DRY_RUN:
            try:
                result = self.kraken.close_position(
                    pair, pos_type, volume, leverage
                )
                print(f"   ✓ Chiusa: {result}")
            except Exception as e:
                print(f"   ❌ Errore: {e}")
                return
        else:
            print(f"   🧪 [SIMULAZIONE]")
        
        entry = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        
        if pos_type == 'long':
            pnl_pct = ((current_price - entry) / entry) * 100 * leverage
        else:
            pnl_pct = ((entry - current_price) / entry) * 100 * leverage
        
        msg = f"""
🔴 <b>POSIZIONE CHIUSA</b>

<b>Par:</b> {pair}
<b>Tipo:</b> {pos_type.upper()}
<b>Entrata:</b> ${entry:.4f}
<b>Uscita:</b> ${current_price:.4f}
<b>PnL:</b> {pnl_pct:+.2f}%
<b>Leva:</b> {leverage}x
<b>Motivo:</b> {reason}
"""
        if self.config.DRY_RUN:
            msg = "🧪 <b>SIMULAZIONE</b>\n" + msg
        
        self.telegram.send(msg)


# ═══════════════════════════════════════════════════════════════════════════
#                    BOT V4 - ADVANCED AI SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class TradingBotV4:
    def __init__(self, config: Config):
        self.config = config
        self.kraken = KrakenClient(
            config.KRAKEN_API_KEY, 
            config.KRAKEN_API_SECRET, 
            config.KRAKEN_API_URL
        )
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.position_mgr = PositionManagerV3(config, self.kraken, self.telegram)
        
        # ═══════════════ INICIALIZAR COMPONENTES V4 ═══════════════
        
        # Sentiment Analyzer
        if config.USE_SENTIMENT_ANALYSIS and SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = SentimentAnalyzer(config.CRYPTOCOMPARE_API_KEY)
            print("   ✓ Sentiment Analyzer attivato")
        else:
            self.sentiment_analyzer = None
        
        # On-Chain Analyzer
        if config.USE_ONCHAIN_ANALYSIS and ONCHAIN_AVAILABLE:
            self.onchain_analyzer = OnChainAnalyzer(config.CRYPTOCOMPARE_API_KEY)
            print("   ✓ On-Chain Analyzer attivato")
        else:
            self.onchain_analyzer = None
        
        # Ensemble System
        if config.USE_ENSEMBLE_SYSTEM and ENSEMBLE_AVAILABLE:
            weights = {
                StrategyType.SWING: config.WEIGHT_SWING,
                StrategyType.MOMENTUM: config.WEIGHT_MOMENTUM,
                StrategyType.MEAN_REVERSION: config.WEIGHT_MEAN_REVERSION,
                StrategyType.TREND_FOLLOWING: config.WEIGHT_TREND_FOLLOWING
            }
            self.ensemble = EnsembleSystem(weights=weights)
            print("   ✓ Ensemble System attivato")
        else:
            self.ensemble = None
        
        # RL Position Sizer
        if config.USE_RL_POSITION_SIZING and RL_AVAILABLE:
            self.rl_sizer = RLPositionSizer(
                learning_rate=config.RL_LEARNING_RATE,
                discount_factor=config.RL_DISCOUNT_FACTOR,
                epsilon=config.RL_EPSILON,
                state_file=config.RL_STATE_FILE
            )
            self.rl_calculator = PositionSizeCalculator(self.rl_sizer)
            print("   ✓ RL Position Sizing attivato")
            
            # ✅ NUEVO: Crear archivo vacío si no existe
            self._initialize_rl_state_file()
        else:
            self.rl_sizer = None
            self.rl_calculator = None
        
        # Historial de trades (para RL y Reportes)
        self.trades_file = "trades_v4.json"
        self.trades_history = self._load_trades_history()
        
    def _load_trades_history(self) -> List[Dict]:
        """Carica lo storico dei trade da file JSON."""
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    trades = json.load(f)
                    # Convertire timestamp stringa in datetime
                    for t in trades:
                        if 'timestamp' in t and isinstance(t['timestamp'], str):
                            t['timestamp'] = datetime.fromisoformat(t['timestamp'])
                    print(f"   📂 Caricati {len(trades)} trade dallo storico.")
                    return trades
        except Exception as e:
            print(f"   ⚠️ Errore caricamento storico: {e}")
        return []

    def _save_trades_history(self):
        """Salva lo storico dei trade su file JSON."""
        try:
            # Serializzazione custom per datetime
            def json_serial(obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")

            with open(self.trades_file, 'w') as f:
                json.dump(self.trades_history, f, default=json_serial, indent=2)
        except Exception as e:
            print(f"   ⚠️ Errore salvataggio storico: {e}")

    def _close_trade_in_history(self, symbol: str, exit_price: float, reason: str):
        """Aggiorna lo storico segnando il trade come chiuso."""
        # Cerca l'ultimo trade aperto per questo simbolo
        for trade in reversed(self.trades_history):
            if trade['symbol'] == symbol and not trade.get('closed', False):
                trade['closed'] = True
                trade['exit_price'] = exit_price
                trade['exit_time'] = datetime.now()
                trade['exit_reason'] = reason
                
                # Calcolo PnL approssimativo (assumiamo LONG come da strategia principale)
                entry = trade['entry_price']
                leverage = trade.get('leverage', 1)
                
                # Se il trade ha info sul tipo (long/short), usalo. Altrimenti default a LONG.
                # Nota: al momento open_position salva 'signal' che è BUY/SELL.
                is_short = trade.get('signal') == 'SELL'
                
                if is_short:
                    pnl_pct = ((entry - exit_price) / entry) * 100 * leverage
                else:
                    pnl_pct = ((exit_price - entry) / entry) * 100 * leverage
                
                trade['pnl_pct'] = pnl_pct
                trade['profit_amount'] = (trade['capital'] * pnl_pct) / 100
                
                print(f"   💾 Trade aggiornato in storico: {symbol} PnL: {pnl_pct:.2f}%")
                self._save_trades_history()
                return
        print(f"   ⚠️ Nessun trade aperto trovato in storico per {symbol}")

    def _initialize_rl_state_file(self):
        """✅ NUOVO: Inizializza file RL state se non esiste"""
        try:
            if not os.path.exists(self.config.RL_STATE_FILE):
                with open(self.config.RL_STATE_FILE, 'w') as f:
                    json.dump({
                        'q_table': {}, 
                        'metadata': {
                            'created': datetime.now().isoformat(),
                            'num_states': 0
                        }
                    }, f)
                print(f"   📝 RL state file inizializzato: {self.config.RL_STATE_FILE}")
        except Exception as e:
            print(f"   ⚠️ Errore inizializzazione RL state: {e}")
    
    def get_market_data(self, symbol: str) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            period=self.config.LOOKBACK_PERIOD, 
            interval=self.config.CANDLE_INTERVAL
        )
        
        if data.empty:
            raise Exception(f"No data for {symbol}")
        
        return data
    
    def analyze_trading_opportunity(self, 
                                pair: TradingPair,
                                data: pd.DataFrame,
                                swing_signal: Tuple) -> Dict:
        """
        ═══════════════════════════════════════════════════════════════
        ANÁLISIS MULTI-LAYER V4
        ═══════════════════════════════════════════════════════════════
        """
        symbol = pair.yf_symbol
        signal, signal_price, swing_confidence = swing_signal
        
        print(f"\n🔍 Analisi Multi-Layer: {symbol}")
        print(f"   Swing Signal: {signal} (conf: {swing_confidence:.2f})")
        
        result = {
            'can_trade': False,
            'final_signal': None,
            'confidence': 0.0,
            'reasons': [],
            'capital': 0.0,
            'leverage': self.config.LEVERAGE,
            'v4_data': {}
        }
        
        # ═══════════════ LAYER 1: SENTIMENT ANALYSIS ═══════════════
        
        if self.sentiment_analyzer:
            print(f"\n   📊 Layer 1: Analisi Sentiment")
            try:
                sentiment = self.sentiment_analyzer.get_sentiment(symbol)
                
                if sentiment:
                    result['v4_data']['sentiment'] = {
                        'overall': sentiment.overall_score,
                        'news': sentiment.news_score,
                        'social': sentiment.social_score,
                        'signal_type': 'BULLISH' if sentiment.is_bullish() else ('BEARISH' if sentiment.is_bearish() else 'NEUTRAL')
                    }
                    
                    can_trade_sentiment = should_trade_based_on_sentiment(
                        sentiment,
                        signal,
                        min_confidence=self.config.MIN_SENTIMENT_CONFIDENCE
                    )
                    
                    if not can_trade_sentiment:
                        result['reasons'].append(
                            f"❌ Sentiment conflictivo: {sentiment.overall_score:.2f}"
                        )
                        print(f"   ❌ Sentiment rifiuta: {signal}")
                        return result
                    
                    result['reasons'].append(
                        f"✓ Sentiment: {sentiment.overall_score:.2f} "
                        f"({result['v4_data']['sentiment']['signal_type']})"
                    )
                    print(f"   ✓ Sentiment conferma")
                else:
                    print(f"   ℹ️ Sentiment non disponibile")
            except Exception as e:
                print(f"   ⚠️ Errore in sentiment: {e}")
        
        # ═══════════════ LAYER 2: ON-CHAIN METRICS ═══════════════
        
        if self.onchain_analyzer:
            print(f"\n   🔗 Layer 2: Metriche On-Chain")
            try:
                onchain = self.onchain_analyzer.get_onchain_signal(symbol)
                
                if onchain:
                    result['v4_data']['onchain'] = {
                        'signal_type': onchain.signal_type,
                        'strength': onchain.strength,
                        'metrics': onchain.metrics
                    }
                    
                    can_trade_onchain = should_trade_based_on_onchain(
                        onchain,
                        signal,
                        min_strength=self.config.MIN_ONCHAIN_STRENGTH
                    )
                    
                    if not can_trade_onchain:
                        result['reasons'].append(
                            f"❌ On-Chain conflittivo: {onchain.signal_type}"
                        )
                        print(f"   ❌ On-Chain rifiuta: {signal}")
                        return result
                    
                    result['reasons'].append(
                        f"✓ On-Chain: {onchain.signal_type} (strength: {onchain.strength:.2f})"
                    )
                    print(f"   ✓ On-Chain conferma")
                else:
                    print(f"   ℹ️ On-Chain non disponibile")
            except Exception as e:
                print(f"   ⚠️ Errore in on-chain: {e}")
        
        # ═══════════════ LAYER 3: ENSEMBLE STRATEGIES ═══════════════
        
        if self.ensemble:
            print(f"\n   🎯 Layer 3: Ensemble Strategies")
            try:
                ensemble_decision = self.ensemble.get_ensemble_decision(
                    data, swing_signal
                )
                
                self.ensemble.print_decision_summary(ensemble_decision)
                
                result['v4_data']['ensemble'] = {
                    'final_signal': ensemble_decision.final_signal,
                    'confidence': ensemble_decision.confidence,
                    'consensus': ensemble_decision.consensus_level,
                    'votes': {str(k): str(v) for k, v in ensemble_decision.votes.items()}
                }
                
                # Verificar consenso y confianza
                if (ensemble_decision.final_signal != signal or
                    ensemble_decision.consensus_level < self.config.MIN_ENSEMBLE_CONSENSUS or
                    ensemble_decision.confidence < self.config.MIN_ENSEMBLE_CONFIDENCE):
                    
                    result['reasons'].append(
                        f"❌ Ensemble: {ensemble_decision.final_signal} "
                        f"(consensus: {ensemble_decision.consensus_level:.2f}, "
                        f"conf: {ensemble_decision.confidence:.2f})"
                    )
                    print(f"   ❌ Ensemble no confirma")
                    return result
                
                result['reasons'].append(
                    f"✓ Ensemble: {ensemble_decision.final_signal} "
                    f"(consensus: {ensemble_decision.consensus_level:.2%}, "
                    f"conf: {ensemble_decision.confidence:.2%})"
                )
                result['confidence'] = ensemble_decision.confidence
                print(f"   ✓ Ensemble conferma con {ensemble_decision.consensus_level:.0%} consenso")
            except Exception as e:
                print(f"   ⚠️ Errore in ensemble: {e}")
                traceback.print_exc()
                result['confidence'] = swing_confidence
        else:
            result['confidence'] = swing_confidence
        
        # ═══════════════ DECISIÓN FINAL ═══════════════
        
        result['can_trade'] = True
        result['final_signal'] = signal
        
        print(f"\n✅ DECISIONE: {signal}")
        print(f"   Confidenza finale: {result['confidence']:.2%}")
        
        return result
    
    def calculate_position_size(self, 
                            pair: TradingPair,
                            data: pd.DataFrame,
                            analysis_result: Dict,
                            available_margin: float) -> Tuple[float, int]:
        """
        ═══════════════════════════════════════════════════════════════
        LAYER 4: RL POSITION SIZING (o tradicional)
        ═══════════════════════════════════════════════════════════════
        """
        
        if self.rl_calculator:
            print(f"\n   🤖 Layer 4: RL Position Sizing")
            try:
                # Obtener posiciones abiertas
                positions = self.kraken.get_open_positions()
                open_positions_count = len(positions)
                
                # Calcular tamaño óptimo con RL
                capital, leverage = self.rl_calculator.get_optimal_size(
                    data=data,
                    signal_confidence=analysis_result['confidence'],
                    available_capital=available_margin,
                    base_leverage=self.config.LEVERAGE,
                    open_positions=open_positions_count,
                    recent_trades=self.trades_history[-20:],  # Últimos 20 trades
                    training=True  # Modo entrenamiento
                )
                
                analysis_result['v4_data']['rl_sizing'] = {
                    'capital': capital,
                    'leverage': leverage,
                    'base_leverage': self.config.LEVERAGE
                }
                
                return capital, leverage
                
            except Exception as e:
                print(f"   ⚠️ Errore in RL sizing: {e}")
                traceback.print_exc()
        
        # Fallback: sizing tradizionale
        print(f"\n   💰 Position Sizing tradizionale")
        allocation = pair.allocation
        capital = available_margin * allocation
        leverage = self.config.LEVERAGE
        
        return capital, leverage
    
    def open_position(self, 
            pair: TradingPair,
            params: dict):
        # Estraiamo i dati dal pacchetto 'params' 📦
        capital = params['capital']
        price = params['price']
        leverage = params['leverage']
        analysis = params['analysis']  # Qui c'è il "perché" dell'AI
        data = params['data']
        signal = analysis.get('final_signal') if 'final_signal' in analysis else None
        confidence = analysis.get('confidence') if 'confidence' in analysis else None

        print(f"DEBUG: L'AI ha deciso di entrare perché: {analysis.get('reason', 'N/A')}")
        # Calcolare volume
        volume = (capital * leverage) / price

        # Verifica volume minimo
        if volume < pair.min_volume:
            print(f"   ⚠️ Volume {volume:.8f} < minimo {pair.min_volume}")
            return

        try:
            print(f"\n🟢 Apertura {signal} su {pair.yf_symbol}")
            print(f"   Prezzo: ${price:.4f}")
            print(f"   Capitale: ${capital:.2f}")
            print(f"   Leva: {leverage}x")
            print(f"   Volume: {volume:.8f}")
            print(f"   Confidenza: {confidence:.2%}")

            # Salva sempre type e vol per la chiusura
            trade_record = {
                'symbol': pair.yf_symbol,
                'entry_price': price,
                'volume': volume,
                'leverage': leverage,
                'capital': capital,
                'signal': signal,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'v4_data': analysis.get('v4_data', {}),
                'closed': False,
                'mode': 'SIMULATION' if self.config.DRY_RUN else 'REAL',
                'type': 'long' if signal == 'BUY' else 'short',
                'vol': volume
            }
            self.trades_history.append(trade_record)
            self._save_trades_history()  # ✅ Salva su file

            if not self.config.DRY_RUN:
                order_type = 'buy' if signal == 'BUY' else 'sell'
                result = self.kraken.place_order(
                    pair=pair.kraken_pair,
                    order_type=order_type,
                    volume=volume,
                    leverage=leverage,
                    reduce_only=False
                )
                print(f"   ✓ Eseguita: {result}")
            else:
                print(f"   🧪 [SIMULAZIONE] Trade registrato")

            # Notifica V4
            self._send_v4_notification(
                pair, signal, price, volume, leverage, 
                confidence, analysis['reasons'], analysis.get('v4_data', {})
            )
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Errore: {error_msg}")
            self.telegram.send(f"❌ Errore in {pair.yf_symbol}: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Errore: {error_msg}")
            self.telegram.send(f"❌ Errore in {pair.yf_symbol}: {error_msg}")
    
    def _send_v4_notification(self, pair, signal, price, volume, 
                            leverage, confidence, reasons, v4_data):
        """Notifica Telegram con dettagli V4."""
        
        reasons_text = "\n".join([f"• {r}" for r in reasons])
        
        # Extraer datos V4 para mostrar
        v4_summary = []
        
        if 'sentiment' in v4_data:
            sent = v4_data['sentiment']
            v4_summary.append(f"Sentiment: {sent['signal_type']} ({sent['overall']:.2f})")
        
        if 'onchain' in v4_data:
            onc = v4_data['onchain']
            v4_summary.append(f"On-Chain: {onc['signal_type']} ({onc['strength']:.2f})")
        
        if 'ensemble' in v4_data:
            ens = v4_data['ensemble']
            v4_summary.append(f"Ensemble: {ens['consensus']:.0%} consenso")
        
        if 'rl_sizing' in v4_data:
            rl = v4_data['rl_sizing']
            v4_summary.append(f"RL: ${rl['capital']:.2f} @ {rl['leverage']}x")
        
        v4_text = "\n".join(v4_summary) if v4_summary else "N/A"
        
        msg = f"""
🟢 <b>NUOVA POSIZIONE V4</b>

<b>Coppia:</b> {pair.yf_symbol} ({pair.kraken_pair})
<b>Segnale:</b> {signal}
<b>Prezzo:</b> ${price:.4f}
<b>Volume:</b> {volume:.8f}
<b>Leva:</b> {leverage}x

<b>🤖 Analisi AI:</b>
<b>Confidenza:</b> {confidence:.1%}
{v4_text}

<b>📊 Validazioni:</b>
{reasons_text}

<b>Data:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        if self.config.DRY_RUN:
            msg = "🧪 <b>SIMULAZIONE</b>\n" + msg
        
        self.telegram.send(msg)
    
    def force_close_all_positions(self, reason: str = "🔄 Reset manuale richiesto") -> int:
        """Chiude forzatamente tutte le posizioni aperte (simulazione o real)."""
        print("\n🧹 Avvio chiusura forzata di tutte le posizioni...")
        closed_count = 0

        # Scarica i dati di mercato necessari per calcolare il PnL corrente
        market_data = {}
        for pair in self.config.TRADING_PAIRS:
            try:
                market_data[pair.yf_symbol] = self.get_market_data(pair.yf_symbol)
            except Exception as e:
                print(f"   ⚠️ Dati mancanti per {pair.yf_symbol}: {e}")

        if self.config.DRY_RUN:
            open_trades = [t for t in self.trades_history if not t.get('closed', False)]

            if not open_trades:
                print("   Nessuna posizione simulata aperta da chiudere.")
                return 0

            for trade in open_trades:
                symbol = trade['symbol']
                data = market_data.get(symbol)
                if data is not None and not data.empty:
                    current_price = float(data['Close'].iloc[-1])
                else:
                    current_price = trade['entry_price']
                entry = trade['entry_price']
                leverage = trade.get('leverage', 1)
                is_short = trade.get('signal') == 'SELL'

                if is_short:
                    pnl_pct = ((entry - current_price) / entry) * 100 * leverage
                else:
                    pnl_pct = ((current_price - entry) / entry) * 100 * leverage

                trade['closed'] = True
                trade['exit_price'] = current_price
                trade['exit_time'] = datetime.now()
                trade['exit_reason'] = reason
                trade['pnl_pct'] = pnl_pct
                trade['profit_amount'] = (trade['capital'] * pnl_pct) / 100

                print(f"   • {symbol} chiuso @ {current_price:.4f} ({pnl_pct:+.2f}%)")
                closed_count += 1

            self._save_trades_history()
            print(f"✅ Chiuse {closed_count} posizioni in simulazione.")
            return closed_count

        # Modalita' REAL: chiude tramite API Kraken
        positions = self.kraken.get_open_positions()
        if not positions:
            print("   Nessuna posizione reale aperta da chiudere.")
            return 0

        for pair_key, pos_data in positions.items():
            trading_pair = next((tp for tp in self.config.TRADING_PAIRS if tp.kraken_pair == pair_key), None)
            if not trading_pair:
                continue

            symbol = trading_pair.yf_symbol
            data = market_data.get(symbol)
            if data is None:
                try:
                    data = self.get_market_data(symbol)
                    market_data[symbol] = data
                except Exception as e:
                    print(f"   ⚠️ Prezzo non disponibile per {symbol}: {e}")
                    continue

            current_price = float(data['Close'].iloc[-1])
            pos_type = pos_data.get('type', 'long')
            volume = float(pos_data.get('vol', 0))

            self.position_mgr.close_position(
                pair_key,
                pos_type,
                volume,
                reason,
                pos_data,
                current_price
            )
            self._close_trade_in_history(symbol, current_price, reason)
            closed_count += 1

        print(f"✅ Chiuse {closed_count} posizioni reali.")
        return closed_count
    
    def run(self):
        """
        ═══════════════════════════════════════════════════════════════
        LOOP PRINCIPAL V4
        ═══════════════════════════════════════════════════════════════
        """
        print("\n" + "="*70)
        print("KRAKEN TRADING BOT V4 - ADVANCED AI SYSTEM")
        print("="*70)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {'🧪 SIMULAZIONE' if self.config.DRY_RUN else '💰 REAL'}")
        print(f"ML Validation: {'✅' if self.config.USE_ML_VALIDATION else '❌'}")
        
        # Mostrar features V4 activas
        print("\n🤖 AI Features V4:")
        print(f"   Sentiment Analysis: {'✅' if self.sentiment_analyzer else '❌'}")
        print(f"   On-Chain Metrics: {'✅' if self.onchain_analyzer else '❌'}")
        print(f"   Ensemble System: {'✅' if self.ensemble else '❌'}")
        print(f"   RL Position Sizing: {'✅' if self.rl_calculator else '❌'}")
        print("="*70)
        
        try:
            # Obtener balance y margen
            balance, currency = self.kraken.get_balance()
            available_margin = self.kraken.get_available_margin()
            
            print(f"\n💰 Saldo: {balance:.2f} {currency}")
            print(f"   Margine disponibile: {available_margin:.2f} {currency}")
            
            if balance < self.config.MIN_BALANCE:
                print(f"⚠️ Saldo insufficiente (min: {self.config.MIN_BALANCE})")
                return
            
            usable_margin = available_margin / self.config.MARGIN_SAFETY_FACTOR
            print(f"   Margine utilizzabile: {usable_margin:.2f} {currency}")
            
            # Scaricare dati di mercato
            print("\n📊 Scaricamento dati multi-asset...")
            market_data = {}
            for pair in self.config.TRADING_PAIRS:
                try:
                    data = self.get_market_data(pair.yf_symbol)
                    market_data[pair.yf_symbol] = data
                    print(f"   ✓ {pair.yf_symbol}: {len(data)} candele")
                except Exception as e:
                    print(f"   ❌ {pair.yf_symbol}: {e}")
            
            if not market_data:
                print("❌ Non è stato possibile scaricare i dati")
                return
            
            # Calcolare correlazioni
            print("\n🔗 Calcolando correlazioni...")
            corr_matrix = CorrelationManager.calculate_correlation_matrix(
                market_data, lookback=30
            )
            
            if not corr_matrix.empty:
                print("   Matrice di correlazione:")
                print(corr_matrix.round(2))
            
            # Verificare posizioni aperte
            print("\n📊 Verificando posizioni APERTE...")
            
            if self.config.DRY_RUN:
                # In simulation mode, build positions from history to avoid duplicates
                positions = {}
                print("   🧪 [SIMULATION] Ricostruzione posizioni dallo storico...")
                for trade in self.trades_history:
                    if not trade.get('closed', False):
                        symbol = trade.get('symbol')
                        tp = next((p for p in self.config.TRADING_PAIRS if p.yf_symbol == symbol), None)
                        if tp:
                            pair_key = tp.kraken_pair
                            vol = float(trade.get('volume', 0))
                            cost = float(trade.get('entry_price', 0)) * vol
                            lev = float(trade.get('leverage', 1))
                            margin = cost / lev if lev > 0 else cost
                            
                            if pair_key in positions:
                                p = positions[pair_key]
                                p['vol'] = str(float(p['vol']) + vol)
                                p['cost'] = str(float(p['cost']) + cost)
                                p['margin'] = float(p.get('margin', 0)) + margin
                            else:
                                positions[pair_key] = {
                                    'pair': pair_key,
                                    'type': trade.get('type', 'long'),
                                    'vol': str(vol),
                                    'cost': str(cost),
                                    'margin': margin,
                                    'leverage': str(lev)
                                }
            else:
                positions = self.kraken.get_open_positions()
            
            open_symbols = []
            total_margin_used = 0.0
            valid_position_count = len(positions)
            
            print(f"✅ {valid_position_count} posizione(i) attive")
            
            if positions:
                for pair_key, pos_data in positions.items():
                    pos_margin = float(pos_data.get('margin', 0))
                    total_margin_used += pos_margin
                    
                    # Encontrar trading pair
                    trading_pair = next(
                        (tp for tp in self.config.TRADING_PAIRS if tp.kraken_pair == pair_key),
                        None
                    )
                    
                    if not trading_pair or trading_pair.yf_symbol not in market_data:
                        continue
                    
                    open_symbols.append(trading_pair.yf_symbol)
                    data = market_data[trading_pair.yf_symbol]
                    current_price = float(data['Close'].iloc[-1])
                    
                    # Detectar régimen
                    regime = RegimeDetector.detect(data, self.config.REGIME_LOOKBACK)
                    regime_params = RegimeDetector.get_adapted_params(
                        regime, 
                        self.config.BASE_STOP_LOSS,
                        self.config.BASE_TAKE_PROFIT,
                        self.config.BASE_TRAILING_STOP
                    )
                    
                    print(f"\n   {trading_pair.yf_symbol} ({pair_key}) - Regime: {regime}")
                    print(f"   Margine usato: {pos_margin:.2f} {currency}")
                    
                    # 1. Verificare SL/TP/Trailing
                    print(f"[DEBUG] Check chiusura: {pair_key} | Prezzo attuale: {current_price} | Dati posizione: {pos_data}")
                    should_close, reason = self.position_mgr.check_position(
                        pair_key, pos_data, current_price, regime_params
                    )
                    print(f"[DEBUG] Risultato check_position: should_close={should_close}, reason={reason}")
                    
                    # 2. Verificare Inversione di Segnale (Signal Reversal)
                    if not should_close:
                        detector = SwingDetectorV3(
                            data, 
                            volume_filter=self.config.USE_VOLUME_FILTER,
                            use_ml=self.config.USE_ML_VALIDATION,
                            ml_threshold=self.config.ML_CONFIDENCE_THRESHOLD
                        )
                        curr_signal, _, curr_conf = detector.get_signal()
                        pos_type = pos_data.get('type', 'long')
                        
                        if pos_type == 'long' and curr_signal == 'SELL':
                            should_close = True
                            reason = f"🔄 Inversione Trend (Segnale SELL, conf {curr_conf:.2f})"
                        elif pos_type == 'short' and curr_signal == 'BUY':
                            should_close = True
                            reason = f"🔄 Inversione Trend (Segnale BUY, conf {curr_conf:.2f})"
                    
                    if should_close:
                        pos_type = pos_data.get('type', 'long')
                        volume = float(pos_data.get('vol', 0))
                        self.position_mgr.close_position(
                            pair_key, pos_type, volume, reason, pos_data, current_price
                        )
                        
                        # ✅ Aggiorna storico trades (anche in simulazione)
                        self._close_trade_in_history(trading_pair.yf_symbol, current_price, reason)
                        
                        total_margin_used -= pos_margin
                        valid_position_count -= 1
                        
                        # Actualizar RL si está activo
                        if self.rl_sizer:
                            self._update_rl_on_close(
                                trading_pair.yf_symbol, 
                                pos_data, 
                                current_price, 
                                reason
                            )
                    else:
                        print(f"   ✓ Mantenere posizione")
            else:
                print("✓ Nessuna posizione aperta")
            
            print(f"\n💰 Margine usato: {total_margin_used:.2f} {currency}")
            print(f"   Margine residuo: {(available_margin - total_margin_used):.2f} {currency}")
            print(f"   Posizioni attive: {valid_position_count}/{self.config.MAX_POSITIONS}")
            
            # Verificar si podemos abrir nuevas posiciones
            if valid_position_count >= self.config.MAX_POSITIONS:
                print(f"\nℹ️ Massimo posizioni raggiunte ({self.config.MAX_POSITIONS})")
                return
            
            margin_for_new = (available_margin - total_margin_used) / self.config.MARGIN_SAFETY_FACTOR
            print(f"   Margine per nuove posizioni: {margin_for_new:.2f} {currency}")
            
            if margin_for_new < self.config.MIN_BALANCE * 0.5:
                print(f"⚠️ Margine insufficiente per nuove posizioni")
                return
            
            # Buscar señales en activos disponibles
            print("\n🔍 Ricerca segnali con analisi V4...")
            
            validated_signals = []
            
            for pair in self.config.TRADING_PAIRS:
                if pair.yf_symbol not in market_data:
                    continue
                
                if pair.yf_symbol in open_symbols:
                    print(f"   ⭕ {pair.yf_symbol}: posizione già aperta")
                    continue
                
                data = market_data[pair.yf_symbol]
                current_price = float(data['Close'].iloc[-1])
                
                # Detectar swing signal (V3)
                detector = SwingDetectorV3(
                    data, 
                    volume_filter=self.config.USE_VOLUME_FILTER,
                    use_ml=self.config.USE_ML_VALIDATION,
                    ml_threshold=self.config.ML_CONFIDENCE_THRESHOLD
                )
                signal, signal_price, confidence = detector.get_signal()
                
                if not signal:
                    print(f"   - {pair.yf_symbol}: nessun segnale swing")
                    continue
                
                print(f"\n   🎯 {pair.yf_symbol}: Segnale {signal} rilevato")
                
                # ═══════════════════════════════════════════════════
                #       ANÁLISIS COMPLETO V4
                # ═══════════════════════════════════════════════════
                
                analysis = self.analyze_trading_opportunity(
                    pair, data, (signal, signal_price, confidence)
                )
                
                if not analysis['can_trade']:
                    print(f"   ❌ Rifiutato da analisi V4")
                    continue
                
                # Verificar correlación
                can_open, max_corr = CorrelationManager.check_position_correlation(
                    open_symbols, pair.yf_symbol, corr_matrix, self.config.MAX_CORRELATION
                )
                
                if not can_open:
                    print(f"   ⚠️ Rifiutato per correlazione ({max_corr:.2f})")
                    continue
                
                # Señal validada
                validated_signals.append({
                    'pair': pair,
                    'data': data,
                    'analysis': analysis,
                    'current_price': current_price
                })
                
                print(f"   ✅ {pair.yf_symbol} validato - Confidenza: {analysis['confidence']:.2%}")
            
            if not validated_signals:
                print("\nℹ️ Nessun segnale valido dopo l'analisi V4")
                return
            
            # Ordenar por confianza
            validated_signals.sort(key=lambda x: x['analysis']['confidence'], reverse=True)
            
            # Abrir posiciones
            positions_to_open = min(
                len(validated_signals), 
                self.config.MAX_POSITIONS - valid_position_count
            )
            
            print(f"\n🎯 Apertura di {positions_to_open} posizione/i con AI V4...")
            
            remaining_margin = margin_for_new
            
            for sig in validated_signals[:positions_to_open]:
                # Calcular tamaño ANTES de abrir
                capital, leverage = self.calculate_position_size(
                    sig['pair'],
                    sig['data'],
                    sig['analysis'],
                    remaining_margin
                )
                
                # Abrir con valores ya calculados
                # Creiamo il pacchetto di parametri per la funzione
                params = {
                    'analysis': sig['analysis'],
                    'data': sig['data'],
                    'price': sig['current_price'],
                    'capital': capital,
                    'leverage': leverage
                }

                # Chiamiamo la funzione passando solo il pair e il dizionario params
                self.open_position(sig['pair'], params)
                
                # Restar margen usado para siguiente posición
                margin_used = capital
                remaining_margin = max(0, remaining_margin - margin_used)
                
                if remaining_margin < self.config.MIN_BALANCE * 0.5:
                    print(f"   ⚠️ Margine residuo insufficiente, interrompo le aperture")
                    break
            
            # ✅ CORREGIDO: Guardar siempre (no solo en modo REAL)
            if self.rl_sizer:
                self.rl_sizer.save_state()
                print(f"\n💾 Stato RL salvato")
            
            print("\n✅ Ciclo completado")
            
        except Exception as e:
            msg = f"Error: {str(e)}"
            print(f"\n❌ {msg}")
            traceback.print_exc()
            self.telegram.send(f"❌ Bot V4 Error: {msg}")
            raise
    
    def _update_rl_on_close(self, symbol: str, pos_data: dict, 
                        exit_price: float, reason: str):
        """
        Attualizza RL agent quando si chiude una posizione.
        """
        if not self.rl_sizer:
            return
        
        try:
            # Calcolare PnL
            entry_price = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
            leverage = float(pos_data.get('leverage', 1))
            pos_type = pos_data.get('type', 'long')
            
            if pos_type == 'long':
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100 * leverage
            else:
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100 * leverage
            
            # Trovare trade nello storico
            matching_trades = [
                t for t in self.trades_history 
                if t['symbol'] == symbol and not t.get('closed', False)
            ]
            
            if not matching_trades:
                return
            
            trade = matching_trades[-1]  # Último trade de este símbolo
            
            # Marcar como cerrado
            trade['closed'] = True
            trade['exit_price'] = exit_price
            trade['pnl_pct'] = pnl_pct
            trade['exit_reason'] = reason
            
            # Calcular reward
            trade_result = {
                'closed': True,
                'pnl_pct': pnl_pct,
                'exit_reason': reason
            }
            
            reward = self.rl_sizer.calculate_reward(trade_result)
            
            print(f"   🤖 RL: reward={reward:.3f} per PnL={pnl_pct:.2f}%")
            self._save_trades_history()  # ✅ Salva aggiornamento chiusura

        except Exception as e:
            print(f"   ⚠️ Error actualizando RL: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Entry point del bot V4."""
    
    print("\n" + "╔"*70)
    print("🚀 INIZIALIZZAZIONE KRAKEN TRADING BOT V4")
    print("╔"*70)
    
    config = Config()
    
    # Verificar credenciales básicas
    if not config.KRAKEN_API_KEY or not config.KRAKEN_API_SECRET:
        print("❌ Mancano credenziali Kraken")
        return
    
    # Verificar APIs V4
    missing_apis = []
    
    if config.USE_SENTIMENT_ANALYSIS and not config.CRYPTOCOMPARE_API_KEY:
        missing_apis.append("CRYPTOCOMPARE_API_KEY (para Sentiment)")
    
    if config.USE_ONCHAIN_ANALYSIS and not config.CRYPTOCOMPARE_API_KEY:
        missing_apis.append("CRYPTOCOMPARE_API_KEY (para On-Chain)")
    
    if missing_apis:
        print("\n⚠️ ATTENZIONE: Funzionalità V4 disattivate per mancanza di API:")
        for api in missing_apis:
            print(f"   - {api}")
        print("\nIl bot funzionerà senza queste funzionalità.")
    
    # Verificar módulos V4
    missing_modules = []
    
    if config.USE_SENTIMENT_ANALYSIS and not SENTIMENT_AVAILABLE:
        missing_modules.append("sentiment_analyzer.py")
    
    if config.USE_ONCHAIN_ANALYSIS and not ONCHAIN_AVAILABLE:
        missing_modules.append("onchain_metrics.py")
    
    if config.USE_ENSEMBLE_SYSTEM and not ENSEMBLE_AVAILABLE:
        missing_modules.append("ensemble_strategies.py")
    
    if config.USE_RL_POSITION_SIZING and not RL_AVAILABLE:
        missing_modules.append("rl_position_sizing.py")
    
    if missing_modules:
        print("\n⚠️ ATTENZIONE: Moduli V4 non trovati:")
        for mod in missing_modules:
            print(f"   - {mod}")
        print("\nIl bot funzionerà senza queste funzionalità.")
    
    print("\n" + "╔"*70)
    
    # Inicializar y ejecutar bot
    bot = TradingBotV4(config)
    bot.run()
    
    print("\n" + "╔"*70)
    print("✅ ESECUZIONE BOT V4 COMPLETATA")
    print("╔"*70)


if __name__ == "__main__":
    main()
