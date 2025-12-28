"""
EJEMPLO DE INTEGRACIÓN V4
Muestra cómo integrar todos los módulos nuevos en el bot V3 existente
"""

import os
from typing import Dict, Optional, Tuple
import pandas as pd

# Importar módulos V4
from sentiment_analyzer import (
    SentimentAnalyzer, 
    should_trade_based_on_sentiment
)
from onchain_metrics import (
    OnChainAnalyzer,
    should_trade_based_on_onchain
)
from ensemble_strategies import (
    EnsembleSystem,
    integrate_ensemble_with_existing
)
from rl_position_sizing import (
    RLPositionSizer,
    PositionSizeCalculator
)

# ═══════════════════════════════════════════════════════════════════════════
#                   MODIFICACIONES AL BOT V3
# ═══════════════════════════════════════════════════════════════════════════

class TradingBotV4:
    """
    Extensión del bot V3 con funcionalidades V4.
    
    Esta clase muestra cómo integrar:
    - Sentiment analysis
    - On-chain metrics
    - Ensemble strategies
    - RL position sizing
    """
    
    def __init__(self, config):
        """
        Inicializar bot V4 con nuevos componentes.
        """
        # Componentes V3 existentes
        self.config = config
        # ... (kraken, telegram, position_mgr, etc.)
        
        # ══════════════════════════════════════════════════════
        #         NUEVOS COMPONENTES V4
        # ══════════════════════════════════════════════════════
        
        # 1. Sentiment Analyzer
        if getattr(config, 'USE_SENTIMENT_ANALYSIS', False):
            self.sentiment_analyzer = SentimentAnalyzer(
                api_key=config.CRYPTOCOMPARE_API_KEY
            )
            print("   ✓ Sentiment Analyzer activado")
        else:
            self.sentiment_analyzer = None
        
        # 2. On-Chain Analyzer
        if getattr(config, 'USE_ONCHAIN_ANALYSIS', False):
            self.onchain_analyzer = OnChainAnalyzer(
                cryptocompare_api_key=config.CRYPTOCOMPARE_API_KEY
            )
            print("   ✓ On-Chain Analyzer activado")
        else:
            self.onchain_analyzer = None
        
        # 3. Ensemble System
        if getattr(config, 'USE_ENSEMBLE_SYSTEM', False):
            weights = {
                'swing': getattr(config, 'WEIGHT_SWING', 0.30),
                'momentum': getattr(config, 'WEIGHT_MOMENTUM', 0.25),
                'mean_reversion': getattr(config, 'WEIGHT_MEAN_REVERSION', 0.25),
                'trend_following': getattr(config, 'WEIGHT_TREND_FOLLOWING', 0.20)
            }
            self.ensemble_system = EnsembleSystem(weights=weights)
            print("   ✓ Ensemble System activado")
        else:
            self.ensemble_system = None
        
        # 4. RL Position Sizer
        if getattr(config, 'USE_RL_POSITION_SIZING', False):
            self.rl_sizer = RLPositionSizer(
                learning_rate=getattr(config, 'RL_LEARNING_RATE', 0.1),
                discount_factor=getattr(config, 'RL_DISCOUNT_FACTOR', 0.95),
                epsilon=getattr(config, 'RL_EPSILON', 0.1),
                state_file=getattr(config, 'RL_STATE_FILE', 'rl_state.json')
            )
            self.rl_calculator = PositionSizeCalculator(self.rl_sizer)
            print("   ✓ RL Position Sizing activado")
        else:
            self.rl_sizer = None
            self.rl_calculator = None
    
    # ══════════════════════════════════════════════════════
    #      MÉTODO PRINCIPAL MODIFICADO
    # ══════════════════════════════════════════════════════
    
    def analyze_trading_opportunity(self, 
                                   pair,
                                   data: pd.DataFrame,
                                   swing_signal: Tuple) -> Dict:
        """
        Analiza oportunidad de trading con TODAS las capas V4.
        
        Args:
            pair: TradingPair object
            data: DataFrame con OHLCV
            swing_signal: (signal, price, confidence) del swing detector
        
        Returns:
            Dict con decisión y metadata
        """
        symbol = pair.yf_symbol
        signal, signal_price, swing_confidence = swing_signal
        
        print(f"\n🔍 Análisis Multi-Layer: {symbol}")
        print(f"   Swing Signal: {signal} (conf: {swing_confidence:.2f})")
        
        result = {
            'can_trade': False,
            'final_signal': None,
            'confidence': 0.0,
            'reasons': [],
            'capital': 0.0,
            'leverage': self.config.LEVERAGE
        }
        
        # ══════════════════════════════════════════════════════
        #       LAYER 1: SENTIMENT ANALYSIS
        # ══════════════════════════════════════════════════════
        
        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer.get_sentiment(symbol)
            
            can_trade_sentiment = should_trade_based_on_sentiment(
                sentiment,
                signal,
                min_confidence=getattr(self.config, 'MIN_SENTIMENT_CONFIDENCE', 0.5)
            )
            
            if not can_trade_sentiment:
                result['reasons'].append(
                    f"❌ Sentiment conflictivo: {sentiment.overall_score:.2f}"
                )
                return result
            
            result['reasons'].append(
                f"✓ Sentiment: {sentiment.overall_score:.2f} ({sentiment.signal_type if hasattr(sentiment, 'signal_type') else 'N/A'})"
            )
        
        # ══════════════════════════════════════════════════════
        #       LAYER 2: ON-CHAIN METRICS
        # ══════════════════════════════════════════════════════
        
        if self.onchain_analyzer:
            onchain = self.onchain_analyzer.get_onchain_signal(symbol)
            
            can_trade_onchain = should_trade_based_on_onchain(
                onchain,
                signal,
                min_strength=getattr(self.config, 'MIN_ONCHAIN_STRENGTH', 0.5)
            )
            
            if not can_trade_onchain:
                result['reasons'].append(
                    f"❌ On-Chain conflictivo: {onchain.signal_type}"
                )
                return result
            
            result['reasons'].append(
                f"✓ On-Chain: {onchain.signal_type} (strength: {onchain.strength:.2f})"
            )
        
        # ══════════════════════════════════════════════════════
        #       LAYER 3: ENSEMBLE STRATEGIES
        # ══════════════════════════════════════════════════════
        
        if self.ensemble_system:
            ensemble_decision = self.ensemble_system.get_ensemble_decision(
                data, swing_signal
            )
            
            self.ensemble_system.print_decision_summary(ensemble_decision)
            
            # Verificar consenso y confianza
            min_consensus = getattr(self.config, 'MIN_ENSEMBLE_CONSENSUS', 0.6)
            min_confidence = getattr(self.config, 'MIN_ENSEMBLE_CONFIDENCE', 0.6)
            
            if (ensemble_decision.final_signal != signal or
                ensemble_decision.consensus_level < min_consensus or
                ensemble_decision.confidence < min_confidence):
                
                result['reasons'].append(
                    f"❌ Ensemble: {ensemble_decision.final_signal} "
                    f"(consensus: {ensemble_decision.consensus_level:.2f}, "
                    f"conf: {ensemble_decision.confidence:.2f})"
                )
                return result
            
            result['reasons'].append(
                f"✓ Ensemble: {ensemble_decision.final_signal} "
                f"(consensus: {ensemble_decision.consensus_level:.2%})"
            )
            result['confidence'] = ensemble_decision.confidence
        else:
            result['confidence'] = swing_confidence
        
        # ══════════════════════════════════════════════════════
        #       LAYER 4: RL POSITION SIZING
        # ══════════════════════════════════════════════════════
        
        if self.rl_calculator:
            # Obtener capital y leverage óptimos
            optimal_capital, optimal_leverage = self.rl_calculator.get_optimal_size(
                data=data,
                signal_confidence=result['confidence'],
                available_capital=self.get_available_capital(),
                base_leverage=self.config.LEVERAGE,
                open_positions=len(self.positions),
                recent_trades=self.get_recent_trades(),
                training=True  # Modo entrenamiento
            )
            
            result['capital'] = optimal_capital
            result['leverage'] = optimal_leverage
            result['reasons'].append(
                f"✓ RL Sizing: ${optimal_capital:.2f} @ {optimal_leverage}x"
            )
        else:
            # Usar sizing tradicional
            allocation = 1.0 / self.config.MAX_POSITIONS
            result['capital'] = self.get_available_capital() * allocation
            result['leverage'] = self.config.LEVERAGE
        
        # ══════════════════════════════════════════════════════
        #       DECISIÓN FINAL
        # ══════════════════════════════════════════════════════
        
        result['can_trade'] = True
        result['final_signal'] = signal
        
        print(f"\n✅ DECISIÓN: {signal}")
        print(f"   Confianza: {result['confidence']:.2%}")
        print(f"   Capital: ${result['capital']:.2f}")
        print(f"   Leverage: {result['leverage']}x")
        print(f"   Razones:")
        for reason in result['reasons']:
            print(f"   {reason}")
        
        return result
    
    # ══════════════════════════════════════════════════════
    #      MODIFICAR MÉTODO DE APERTURA DE POSICIÓN
    # ══════════════════════════════════════════════════════
    
    def open_position_v4(self, pair, analysis_result: Dict, current_price: float):
        """
        Abre posición usando resultados del análisis V4.
        
        Args:
            pair: TradingPair object
            analysis_result: Dict del analyze_trading_opportunity
            current_price: Precio actual
        """
        signal = analysis_result['final_signal']
        capital = analysis_result['capital']
        leverage = analysis_result['leverage']
        confidence = analysis_result['confidence']
        
        # Calcular volumen basado en capital RL
        volume = (capital * leverage) / current_price
        
        # Verificar volumen mínimo
        if volume < pair.min_volume:
            print(f"   ⚠️ Volumen {volume:.8f} < mínimo {pair.min_volume}")
            return
        
        try:
            print(f"\n🟢 Abriendo {signal} en {pair.yf_symbol}")
            print(f"   Precio: ${current_price:.4f}")
            print(f"   Capital RL: ${capital:.2f}")
            print(f"   Leverage RL: {leverage}x")
            print(f"   Volumen: {volume:.8f}")
            print(f"   Confianza Ensemble: {confidence:.2%}")
            
            if not self.config.DRY_RUN:
                order_type = 'buy' if signal == 'BUY' else 'sell'
                
                result = self.kraken.place_order(
                    pair=pair.kraken_pair,
                    order_type=order_type,
                    volume=volume,
                    leverage=leverage,
                    reduce_only=False
                )
                
                print(f"   ✓ Ejecutada: {result}")
                
                # Guardar para actualización RL posterior
                if self.rl_calculator:
                    self._save_trade_for_rl_update({
                        'symbol': pair.yf_symbol,
                        'entry_price': current_price,
                        'volume': volume,
                        'leverage': leverage,
                        'capital': capital,
                        'confidence': confidence
                    })
            else:
                print(f"   🧪 [SIMULACIÓN]")
            
            # Notificar con detalles V4
            self._send_v4_notification(pair, signal, current_price, 
                                      volume, leverage, confidence,
                                      analysis_result['reasons'])
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # ══════════════════════════════════════════════════════
    #      ACTUALIZACIÓN RL AL CERRAR POSICIÓN
    # ══════════════════════════════════════════════════════
    
    def close_position_v4(self, pair, pos_data: Dict, exit_price: float, reason: str):
        """
        Cierra posición y actualiza RL agent.
        """
        # Cerrar posición normalmente (código V3)
        # ... 
        
        # Calcular PnL
        entry_price = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        leverage = float(pos_data.get('leverage', 1))
        pos_type = pos_data.get('type', 'long')
        
        if pos_type == 'long':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100 * leverage
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100 * leverage
        
        # ══════════════════════════════════════════════════════
        #       ACTUALIZAR RL AGENT
        # ══════════════════════════════════════════════════════
        
        if self.rl_sizer:
            trade_result = {
                'closed': True,
                'pnl_pct': pnl_pct,
                'exit_reason': reason
            }
            
            reward = self.rl_sizer.calculate_reward(trade_result)
            
            # Obtener el historial de esta posición
            trade_history = self._get_trade_history(pair.yf_symbol)
            
            if trade_history:
                state = trade_history['state']
                action_idx = trade_history['action_idx']
                
                # Calcular next_state (estado actual)
                current_data = self.get_market_data(pair.yf_symbol)
                next_state = self.rl_calculator.calculate_market_state(
                    data=current_data,
                    signal_confidence=0.0,  # No hay señal ahora
                    open_positions=len(self.positions),
                    recent_trades=self.trades
                )
                
                # Actualizar Q-values
                self.rl_sizer.update_q_value(state, action_idx, reward, next_state)
                
                print(f"   🤖 RL Updated: reward={reward:.3f}")
            
            # Guardar estado periódicamente
            if len(self.trades) % 5 == 0:  # Cada 5 trades
                self.rl_sizer.save_state()
    
    # ══════════════════════════════════════════════════════
    #      HELPERS
    # ══════════════════════════════════════════════════════
    
    def _send_v4_notification(self, pair, signal, price, volume, 
                             leverage, confidence, reasons):
        """Notificación Telegram con detalles V4."""
        
        reasons_text = "\n".join([f"• {r}" for r in reasons])
        
        msg = f"""
🟢 <b>NUEVA POSICIÓN V4</b>

<b>Par:</b> {pair.yf_symbol} ({pair.kraken_pair})
<b>Señal:</b> {signal}
<b>Precio:</b> ${price:.4f}
<b>Volumen:</b> {volume:.8f}

<b>🤖 AI Decision:</b>
<b>Confianza:</b> {confidence:.1%}
<b>Capital RL:</b> ${volume * price / leverage:.2f}
<b>Leverage RL:</b> {leverage}x

<b>📊 Análisis:</b>
{reasons_text}

<b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        if self.config.DRY_RUN:
            msg = "🧪 <b>SIMULACIÓN</b>\n" + msg
        
        self.telegram.send(msg)
    
    def _save_trade_for_rl_update(self, trade_data: Dict):
        """Guarda datos del trade para actualización RL posterior."""
        # Implementar según tu sistema de persistencia
        pass
    
    def _get_trade_history(self, symbol: str) -> Optional[Dict]:
        """Recupera historial de trade para RL update."""
        # Implementar según tu sistema de persistencia
        pass
    
    def get_available_capital(self) -> float:
        """Obtiene capital disponible."""
        # Implementar según tu lógica
        return 1000.0
    
    def get_recent_trades(self) -> list:
        """Obtiene lista de trades recientes."""
        # Implementar según tu lógica
        return []
    
    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Obtiene datos de mercado."""
        # Implementar según tu lógica
        pass


# ═══════════════════════════════════════════════════════════════════════════
#                   EJEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════════════

def example_usage():
    """
    Ejemplo de cómo usar el bot V4.
    """
    
    # Configuración
    class ConfigV4:
        # Existing V3 config
        KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
        KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')
        CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')
        
        # V4 features
        USE_SENTIMENT_ANALYSIS = True
        USE_ONCHAIN_ANALYSIS = True
        USE_ENSEMBLE_SYSTEM = True
        USE_RL_POSITION_SIZING = True
        
        MIN_SENTIMENT_CONFIDENCE = 0.5
        MIN_ONCHAIN_STRENGTH = 0.5
        MIN_ENSEMBLE_CONSENSUS = 0.6
        MIN_ENSEMBLE_CONFIDENCE = 0.6
        
        WEIGHT_SWING = 0.30
        WEIGHT_MOMENTUM = 0.25
        WEIGHT_MEAN_REVERSION = 0.25
        WEIGHT_TREND_FOLLOWING = 0.20
        
        RL_LEARNING_RATE = 0.1
        RL_EPSILON = 0.1
        
        MAX_POSITIONS = 3
        LEVERAGE = 3
        DRY_RUN = True
    
    # Inicializar bot V4
    bot = TradingBotV4(ConfigV4())
    
    # En tu loop principal, reemplazar:
    # if swing_signal:
    #     open_position(...)
    
    # Por:
    # if swing_signal:
    #     analysis = bot.analyze_trading_opportunity(pair, data, swing_signal)
    #     if analysis['can_trade']:
    #         bot.open_position_v4(pair, analysis, current_price)


if __name__ == "__main__":
    print("Este es un archivo de ejemplo de integración.")
    print("Copia las funciones relevantes a tu bot principal.")
