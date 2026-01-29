"""
SENTIMENT ANALYZER - CryptoCompare + NewsData.io + Alternative.me Fear & Greed Integration
Analiza el sentiment de redes sociales y noticias para crypto
"""

import os
import requests
import re
import time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class SentimentScore:
    overall_score: float  # -1 (muy negativo) a +1 (muy positivo)
    news_score: float
    social_score: float
    newsdata_score: float  # NUEVO: Score de NewsData.io
    fng_score: float       # NUEVO: Score de Alternative.me (Crypto Fear & Greed)
    confidence: float
    timestamp: datetime
    news_count: int = 0  # NUEVO: Cantidad de noticias analizadas
    
    def is_bullish(self, threshold: float = 0.2) -> bool:
        return self.overall_score > threshold
    
    def is_bearish(self, threshold: float = -0.2) -> bool:
        return self.overall_score < threshold


class AlternativeMeFearAndGreedScraper:
    """
    Scraper para el índice Fear & Greed de Crypto (Alternative.me).
    Esta fuente es usada como estándar en la industria ("Coinglass uses similar data").
    URL: https://api.alternative.me/fng/
    """
    def __init__(self):
        self.url = "https://api.alternative.me/fng/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_index(self) -> Optional[float]:
        """
        Obtiene el valor actual del índice (0-100).
        Retorna float o None si falla.
        """
        try:
            response = self.session.get(self.url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Alternative.me devuelve: {"data": [{"value": "26", "value_classification": "Fear", ...}]}
            if 'data' in data and len(data['data']) > 0:
                val = data['data'][0].get('value')
                if val is not None:
                    return float(val)
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Error scraping Alternative.me Fear & Greed: {e}")
            return None


class SentimentAnalyzer:
    """
    Integra datos de sentiment de múltiples fuentes:
    - CryptoCompare API (noticias y social)
    - NewsData.io API (noticias globales)
    - Alternative.me Fear & Greed Index (Crypto Market)
    """
    
    def __init__(self, cryptocompare_api_key: str, newsdata_api_key: Optional[str] = None):
        self.cryptocompare_key = cryptocompare_api_key
        self.newsdata_key = newsdata_api_key
        self.base_url = "https://min-api.cryptocompare.com"
        self.newsdata_url = "https://newsdata.io/api/1/news"
        self.session = requests.Session()
        self.cache = {}
        self.cache_duration = 300  # 5 minutos
        
        self.fng_scraper = AlternativeMeFearAndGreedScraper() # Inicializar FnG Scraper
        
        # Keywords para análisis de sentiment

        self.positive_keywords = [
            'bullish', 'surge', 'rally', 'gain', 'pump', 'moon', 'adoption',
            'breakthrough', 'soar', 'skyrocket', 'boom', 'breakthrough',
            'positive', 'upgrade', 'partnership', 'growth', 'innovation',
            'institutional', 'buying', 'accumulation', 'breakout'
        ]
        
        self.negative_keywords = [
            'bearish', 'crash', 'dump', 'drop', 'fall', 'decline', 'regulation',
            'ban', 'crackdown', 'plunge', 'collapse', 'fear', 'panic',
            'lawsuit', 'hack', 'scam', 'fraud', 'bubble', 'selloff',
            'correction', 'weakness', 'concern', 'risk'
        ]
        
        # Crypto symbols mapping
        self.crypto_names = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'ADA': ['cardano', 'ada'],
            'SOL': ['solana', 'sol'],
            'XRP': ['ripple', 'xrp'],
            'MATIC': ['polygon', 'matic'],
            'AVAX': ['avalanche', 'avax'],
            'LINK': ['chainlink', 'link'],
            'DOT': ['polkadot', 'dot']
        }
    
    def get_sentiment(self, symbol: str) -> Optional[SentimentScore]:
        """
        Obtiene sentiment score agregado de todas las fuentes.
        
        Args:
            symbol: Símbolo crypto (BTC, ETH, etc.)
        
        Returns:
            SentimentScore o None si hay error
        """
        # Normalizar símbolo (BTC-USD -> BTC)
        clean_symbol = symbol.split('-')[0].upper()
        
        # Check cache
        cache_key = f"{clean_symbol}_{int(time.time() / self.cache_duration)}"
        if cache_key in self.cache:
            # print(f"   📋 Sentiment cache hit: {clean_symbol}")
            return self.cache[cache_key]
        
        try:
            scores = []
            weights = []
            news_count = 0
            
            # 1. CryptoCompare News sentiment
            cc_news_score = self._get_cryptocompare_news_sentiment(clean_symbol)
            if cc_news_score is not None:
                scores.append(cc_news_score)
                weights.append(0.25)  # 25% peso
            
            # 2. CryptoCompare Social sentiment
            social_score = self._get_social_sentiment(clean_symbol)
            if social_score is not None:
                scores.append(social_score)
                weights.append(0.25)  # 25% peso
            
            # 3. NewsData.io sentiment (NUEVO)
            newsdata_score = None
            if self.newsdata_key:
                newsdata_score, news_count = self._get_newsdata_sentiment(clean_symbol)
                if newsdata_score is not None:
                    scores.append(newsdata_score)
                    weights.append(0.30)  # 30% peso
            
            # 4. Alternative.me Fear & Greed Index (Crypto)
            fng_raw_score = self.fng_scraper.get_index()
            fng_norm_score = 0.0
            if fng_raw_score is not None:
                # Normalizar 0-100 a -1 a +1
                # 0 = -1 (Extreme Fear), 50 = 0 (Neutral), 100 = 1 (Extreme Greed)
                fng_norm_score = (fng_raw_score - 50) / 50.0
                scores.append(fng_norm_score)
                weights.append(0.40) # 40% peso (Indicador fuerte de mercado CS)
            
            # Verificar que tengamos al menos una fuente
            if not scores:
                return None
            
            # Calculate weighted average
            total_weight = sum(weights)
            overall = sum(s * w for s, w in zip(scores, weights)) / total_weight if total_weight > 0 else 0
            
            # Confidence basada en disponibilidad de datos
            # Ahora tenemos 4 fuentes posibles
            confidence = len(scores) / 4.0 
            if confidence > 1.0: confidence = 1.0
            
            # Ajustar confidence si tenemos FnG (es muy fiable como dato)
            if fng_raw_score is not None:
                confidence = max(confidence, 0.6)

            sentiment = SentimentScore(
                overall_score=overall,
                news_score=cc_news_score or 0.0,
                social_score=social_score or 0.0,
                newsdata_score=newsdata_score or 0.0,
                fng_score=fng_norm_score,
                confidence=confidence,
                timestamp=datetime.now(),
                news_count=news_count
            )
            
            # Cache result
            self.cache[cache_key] = sentiment
            
            print(f"   💭 Sentiment {clean_symbol}: Overall={overall:.2f}")
            print(f"      CC_News={cc_news_score}, Social={social_score}, NewsData={newsdata_score}")
            print(f"      Crypto Fear&Greed={fng_raw_score} (norm: {fng_norm_score:.2f})")
            
            return sentiment
            
        except Exception as e:
            print(f"   ⚠️ Error obteniendo sentiment para {clean_symbol}: {e}")
            return None
    
    def _get_cryptocompare_news_sentiment(self, symbol: str) -> Optional[float]:
        """
        Analiza sentiment de noticias de CryptoCompare.
        
        Returns:
            Score de -1 a +1, o None si error
        """
        try:
            url = f"{self.base_url}/data/v2/news/"
            params = {
                'categories': symbol,
                'lang': 'EN'
            }
            
            headers = {
                'authorization': f'Apikey {self.cryptocompare_key}'
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('Response') != 'Success':
                return None
            
            news_items = data.get('Data', [])
            
            if not news_items:
                return None
            
            sentiment_scores = []
            
            for item in news_items[:20]:  # Últimas 20 noticias
                title = item.get('title', '').lower()
                body = item.get('body', '').lower()
                text = title + ' ' + body
                
                score = self._calculate_text_sentiment(text)
                if score != 0.0:
                    sentiment_scores.append(score)
            
            if not sentiment_scores:
                return 0.0
            
            return sum(sentiment_scores) / len(sentiment_scores)
            
        except Exception as e:
            print(f"   ⚠️ Error en CC news sentiment: {e}")
            return None
    
    def _get_newsdata_sentiment(self, symbol: str) -> tuple[Optional[float], int]:
        """
        Analiza sentiment de noticias desde NewsData.io.
        
        Returns:
            (score, news_count) o (None, 0) si error
        """
        if not self.newsdata_key:
            return None, 0
        
        try:
            # Obtener nombres asociados al símbolo
            search_terms = self.crypto_names.get(symbol, [symbol.lower()])
            
            # Construir query
            query = ' OR '.join(search_terms)
            
            params = {
                'apikey': self.newsdata_key,
                'q': query,
                'language': 'en',
                'category': 'business,technology',
                'size': 10  # Últimas 10 noticias
            }
            
            response = self.session.get(self.newsdata_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'success':
                return None, 0
            
            news_items = data.get('results', [])
            
            if not news_items:
                return 0.0, 0
            
            sentiment_scores = []
            
            for item in news_items:
                # Combinar título y descripción
                title = item.get('title', '').lower()
                description = item.get('description', '').lower()
                content = item.get('content', '').lower() if item.get('content') else ''
                
                text = f"{title} {description} {content}"
                
                # Verificar relevancia (menciona el crypto?)
                is_relevant = any(term in text for term in search_terms)
                
                if is_relevant:
                    score = self._calculate_text_sentiment(text)
                    sentiment_scores.append(score)
            
            if not sentiment_scores:
                return 0.0, 0
            
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            
            return avg_score, len(sentiment_scores)
            
        except Exception as e:
            print(f"   ⚠️ Error en NewsData sentiment: {e}")
            return None, 0
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """
        Calcula sentiment de un texto usando keyword matching.
        
        Returns:
            Score de -1 a +1
        """
        text_lower = text.lower()
        
        pos_count = sum(1 for word in self.positive_keywords if word in text_lower)
        neg_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        # Score normalizado
        score = (pos_count - neg_count) / (pos_count + neg_count)
        
        # Aplicar peso por cantidad de keywords (más keywords = más confianza)
        total_keywords = pos_count + neg_count
        confidence_multiplier = min(1.0, total_keywords / 5.0)  # Max 5 keywords
        
        return score * confidence_multiplier
    
    def _get_social_sentiment(self, symbol: str) -> Optional[float]:
        """
        Obtiene métricas sociales (Twitter, Reddit, etc.) de CryptoCompare.
        
        Returns:
            Score de -1 a +1, o None si error
        """
        try:
            url = f"{self.base_url}/data/social/coin/latest"
            params = {
                'coinId': self._get_coin_id(symbol)
            }
            
            headers = {
                'authorization': f'Apikey {self.cryptocompare_key}'
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('Response') != 'Success':
                return None
            
            social_data = data.get('Data', {})
            
            # Extraer métricas clave
            twitter_followers = social_data.get('Twitter', {}).get('followers', 0)
            reddit_subscribers = social_data.get('Reddit', {}).get('subscribers', 0)
            twitter_points = social_data.get('Twitter', {}).get('Points', 0)
            reddit_points = social_data.get('Reddit', {}).get('Points', 0)
            
            if twitter_followers == 0 and reddit_subscribers == 0:
                return None
            
            # Score basado en engagement points
            total_points = twitter_points + reddit_points
            
            # Normalizar
            if total_points > 10000:
                score = 0.5
            elif total_points > 5000:
                score = 0.3
            elif total_points > 1000:
                score = 0.1
            elif total_points < 100:
                score = -0.2
            else:
                score = 0.0
            
            return score
            
        except Exception as e:
            print(f"   ⚠️ Error en social sentiment: {e}")
            return None
    
    def _get_coin_id(self, symbol: str) -> int:
        """
        Mapea símbolos a CryptoCompare coin IDs.
        """
        coin_map = {
            'BTC': 1182,
            'ETH': 7605,
            'ADA': 5038,
            'SOL': 151340,
            'XRP': 4614,
            'MATIC': 202330,
            'AVAX': 166503,
            'LINK': 3808,
            'DOT': 165542
        }
        
        return coin_map.get(symbol, 1182)  # Default BTC
    
    def get_market_sentiment_summary(self, symbols: list) -> Dict[str, str]:
        """
        Obtiene resumen de sentiment del mercado para múltiples símbolos.
        
        Returns:
            Dict con símbolo -> clasificación (BULLISH/NEUTRAL/BEARISH)
        """
        summary = {}
        
        for symbol in symbols:
            sentiment = self.get_sentiment(symbol)
            
            if sentiment is None:
                summary[symbol] = 'UNKNOWN'
            elif sentiment.is_bullish():
                summary[symbol] = 'BULLISH'
            elif sentiment.is_bearish():
                summary[symbol] = 'BEARISH'
            else:
                summary[symbol] = 'NEUTRAL'
        
        return summary


# Función helper para usar en el bot
def should_trade_based_on_sentiment(sentiment: Optional[SentimentScore],
                                   signal: str,
                                   min_confidence: float = 0.5) -> bool:
    """
    Decide si operar basado en sentiment y señal técnica.
    
    Args:
        sentiment: SentimentScore del asset
        signal: 'BUY' o 'SELL'
        min_confidence: Confianza mínima requerida
    
    Returns:
        True si sentiment confirma la señal
    """
    if sentiment is None:
        return True  # No bloquear si no hay datos
    
    if sentiment.confidence < min_confidence:
        return True  # No bloquear si confianza baja
    
    # Confirmar señal con sentiment
    if signal == 'BUY':
        return sentiment.overall_score >= 0.0  # Sentiment neutral o positivo
    else:  # SELL
        return sentiment.overall_score <= 0.0  # Sentiment neutral o negativo
