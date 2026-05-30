# 🚀 Guida completa all'integrazione - Bot V4

## ✅ Archivos Necesarios

### Estructura Final del Repositorio

```
tu-repo/
├── .github/
│   └── workflows/
│       └── trading-bot-v4.yml          ✅ NUEVO - Workflow V4
│
├── kraken_bot_v4_advanced.py           ✅ NUEVO - Bot principal V4
├── sentiment_analyzer.py                ✅ NUEVO - Análisis sentiment
├── onchain_metrics.py                   ✅ NUEVO - Métricas blockchain
├── ensemble_strategies.py               ✅ NUEVO - Sistema ensemble
├── rl_position_sizing.py                ✅ NUEVO - RL position sizing
│
├── requirements.txt                     ✅ ACTUALIZADO
├── README_V4.md                         ✅ NUEVO - Documentación
│
└── (Archivos V3 opcionales para referencia)
    ├── kraken_bot_v3_multi_asset.py
    ├── analyze_correlations.py
    └── README_V3.md
```

---

## 📋 Checklist de Configuración

### 1. GitHub Secrets

Ve a **Settings → Secrets and variables → Actions** y agrega:

```bash
# ✅ OBLIGATORIOS
KRAKEN_API_KEY=tu_kraken_api_key
KRAKEN_API_SECRET=tu_kraken_api_secret

# ✅ OBLIGATORIO PARA V4
CRYPTOCOMPARE_API_KEY=tu_cryptocompare_key

# ⚠️ OPCIONAL (recomendado)
TELEGRAM_BOT_TOKEN=tu_telegram_token
TELEGRAM_CHAT_ID=tu_chat_id
```

#### Cómo obtener CryptoCompare API Key

1. Ve a <https://www.cryptocompare.com/>
2. Crea cuenta gratuita
3. Ve a <https://www.cryptocompare.com/cryptopian/api-keys>
4. Crea nueva API key
5. Free tier: 100,000 calls/mes (suficiente)

---

### 2. Archivos a Subir

#### Archivos NUEVOS (copiar exactamente)

1. **kraken_bot_v4_advanced.py** - Bot principal completo
2. **sentiment_analyzer.py** - Del documento que te pasé
3. **onchain_metrics.py** - Del documento que te pasé
4. **ensemble_strategies.py** - Del documento que te pasé
5. **rl_position_sizing.py** - Del documento que te pasé
6. **trading-bot-v4.yml** - Workflow de GitHub Actions

#### Archivos a ACTUALIZAR

1. **requirements.txt** - Agregar:

```txt
requests>=2.31.0
yfinance>=1.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

---

## 🔄 Diferencias Clave V3 vs V4

### Arquitectura

**V3:**

```
Swing Detection → Open Position
```

**V4:**

```
Swing Detection 
    ↓
Sentiment Analysis (LAYER 1)
    ↓
On-Chain Metrics (LAYER 2)
    ↓
Ensemble Strategies (LAYER 3)
    ↓
RL Position Sizing (LAYER 4)
    ↓
Open Position
```

### Lo que V4 tiene que V3 NO

1. **Sentiment Analysis**
   - Analiza noticias y redes sociales
   - Filtra señales según sentiment del mercado

2. **On-Chain Metrics**
   - Exchange flows
   - Active addresses
   - Whale activity

3. **Multi-Strategy Ensemble**
   - 4 estrategias corriendo en paralelo
   - Sistema de votación ponderada
   - Mayor robustez en señales

4. **RL Position Sizing**
   - Tamaño dinámico de posiciones
   - Aprende de trades anteriores
   - Q-Learning implementado

---

## ⚙️ Configuración del Workflow

### Variables Clave en `trading-bot-v4.yml`

```yaml
# ═══════════════ FEATURES V4 ═══════════════
USE_SENTIMENT_ANALYSIS: 'true'      # Activar/desactivar sentiment
USE_ONCHAIN_ANALYSIS: 'true'        # Activar/desactivar on-chain
USE_ENSEMBLE_SYSTEM: 'true'         # Activar/desactivar ensemble
USE_RL_POSITION_SIZING: 'true'      # Activar/desactivar RL

# ═══════════════ THRESHOLDS ═══════════════
MIN_SENTIMENT_CONFIDENCE: '0.5'     # Confianza mínima sentiment
MIN_ONCHAIN_STRENGTH: '0.5'         # Fuerza mínima on-chain
MIN_ENSEMBLE_CONSENSUS: '0.6'       # Consenso mínimo (60%)
MIN_ENSEMBLE_CONFIDENCE: '0.6'      # Confianza mínima ensemble

# ═══════════════ ENSEMBLE WEIGHTS ═══════════════
WEIGHT_SWING: '0.30'                # 30% peso swing
WEIGHT_MOMENTUM: '0.25'             # 25% peso momentum
WEIGHT_MEAN_REVERSION: '0.25'       # 25% peso mean reversion
WEIGHT_TREND_FOLLOWING: '0.20'      # 20% peso trend following

# ═══════════════ RL CONFIG ═══════════════
RL_LEARNING_RATE: '0.1'             # Alpha
RL_EPSILON: '0.1'                   # Exploración (10%)
```

---

## 🧪 Testing - Pasos Recomendados

### Fase 1: Test Básico (Sin V4)

```yaml
# En trading-bot-v4.yml, temporalmente:
USE_SENTIMENT_ANALYSIS: 'false'
USE_ONCHAIN_ANALYSIS: 'false'
USE_ENSEMBLE_SYSTEM: 'false'
USE_RL_POSITION_SIZING: 'false'
DRY_RUN: 'true'
```

**Verificar:**

- ✅ Bot inicia correctamente
- ✅ Descarga datos de mercado
- ✅ Detecta swing signals
- ✅ Calcula correlaciones
- ✅ Notificaciones Telegram funcionan

### Fase 2: Test con Sentiment

```yaml
USE_SENTIMENT_ANALYSIS: 'true'
USE_ONCHAIN_ANALYSIS: 'false'
USE_ENSEMBLE_SYSTEM: 'false'
USE_RL_POSITION_SIZING: 'false'
DRY_RUN: 'true'
```

**Verificar:**

- ✅ CryptoCompare API responde
- ✅ Sentiment scores se calculan
- ✅ Señales se filtran correctamente

### Fase 3: Test con On-Chain

```yaml
USE_SENTIMENT_ANALYSIS: 'true'
USE_ONCHAIN_ANALYSIS: 'true'
USE_ENSEMBLE_SYSTEM: 'false'
USE_RL_POSITION_SIZING: 'false'
DRY_RUN: 'true'
```

**Verificar:**

- ✅ On-chain metrics se obtienen
- ✅ Ambas capas funcionan juntas

### Fase 4: Test con Ensemble

```yaml
USE_SENTIMENT_ANALYSIS: 'true'
USE_ONCHAIN_ANALYSIS: 'true'
USE_ENSEMBLE_SYSTEM: 'true'
USE_RL_POSITION_SIZING: 'false'
DRY_RUN: 'true'
```

**Verificar:**

- ✅ Las 4 estrategias generan señales
- ✅ Sistema de votación funciona
- ✅ Consenso se calcula correctamente

### Fase 5: Test Completo V4

```yaml
USE_SENTIMENT_ANALYSIS: 'true'
USE_ONCHAIN_ANALYSIS: 'true'
USE_ENSEMBLE_SYSTEM: 'true'
USE_RL_POSITION_SIZING: 'true'
DRY_RUN: 'true'
```

**Verificar:**

- ✅ RL agent inicializa
- ✅ Position sizing dinámico funciona
- ✅ Todas las capas trabajan juntas
- ✅ No hay errores en logs

### Fase 6: LIVE (con precaución)

```yaml
# Todos activados
DRY_RUN: 'false'  # ⚠️ CUIDADO
```

**Recomendaciones:**

- Empieza con capital pequeño
- Monitorea constantemente las primeras horas
- Ten `MAX_POSITIONS: '1'` inicialmente
- Aumenta gradualmente

---

## 🐛 Troubleshooting Común

### Error: "sentiment_analyzer.py not found"

**Causa:** Archivo no está en el repo

**Solución:**

```bash
# Verifica que el archivo existe
ls -la sentiment_analyzer.py

# Si no existe, súbelo
git add sentiment_analyzer.py
git commit -m "Add sentiment analyzer"
git push
```

### Error: "Invalid CryptoCompare API key"

**Causa:** Secret mal configurado o key inválida

**Solución:**

1. Verifica en CryptoCompare que la key esté activa
2. Revisa que el secret en GitHub no tenga espacios
3. Regenera la key si es necesario

### Error: "Rate limit exceeded"

**Causa:** Demasiadas llamadas a CryptoCompare

**Solución:**

- Free tier: 100k calls/mes
- Reduce frecuencia de ejecución (cada 30 min en lugar de 15)
- O aumenta cache duration en los analyzers

### RL no mejora

**Síntomas:** Q-values no cambian

**Solución:**

```bash
# Elimina el estado RL para empezar de cero
rm rl_state.json

# O aumenta exploración
RL_EPSILON: '0.2'  # 20% exploración
```

### Ensemble siempre rechaza

**Síntomas:** Nunca hay consenso

**Solución:**

```yaml
# Reduce thresholds
MIN_ENSEMBLE_CONSENSUS: '0.5'  # 50% en lugar de 60%
MIN_ENSEMBLE_CONFIDENCE: '0.5'
```

---

## 📊 Monitoring

### Logs a Revisar

1. **GitHub Actions Logs:**
   - Actions → Workflow runs → Última ejecución
   - Busca secciones con emojis: 🔍 📊 ✅ ❌

2. **Telegram Notifications:**
   - Nuevas posiciones (🟢)
   - Posiciones cerradas (🔴)
   - Errores (❌)

3. **Artifacts:**
   - Actions → Workflow run → Artifacts
   - Descarga `trading-logs-XXX.zip`
   - Contiene `rl_state.json` y logs

### Métricas Clave V4

```
📊 Cada ejecución debe mostrar:

1. AI Features activas (✅/❌)
2. Balance y margen disponible
3. Datos descargados por símbolo
4. Posiciones abiertas
5. Para cada señal:
   - ✓ Sentiment confirma/rechaza
   - ✓ On-Chain confirma/rechaza
   - ✓ Ensemble: consenso y confianza
   - ✓ RL: capital y leverage asignados
6. Decisión final
```

---

## 🎯 Perfiles de Configuración

### Conservador (Recomendado para empezar)

```yaml
MAX_POSITIONS: '1'
LEVERAGE: '2'

MIN_SENTIMENT_CONFIDENCE: '0.7'
MIN_ONCHAIN_STRENGTH: '0.7'
MIN_ENSEMBLE_CONSENSUS: '0.75'
MIN_ENSEMBLE_CONFIDENCE: '0.7'

RL_EPSILON: '0.05'  # Poca exploración
```

### Balanceado (Producción)

```yaml
MAX_POSITIONS: '3'
LEVERAGE: '3'

MIN_SENTIMENT_CONFIDENCE: '0.5'
MIN_ONCHAIN_STRENGTH: '0.5'
MIN_ENSEMBLE_CONSENSUS: '0.6'
MIN_ENSEMBLE_CONFIDENCE: '0.6'

RL_EPSILON: '0.1'
```

### Agresivo (Experimental)

```yaml
MAX_POSITIONS: '4'
LEVERAGE: '5'

MIN_SENTIMENT_CONFIDENCE: '0.3'
MIN_ONCHAIN_STRENGTH: '0.3'
MIN_ENSEMBLE_CONSENSUS: '0.5'
MIN_ENSEMBLE_CONFIDENCE: '0.5'

RL_EPSILON: '0.2'  # Más exploración
```

---

## 🔐 Seguridad

### ✅ Buenas Prácticas

1. **Nunca** hardcodees API keys en el código
2. **Siempre** usa GitHub Secrets
3. Empieza con `DRY_RUN: 'true'`
4. Monitorea las primeras 24h constantemente
5. Ten stop-loss configurados
6. No inviertas más de lo que puedes perder

### ⚠️ Riesgos V4

El sistema V4 es **más complejo** = **más puntos de falla**:

- APIs externas pueden fallar (CryptoCompare)
- RL puede tomar malas decisiones inicialmente
- Ensemble puede ser demasiado conservador
- Sentiment puede ser engañoso

**Mitigación:**

- Prueba MUCHO en simulación
- Empieza con capital pequeño
- Revisa logs diariamente
- Ten plan de salida claro

---

## 📞 Soporte

### Si algo falla

1. **Revisa los logs** en GitHub Actions
2. **Verifica secrets** están configurados
3. **Prueba módulos individualmente** (fases de testing)
4. **Compara con esta guía** paso a paso

### Para reportar bugs

1. Logs completos de la ejecución
2. Configuración usada (sin exponer keys)
3. Qué esperabas vs qué obtuviste
4. ¿En qué fase de testing estás?

---

## ✅ Checklist Final

Antes de activar el bot en LIVE:

- [ ] Todos los archivos V4 subidos al repo
- [ ] Secrets configurados en GitHub
- [ ] CryptoCompare API key válida y activa
- [ ] Requirements.txt actualizado
- [ ] Workflow V4 en `.github/workflows/`
- [ ] Tests en simulación completados (Fases 1-5)
- [ ] Logs revisados sin errores críticos
- [ ] Telegram notifications funcionando
- [ ] RL state guarda/carga correctamente
- [ ] Entiendes cada parámetro de configuración
- [ ] Capital de prueba pequeño preparado
- [ ] Plan de monitoreo definido

---

## 🚀 ¡Listo para Empezar

```bash
# 1. Commit todos los archivos nuevos
git add .
git commit -m "Add V4 Advanced AI System"
git push

# 2. Ve a GitHub Actions
# 3. Run workflow manualmente con:
#    - dry_run: true
#    - use_sentiment: true
#    - use_onchain: true
#    - use_ensemble: true
#    - use_rl: true

# 4. Revisa logs

# 5. Si todo OK → Ejecuta cada 15 min automáticamente
```

**🎉 ¡Bot V4 completo y listo!**

---

*Última actualización: Diciembre 2024 - V4.0*
