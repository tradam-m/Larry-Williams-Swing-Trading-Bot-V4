# Larry Williams Swing Trading Bot V4 — CLAUDE.md

Documentazione tecnica per sessioni future con Claude Code.

---

## Architettura del progetto

| File | Ruolo |
|------|-------|
| `kraken_bot_v4_advanced.py` | Motore principale (TradingBotV4, Config, SwingDetectorV3) |
| `start_live_simulation.py` | Loop locale continuo ogni 15 min, legge da `.env` |
| `ensemble_strategies.py` | Strategie Momentum, MeanReversion, TrendFollowing, Swing |
| `sentiment_analyzer.py` | Fear&Greed + CryptoCompare + NewsData.io |
| `onchain_metrics.py` | On-chain via CryptoCompare |
| `rl_position_sizing.py` | Q-learning per sizing posizioni |
| `telegram_perf_report.py` | Report performance → Telegram |
| `report_v4.py` | Report locale da CLI |
| `trades_v4_live.json` | Storico trade committato nel repo (aggiornato da GH Actions) |
| `sync_live_trades.ps1` | Copia trades dal runner self-hosted alla cartella locale |
| `schedule_sync_live_trades.ps1` | Registra task Windows per sync automatico ogni N minuti |

---

## Workflow GitHub Actions

### `trading-bot-v4.yml` — principale
- **Schedule**: ogni 15 minuti (`cron: "*/15 * * * *"`)
- **Runner**: `ubuntu-latest`
- **DRY_RUN**: `true` per default (sicuro — nessun ordine reale)
- **Trigger manuale**: workflow_dispatch con parametri (dry_run, use_sentiment, ecc.)
- Dopo ogni run: invia report Telegram + committa `trades_v4_live.json`

### `performance-report.yml`
- Solo trigger manuale
- Parametro `period`: `24h / 48h / 7d / all`
- Invia report Telegram con metriche aggregate

### `bot_runner_finale.yml`
- Runner **self-hosted** (Windows locale)
- Usa `start_live_simulation.py` in loop continuo
- Scrive `.env` con le variabili configurate

---

## Configurazione GitHub Actions (trading-bot-v4.yml)

```
CANDLE_INTERVAL: 4h
PAIR: BTC + ETH + ADA + SOL + XRP (tutti attivi)
USE_ENSEMBLE_SYSTEM: true
STOP_LOSS_PCT: 4.0%  |  TAKE_PROFIT_PCT: 8.0%
TRAILING_STOP_PCT: 2.5%  |  MIN_PROFIT_FOR_TRAILING: 3.0%
MAX_POSITIONS: 5  |  LEVERAGE: 3x
DRY_RUN: true (default)
```

## Configurazione live locale (`.env`)

```
DRY_RUN: False  (trading reale)
CANDLE_INTERVAL: 1h
LEVERAGE: 1x  |  MAX_POSITIONS: 1
CAPITAL_PER_TRADE: 20%
Coppie: BTC-EUR, ETH-EUR, ADA-EUR, SOL-EUR, XRP-EUR
```

---

## Filtri attivi nel codice (`kraken_bot_v4_advanced.py`)

1. **Filtro SELL spot** (~linea 1488): SELL ignorato se `LEVERAGE <= 1` (no short su spot)
2. **Filtro Regime VOLATILE** (~linea 1496): nessuna apertura se `RegimeDetector` rileva VOLATILE
3. **Filtro SMA200** (~linea 1501): BUY solo se `prezzo > SMA200`; SELL solo se `prezzo < SMA200`
4. **Cap CAPITAL_PER_TRADE** (~linea 1021): il RL non supera la % configurata in `CAPITAL_PER_TRADE`

---

## Secrets GitHub (repo `tradam-m/Larry-Williams-Swing-Trading-Bot-V4`)

Tutti configurati:
- `KRAKEN_API_KEY` / `KRAKEN_API_SECRET`
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
- `CRYPTOCOMPARE_API_KEY`
- `OPENAI_API_KEY`, `GEMINI_API_KEY`, `NEWSDATA_API_KEY`

**IMPORTANTE**: la API key Kraken (`APIkeyZM`) deve avere la **restrizione IP disabilitata**
per funzionare su GitHub Actions (IP variabili). Verificare su Kraken → Security → API.

---

## Storico fix principali

### Maggio 2026 (revisione dopo 5 mesi di stop)
- **Schedule riabilitato**: era commentato dal commit `5e54b30` (dic 2025)
- **`bot_runner.yml` rimosso**: file YAML corrotto (2 righe) causava errore ad ogni push
- **Restrizione IP API key**: rimossa da Kraken Pro — bloccava GH Actions con `EGeneral:Permission denied`
- **`requirements.txt`**: rimossi `scikit-learn` e `matplotlib` (non importati dal bot), `yfinance` pinned a `>=1.1.0`
- **GitHub Actions aggiornate**: `checkout@v4.2.2`, `setup-python@v5`, `cache@v4.2.3`, `upload-artifact@v4.6.2` (Node.js 24, scadenza 16 giu 2026)
- **`schedule_sync_live_trades.ps1`** aggiunto al tracking git

### Marzo 2026
- Coppie cambiate da USD a EUR (`BTC-EUR`, ecc.) — fix `EOrder:Insufficient funds`
- Filtro SELL spot aggiunto
- Fix doppio blocco `except` in `open_position`
- Cap `CAPITAL_PER_TRADE` sul sizing RL

### Febbraio–Marzo 2026
- Filtro Regime VOLATILE aggiunto
- Filtro SMA200 aggiunto
- Peso Fear&Greed in `sentiment_analyzer.py` ridotto da 40% a 20%

---

## Come vedere i risultati

```bash
# Report locale
python report_v4.py --input trades_v4_live.json

# Avvio bot locale (loop 15 min)
python start_live_simulation.py

# Forza run su GitHub Actions
gh workflow run trading-bot-v4.yml -R tradam-m/Larry-Williams-Swing-Trading-Bot-V4 -f dry_run=true

# Ultimi run
gh run list --limit 10 -R tradam-m/Larry-Williams-Swing-Trading-Bot-V4

# Registra task Windows per sync automatico
.\schedule_sync_live_trades.ps1 -IntervalMinutes 15
.\schedule_sync_live_trades.ps1 -Remove
```

---

## Note tecniche

- `USE_ML_VALIDATION=true` usa un **rule-based scoring** interno, non scikit-learn
- `CRYPTOCOMPARE_API_KEY` nei secrets GitHub è la stessa stringa della `KRAKEN_API_SECRET` nel `.env` locale — è errata per CryptoCompare ma il bot degrada gracefully senza sentiment/onchain
- Il `trades_v4_live.json` si resetta a `[]` se il workflow non trova dati in cache — normale dopo un lungo stop
- Due remotes: `origin` = `tradam-m` (tuo fork attivo), `upstream` = `winningtrendingbots`
