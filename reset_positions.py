"""Utility per chiudere automaticamente tutte le posizioni aperte.

Esecuzione:
    python reset_positions.py

Il comportamento si adatta automaticamente alla modalita' del bot:
- In simulazione (DRY_RUN=true) marca tutti i trade ancora aperti in trades_v4.json
  come chiusi, calcolando il PnL corrente.
- In modalita' reale richiama le API Kraken per chiudere ogni posizione aperta.
"""

from kraken_bot_v4_advanced import TradingBotV4, Config


def main():
    bot = TradingBotV4(Config())
    closed = bot.force_close_all_positions()
    if closed:
        print(f"\n✅ Reset completato: {closed} posizioni chiuse.")
    else:
        print("\nℹ️ Nessuna posizione da chiudere.")


if __name__ == "__main__":
    main()
