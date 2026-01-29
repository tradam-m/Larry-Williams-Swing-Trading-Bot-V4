import time
from kraken_bot_v4_advanced import TradingBotV4, Config

def main():
    print("\n" + "═"*70)
    print("🚀 AVVIO CICLO SINGOLO - GITHUB ACTIONS")
    print(f"   Timeframe dati: {Config.CANDLE_INTERVAL}")
    print("═"*70)
    bot = TradingBotV4(Config())
    bot.run()
    print("\n✅ Ciclo completato. Stato e storico aggiornati.")

if __name__ == "__main__":
    main()
