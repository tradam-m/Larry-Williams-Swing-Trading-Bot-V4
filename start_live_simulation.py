import time
import sys
import traceback
from datetime import datetime
from kraken_bot_v4_advanced import TradingBotV4, Config, SENTIMENT_AVAILABLE, ONCHAIN_AVAILABLE, ENSEMBLE_AVAILABLE, RL_AVAILABLE

def log_to_file(message):
    try:
        with open("bot_activity.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    except:
        pass

def run_bot_cycle():
    """Esegue un ciclo completo del bot."""
    start_msg = f"⏰ Inizio ciclo"
    print(f"\n{start_msg}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_file("Ciclo avviato")
    
    try:
        config = Config()
        
        # Verificar credenciales básicas
        if not config.KRAKEN_API_KEY or not config.KRAKEN_API_SECRET:
            print("❌ Mancano credenziali Kraken")
            log_to_file("❌ Errore: Mancano credenziali Kraken")
            return
            
        bot = TradingBotV4(config)
        bot.run()
        log_to_file("Ciclo completato con successo")
        
    except Exception as e:
        err_msg = f"❌ Errore critico nel ciclo: {e}"
        print(err_msg)
        log_to_file(err_msg)
        traceback.print_exc()

def main():
    print("\n" + "═"*70)
    print("🚀 AVVIO SIMULAZIONE LIVE (PAPER TRADING) - 24 ORE")
    print(f"   Intervallo: {Config.CANDLE_INTERVAL}")
    print("   Premi Ctrl+C per fermare")
    print("═"*70)
    log_to_file("🤖 BOT AVVIATO")

    while True:
        try:
            run_bot_cycle()
            
            # Calcolare tempo di attesa (15 minuti)
            wait_minutes = 15
            print(f"\n💤 Attesa {wait_minutes} minuti per il prossimo ciclo...")
            
            # Cuenta regresiva simple
            # Separata try block per permettere CTRL+C durante l'attesa
            try:
                for i in range(wait_minutes, 0, -1):
                    print(f"   Prossimo ciclo tra {i} min...   ", end='\r')
                    time.sleep(60)
            except KeyboardInterrupt:
                raise # Rilancia al blocco principale per uscita pulita
                
        except KeyboardInterrupt:
            print("\n\n🛑 Bot fermato dall'utente.")
            log_to_file("🛑 Bot fermato dall'utente")
            break
        except Exception as e:
            # Su VPS: Cattura errori fatali e riprova invece di chiudersi
            fatal_err = f"\n❌ ERRORE CRITICO: {e}. Il bot riproverà tra 60 secondi..."
            print(fatal_err)
            log_to_file(fatal_err)
            try:
                traceback.print_exc()
            except:
                pass
            time.sleep(60) # Attesa di sicurezza prima di riprovare

if __name__ == "__main__":
    main()
