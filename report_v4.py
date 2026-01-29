import json
import pandas as pd
import os
from datetime import datetime

def generate_report():
    trades_file = "trades_v4.json"
    
    print("\n" + "═"*60)
    print(f"📊 REPORT ATTIVITÀ DI TRADING V4 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═"*60)
    
    if not os.path.exists(trades_file):
        print("ℹ️ Nessun file storico trade trovato (trades_v4.json).")
        print("   Il bot deve eseguire almeno un'operazione per generare il file.")
        return

    try:
        with open(trades_file, 'r') as f:
            data = json.load(f)
            
        if not data:
            print("ℹ️ Lo storico trade è vuoto.")
            return

        df = pd.DataFrame(data)
        
        # Conversione tipi
        df['capital'] = pd.to_numeric(df['capital'])
        df['pnl_pct'] = pd.to_numeric(df.get('pnl_pct', 0))
        
        # Filtri
        total_trades = len(df)
        closed_trades = df[df.get('closed', False) == True].copy()
        open_trades = df[df.get('closed', False) == False].copy()
        
        print(f"🔸 Totale Operazioni Gestite: {total_trades}")
        print(f"🔸 Posizioni Aperte: {len(open_trades)}")
        print(f"🔸 Posizioni Chiuse: {len(closed_trades)}")
        
        if not closed_trades.empty:
            # Calcolo metriche su posizioni chiuse
            wins = closed_trades[closed_trades['pnl_pct'] > 0]
            losses = closed_trades[closed_trades['pnl_pct'] <= 0]
            
            win_rate = (len(wins) / len(closed_trades)) * 100
            
            # PnL medio e totale (basato su %)
            avg_pnl = closed_trades['pnl_pct'].mean()
            total_pnl_sum = closed_trades['pnl_pct'].sum()
            
            best_trade = closed_trades['pnl_pct'].max()
            worst_trade = closed_trades['pnl_pct'].min()
            
            print("\n📈 PERFORMANCE (Su posizioni chiuse):")
            print(f"   • Win Rate:      {win_rate:.2f}%")
            print(f"   • PnL Cumulativo (Somma %): {total_pnl_sum:+.2f}%")
            print(f"   • PnL Medio:     {avg_pnl:+.2f}%")
            print(f"   • Miglior Trade: {best_trade:+.2f}%")
            print(f"   • Peggior Trade: {worst_trade:+.2f}%")
            
            if len(losses) > 0:
                avg_win = wins['pnl_pct'].mean() if not wins.empty else 0
                avg_loss = losses['pnl_pct'].mean()
                if avg_loss != 0:
                    profit_factor = abs(wins['pnl_pct'].sum() / losses['pnl_pct'].sum())
                    print(f"   • Profit Factor: {profit_factor:.2f}")
        else:
            print("\nℹ️ Nessuna posizione chiusa per calcolare Win Rate/PnL.")

        if not open_trades.empty:
            print("\n🔓 POSIZIONI APERTE (Prime 10):")
            for _, row in open_trades.head(10).iterrows():
                conf = row.get('confidence', 0) * 100 if row.get('confidence') else 0
                print(f"   • {row['symbol']} ({row['signal']}) @ ${row['entry_price']:.4f} | Conf: {conf:.1f}% | Size: ${row['capital']:.2f}")
            if len(open_trades) > 10:
                print(f"   ...e altre {len(open_trades)-10} posizioni aperte")

    except Exception as e:
        print(f"❌ Errore nella generazione del report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_report()
