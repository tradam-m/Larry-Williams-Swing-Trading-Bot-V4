import argparse
import json
import os
from datetime import datetime

import pandas as pd


def _parse_iso(value):
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def generate_report(trades_file="trades_v4.json", daily=False, target_date=None, save_to_file=False):
    now = datetime.now()
    report_lines = []

    if daily:
        if target_date:
            day = datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            day = now.date()
        title = f"📊 REPORT GIORNALIERO V4 - {day.isoformat()}"
    else:
        day = None
        title = f"📊 REPORT ATTIVITÀ DI TRADING V4 - {now.strftime('%Y-%m-%d %H:%M')}"

    report_lines.append("\n" + "═" * 60)
    report_lines.append(title)
    report_lines.append("═" * 60)

    if not os.path.exists(trades_file):
        report_lines.append("ℹ️ Nessun file storico trade trovato (trades_v4.json).")
        report_lines.append("   Il bot deve eseguire almeno un'operazione per generare il file.")
        print("\n".join(report_lines))
        return

    try:
        with open(trades_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            report_lines.append("ℹ️ Lo storico trade è vuoto.")
            print("\n".join(report_lines))
            return

        df = pd.DataFrame(data)

        if "capital" not in df:
            df["capital"] = 0
        if "pnl_pct" not in df:
            df["pnl_pct"] = 0
        if "closed" not in df:
            df["closed"] = False

        df["capital"] = pd.to_numeric(df["capital"], errors="coerce").fillna(0.0)
        df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)

        # Data parsing per report giornaliero
        df["entry_dt"] = df.get("timestamp", pd.Series([None] * len(df))).apply(_parse_iso)
        df["exit_dt"] = df.get("exit_time", pd.Series([None] * len(df))).apply(_parse_iso)

        if daily:
            opened_today = df[df["entry_dt"].apply(lambda x: x.date() == day if x else False)].copy()
            closed_today = df[(df["closed"] == True) & df["exit_dt"].apply(lambda x: x.date() == day if x else False)].copy()
            open_now = df[df["closed"] == False].copy()

            report_lines.append(f"🔸 Operazioni Aperte nel giorno: {len(opened_today)}")
            report_lines.append(f"🔸 Operazioni Chiuse nel giorno: {len(closed_today)}")
            report_lines.append(f"🔸 Posizioni Aperte adesso: {len(open_now)}")

            if not closed_today.empty:
                wins = closed_today[closed_today["pnl_pct"] > 0]
                losses = closed_today[closed_today["pnl_pct"] <= 0]

                win_rate = (len(wins) / len(closed_today)) * 100
                avg_pnl = closed_today["pnl_pct"].mean()
                total_pnl_sum = closed_today["pnl_pct"].sum()
                best_trade = closed_today["pnl_pct"].max()
                worst_trade = closed_today["pnl_pct"].min()

                report_lines.append("\n📈 PERFORMANCE GIORNALIERA (chiusure del giorno):")
                report_lines.append(f"   • Win Rate:      {win_rate:.2f}%")
                report_lines.append(f"   • PnL Giornaliero (Somma %): {total_pnl_sum:+.2f}%")
                report_lines.append(f"   • PnL Medio:     {avg_pnl:+.2f}%")
                report_lines.append(f"   • Miglior Trade: {best_trade:+.2f}%")
                report_lines.append(f"   • Peggior Trade: {worst_trade:+.2f}%")

                if len(losses) > 0 and losses["pnl_pct"].sum() != 0:
                    profit_factor = abs(wins["pnl_pct"].sum() / losses["pnl_pct"].sum()) if not wins.empty else 0
                    report_lines.append(f"   • Profit Factor: {profit_factor:.2f}")
            else:
                report_lines.append("\nℹ️ Nessuna posizione chiusa nel giorno selezionato.")

        else:
            total_trades = len(df)
            closed_trades = df[df["closed"] == True].copy()
            open_trades = df[df["closed"] == False].copy()

            report_lines.append(f"🔸 Totale Operazioni Gestite: {total_trades}")
            report_lines.append(f"🔸 Posizioni Aperte: {len(open_trades)}")
            report_lines.append(f"🔸 Posizioni Chiuse: {len(closed_trades)}")

            if not closed_trades.empty:
                wins = closed_trades[closed_trades["pnl_pct"] > 0]
                losses = closed_trades[closed_trades["pnl_pct"] <= 0]

                win_rate = (len(wins) / len(closed_trades)) * 100
                avg_pnl = closed_trades["pnl_pct"].mean()
                total_pnl_sum = closed_trades["pnl_pct"].sum()
                best_trade = closed_trades["pnl_pct"].max()
                worst_trade = closed_trades["pnl_pct"].min()

                report_lines.append("\n📈 PERFORMANCE (Su posizioni chiuse):")
                report_lines.append(f"   • Win Rate:      {win_rate:.2f}%")
                report_lines.append(f"   • PnL Cumulativo (Somma %): {total_pnl_sum:+.2f}%")
                report_lines.append(f"   • PnL Medio:     {avg_pnl:+.2f}%")
                report_lines.append(f"   • Miglior Trade: {best_trade:+.2f}%")
                report_lines.append(f"   • Peggior Trade: {worst_trade:+.2f}%")

                if len(losses) > 0 and losses["pnl_pct"].sum() != 0:
                    profit_factor = abs(wins["pnl_pct"].sum() / losses["pnl_pct"].sum()) if not wins.empty else 0
                    report_lines.append(f"   • Profit Factor: {profit_factor:.2f}")
            else:
                report_lines.append("\nℹ️ Nessuna posizione chiusa per calcolare Win Rate/PnL.")

            if not open_trades.empty:
                report_lines.append("\n🔓 POSIZIONI APERTE (Prime 10):")
                for _, row in open_trades.head(10).iterrows():
                    conf = _safe_float(row.get("confidence"), 0.0) * 100
                    report_lines.append(
                        f"   • {row.get('symbol', 'N/A')} ({row.get('signal', 'N/A')}) @ ${_safe_float(row.get('entry_price')):.4f} | "
                        f"Conf: {conf:.1f}% | Size: ${_safe_float(row.get('capital')):.2f}"
                    )
                if len(open_trades) > 10:
                    report_lines.append(f"   ...e altre {len(open_trades) - 10} posizioni aperte")

        output = "\n".join(report_lines)
        print(output)

        if save_to_file:
            os.makedirs("reports", exist_ok=True)
            if daily:
                report_path = os.path.join("reports", f"report_daily_{day.isoformat()}.txt")
            else:
                report_path = os.path.join("reports", f"report_full_{now.strftime('%Y-%m-%d_%H-%M')}.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(output + "\n")
            print(f"\n💾 Report salvato in: {report_path}")

    except Exception as e:
        print(f"❌ Errore nella generazione del report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera report operativo/performance del bot V4")
    parser.add_argument("--input", default="trades_v4.json", help="Path file storico trade JSON")
    parser.add_argument("--daily", action="store_true", help="Genera report giornaliero")
    parser.add_argument("--date", default=None, help="Data target formato YYYY-MM-DD (valido con --daily)")
    parser.add_argument("--save", action="store_true", help="Salva anche il report su file in cartella reports/")
    args = parser.parse_args()

    generate_report(
        trades_file=args.input,
        daily=args.daily,
        target_date=args.date,
        save_to_file=args.save,
    )
