"""
Invia report performance su Telegram dopo ogni run del bot.
Legge trades_v4.json e calcola: Win Rate, Profit Factor, Drawdown, PnL, ecc.
"""
import json
import os
import sys
from datetime import datetime, timedelta

import requests


def load_trades(path="trades_v4.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calc_metrics(trades):
    closed = [t for t in trades if t.get("closed") is True and "pnl_pct" in t]
    open_t = [t for t in trades if t.get("closed") is False]

    if not closed:
        return None, open_t

    wins   = [t for t in closed if t["pnl_pct"] > 0]
    losses = [t for t in closed if t["pnl_pct"] <= 0]

    gross_win  = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))

    pf       = round(gross_win / gross_loss, 2) if gross_loss else float("inf")
    wr       = round(len(wins) / len(closed) * 100, 1)
    avg_win  = round(gross_win / len(wins), 2) if wins else 0
    avg_loss = round(gross_loss / len(losses), 2) if losses else 0
    pnl_sum  = round(sum(t["pnl_pct"] for t in closed), 2)
    pnl_usd  = round(sum(t.get("profit_amount", 0) for t in closed), 2)
    best     = round(max(t["pnl_pct"] for t in closed), 2)
    worst    = round(min(t["pnl_pct"] for t in closed), 2)

    # Max Drawdown su equity curve cumulativa
    cum = 0
    peak = 0
    max_dd = 0
    for t in closed:
        cum += t["pnl_pct"]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    max_dd = round(max_dd, 2)

    return {
        "total": len(closed),
        "open":  len(open_t),
        "wins":  len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "profit_factor": pf,
        "pnl_pct": pnl_sum,
        "pnl_usd": pnl_usd,
        "max_dd": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best": best,
        "worst": worst,
    }, open_t


def filter_last_24h(trades):
    cutoff = datetime.now() - timedelta(hours=24)
    result = []
    for t in trades:
        for key in ("exit_time", "timestamp"):
            val = t.get(key)
            if val:
                try:
                    if datetime.fromisoformat(val) >= cutoff:
                        result.append(t)
                        break
                except Exception:
                    pass
    return result


def send_telegram(token, chat_id, message):
    if not token or not chat_id:
        print(message)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
        r.raise_for_status()
        print("Report inviato su Telegram.")
    except Exception as e:
        print(f"Errore Telegram: {e}")
        print(message)


def format_report(label, m, open_trades):
    if m is None:
        return f"<b>{label}</b>\nNessun trade chiuso."

    pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float("inf") else "inf"
    sign = "+" if m["pnl_usd"] >= 0 else ""

    lines = [
        f"<b>{label}</b>",
        f"Trade chiusi: {m['total']}  |  Aperti: {m['open']}",
        f"Vincenti: {m['wins']}  |  Perdenti: {m['losses']}",
        "",
        f"Win Rate:       {m['win_rate']}%",
        f"Profit Factor:  {pf_str}",
        f"Max Drawdown:   -{m['max_dd']}%",
        f"PnL cumulativo: {m['pnl_pct']:+.2f}%",
        f"PnL netto USD:  {sign}{m['pnl_usd']:.2f}$",
        "",
        f"Avg Win:   +{m['avg_win']:.2f}%",
        f"Avg Loss:  -{m['avg_loss']:.2f}%",
        f"Best:  {m['best']:+.2f}%  |  Worst: {m['worst']:+.2f}%",
    ]

    if open_trades:
        lines.append("")
        lines.append(f"Posizioni aperte ({len(open_trades)}):")
        for t in open_trades[:5]:
            ts = (t.get("timestamp") or "")[:16]
            lines.append(f"  {t.get('symbol','?')} @ {t.get('entry_price',0):.4f}  [{ts}]")

    return "\n".join(lines)


def main():
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    all_trades = load_trades("trades_v4.json")

    # Report ultime 24h
    recent = filter_last_24h(all_trades)
    m24, open24 = calc_metrics(recent)
    msg24 = format_report("REPORT 24H", m24, open24)

    # Report totale (escluse chiusure da reset manuale)
    valid = [t for t in all_trades if "Reset manuale" not in (t.get("exit_reason") or "")]
    m_all, open_all = calc_metrics(valid)
    msg_all = format_report("REPORT TOTALE (esclusi reset)", m_all, open_all)

    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    full_msg = f"<b>BOT V4 Performance - {now}</b>\n\n{msg24}\n\n{msg_all}"

    send_telegram(token, chat_id, full_msg)


if __name__ == "__main__":
    main()
