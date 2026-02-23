"""Realtime watcher for trades_v4.json.

Usage:
    python watch_trades.py [--file trades_v4.json] [--interval 30]

The watcher prints a summary whenever the file changes (size/timestamp).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def load_trades(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(trades: List[Dict[str, Any]]) -> str:
    total = len(trades)
    if total == 0:
        return "Nessun trade registrato finora."

    closed = sum(1 for t in trades if t.get("closed"))
    open_ = total - closed
    last = trades[-1]
    ts = last.get("timestamp") or last.get("exit_time") or "n/a"
    symbol = last.get("symbol", "?")
    status = "closed" if last.get("closed") else "open"
    pnl = last.get("pnl_pct")
    pnl_txt = f" | PnL {pnl:+.2f}%" if isinstance(pnl, (int, float)) else ""

    return (
        f"Totale={total} | Chiusi={closed} | Aperti={open_} | "
        f"Ultimo: {symbol} ({status}) @ {ts}{pnl_txt}"
    )


def watch(path: Path, interval: int) -> None:
    print(f"👀 Monitoraggio {path} ogni {interval}s. Ctrl+C per uscire.")
    last_mtime = None
    last_size = None

    while True:
        try:
            stat = path.stat()
        except FileNotFoundError:
            print("⚠️ File non trovato. Attendo che venga creato...")
            time.sleep(interval)
            continue

        if last_mtime is None or stat.st_mtime > last_mtime or stat.st_size != last_size:
            last_mtime = stat.st_mtime
            last_size = stat.st_size

            try:
                trades = load_trades(path)
            except json.JSONDecodeError as exc:
                print(f"❌ JSON non valido: {exc}")
                time.sleep(interval)
                continue

            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{stamp}] Aggiornamento rilevato -> {summarize(trades)}")

        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watcher trades_v4.json")
    parser.add_argument(
        "--file",
        default="trades_v4.json",
        help="Percorso del file da monitorare (default: trades_v4.json)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Secondi tra un controllo e l'altro (default: 30)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target = Path(args.file)
    watch(target, args.interval)
