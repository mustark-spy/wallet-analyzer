#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified version for Render.com deployment with Telegram bot integration.
Handles /stats <wallet> command, analyzes the wallet, and sends a ZIP archive with enhanced.json and dataset.json.

Deployment notes:
- Set env vars on Render: TELEGRAM_TOKEN, HELIUS_API_KEY, WEBHOOK_URL (e.g., https://your-service.onrender.com/webhook)
- Use Python 3.12+, install dependencies: pip install pyTelegramBotAPI flask requests
- Deploy as Web Service on Render.com, with start command: python app.py (rename this file to app.py if needed)

Features retained:
- Analyzes swaps on Meteora and Pump.fun
- Computes dataset

Changes:
- No CLI, instead Telegram bot handler
- Returns data instead of printing/saving CSVs
- Zips JSONs and sends via Telegram
"""

import os
import json
import time
import zipfile
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import statistics

import requests
import telebot
from telebot.types import Message
from flask import Flask, request

METEORA_PROGRAM_ID = "cpamdpZCGKUy5JxQXB4dcpGPiikHawvSWAd6mEn1sGG"
PUMPFUN_AMM = "G5UZAVbAf46s7cKWoyKu8kYTip9DGTpbLZ2qa9Aq69dP"
PUMPFUN_PROGRAM_IDS = {
    PUMPFUN_AMM,
    "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",
    "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ",
}
SOL_MINT = "So11111111111111111111111111111111111111112"  # WSOL mint
HEL_BASE = "https://api.helius.xyz/v0"

@dataclass
class SwapRecord:
    platform: str               # "Meteora" or "Pump.fun"
    signature: str
    timestamp: Optional[int]
    program_ids: List[str]
    buy_mint: Optional[str]
    buy_amount: float
    sell_mint: Optional[str]
    sell_amount: float
    sol_delta: float            # net SOL change (SOL in - SOL out) in SOL
    kind: str                   # 'BUY' or 'SELL' or 'UNKNOWN'
    source: Optional[str] = None

# ----------------------- Helpers -----------------------

def lamports_to_sol(lamports: int | float | None) -> float:
    if lamports is None:
        return 0.0
    try:
        return float(lamports) / 1e9
    except Exception:
        return 0.0

def fetch_address_transactions(wallet: str, api_key: str, limit: int = 100, pages: int = 5) -> List[Dict[str, Any]]:
    """Pull recent transactions (any type) for a wallet via Helius addresses endpoint with pagination."""
    all_txs: List[Dict[str, Any]] = []
    before: Optional[str] = None
    per_page = min(max(limit, 1), 1000)
    for _ in range(pages):
        url = f"{HEL_BASE}/addresses/{wallet}/transactions?api-key={api_key}&limit={per_page}"
        if before:
            url += f"&before={before}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        all_txs.extend(batch)
        before = batch[-1].get("signature")
        time.sleep(0.2)
    return all_txs

def load_enhanced_from_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", [])
    if not isinstance(data, list):
        raise ValueError("Enhanced file must be a JSON array or an object with 'data' array.")
    return data

def ensure_enhanced(signatures: List[str], api_key: str) -> List[Dict[str, Any]]:
    """Fetch enhanced transactions for given signatures; Helius bulk uses key 'transactions'."""
    enhanced: List[Dict[str, Any]] = []
    url = f"{HEL_BASE}/transactions?api-key={api_key}"
    for i in range(0, len(signatures), 100):
        batch = signatures[i:i + 100]
        payload = {"transactions": batch}
        try:
            r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=40)
            r.raise_for_status()
            result = r.json()
            if isinstance(result, list):
                enhanced.extend(result)
            else:
                pass  # Skip warning prints
        except Exception as e:
            pass  # Skip warnings
        time.sleep(0.25)
    return enhanced

def extract_program_ids(tx: Dict[str, Any]) -> List[str]:
    progs: set[str] = set()
    stack = list(tx.get("instructions") or [])
    while stack:
        ins = stack.pop()
        pid = ins.get("programId")
        if pid:
            progs.add(str(pid))
        for iins in ins.get("innerInstructions", []) or []:
            stack.append(iins)
    return sorted(progs)

# ------------------- Meteora detection -------------------

def is_meteora_tx(tx: Dict[str, Any]) -> bool:
    if METEORA_PROGRAM_ID in extract_program_ids(tx):
        return True
    src = (tx.get("source") or "") + json.dumps(tx.get("description", {}))
    return "meteora" in src.lower()

def classify_swap_meteora(tx: Dict[str, Any], wallet: str) -> Optional[Tuple[SwapRecord, dict]]:
    if not is_meteora_tx(tx):
        return None

    token_transfers = tx.get("tokenTransfers", []) or []
    native_transfers = tx.get("nativeTransfers", []) or []

    net_by_mint: dict[str, float] = defaultdict(float)
    for tt in token_transfers:
        mint = tt.get("mint")
        amt = float(tt.get("tokenAmount") or 0)
        frm = tt.get("fromUserAccount")
        to = tt.get("toUserAccount")
        if not mint:
            continue
        if to == wallet:
            net_by_mint[mint] += amt
        if frm == wallet:
            net_by_mint[mint] -= amt

    net_sol = 0.0
    for nt in native_transfers:
        lam = int(nt.get("amount") or 0)
        frm = nt.get("fromUserAccount")
        to = nt.get("toUserAccount")
        if to == wallet:
            net_sol += lam
        if frm == wallet:
            net_sol -= lam
    net_sol = lamports_to_sol(net_sol)

    non_sol_mints = {m: a for m, a in net_by_mint.items() if m != SOL_MINT and abs(a) > 0}
    if not non_sol_mints:
        return None

    buy_token_mint = max(non_sol_mints, key=lambda m: abs(non_sol_mints[m]))
    token_delta = non_sol_mints[buy_token_mint]

    kind = "BUY" if (token_delta > 0 and net_sol < 0) else ("SELL" if (token_delta < 0 and net_sol > 0) else "UNKNOWN")

    if kind == "BUY":
        buy_mint = buy_token_mint; buy_amt = abs(token_delta)
        sell_mint = "SOL";        sell_amt = abs(net_sol)
    elif kind == "SELL":
        buy_mint = "SOL";        buy_amt = abs(net_sol)
        sell_mint = buy_token_mint; sell_amt = abs(token_delta)
    else:
        buy_mint = buy_token_mint if token_delta > 0 else "SOL"
        buy_amt = abs(token_delta) if token_delta > 0 else abs(net_sol)
        sell_mint = "SOL" if token_delta > 0 else buy_token_mint
        sell_amt = abs(net_sol) if token_delta > 0 else abs(token_delta)

    rec = SwapRecord(
        platform="Meteora",
        signature=tx.get("signature",""),
        timestamp=tx.get("timestamp"),
        program_ids=extract_program_ids(tx),
        buy_mint=buy_mint,
        buy_amount=float(buy_amt or 0),
        sell_mint=sell_mint,
        sell_amount=float(sell_amt or 0),
        sol_delta=net_sol,
        kind=kind,
        source=tx.get("source")
    )
    debug = {
        "net_by_mint": net_by_mint,
        "native_sol_net": net_sol,
        "program_ids": rec.program_ids,
    }
    return rec, debug

# ------------------- Pump.fun detection -------------------

def is_pumpfun_tx(tx: Dict[str, Any]) -> bool:
    src = (tx.get("source") or "").upper()
    if "PUMP_AMM" in src:
        return True
    pids = set(extract_program_ids(tx))
    if pids & PUMPFUN_PROGRAM_IDS:
        return True
    for tt in (tx.get("tokenTransfers") or []):
        mint = (tt.get("mint") or "").lower()
        if mint.endswith("pump"):
            return True
    return False

def classify_swap_pumpfun(tx: Dict[str, Any], wallet: str) -> Optional[Tuple[SwapRecord, dict]]:
    if not is_pumpfun_tx(tx):
        return None

    token_transfers = tx.get("tokenTransfers", []) or []
    native_transfers = tx.get("nativeTransfers", []) or []

    net_by_mint: dict[str, float] = defaultdict(float)
    for tt in token_transfers:
        mint = tt.get("mint")
        amt = float(tt.get("tokenAmount") or 0)
        frm = tt.get("fromUserAccount")
        to = tt.get("toUserAccount")
        if not mint:
            continue
        if to == wallet:
            net_by_mint[mint] += amt
        if frm == wallet:
            net_by_mint[mint] -= amt

    net_sol = 0.0
    for nt in native_transfers:
        lam = int(nt.get("amount") or 0)
        frm = nt.get("fromUserAccount")
        to = nt.get("toUserAccount")
        if to == wallet:
            net_sol += lam
        if frm == wallet:
            net_sol -= lam
    net_sol = lamports_to_sol(net_sol)

    non_sol_mints = {m: a for m, a in net_by_mint.items() if m != SOL_MINT and abs(a) > 0}
    if not non_sol_mints:
        return None

    buy_token_mint = max(non_sol_mints, key=lambda m: abs(non_sol_mints[m]))
    token_delta = non_sol_mints[buy_token_mint]

    kind = "BUY" if (token_delta > 0 and net_sol < 0) else ("SELL" if (token_delta < 0 and net_sol > 0) else "UNKNOWN")

    if kind == "BUY":
        buy_mint = buy_token_mint; buy_amt = abs(token_delta)
        sell_mint = "SOL";        sell_amt = abs(net_sol)
    elif kind == "SELL":
        buy_mint = "SOL";        buy_amt = abs(net_sol)
        sell_mint = buy_token_mint; sell_amt = abs(token_delta)
    else:
        buy_mint = buy_token_mint if token_delta > 0 else "SOL"
        buy_amt = abs(token_delta) if token_delta > 0 else abs(net_sol)
        sell_mint = "SOL" if token_delta > 0 else buy_token_mint
        sell_amt = abs(net_sol) if token_delta > 0 else abs(token_delta)

    rec = SwapRecord(
        platform="Pump.fun",
        signature=tx.get("signature",""),
        timestamp=tx.get("timestamp"),
        program_ids=extract_program_ids(tx),
        buy_mint=buy_mint,
        buy_amount=float(buy_amt or 0),
        sell_mint=sell_mint,
        sell_amount=float(sell_amt or 0),
        sol_delta=net_sol,
        kind=kind,
        source=tx.get("source")
    )
    debug = {
        "net_by_mint": net_by_mint,
        "native_sol_net": net_sol,
        "program_ids": rec.program_ids,
    }
    return rec, debug

# ------------------- Dataset computation -------------------

def compute_dataset(wallet: str, swaps: List[SwapRecord], enhanced: List[Dict[str, Any]]):
    # 1) Basic PnL & counts
    buys = [s for s in swaps if s.kind == "BUY"]
    sells = [s for s in swaps if s.kind == "SELL"]

    gross_spent = sum(s.sell_amount for s in buys if s.sell_mint == "SOL")
    gross_recv  = sum(s.buy_amount  for s in sells if s.buy_mint  == "SOL")
    pnl_sol     = gross_recv - gross_spent

    # 2) Exposure (top mints)
    exposure = Counter()
    for s in swaps:
        if s.kind == "BUY" and s.buy_mint and s.buy_mint != "SOL":
            exposure[s.buy_mint] += s.buy_amount
        if s.kind == "SELL" and s.sell_mint and s.sell_mint != "SOL":
            exposure[s.sell_mint] -= s.sell_amount

    # 3) Realized trades, winrate, holding durations, r-multiples
    realized = []
    mint_positions: dict[str, list[SwapRecord]] = defaultdict(list)
    for s in sorted(swaps, key=lambda x: (x.timestamp or 0)):
        mint = s.buy_mint if (s.buy_mint and s.buy_mint != "SOL") else s.sell_mint
        if not mint or mint == "SOL":
            continue
        if s.kind == "BUY":
            mint_positions[mint].append(s)
        elif s.kind == "SELL" and mint_positions[mint]:
            buy = mint_positions[mint].pop(0)
            # Value terms are SOL legs: buy.sell_amount (SOL out) vs sell.buy_amount (SOL in)
            entry_sol = buy.sell_amount if buy.sell_mint == "SOL" else 0.0
            exit_sol  = s.buy_amount  if s.buy_mint  == "SOL" else 0.0
            r_mult = (exit_sol / entry_sol) if entry_sol else 0.0
            realized.append({
                "mint": mint,
                "buy_sig": buy.signature,
                "sell_sig": s.signature,
                "buy_time": buy.timestamp,
                "sell_time": s.timestamp,
                "duration_h": ((s.timestamp or 0) - (buy.timestamp or 0)) / 3600.0,
                "r_multiple": r_mult,
                "win": exit_sol > entry_sol,
            })

    realized_count = len(realized)
    winrate = (sum(1 for r in realized if r["win"]) / realized_count) if realized_count else 0.0
    median_hold = statistics.median([r["duration_h"] for r in realized]) if realized_count else 0.0
    avg_r = statistics.mean([r["r_multiple"] for r in realized]) if realized_count else 0.0

    # 4) Peak minute activity
    per_min = Counter()
    for s in swaps:
        ts = s.timestamp or 0
        if ts:
            minute = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M")
            per_min[minute] += 1
    if per_min:
        peak_minute, peak_tx = max(per_min.items(), key=lambda x: x[1])
    else:
        peak_minute, peak_tx = None, 0

    # 5) Top counterparties (from/to), derived from enhanced transfers
    top_from, top_to = Counter(), Counter()
    for tx in enhanced:
        for nt in tx.get("nativeTransfers", []) or []:
            frm, to = nt.get("fromUserAccount"), nt.get("toUserAccount")
            if frm == wallet and to:
                top_to[to] += 1
            if to == wallet and frm:
                top_from[frm] += 1
        for tt in tx.get("tokenTransfers", []) or []:
            frm, to = tt.get("fromUserAccount"), tt.get("toUserAccount")
            if frm == wallet and to:
                top_to[to] += 1
            if to == wallet and frm:
                top_from[frm] += 1

    dataset = {
        "n_swaps": len(swaps),
        "gross_spent_sol": float(gross_spent),
        "gross_recv_sol": float(gross_recv),
        "pnl_sol": float(pnl_sol),
        "exposure_top": exposure.most_common(10),
        "realized_trades": realized_count,
        "winrate": float(winrate),
        "median_hold_hours": float(median_hold),
        "avg_r_multiple": float(avg_r),
        "peak_minute": {"human_time": peak_minute, "tx_per_min": peak_tx},
        "top_from": dict(top_from.most_common(5)),
        "top_to": dict(top_to.most_common(5)),
    }
    return dataset

# ------------------- Main analysis -------------------

def analyze(wallet: str, api_key: str, limit: int = 100, pages: int = 15, from_file: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if from_file:
        enhanced = load_enhanced_from_file(from_file)
    else:
        txs = fetch_address_transactions(wallet, api_key, limit=limit, pages=pages)
        sigs = [t.get("signature") for t in txs if t.get("signature")]
        enhanced = ensure_enhanced(sigs, api_key)

    meteora_swaps: List[SwapRecord] = []
    pump_swaps: List[SwapRecord] = []

    # unknown_debug_meteora and pump skipped as not used in output

    for tx in enhanced:
        res_m = classify_swap_meteora(tx, wallet)
        if res_m:
            rec, dbg = res_m
            meteora_swaps.append(rec)
            continue  # avoid double-counting
        res_p = classify_swap_pumpfun(tx, wallet)
        if res_p:
            rec, dbg = res_p
            pump_swaps.append(rec)

    all_swaps = meteora_swaps + pump_swaps
    dataset = compute_dataset(wallet, all_swaps, enhanced)
    return enhanced, dataset

# ------------------- Telegram Bot Setup -------------------

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY')
WEBHOOK_URL = os.getenv('WEBHOOK_URL')  # Set to your Render URL + /webhook

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

@bot.message_handler(commands=['stats'])
def handle_stats(message: Message):
    parts = message.text.split()
    if len(parts) != 2:
        bot.reply_to(message, "Usage: /stats <wallet_address>")
        return

    wallet = parts[1]
    if not HELIUS_API_KEY:
        bot.reply_to(message, "Helius API key not configured.")
        return

    try:
        bot.reply_to(message, f"Analyzing wallet {wallet}... This may take a moment.")
        enhanced, dataset = analyze(wallet, HELIUS_API_KEY)

        # Create in-memory ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('enhanced.json', json.dumps(enhanced, ensure_ascii=False, indent=2))
            zipf.writestr('dataset.json', json.dumps(dataset, ensure_ascii=False, indent=2))

        zip_buffer.seek(0)
        # Compatible avec toutes les versions de pyTelegramBotAPI
        # Méthode compatible avec pyTelegramBotAPI >= 4.0
        statsfile = f"{wallet[:6]}_stats.zip"
        bot.send_document(
            chat_id=message.chat.id,
            document=(statsfile, zip_buffer, 'application/zip'),
            caption=f"Analysis complete for {wallet[:6]}..."
        )
        
        bot.reply_to(message, f"{dataset}")
    except Exception as e:
        bot.reply_to(message, f"Error analyzing wallet: {str(e)}")

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    else:
        return '', 403

if __name__ == '__main__':
    # Set webhook
    bot.remove_webhook()
    bot.set_webhook(WEBHOOK_URL)
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
