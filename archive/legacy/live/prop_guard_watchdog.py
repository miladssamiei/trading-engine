"""Standalone account watchdog and secondary circuit breaker for the
PropFirmGuard execution layer (SPEC_EXECUTION_INFRASTRUCTURE, Path C).

Runs as an independent process polling the MT5 account every second.
Keeps its own High-Water Mark and server-day rollover balance in SQLite
(data/queue/account_state.db) so it cannot be laundered by terminal
restarts, monitors the terminal global variable PROP_GUARD_BREACHED, and
provides fail-safe redundancy: if the EA-side guard is hung or the
connection stalls and daily drawdown >= 3.7% or trailing drawdown >= 8.2%
(0.2% buffer past the EA's 3.5%/8.0% triggers), the watchdog flattens
every open position with opposite-side market deals and cancels every
pending order directly from Python, then latches off in its own state
table until a human runs it with --reset.

Refuses to run against a non-demo account unless --allow-live is given.
"""

import argparse
import datetime as dt
import os
import sqlite3
import sys
import time

REPO = r"C:\Users\Milad\Desktop\RR"
TERMINAL_EXE = r"C:\Program Files\MetaTrader 5\terminal64.exe"

GTV_BREACHED = "PROP_GUARD_BREACHED"
EA_DAILY_PCT = 3.5
EA_TRAILING_PCT = 8.0
WD_DAILY_PCT = 3.7
WD_TRAILING_PCT = 8.2
WD_MAGIC = 26082999
MAX_SLIPPAGE_INDEX_PT = 0.20
FLATTEN_ATTEMPTS = 3
LOG_ROTATE_BYTES = 10 * 1024 * 1024
PRIMARY_SYMBOL = "US500.c"


def utcnow():
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


class WdLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, msg):
        if (os.path.exists(self.path)
                and os.path.getsize(self.path) > LOG_ROTATE_BYTES):
            os.replace(self.path, self.path + ".1")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("%s %s\n" % (utcnow(), msg))
        print("%s %s" % (utcnow(), msg), flush=True)


class Watchdog:
    def __init__(self, args):
        self.args = args
        self.log = WdLogger(args.log)
        os.makedirs(os.path.dirname(os.path.join(REPO, args.db)), exist_ok=True)
        self.db = sqlite3.connect(os.path.join(REPO, args.db))
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS account_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                hwm REAL NOT NULL,
                eod_balance REAL NOT NULL,
                eod_date TEXT NOT NULL,
                watchdog_latched INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS poll_log (
                ts TEXT, equity REAL, balance REAL, margin REAL,
                margin_free REAL, daily_dd_pct REAL, trailing_dd_pct REAL,
                positions INTEGER, pendings INTEGER, spread REAL,
                ea_latched INTEGER, action TEXT
            );
            CREATE TABLE IF NOT EXISTS flatten_events (
                ts TEXT, trigger TEXT, detail TEXT
            );
            """
        )
        self.db.commit()

    def state(self):
        row = self.db.execute(
            "SELECT hwm, eod_balance, eod_date, watchdog_latched "
            "FROM account_state WHERE id = 1").fetchone()
        return row

    def server_date(self, mt5):
        tick = mt5.symbol_info_tick(PRIMARY_SYMBOL)
        if tick is None:
            for p in mt5.positions_get() or []:
                tick = mt5.symbol_info_tick(p.symbol)
                if tick:
                    break
        if tick is None:
            return None
        return dt.datetime.utcfromtimestamp(tick.time).strftime("%Y-%m-%d")

    def update_state(self, equity, balance, sdate):
        row = self.state()
        if row is None:
            self.db.execute(
                "INSERT INTO account_state (id, hwm, eod_balance, eod_date,"
                " watchdog_latched, updated_at) VALUES (1, ?, ?, ?, 0, ?)",
                (max(equity, balance), balance, sdate, utcnow()))
            return max(equity, balance), balance, 0
        hwm, eod_bal, eod_date, latched = row
        if sdate is not None and sdate != eod_date:
            eod_bal = balance
            eod_date = sdate
        if equity > hwm:
            hwm = equity
        self.db.execute(
            "UPDATE account_state SET hwm = ?, eod_balance = ?, eod_date = ?,"
            " updated_at = ? WHERE id = 1", (hwm, eod_bal, eod_date, utcnow()))
        return hwm, eod_bal, latched

    def spread(self, mt5):
        tick = mt5.symbol_info_tick(PRIMARY_SYMBOL)
        if tick is None or tick.bid <= 0:
            return None
        return tick.ask - tick.bid

    def deviation_points(self, mt5, symbol):
        info = mt5.symbol_info(symbol)
        if info is None or info.point <= 0:
            return 20
        return max(1, round(MAX_SLIPPAGE_INDEX_PT / info.point))

    def flatten_all(self, mt5, trigger, detail):
        events = []
        for attempt in range(FLATTEN_ATTEMPTS):
            positions = mt5.positions_get() or []
            pendings = mt5.orders_get() or []
            if not positions and not pendings:
                break
            for p in positions:
                tick = mt5.symbol_info_tick(p.symbol)
                if tick is None:
                    continue
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": p.symbol,
                    "volume": p.volume,
                    "type": (mt5.ORDER_TYPE_SELL if p.type == 0
                             else mt5.ORDER_TYPE_BUY),
                    "position": p.ticket,
                    "price": tick.bid if p.type == 0 else tick.ask,
                    "deviation": self.deviation_points(mt5, p.symbol),
                    "magic": WD_MAGIC,
                    "comment": "WD emergency flatten",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res = None
                if self.args.dry_run:
                    events.append("dryrun close ticket=%d" % p.ticket)
                else:
                    res = mt5.order_send(req)
                    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                        req["type_filling"] = mt5.ORDER_FILLING_FOK
                        res = mt5.order_send(req)
                    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                        req["type_filling"] = mt5.ORDER_FILLING_RETURN
                        res = mt5.order_send(req)
                    events.append(
                        "close ticket=%d retcode=%s"
                        % (p.ticket,
                           getattr(res, "retcode", "send-failed")))
            for o in pendings:
                if self.args.dry_run:
                    events.append("dryrun remove order=%d" % o.ticket)
                else:
                    res = mt5.order_send({
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order": o.ticket,
                    })
                    events.append(
                        "remove order=%d retcode=%s"
                        % (o.ticket,
                           getattr(res, "retcode", "send-failed")))
            time.sleep(1.0)
        self.db.execute(
            "INSERT INTO flatten_events (ts, trigger, detail) "
            "VALUES (?, ?, ?)", (utcnow(), trigger, "; ".join(events)))
        self.db.execute(
            "UPDATE account_state SET watchdog_latched = 1,"
            " updated_at = ? WHERE id = 1", (utcnow(),))
        self.db.commit()
        self.log.write("EMERGENCY FLATTEN trigger=%s %s"
                       % (trigger, "; ".join(events) or "no open tickets"))

    def ea_latched(self, mt5):
        try:
            if hasattr(mt5, "global_variable_check"):
                return (mt5.global_variable_check(GTV_BREACHED)
                        and mt5.global_variable_get(GTV_BREACHED) > 0)
        except Exception:
            pass
        gv = os.path.join(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Terminal"
                          r"\D0E8209F77C8CF37AD8BF550E51FF075",
                          "gvariables.dat")
        try:
            if os.path.exists(gv):
                blob = open(gv, "rb").read()
                return GTV_BREACHED.encode("ascii") in blob
        except OSError:
            pass
        return False

    def poll_once(self, mt5):
        acc = mt5.account_info()
        if acc is None:
            return "api-error"
        sdate = self.server_date(mt5)
        hwm, eod_bal, latched = self.update_state(acc.equity, acc.balance,
                                                  sdate)
        daily_dd = (eod_bal - acc.equity) / eod_bal * 100.0 if eod_bal > 0 else 0.0
        trail_dd = (hwm - acc.equity) / hwm * 100.0 if hwm > 0 else 0.0
        positions = mt5.positions_get() or []
        pendings = mt5.orders_get() or []
        spread = self.spread(mt5)
        ea = self.ea_latched(mt5)
        action = "poll"

        if latched:
            action = "latched"
        elif not self.args.dry_run or True:
            trigger = None
            if daily_dd >= WD_DAILY_PCT:
                trigger = "DAILY_DD %.2f%% >= %.1f%%" % (daily_dd,
                                                          WD_DAILY_PCT)
            elif trail_dd >= WD_TRAILING_PCT:
                trigger = "TRAILING_DD %.2f%% >= %.1f%%" % (trail_dd,
                                                            WD_TRAILING_PCT)
            elif ea and positions:
                trigger = "EA_LATCHED_WITH_OPEN_POSITIONS"
            if trigger:
                action = "FLATTEN"
                self.flatten_all(mt5, trigger, "equity=%.2f" % acc.equity)
                latched = 1

        self.db.execute(
            "INSERT INTO poll_log (ts, equity, balance, margin, margin_free,"
            " daily_dd_pct, trailing_dd_pct, positions, pendings, spread,"
            " ea_latched, action) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (utcnow(), acc.equity, acc.balance, acc.margin,
             acc.margin_free, daily_dd, trail_dd, len(positions),
             len(pendings), spread, 1 if ea else 0, action))
        self.db.commit()
        self.log.write(
            "eq=%.2f bal=%.2f marg=%.2f mfr=%.2f pos=%d pend=%d "
            "sprd=%s dDD=%.2f%% tDD=%.2f%% ea=%d act=%s%s"
            % (acc.equity, acc.balance, acc.margin, acc.margin_free,
               len(positions), len(pendings),
               "%.3f" % spread if spread is not None else "n/a",
               daily_dd, trail_dd, 1 if ea else 0, action,
               " DRYRUN" if self.args.dry_run else ""))
        return action

    def run(self):
        import MetaTrader5 as mt5
        if not mt5.initialize(TERMINAL_EXE):
            self.log.write("INIT FAILED %s" % (mt5.last_error(),))
            return 2
        acc = None
        for _ in range(30):
            acc = mt5.account_info()
            if acc is not None and acc.login != 0:
                break
            time.sleep(1.0)
        if acc is None:
            self.log.write("NO ACCOUNT INFO")
            mt5.shutdown()
            return 2
        if acc.trade_mode != 0 and not self.args.allow_live:
            self.log.write("REFUSING NON-DEMO ACCOUNT login=%d trade_mode=%d"
                           " (use --allow-live to override)"
                           % (acc.login, acc.trade_mode))
            mt5.shutdown()
            return 3
        self.log.write("watchdog start login=%d server=%s equity=%.2f"
                       " interval=%.1fs dry_run=%s"
                       % (acc.login, acc.server, acc.equity,
                          self.args.interval, self.args.dry_run))
        deadline = time.time() + self.args.max_runtime
        code = 0
        try:
            while True:
                act = self.poll_once(mt5)
                if act == "api-error":
                    code = 1
                if self.args.once:
                    break
                if time.time() >= deadline:
                    self.log.write("max runtime reached, exiting")
                    break
                time.sleep(self.args.interval)
        except KeyboardInterrupt:
            self.log.write("interrupted by user")
        self.log.write("watchdog stop")
        mt5.shutdown()
        return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="clear the watchdog latch (human action)")
    ap.add_argument("--max-runtime", type=float, default=86400.0)
    ap.add_argument("--db", default=r"data\queue\account_state.db")
    ap.add_argument("--log", default=r"data\logs\watchdog.log")
    ap.add_argument("--allow-live", action="store_true")
    args = ap.parse_args()

    wd = Watchdog(args)
    if args.reset:
        wd.db.execute("UPDATE account_state SET watchdog_latched = 0,"
                      " updated_at = ? WHERE id = 1", (utcnow(),))
        wd.db.commit()
        wd.log.write("watchdog latch cleared by --reset")
        return 0
    sys.exit(wd.run())


if __name__ == "__main__":
    main()
