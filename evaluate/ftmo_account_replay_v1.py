"""Offline FTMO accounting; never connects to a broker or sends orders.

Completeness/origin assertions must come from independently reviewed evidence.
These functions validate their supplied contract, not the truth of an assertion.
Equity includes already-incurred floating P/L and costs. Exposure risk is only
additional loss from that marked equity, including future execution costs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import json
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

POLICY_VERSION = "FTMO_2STEP_SWING_10K_V1"
INITIAL_BALANCE = Decimal("10000")
DAILY_ALLOWANCE = Decimal("500")
TOTAL_FLOOR = Decimal("9000")


class ValidationError(ValueError):
    pass


class RiskRefusal(ValidationError):
    pass


def dec(value: Any, name: str = "value") -> Decimal:
    if isinstance(value, bool):
        raise ValidationError(f"{name}: boolean is not numeric")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValidationError(f"{name}: invalid number") from None
    if not number.is_finite():
        raise ValidationError(f"{name}: nonfinite number")
    return number


def nonnegative(value: Any, name: str = "value") -> Decimal:
    number = dec(value, name)
    if number < 0:
        raise ValidationError(f"{name}: must be nonnegative")
    return number


def utc_time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("timestamp must be timezone aware")
    return value.astimezone(timezone.utc)


def _prague() -> ZoneInfo:
    try:
        return ZoneInfo("Europe/Prague")
    except ZoneInfoNotFoundError:
        raise RiskRefusal("required Europe/Prague timezone unavailable") from None


def _currency(value: str) -> None:
    if value != "USD":
        raise RiskRefusal("verified USD account currency required")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"nonempty {name} required")
    return value


@dataclass(frozen=True)
class IdentityAttestation:
    account_fingerprint: str
    server: str
    broker: str
    platform: str
    currency: str
    initial_balance: Decimal
    program: str
    account_type: str
    phase: str
    independently_reviewed: bool = False

    def __post_init__(self):
        for name in ("account_fingerprint", "server"):
            _text(getattr(self, name), name)
        _currency(self.currency)
        if (self.broker != "FTMO" or self.platform != "MT5" or
                dec(self.initial_balance) != INITIAL_BALANCE or self.program != "2-Step" or
                self.account_type != "Swing" or self.phase not in ("Challenge", "Verification")):
            raise RiskRefusal("identity does not match supported FTMO policy")
        if self.independently_reviewed is not True:
            raise RiskRefusal("independent identity review required")


def attest_identity(actual: Mapping[str, Any], expected: Mapping[str, Any], *,
                    independently_reviewed: bool = False) -> IdentityAttestation:
    """Match externally reviewed identities; matching JSON alone is insufficient."""
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise ValidationError("identity mappings required")
    fields = tuple(k for k in IdentityAttestation.__dataclass_fields__ if k != "independently_reviewed")
    for name in fields:
        if actual.get(name) is None or expected.get(name) is None or actual[name] != expected[name]:
            raise RiskRefusal(f"identity mismatch or unresolved field: {name}")
    return IdentityAttestation(**{k: actual[k] for k in fields}, independently_reviewed=independently_reviewed)


def _identity(value: IdentityAttestation | None) -> None:
    if not isinstance(value, IdentityAttestation):
        raise RiskRefusal("reviewed account identity required")
    value.__post_init__()


def prague_midnight(value: datetime) -> datetime:
    local = utc_time(value).astimezone(_prague())
    return datetime.combine(local.date(), datetime.min.time(), tzinfo=_prague()).astimezone(timezone.utc)


@dataclass(frozen=True)
class BalanceObservation:
    timestamp: datetime
    balance: Decimal
    exact_midnight: bool = False

    def __post_init__(self):
        utc_time(self.timestamp)
        dec(self.balance, "balance")
        if type(self.exact_midnight) is not bool:
            raise ValidationError("exact-midnight flag must be boolean")


@dataclass(frozen=True)
class LedgerEvent:
    timestamp: datetime
    delta: Decimal
    event_id: str

    def __post_init__(self):
        utc_time(self.timestamp)
        dec(self.delta, "ledger delta")
        _text(self.event_id, "event ID")


def daily_anchor(observations: Iterable[BalanceObservation], at: datetime,
                 ledger: Iterable[LedgerEvent] = (), prior_balance: Any = None, *,
                 prior_at: datetime | None = None, coverage_end: datetime | None = None,
                 history_complete: bool = False) -> Decimal:
    """Midnight balance before events at reset; prior balance is after prior_at events.

    Reconstructed interval is (prior_at, reset). Complete coverage must extend
    through reset. An empty but explicitly complete event interval is legitimate.
    Exact-midnight observations mean the balance before any event at that instant.
    """
    now = utc_time(at)
    midnight = prague_midnight(now)
    observations = list(observations)
    times = [utc_time(o.timestamp) for o in observations]
    if times != sorted(times) or any(t > now for t in times):
        raise ValidationError("out-of-order or future balance observation")
    exact = [o for o in observations if o.exact_midnight and utc_time(o.timestamp) == midnight]
    if any(o.exact_midnight and utc_time(o.timestamp) != prague_midnight(o.timestamp) for o in observations):
        raise ValidationError("observation falsely labelled exact midnight")
    if exact:
        if len(exact) != 1:
            raise RiskRefusal("ambiguous midnight balance")
        return dec(exact[0].balance)
    if history_complete is not True or prior_balance is None or prior_at is None or coverage_end is None:
        raise RiskRefusal("complete timestamped reconstruction history required")
    start, end = utc_time(prior_at), utc_time(coverage_end)
    if start >= midnight or end < midnight or end > now:
        raise RiskRefusal("balance history does not cover reset")
    events = list(ledger)
    event_times = [utc_time(e.timestamp) for e in events]
    if event_times != sorted(event_times) or len({e.event_id for e in events}) != len(events):
        raise ValidationError("out-of-order or duplicate ledger event")
    if any(t <= start or t > end for t in event_times):
        raise ValidationError("ledger event outside attested coverage")
    return dec(prior_balance) + sum((dec(e.delta) for e in events if utc_time(e.timestamp) < midnight), Decimal(0))


def firm_status(equity: Any, midnight_balance: Any, *, currency: str) -> dict:
    _currency(currency)
    eq, mid = dec(equity), dec(midnight_balance)
    floor = mid - DAILY_ALLOWANCE
    return {"daily_floor": floor, "total_floor": TOTAL_FLOOR,
            "daily_breach": eq < floor, "total_breach": eq < TOTAL_FLOOR,
            "no_headroom": eq <= max(floor, TOTAL_FLOOR)}


@dataclass(frozen=True)
class RiskLimits:
    per_position: Decimal = Decimal("10")
    aggregate: Decimal = Decimal("50")
    daily: Decimal = Decimal("100")
    weekly: Decimal = Decimal("200")
    strategy_hwm: Decimal = Decimal("300")
    floor_reserve: Decimal = Decimal("150")

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            nonnegative(getattr(self, name), name)


@dataclass(frozen=True)
class RiskSnapshot:
    equity: Decimal
    midnight_balance: Decimal
    weekly_anchor: Decimal | None
    strategy_hwm: Decimal | None
    observed_at: datetime
    anchor_at: datetime
    identity: IdentityAttestation | None = None
    existing_risk: Decimal | None = None
    history_complete: bool = False
    exposure_complete: bool = False


def risk_allowance(snapshot: RiskSnapshot, limits: RiskLimits = RiskLimits(), *,
                   as_of: datetime, max_age_seconds: Any = 5,
                   additional_liquidation_reserve: Any = 0) -> Decimal:
    _identity(snapshot.identity)
    now, observed = utc_time(as_of), utc_time(snapshot.observed_at)
    age = dec((now - observed).total_seconds())
    if age < 0 or age > nonnegative(max_age_seconds, "maximum snapshot age"):
        raise RiskRefusal("stale or future account snapshot")
    if utc_time(snapshot.anchor_at) != prague_midnight(now):
        raise RiskRefusal("missing current Prague reset anchor")
    if snapshot.history_complete is not True or snapshot.exposure_complete is not True:
        raise RiskRefusal("complete anchor history and account exposure required")
    if snapshot.weekly_anchor is None or snapshot.strategy_hwm is None or snapshot.existing_risk is None:
        raise RiskRefusal("weekly/HWM anchors and bounded exposure required")
    eq, mid = dec(snapshot.equity), dec(snapshot.midnight_balance)
    weekly, hwm = dec(snapshot.weekly_anchor), dec(snapshot.strategy_hwm)
    existing = nonnegative(snapshot.existing_risk, "existing incremental risk")
    reserve = nonnegative(additional_liquidation_reserve, "additional liquidation reserve")
    limits.__post_init__()
    if firm_status(eq, mid, currency=snapshot.identity.currency)["no_headroom"]:
        return Decimal(0)
    caps = [dec(limits.per_position), dec(limits.aggregate) - existing,
            eq - (mid - dec(limits.daily)) - existing,
            eq - (weekly - dec(limits.weekly)) - existing,
            eq - (hwm - dec(limits.strategy_hwm)) - existing,
            eq - (mid - DAILY_ALLOWANCE) - dec(limits.floor_reserve) - existing,
            eq - TOTAL_FLOOR - dec(limits.floor_reserve) - existing]
    return max(Decimal(0), min(caps) - reserve)


def risk_allowance_from_exposure(snapshot: RiskSnapshot, open_risk: Iterable[Any],
                                 pending_risk: Iterable[Any], *, exposure_complete: bool = False,
                                 **kwargs) -> Decimal:
    """Add all incremental manual/open/partial/pending losses; no correlation netting."""
    if exposure_complete is not True:
        raise RiskRefusal("complete all-account exposure declaration required")
    risks = [nonnegative(r, "bounded exposure risk") for r in list(open_risk) + list(pending_risk)]
    return risk_allowance(replace(snapshot, existing_risk=sum(risks, Decimal(0)), exposure_complete=True), **kwargs)


def size_lots(allowance: Any, risk_per_lot: Any, volume_step: Any, volume_min: Any,
              volume_max: Any, *, identity: IdentityAttestation | None = None) -> Decimal:
    _identity(identity)
    a = nonnegative(allowance, "allowance")
    r, step, vmin, vmax = map(dec, (risk_per_lot, volume_step, volume_min, volume_max))
    if min(r, step, vmin) <= 0 or vmax < vmin:
        raise ValidationError("invalid risk or lot constraints")
    lots = (min(a / r, vmax) / step).to_integral_value(rounding=ROUND_DOWN) * step
    return Decimal(0) if lots < vmin else lots


def risk_per_lot(tick_size: Any, tick_value_loss: Any, stop_distance: Any, costs: Any = 0,
                 *, currency: str) -> Decimal:
    """Tick value must be the verified USD value per tick per lot; costs per lot."""
    _currency(currency)
    tick, value, distance = map(dec, (tick_size, tick_value_loss, stop_distance))
    if min(tick, value, distance) <= 0:
        raise ValidationError("positive tick economics and stop distance required")
    return distance / tick * value + nonnegative(costs, "future costs per lot")


def midpoint_spread_cost(entry_full_spread: Any, exit_full_spread: Any, lots: Any = 1,
                         value_per_point: Any = 1, *, currency: str) -> Decimal:
    _currency(currency)
    entry, exit_, volume, value = map(nonnegative, (entry_full_spread, exit_full_spread, lots, value_per_point))
    if value <= 0:
        raise ValidationError("positive account-currency point value required")
    return (entry + exit_) / 2 * volume * value


def actual_fill_cost(commission: Any = 0, slippage: Any = 0, financing: Any = 0,
                     *, currency: str) -> Decimal:
    """Additional cost only, with financing positive for debit/negative for credit.

    Slippage is incremental relative to the observed executable bid/ask. Pass
    zero if already included in the chosen actual-fill P/L ledger. No spread term.
    """
    _currency(currency)
    return nonnegative(commission) + nonnegative(slippage) + dec(financing)


def qualification_complete(balance: Any, phase: str, trading_dates: Iterable[date | datetime], *,
                           open_positions: int, pending_orders: int, known_breach: bool,
                           history_complete: bool = False, identity: IdentityAttestation | None = None) -> bool:
    """Bounded arithmetic check, not certification of unobserved intraday compliance."""
    _identity(identity)
    if phase != identity.phase or phase not in ("Challenge", "Verification"):
        raise ValidationError("phase does not match reviewed identity")
    for value in (open_positions, pending_orders):
        if type(value) is not int or value < 0:
            raise ValidationError("nonnegative integer position/order counts required")
    if type(known_breach) is not bool:
        raise ValidationError("explicit known-breach boolean required")
    dates = set()
    for value in trading_dates:
        if isinstance(value, datetime):
            dates.add(utc_time(value).astimezone(_prague()).date())
        elif type(value) is date:
            dates.add(value)  # caller attests these are Prague trade-entry dates
        else:
            raise ValidationError("Prague trade dates or aware timestamps required")
    target = INITIAL_BALANCE + (Decimal("1000") if phase == "Challenge" else Decimal("500"))
    return (dec(balance) >= target and len(dates) >= 4 and open_positions == 0 and
            pending_orders == 0 and known_breach is False and history_complete is True)


def _namespace(server: str, account_fingerprint: str, policy_version: str) -> dict:
    for name, value in (("server", server), ("account fingerprint", account_fingerprint)):
        _text(value, name)
    if policy_version != POLICY_VERSION:
        raise RiskRefusal("unsupported persisted-state policy")
    return {"server": server, "account_fingerprint": account_fingerprint, "policy_version": policy_version}


def _state_payload(payload: Any, stamp: datetime) -> dict:
    if not isinstance(payload, dict):
        raise RiskRefusal("invalid persisted-state payload")
    try:
        balance = dec(payload["midnight_balance"])
        anchor = utc_time(datetime.fromisoformat(payload["anchor_at"]))
    except (KeyError, TypeError, ValueError):
        raise RiskRefusal("missing or corrupt persisted anchor") from None
    if anchor != prague_midnight(stamp):
        raise RiskRefusal("persisted anchor does not match state reset date")
    return {"midnight_balance": str(balance), "anchor_at": anchor.isoformat()}


def serialize_state(payload: Mapping[str, Any], *, server: str, account_fingerprint: str,
                    policy_version: str, asof_utc: datetime) -> str:
    stamp = utc_time(asof_utc)
    return json.dumps({"schema": 1, "namespace": _namespace(server, account_fingerprint, policy_version),
                       "asof_utc": stamp.isoformat(), "payload": _state_payload(payload, stamp)}, sort_keys=True)


def deserialize_state(raw: str, *, server: str, account_fingerprint: str, policy_version: str,
                      now_utc: datetime, max_age_seconds: Any = 5) -> Mapping[str, Any]:
    namespace = _namespace(server, account_fingerprint, policy_version)
    try:
        obj = json.loads(raw)
        if (not isinstance(obj, dict) or type(obj.get("schema")) is not int or obj["schema"] != 1 or
                obj.get("namespace") != namespace):
            raise RiskRefusal("corrupt or mismatched state")
        stamp = utc_time(datetime.fromisoformat(obj["asof_utc"]))
        payload = _state_payload(obj.get("payload"), stamp)
    except (TypeError, KeyError, ValueError):
        raise RiskRefusal("corrupt, missing or mismatched persisted state") from None
    now = utc_time(now_utc)
    age = dec((now - stamp).total_seconds())
    if age < 0 or age > nonnegative(max_age_seconds) or prague_midnight(now) != prague_midnight(stamp):
        raise RiskRefusal("stale state; reconstruct current reset from complete history")
    return payload


compute_risk_allowance = risk_allowance
calculate_lot_size = size_lots
reconstruct_midnight_balance = daily_anchor
