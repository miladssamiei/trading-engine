"""Stdlib-only local FTMO evidence importer and bounded coverage audit."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re,statistics,sys
from datetime import datetime,timezone,timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
NY=ZoneInfo("America/New_York"); PRAGUE=ZoneInfo("Europe/Prague")
SECRET_RE=re.compile(r"(?:password|passwd|token|login|private[_ -]?key|secret|access[_ -]?key|api[_ -]?key|account[_ -]?(?:id|number))",re.I)
FIELDS={"provenance":("source_kind","source","received_at"),"account":("account_fingerprint","server","broker","platform","currency","initial_balance","program","account_type","phase","captured_at","balance","equity"),"symbol":("name","contract_id","quote_currency","profit_currency","tick_size","tick_value_profit","tick_value_loss","contract_size","volume_min","volume_max","volume_step","margin_initial","stops_level","freeze_level","filling_modes","trading_sessions","swap_mode","swap_long","swap_short","rollover_multipliers")}
def _finite(x,positive=False):
    if isinstance(x,bool):return False
    try:d=Decimal(str(x))
    except Exception:return False
    return d.is_finite() and (not positive or d>0)
def _utc(x,label):
    if not isinstance(x,str):raise ValueError(f"{label} must be ISO8601 UTC")
    try:d=datetime.fromisoformat(x.replace("Z","+00:00"))
    except Exception as e:raise ValueError(f"{label} is malformed") from e
    if d.tzinfo is None or d.utcoffset()!=timezone.utc.utcoffset(d):raise ValueError(f"{label} must include UTC timezone")
    return d.astimezone(timezone.utc)
def _secret(x,path=""):
    if isinstance(x,dict):
        for k,v in x.items():
            if SECRET_RE.search(str(k)):return f"credential field at {path}{k}"
            z=_secret(v,path+str(k)+".")
            if z:return z
    elif isinstance(x,list):
        for i,v in enumerate(x):
            z=_secret(v,f"{path}{i}.")
            if z:return z
    return None
def _read_json(p):
    try:return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:raise ValueError(f"invalid JSON input: {Path(p).name}") from e
def _as_of(value):
    value = value or datetime.now(timezone.utc)
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("as-of must be timezone-aware")
    return value.astimezone(timezone.utc)


def validate_snapshot(data,manifest,as_of=None):
    errors=[];missing=[];as_of=_as_of(as_of)
    if not isinstance(data,dict):return ["snapshot must be an object"],[]
    if data.get("schema_version")!=1:errors.append("snapshot schema_version must be 1")
    sections={}
    for sec,fields in FIELDS.items():
        if sec not in data or data[sec] is None:missing.extend(f"{sec}.{f}" for f in fields);sections[sec]={}
        elif not isinstance(data[sec],dict):errors.append(f"{sec} must be an object");sections[sec]={}
        else:sections[sec]=data[sec];missing.extend(f"{sec}.{f}" for f in fields if f not in data[sec] or data[sec][f] is None)
    p,a,s=sections["provenance"],sections["account"],sections["symbol"]
    strings={"provenance":("source_kind","source"),"account":("account_fingerprint","server","broker","platform","currency","program","account_type","phase"),"symbol":("name","contract_id","quote_currency","profit_currency","swap_mode")}
    for sec,ks in strings.items():
        for k in ks:
            if k in sections[sec] and sections[sec][k] is not None and (not isinstance(sections[sec][k],str) or not sections[sec][k].strip()):errors.append(f"{sec}.{k} must be nonempty string")
    if p.get("source_kind") not in (None,"ftmo_export","synthetic_fixture"):errors.append("invalid provenance source_kind")
    times={}
    for sec,k in (("provenance","received_at"),("account","captured_at")):
        if sections[sec].get(k) is not None:
            try:times[k]=_utc(sections[sec][k],f"{sec}.{k}")
            except ValueError as e:errors.append(str(e))
    if times.get("captured_at") and times["captured_at"]>as_of:errors.append("snapshot capture is after as-of")
    if times.get("received_at") and times["received_at"]>as_of:errors.append("snapshot receipt is after as-of")
    if times.get("captured_at") and times.get("received_at") and times["captured_at"]>times["received_at"]:errors.append("snapshot receipt precedes capture")
    for k in ("initial_balance","balance","equity"):
        if k in a and a[k] is not None and not _finite(a[k],k=="initial_balance"):errors.append(f"account.{k} invalid numeric")
    for k in ("tick_size","tick_value_profit","tick_value_loss","contract_size","volume_min","volume_max","volume_step"):
        if k in s and s[k] is not None and not _finite(s[k],True):errors.append(f"symbol.{k} invalid numeric")
    for k in ("margin_initial","stops_level","freeze_level"):
        if k in s and s[k] is not None and (not _finite(s[k]) or Decimal(str(s[k]))<0):errors.append(f"symbol.{k} invalid nonnegative numeric")
    for k in ("swap_long","swap_short"):
        if k in s and s[k] is not None and not _finite(s[k]):errors.append(f"symbol.{k} invalid numeric")
    if all(_finite(s.get(k),True) for k in ("volume_min","volume_max","volume_step")) and Decimal(str(s["volume_min"]))>Decimal(str(s["volume_max"])):errors.append("volume bounds contradictory")
    fm=s.get("filling_modes")
    if fm == []: missing.append("symbol.filling_modes")
    elif fm is not None and (not isinstance(fm,list) or any(not isinstance(x,str) or not x.strip() for x in fm)):errors.append("filling_modes invalid")
    ts=s.get("trading_sessions")
    if ts == []: missing.append("symbol.trading_sessions")
    elif ts is not None and (not isinstance(ts,list) or any(not isinstance(x,dict) or not isinstance(x.get("weekday"),int) or isinstance(x.get("weekday"),bool) or not 0<=x["weekday"]<=6 or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d",str(x.get("open",""))) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d",str(x.get("close",""))) for x in ts)):errors.append("trading_sessions invalid")
    rm=s.get("rollover_multipliers")
    if rm == {}: missing.append("symbol.rollover_multipliers")
    elif rm is not None and (not isinstance(rm,dict) or set(rm)!={str(i) for i in range(7)} or any(not _finite(v) or Decimal(str(v))<0 for v in rm.values())):errors.append("rollover_multipliers invalid")
    if s.get("swap_mode") is not None and s["swap_mode"].lower() not in ("points","currency","money","percentage"):errors.append("unsupported swap_mode")
    expected=manifest.get("identity",{}) if isinstance(manifest,dict) else {}
    if not isinstance(expected, dict):
        errors.append("manifest identity is malformed")
        expected = {}
    for k,v in expected.items():
        if v is not None and k not in a:missing.append(f"account.{k}")
        elif v is not None and a.get(k)!=v:errors.append(f"identity mismatch: {k}")
    return errors,sorted(set(missing))
def _window(dt,name):
    n=dt.astimezone(NY);p=dt.astimezone(PRAGUE);m=n.hour*60+n.minute
    if name=="entry_exit_ny":return 570<=m<580,n.date()
    if name=="stress_ny":return 955<=m<965,n.date()
    pm=p.hour*60+p.minute;return pm>=1435 or pm<5,p.date()+timedelta(days=1) if pm>=1435 else p.date()
def _csv_read(path):
    try:
        with Path(path).open(newline="",encoding="utf-8-sig") as f:
            d=csv.Sniffer().sniff(f.read(4096),delimiters=",;");f.seek(0);return list(csv.DictReader(f,dialect=d))
    except Exception as e:raise ValueError("invalid quote CSV") from e
def validate_quotes(path,snapshot,as_of=None):
    errors=[];rows=_csv_read(path);as_of=_as_of(as_of);symbol=(snapshot or {}).get("symbol",{}) if isinstance(snapshot or {},dict) else {};req={"timestamp_utc","received_at_utc","bid","ask","symbol","contract_id","source_kind"};lo=lr=None;obs=[]
    if not rows:errors.append("quote CSV is empty")
    for i,r in enumerate(rows,1):
        if not req.issubset(r):errors.append(f"quote row {i} missing fields");continue
        try:o=_utc(r["timestamp_utc"],"quote timestamp");rc=_utc(r["received_at_utc"],"quote receipt")
        except ValueError as e:errors.append(f"quote row {i}: {e}");continue
        valid=True
        try:b=float(r["bid"]);ask=float(r["ask"]);valid=math.isfinite(b) and math.isfinite(ask) and b>0 and ask>=b
        except Exception:valid=False;b=ask=float("nan")
        if not valid:errors.append(f"quote row {i}: invalid bid/ask")
        if rc<o:errors.append(f"quote row {i}: receipt precedes observation");valid=False
        if o>as_of or rc>as_of:errors.append(f"quote row {i}: after as-of");valid=False
        if symbol.get("name") is not None and r["symbol"]!=symbol["name"]:errors.append(f"quote row {i}: symbol mismatch");valid=False
        if symbol.get("contract_id") is not None and r["contract_id"]!=symbol["contract_id"]:errors.append(f"quote row {i}: contract mismatch");valid=False
        if r["source_kind"] not in ("ftmo_export","synthetic_fixture"):errors.append(f"quote row {i}: invalid provenance");valid=False
        if lo and o<=lo:errors.append(f"quote row {i}: observations not chronological");valid=False
        if lr and rc<lr:errors.append(f"quote row {i}: receipts not chronological");valid=False
        lo,lr=o,rc
        if valid:obs.append((o,rc,b,ask,r["source_kind"]))
    return errors,obs
def audit(manifest,snapshot=None,quotes=None,as_of=None):
    as_of=_as_of(as_of);errors=[];missing=[];g={x:"INCOMPLETE" for x in ("G-D1","G-D2","G-D3","G-D4")}
    if not isinstance(manifest,dict) or not isinstance(manifest.get("identity"),dict):errors.append("manifest identity is malformed")
    if snapshot is None:missing.append("authentic snapshot")
    else:
        e,m=validate_snapshot(snapshot,manifest,as_of);errors+=e;missing+=m
        if any("identity mismatch" in x or x.startswith("account.") or x == "account must be an object" for x in e):g["G-D1"]="FAIL"
        elif not m:g["G-D1"]="INCOMPLETE"
        if any(x.startswith(("symbol.","provenance.","account.")) or "schema" in x or
               x in ("provenance must be an object", "symbol must be an object") for x in e):g["G-D2"]="FAIL"
        elif not m:g["G-D2"]="PASS"
        provenance = snapshot.get("provenance") if isinstance(snapshot,dict) else None
        if not isinstance(provenance,dict) or provenance.get("source_kind")!="ftmo_export":missing.append("independent authentic provenance")
        expected = manifest.get("identity", {}) if isinstance(manifest, dict) else {}
        if isinstance(expected, dict):
            for key in ("account_fingerprint", "server"):
                if expected.get(key) is None: missing.append(f"expected identity.{key}")
    cov={x:{"dates":0,"observations":0} for x in ("entry_exit_ny","rollover_prague","stress_ny")};summary={"count":0,"first_timestamp_utc":None,"last_timestamp_utc":None,"cash_session_weekday_dates":0,"holiday_calendar_validated":False,"spread":{"min":None,"median":None,"max":None},"receipt_latency_seconds":{"min":None,"median":None,"max":None}}
    if quotes is None:missing.append("quotes")
    else:
        e,ob=validate_quotes(quotes,snapshot or {},as_of);errors+=e
        if e:g["G-D3"]="FAIL"
        summary["count"]=len(ob)
        if ob:
            summary["first_timestamp_utc"]=ob[0][0].isoformat().replace("+00:00","Z");summary["last_timestamp_utc"]=ob[-1][0].isoformat().replace("+00:00","Z");ss=sorted(a-b for _,_,b,a,_ in ob);ll=sorted((r-o).total_seconds() for o,r,_,_,_ in ob)
            for dest,v in ((summary["spread"],ss),(summary["receipt_latency_seconds"],ll)):dest.update(min=v[0],median=statistics.median(v),max=v[-1])
        for n in cov:
            ds=set()
            for o in ob:
                ok,d=_window(o[0],n)
                if ok and d.weekday()<5:ds.add(d);cov[n]["observations"]+=1
            cov[n]["dates"]=len(ds)
        summary["cash_session_weekday_dates"]=len({o[0].astimezone(NY).date() for o in ob if o[0].astimezone(NY).weekday()<5 and 570<=o[0].astimezone(NY).hour*60+o[0].astimezone(NY).minute<960})
        if not e and all(x["dates"]>=20 for x in cov.values()):g["G-D3"]="INCOMPLETE";missing.append("independent uninterrupted coverage evidence")
    missing += ["execution/slippage/commission reconciliation","financing reconciliation","historical availability limitations","independent account-rule review"]
    return {"status":"FAIL" if errors else "INCOMPLETE","downstream_model_allowed":False,"errors":errors,"missing_evidence":sorted(set(missing)),"gates":g,"coverage":cov,"quote_summary":summary,"as_of":as_of.isoformat().replace("+00:00","Z")}
def _archive(path,out,name):
    raw=Path(path).read_bytes();(out/(name+".sha256")).write_text(hashlib.sha256(raw).hexdigest()+"\n",encoding="ascii");(out/name).write_bytes(raw)
def _csv_has_credentials(path):
    try:
        with Path(path).open(newline="",encoding="utf-8-sig") as f:
            sample=f.read(4096);f.seek(0)
            dialect=csv.Sniffer().sniff(sample,delimiters=",;")
            header=next(csv.reader(f,dialect=dialect),[])
        return any(SECRET_RE.search(x) for x in header)
    except Exception as e:raise ValueError("invalid quote CSV") from e
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--manifest",required=True);ap.add_argument("--snapshot");ap.add_argument("--quotes");ap.add_argument("--output-dir",required=True);ap.add_argument("--as-of");a=ap.parse_args(argv);out=Path(a.output_dir)
    if out.exists():print("output directory already exists",file=sys.stderr);return 1
    try:
        man=_read_json(a.manifest);snap=_read_json(a.snapshot) if a.snapshot else None
        if _secret(man) or (snap and _secret(snap)) or (a.quotes and _csv_has_credentials(a.quotes)):raise ValueError("credential-bearing input rejected before archival")
        ao=_utc(a.as_of,"as-of") if a.as_of else datetime.now(timezone.utc);result=audit(man,snap,a.quotes,ao);out.mkdir(parents=True)
        for p,n in ((a.manifest,"raw_manifest.json"),(a.snapshot,"raw_snapshot.json"),(a.quotes,"raw_quotes.csv")):
            if p:_archive(p,out,n)
        (out/"audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps({"status":result["status"],"downstream_model_allowed":False}));return 1 if result["status"]=="FAIL" else 2
    except Exception as e:print(str(e),file=sys.stderr);return 1
if __name__=="__main__":sys.exit(main())
