import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pyotp
from SmartApi.smartConnect import SmartConnect
import time
import io

# Import stock list from separate file
from Stock_list_token import stock_list


# ------------------- CONFIG -------------------
st.set_page_config(page_title="🕯️ Nandi Breakout Scanner", layout="wide")
st.title("🕯️ Nandi White Candle → High Breakout Scanner (Batch of 100)")
st.caption("Shows only those stocks where White Candle formed recently AND its HIGH is crossed within X days.")


# ------------------- ANGEL ONE LOGIN (YOUR CREDENTIALS) -------------------
api_key = "EKa93pFu"
client_id = "R59803990"
password = "1234"
totp_secret = "5W4MC6MMLANC3UYOAW2QDUIFEU"

try:
    totp = pyotp.TOTP(totp_secret).now()
    obj = SmartConnect(api_key=api_key)
    session_data = obj.generateSession(client_id, password, totp)
    st.success("✅ Angel One Login Successful")
except Exception as e:
    st.error(f"❌ Login Failed: {e}")
    st.stop()


# ------------------- HELPERS -------------------
def _make_from_to_datetime(from_date: dt.date, to_date: dt.date):
    """
    Candle API expects datetime-like strings.
    We'll use market-ish times to reduce empty data issues.
    """
    from_dt = dt.datetime.combine(from_date, dt.time(9, 15))
    to_dt = dt.datetime.combine(to_date, dt.time(15, 30))
    return from_dt, to_dt


def fetch_ohlc_data_by_token(symboltoken, interval, from_date, to_date):
    try:
        from_dt, to_dt = _make_from_to_datetime(from_date, to_date)

        params = {
            "exchange": "NSE",
            "symboltoken": str(symboltoken),
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
        }
        response = obj.getCandleData(params)

        if (not response) or (response.get("status") != True) or ("data" not in response) or (not response["data"]):
            return None

        data = pd.DataFrame(response["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data[["open", "high", "low", "close", "volume"]] = data[["open", "high", "low", "close", "volume"]].astype(float)
        return data
    except Exception:
        return None


def compute_cmo(close: pd.Series, length: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)

    sum_up = up.rolling(length).sum()
    sum_down = down.rolling(length).sum()

    denom = (sum_up + sum_down).replace(0, np.nan)
    cmo = 100 * (sum_up - sum_down) / denom
    return cmo.fillna(0)


def add_nandi_white_candle(df: pd.DataFrame, len_=20, mult=2.0, cmoLen=9, volMultiplier=1.5):
    """
    Same logic as your project:
    condition1: absKRI > dev and prev absKRI <= prev dev and changePerc >= 0
    condition2: CMO crosses from negative to positive
    whiteCandle = condition1 & condition2
    """
    close = df["close"]

    smaVal = close.rolling(len_).mean()
    dev = mult * (close.sub(smaVal).abs().rolling(len_).std())

    close_prev = close.shift(1)
    smaPrev = close_prev.rolling(len_).mean()
    devPrev = mult * (close_prev.sub(smaPrev).abs().rolling(len_).std())

    kri = close - smaVal
    absKRI = kri.abs()

    kriPrev = close_prev - smaPrev
    absKRIprev = kriPrev.abs()

    changePerc = (close.sub(close.shift(1)) / close.shift(1)) * 100

    condition1 = (absKRI > dev) & (absKRIprev <= devPrev) & (changePerc >= 0)

    cmo = compute_cmo(close, cmoLen)
    condition2 = (cmo > 0) & (cmo.shift(1) < 0)

    whiteCandle = condition1 & condition2

    vol_sma = df["volume"].rolling(20).mean()
    volSpike = df["volume"] > (vol_sma * volMultiplier)

    out = df.copy()
    out["smaVal"] = smaVal
    out["dev"] = dev
    out["cmo"] = cmo
    out["changePerc"] = changePerc
    out["condition1"] = condition1
    out["condition2"] = condition2
    out["whiteCandle"] = whiteCandle
    out["volSpike"] = volSpike
    return out


def find_high_cross_after(df_: pd.DataFrame, start_ts: pd.Timestamp, level: float, within_days: int):
    """
    Returns (cross_ts, cross_high, cross_close) if df high crosses level
    within 'within_days' days after start_ts; else None.
    """
    end_ts = start_ts + pd.Timedelta(days=int(within_days))
    future = df_[(df_["timestamp"] > start_ts) & (df_["timestamp"] <= end_ts)].copy()
    if future.empty:
        return None

    crossed = future[future["high"] >= level].sort_values("timestamp")
    if crossed.empty:
        return None

    row = crossed.iloc[0]
    return row["timestamp"], float(row["high"]), float(row["close"])


def excel_safe(df_: pd.DataFrame) -> pd.DataFrame:
    dfx = df_.copy()

    for col in dfx.columns:
        if pd.api.types.is_datetime64_any_dtype(dfx[col]):
            try:
                if getattr(dfx[col].dt, "tz", None) is not None:
                    dfx[col] = dfx[col].dt.tz_localize(None)
            except Exception:
                pass

    for col in dfx.columns:
        if dfx[col].dtype == "object":
            dfx[col] = dfx[col].map(
                lambda v: v.tz_localize(None)
                if isinstance(v, pd.Timestamp) and v.tzinfo is not None
                else v
            )
    return dfx


# ------------------- UI -------------------
items = list(stock_list.items())  # (symbol, token)
total = len(items)
batch_size = 100
batches = [items[i:i + batch_size] for i in range(0, total, batch_size)]
total_batches = len(batches)

r1c1, r1c2, r1c3, r1c4 = st.columns([1.2, 1, 1, 1])

with r1c1:
    batch_no = st.selectbox("📦 Batch Number (100 stocks)", list(range(1, total_batches + 1)), index=0)
    selected_batch = batches[batch_no - 1]
    st.write(f"Batch {batch_no}: {((batch_no-1)*batch_size)+1} → {min(batch_no*batch_size, total)}")

with r1c2:
    scan_interval = st.selectbox(
        "🕒 Interval",
        ["ONE_DAY", "ONE_HOUR", "THIRTY_MINUTE", "FIFTEEN_MINUTE", "FIVE_MINUTE", "ONE_MINUTE"],
        index=0
    )

with r1c3:
    scan_from_date = st.date_input("📆 Scan From", value=dt.date.today() - dt.timedelta(days=30))

with r1c4:
    scan_to_date = st.date_input("📆 Scan To", value=dt.date.today())

if scan_from_date > scan_to_date:
    st.warning("⚠️ Scan From cannot be after Scan To.")
    st.stop()

p1, p2, p3, p4 = st.columns(4)
with p1:
    len_ = st.number_input("SMA/StdDev Length (len)", min_value=1, value=20, step=1)
with p2:
    mult = st.number_input("Std Dev Multiplier (mult)", min_value=0.1, value=2.0, step=0.1)
with p3:
    cmoLen = st.number_input("CMO Length (cmoLen)", min_value=1, value=9, step=1)
with p4:
    volMultiplier = st.number_input("Volume Spike Multiplier", min_value=1.0, value=1.5, step=0.1)

b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
with b1:
    signal_lookback_days = st.number_input("🗓️ White Candle must be in last N days", min_value=1, max_value=180, value=10)
with b2:
    cross_within_days = st.number_input("🚀 X days to cross White Candle HIGH", min_value=1, max_value=60, value=3)
with b3:
    pick_signal = st.selectbox("If multiple signals, use:", ["LAST", "FIRST"], index=0)
with b4:
    per_request_sleep = st.number_input("⏳ Sleep/request (sec)", min_value=0.0, max_value=3.0, value=0.15, step=0.05)

cutoff_signal_date = scan_to_date - dt.timedelta(days=int(signal_lookback_days))
st.success(
    f"✅ Signal cutoff: {cutoff_signal_date} → White Candle date must be ≥ cutoff\n"
    f"✅ Breakout rule: HIGH must be crossed within {int(cross_within_days)} days after the signal candle."
)

run_scan = st.button(f"🚀 Scan Batch {batch_no}", use_container_width=True)


# ------------------- RUN SCAN -------------------
if run_scan:
    progress = st.progress(0)
    status = st.empty()

    results = []
    failed = 0
    n = len(selected_batch)

    status.info(f"Scanning Batch {batch_no}/{total_batches} … ({n} stocks)")

    for i, (sym, token) in enumerate(selected_batch, start=1):
        df_scan = fetch_ohlc_data_by_token(token, scan_interval, scan_from_date, scan_to_date)

        if df_scan is None or df_scan.empty:
            failed += 1
        else:
            df_n = add_nandi_white_candle(
                df_scan,
                len_=int(len_),
                mult=float(mult),
                cmoLen=int(cmoLen),
                volMultiplier=float(volMultiplier),
            )

            wc = df_n[df_n["whiteCandle"] == True].copy()
            if not wc.empty:
                wc["date"] = wc["timestamp"].dt.date
                wc = wc[wc["date"] >= cutoff_signal_date]

                if not wc.empty:
                    wc = wc.sort_values("timestamp")
                    signal_row = wc.iloc[-1] if pick_signal == "LAST" else wc.iloc[0]

                    wc_ts = signal_row["timestamp"]
                    wc_high = float(signal_row["high"])
                    wc_low = float(signal_row["low"])

                    cross = find_high_cross_after(df_scan, wc_ts, wc_high, int(cross_within_days))

                    # ✅ Keep ONLY if breakout happened within X days
                    if cross is not None:
                        cross_ts, cross_high, cross_close = cross
                        days_to_cross = (cross_ts.date() - wc_ts.date()).days

                        results.append({
                            "Symbol": sym,
                            "Token": str(token),

                            "WhiteCandle_Timestamp": wc_ts,
                            "WhiteCandle_High": wc_high,
                            "WhiteCandle_Low": wc_low,
                            "Signal_Close": float(signal_row["close"]),
                            "CMO": float(signal_row["cmo"]),
                            "Change%": float(signal_row["changePerc"]),
                            "VolSpike_On_WhiteCandle": bool(signal_row["volSpike"]),

                            "Cross_Timestamp": cross_ts,
                            "Cross_High": cross_high,
                            "Cross_Close": cross_close,
                            "Days_To_Cross": int(days_to_cross),

                            "Interval": scan_interval,
                            "Scan_From": scan_from_date,
                            "Scan_To": scan_to_date,
                            "Batch_No": batch_no,
                            "Cross_Within_Days": int(cross_within_days),
                        })

        progress.progress(i / n)
        if per_request_sleep > 0:
            time.sleep(float(per_request_sleep))

    status.success(f"✅ Done. Found: {len(results)} | Failed: {failed}")

    if not results:
        st.warning("No stocks matched: White Candle HIGH was not crossed within X days.")
    else:
        out_df = pd.DataFrame(results).sort_values(["Cross_Timestamp", "Days_To_Cross"], ascending=[False, True])
        st.dataframe(out_df, use_container_width=True)

        out_df_excel = excel_safe(out_df)
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            out_df_excel.to_excel(writer, index=False, sheet_name=f"Nandi_Breakout_B{batch_no}")

        st.download_button(
            "⬇️ Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"nandi_whitecandle_high_breakout_batch_{batch_no}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
