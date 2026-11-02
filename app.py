import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Scanner", page_icon="📊", layout="wide")

st.markdown("""
<style>
/* Mobile-friendly layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { min-width: 260px; }
div[data-testid="stMetric"] { padding: 0.65rem; border: 1px solid rgba(128,128,128,.22); border-radius: 10px; }
.stButton > button { width: 100%; min-height: 2.7rem; }
div[data-testid="stDataFrame"] { width: 100%; }
@media (max-width: 768px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  h1 { font-size: 1.65rem !important; }
  h2, h3 { font-size: 1.2rem !important; }
  [data-testid="stHorizontalBlock"] { gap: 0.5rem; }
  div[data-testid="stMetric"] { padding: 0.45rem; }
  div[data-testid="stMetricValue"] { font-size: 1.15rem; }
  [data-testid="stSidebar"] { min-width: 0; }
  .stPlotlyChart { overflow-x: auto; }
}
</style>
""", unsafe_allow_html=True)

st.title("Crypto Scanner")
st.caption("Pemindai market dan sinyal teknikal. Sinyal bersifat heuristik, bukan jaminan harga akan naik.")

BASE_HEADERS = {"User-Agent": "Mozilla/5.0 CryptoScanner/1.0"}

@st.cache_data(ttl=180)
def get_market_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd", "order": "volume_desc", "per_page": 250,
        "page": 1, "sparkline": "false", "price_change_percentage": "24h,7d"
    }
    r = requests.get(url, params=params, headers=BASE_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=900)
def get_binance_usdt_pairs():
    """Ambil pasangan spot USDT yang aktif dari metadata Binance."""
    url = "https://data-api.binance.vision/api/v3/exchangeInfo"
    r = requests.get(url, headers=BASE_HEADERS, timeout=25)
    r.raise_for_status()
    payload = r.json()
    rows = []
    for item in payload.get("symbols", []):
        if (item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", True)):
            rows.append({
                "base": item.get("baseAsset", ""),
                "symbol": item.get("symbol", ""),
                "base_name": item.get("baseAsset", "")
            })
    return rows

@st.cache_data(ttl=90)
def get_klines(symbol, interval="1h", limit=100):
    url = "https://data-api.binance.vision/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit},
                    headers=BASE_HEADERS, timeout=25)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        return pd.DataFrame()
    cols = ["open_time","Open","High","Low","Close","Volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    d = pd.DataFrame(raw, columns=cols)
    for col in ["Open","High","Low","Close","Volume","quote_volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["Time"] = pd.to_datetime(d["open_time"], unit="ms")
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    d["RSI14"] = 100 - (100 / (1 + rs))
    d["VolAvg20"] = d["Volume"].shift(1).rolling(20).mean()
    d["VolumeSpike"] = d["Volume"] / d["VolAvg20"].replace(0, pd.NA)
    d["Resistance20"] = d["High"].shift(1).rolling(20).max()
    d["Breakout"] = d["Close"] > d["Resistance20"]
    prev_close = d["Close"].shift(1)
    true_range = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["ATR14"] = true_range.rolling(14).mean()
    return d

def evaluate_signal(symbol, interval="1h"):
    candles = get_klines(symbol, interval, 100)
    if candles.empty or len(candles) < 55:
        return None
    # Use latest closed candle, not the currently forming candle.
    row = candles.iloc[-2]
    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    rsi = float(row["RSI14"]) if pd.notna(row["RSI14"]) else 50.0
    spike = float(row["VolumeSpike"]) if pd.notna(row["VolumeSpike"]) else 0.0
    breakout = bool(row["Breakout"]) if pd.notna(row["Breakout"]) else False
    bullish = close > ema20 > ema50
    bearish = close < ema20 < ema50
    score = 0
    score += 30 if bullish else (0 if bearish else 12)
    score += 20 if 50 <= rsi <= 68 else (10 if 45 <= rsi < 50 or 68 < rsi <= 74 else 0)
    score += 25 if spike >= 2 else (15 if spike >= 1.5 else (5 if spike >= 1.2 else 0))
    score += 25 if breakout else (10 if close > ema20 else 0)
    score = min(100, score)

    if bullish and breakout and spike >= 1.5 and 50 <= rsi <= 70:
        signal = "STRONG BUY"
    elif bullish and score >= 55 and rsi < 72:
        signal = "BUY"
    elif bearish and rsi < 48:
        signal = "STRONG SELL" if spike >= 1.5 else "SELL"
    elif not bullish and not bearish and 45 <= rsi <= 55:
        signal = "SIDEWAYS"
    elif close < ema20 and rsi < 50:
        signal = "SELL"
    else:
        signal = "WAIT"

    return {
        "Pair": symbol, "Signal": signal, "Score": score, "Price": close,
        "RSI 14": round(rsi, 1), "Volume Spike": round(spike, 2),
        "Breakout": "Yes" if breakout else "No",
        "Trend": "Bullish" if bullish else ("Bearish" if bearish else "Mixed"),
        "Time": row["Time"]
    }

def usd(v):
    if pd.isna(v): return "-"
    return f"${v:,.0f}"

# Sidebar controls
st.sidebar.header("Filter Market")
search = st.sidebar.text_input("Cari coin atau simbol")
min_volume = st.sidebar.number_input("Volume minimum 24 jam (USD)", min_value=0, value=100000, step=100000)
max_cap = st.sidebar.number_input("Market cap maksimum (USD)", min_value=0, value=50000000, step=5000000)
min_cap = st.sidebar.number_input("Market cap minimum (USD)", min_value=0, value=0, step=1000000)

if st.button("Refresh Market"):
    st.cache_data.clear()
    st.rerun()

try:
    market = pd.DataFrame(get_market_data())
    market = market.rename(columns={
        "name":"Coin", "symbol":"Symbol", "current_price":"Price USD",
        "market_cap":"Market Cap", "total_volume":"Volume 24h",
        "price_change_percentage_24h":"Change 24h",
        "price_change_percentage_7d_in_currency":"Change 7d"
    })
    for col in ["Price USD","Market Cap","Volume 24h","Change 24h","Change 7d"]:
        if col not in market: market[col] = pd.NA
        market[col] = pd.to_numeric(market[col], errors="coerce")
    market["Symbol"] = market["Symbol"].astype(str).str.upper()
    market["Volume/Market Cap %"] = (market["Volume 24h"] / market["Market Cap"].replace(0, pd.NA) * 100).fillna(0)
    c24 = market["Change 24h"].fillna(0).clip(-20, 30)
    c7 = market["Change 7d"].fillna(0).clip(-30, 50)
    vr = market["Volume/Market Cap %"].clip(0, 30)
    market["Momentum Score"] = (((c24+20)/50)*40 + ((c7+30)/80)*30 + (vr/30)*30).clip(0,100).round(1)

    pair_rows = get_binance_usdt_pairs()
    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        st.error("Gagal mengambil daftar pasangan Binance. Coba Refresh Market atau periksa koneksi.")
        st.stop()

    # Map symbol to available Binance pairs; handle duplicate ticker symbols conservatively.
    pair_map = pair_df.groupby("base")["symbol"].apply(list).to_dict()
    market["Binance USDT Pair"] = market["Symbol"].map(
        lambda s: pair_map.get(s, [None])[0] if pair_map.get(s) else None
    )
    market["Binance USDT"] = market["Binance USDT Pair"].notna()

    filtered = market[
        (market["Volume 24h"].fillna(0) >= min_volume)
        & (market["Market Cap"].fillna(0) >= min_cap)
        & (market["Market Cap"].fillna(float("inf")) <= max_cap)
    ].copy()
    if search:
        filtered = filtered[
            filtered["Coin"].astype(str).str.contains(search, case=False, na=False)
            | filtered["Symbol"].str.contains(search, case=False, na=False)
        ]

    st.subheader("Status Pasangan USDT")
    st.caption("Status berasal dari daftar spot aktif Binance. Ticker yang sama bisa merujuk ke token berbeda, jadi periksa nama dan alamat kontrak sebelum transaksi.")
    pair_col1, pair_col2 = st.columns(2)
    pair_col1.metric("Coin dengan pasangan Binance USDT", int(market["Binance USDT"].sum()))
    pair_col2.metric("Coin tanpa pasangan Binance USDT", int((~market["Binance USDT"]).sum()))

    st.subheader("Market Watchlist")
    watch_cols = ["Coin","Symbol","Binance USDT Pair","Price USD","Change 24h","Change 7d","Volume 24h","Market Cap","Momentum Score"]
    st.dataframe(
        filtered[watch_cols].sort_values("Volume 24h", ascending=False),
        width="stretch", hide_index=True,
        column_config={
            "Price USD": st.column_config.NumberColumn(format="$%.8f"),
            "Change 24h": st.column_config.NumberColumn(format="%.2f%%"),
            "Change 7d": st.column_config.NumberColumn(format="%.2f%%"),
            "Volume 24h": st.column_config.NumberColumn(format="$%.0f"),
            "Market Cap": st.column_config.NumberColumn(format="$%.0f"),
            "Momentum Score": st.column_config.ProgressColumn(min_value=0,max_value=100)
        }
    )

    st.divider()
    st.subheader("Pemindai Sinyal Harian")
    st.write("Memindai kandidat yang memenuhi filter dan memiliki pasangan spot USDT Binance. Pemindaian menggunakan candle 1 jam yang sudah ditutup.")
    scan_limit = st.selectbox("Jumlah coin yang dipindai", [10, 20, 30], index=1)
    if st.button("Scan Rekomendasi 24 Jam", type="primary"):
        scan_universe = filtered[filtered["Binance USDT"]].sort_values("Volume 24h", ascending=False).head(scan_limit)
        if scan_universe.empty:
            st.warning("Tidak ada kandidat yang lolos filter dan memiliki pasangan USDT.")
        else:
            results = []
            progress = st.progress(0)
            status = st.empty()
            for i, (_, item) in enumerate(scan_universe.iterrows(), start=1):
                pair = item["Binance USDT Pair"]
                status.write(f"Memindai {pair} ({i}/{len(scan_universe)})")
                try:
                    result = evaluate_signal(pair, "1h")
                    if result:
                        result["Coin"] = item["Coin"]
                        result["Change 24h %"] = item["Change 24h"]
                        result["Market Cap"] = item["Market Cap"]
                        result["Volume 24h USD"] = item["Volume 24h"]
                        results.append(result)
                except requests.HTTPError:
                    pass
                except Exception:
                    pass
                progress.progress(i / len(scan_universe))
                time.sleep(0.12)
            status.empty()
            progress.empty()
            if results:
                signals = pd.DataFrame(results)
                st.session_state["daily_signals"] = signals
                st.session_state["scan_time"] = pd.Timestamp.now()
            else:
                st.warning("Data candle tidak tersedia untuk kandidat yang dipindai.")

    if "daily_signals" in st.session_state:
        signals = st.session_state["daily_signals"].copy()
        st.caption(f"Snapshot pemindaian: {st.session_state.get('scan_time')}")
        order = {"STRONG BUY":0, "BUY":1, "WAIT":2, "SIDEWAYS":3, "SELL":4, "STRONG SELL":5}
        signals["_order"] = signals["Signal"].map(order).fillna(6)
        signals = signals.sort_values(["_order","Score"], ascending=[True,False]).drop(columns="_order")
        buys = signals[signals["Signal"].isin(["BUY","STRONG BUY"])]
        if not buys.empty:
            st.success(f"Ada {len(buys)} kandidat berstatus BUY atau STRONG BUY pada snapshot ini.")
        else:
            st.info("Belum ada kandidat BUY. Jangan memaksakan entry. Tunggu sinyal berikutnya.")
        st.subheader("Kartu Rekomendasi")
        st.caption("Format kartu lebih mudah dibaca di layar HP.")
        for _, item in signals.iterrows():
            with st.container(border=True):
                head1, head2 = st.columns([3, 1])
                head1.markdown(f"**{item['Coin']} · {item['Pair']}**")
                head2.metric("Sinyal", str(item["Signal"]))
                a, b, c = st.columns(3)
                a.metric("Skor", int(item["Score"]))
                b.metric("RSI", f"{item['RSI 14']:.1f}")
                c.metric("Volume", f"{item['Volume Spike']:.2f}x")
                st.caption(
                    f"Harga ${item['Price']:.8f} · Perubahan 24 jam "
                    f"{item['Change 24h %']:.2f}% · Breakout {item['Breakout']} · "
                    f"Tren {item['Trend']}"
                )

        with st.expander("Tabel detail untuk layar besar"):
            st.dataframe(
                signals[["Coin","Pair","Signal","Score","Price","Change 24h %","RSI 14","Volume Spike","Breakout","Trend","Market Cap","Volume 24h USD"]],
                width="stretch", hide_index=True,
                column_config={
                    "Score":st.column_config.ProgressColumn(min_value=0,max_value=100),
                    "Price":st.column_config.NumberColumn(format="$%.8f"),
                    "Change 24h %":st.column_config.NumberColumn(format="%.2f%%"),
                    "Market Cap":st.column_config.NumberColumn(format="$%.0f"),
                    "Volume 24h USD":st.column_config.NumberColumn(format="$%.0f")
                }
            )
        st.download_button("Download hasil scan CSV", signals.to_csv(index=False).encode("utf-8"), "crypto_signals.csv", "text/csv")

        st.subheader("Rencana Entry Kandidat")
        st.caption("Rencana dihitung dari candle 1 jam terakhir yang sudah ditutup. Gunakan sebagai acuan, bukan perintah transaksi.")
        candidates = signals[signals["Signal"].isin(["STRONG BUY", "BUY", "WAIT", "SIDEWAYS", "SELL", "STRONG SELL"])].copy()
        if not candidates.empty:
            candidates["Label"] = candidates.apply(
                lambda r: f'{r["Coin"]} ({r["Pair"]}) | {r["Signal"]} | Skor {r["Score"]}',
                axis=1
            )
            chosen_label = st.selectbox("Pilih kandidat untuk dibuatkan rencana", candidates["Label"].tolist(), key="plan_candidate")
            chosen = candidates[candidates["Label"] == chosen_label].iloc[0]
            if st.button("Buat Rencana Entry", key="make_entry_plan"):
                try:
                    plan_candles = get_klines(chosen["Pair"], "1h", 100)
                    if plan_candles.empty or pd.isna(plan_candles.iloc[-2]["ATR14"]):
                        st.warning("Data ATR belum tersedia untuk kandidat ini.")
                    else:
                        candle = plan_candles.iloc[-2]
                        entry = float(candle["Close"])
                        atr = float(candle["ATR14"])
                        signal_name = chosen["Signal"]
                        if signal_name in ["BUY", "STRONG BUY"]:
                            side = "LONG"
                            stop = entry - 1.5 * atr
                            risk = entry - stop
                            tp1 = entry + 1.5 * risk
                            tp2 = entry + 2.5 * risk
                            zone_low, zone_high = entry - 0.25 * atr, entry + 0.25 * atr
                        elif signal_name in ["SELL", "STRONG SELL"]:
                            side = "SHORT"
                            stop = entry + 1.5 * atr
                            risk = stop - entry
                            tp1 = entry - 1.5 * risk
                            tp2 = entry - 2.5 * risk
                            zone_low, zone_high = entry - 0.25 * atr, entry + 0.25 * atr
                        else:
                            side = "WAIT"
                            stop = tp1 = tp2 = None
                            zone_low, zone_high = entry - 0.25 * atr, entry + 0.25 * atr
                        st.session_state["entry_plan"] = {
                            "pair": chosen["Pair"], "signal": signal_name, "side": side,
                            "entry": entry, "zone_low": zone_low, "zone_high": zone_high,
                            "stop": stop, "tp1": tp1, "tp2": tp2, "atr": atr,
                            "score": chosen["Score"]
                        }
                except Exception as e:
                    st.error(f"Gagal membuat rencana entry: {e}")

        plan = st.session_state.get("entry_plan")
        if plan:
            st.markdown(f'**{plan["pair"]} | {plan["signal"]} | Skor {plan["score"]}**')
            if plan["side"] == "WAIT":
                st.info(f'Belum ada setup entry yang layak. Harga acuan: ${plan["entry"]:.8f}. Sinyal saat ini {plan["signal"]}. Tunggu konfirmasi.')
            else:
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Arah", plan["side"])
                p2.metric("Zona Entry", f'${plan["zone_low"]:.8f} - ${plan["zone_high"]:.8f}')
                p3.metric("Stop Loss", f'${plan["stop"]:.8f}')
                p4.metric("Take Profit 1", f'${plan["tp1"]:.8f}')
                st.write(f'Take Profit 2: ${plan["tp2"]:.8f}. ATR 14: ${plan["atr"]:.8f}. Rasio target terhadap risiko: TP1 1:1.5, TP2 1:2.5.')
                st.caption("Rencana ini memakai ATR 14 pada timeframe 1 jam. Jika harga sudah bergerak jauh dari zona entry, jangan mengejar harga. Periksa spread, likuiditas, dan konfirmasi timeframe 4 jam.")

    st.divider()
    st.subheader("Analisis Coin Manual")
    available = filtered[filtered["Binance USDT"]]
    if available.empty:
        st.info("Tidak ada coin yang memenuhi filter dan tersedia di Binance USDT.")
    else:
        choices = available[["Coin","Symbol","Binance USDT Pair"]].drop_duplicates("Binance USDT Pair")
        labels = {row["Binance USDT Pair"]: f'{row["Coin"]} ({row["Binance USDT Pair"]})' for _,row in choices.iterrows()}
        selected_pair = st.selectbox("Pilih coin yang tersedia di Binance USDT", choices["Binance USDT Pair"].tolist(), format_func=lambda x: labels.get(x,x))
        interval_label = st.selectbox("Timeframe chart", ["15 menit","1 jam","4 jam"], index=1)
        interval_map = {"15 menit":"15m","1 jam":"1h","4 jam":"4h"}
        if st.button("Analisis Coin"):
            try:
                candles = get_klines(selected_pair, interval_map[interval_label], 200)
                if candles.empty:
                    st.warning("Data candle tidak tersedia.")
                else:
                    row = candles.iloc[-2]
                    sig = evaluate_signal(selected_pair, interval_map[interval_label])
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("Sinyal", sig["Signal"])
                    m2.metric("RSI 14", f'{sig["RSI 14"]:.1f}')
                    m3.metric("Volume Spike", f'{sig["Volume Spike"]:.2f}x')
                    m4.metric("Momentum Score", sig["Score"])
                    st.write(f'Trend: {sig["Trend"]}. Breakout resistance: {sig["Breakout"]}. Harga candle terakhir ditutup: ${sig["Price"]:.8f}')
                    fig = go.Figure(data=[go.Candlestick(x=candles["Time"],open=candles["Open"],high=candles["High"],low=candles["Low"],close=candles["Close"],name="Candlestick")])
                    fig.add_trace(go.Scatter(x=candles["Time"],y=candles["EMA20"],name="EMA 20"))
                    fig.add_trace(go.Scatter(x=candles["Time"],y=candles["EMA50"],name="EMA 50"))
                    fig.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
                    st.plotly_chart(fig,width="stretch")
            except Exception as e:
                st.error(f"Gagal mengambil candle untuk {selected_pair}: {e}")

    st.caption("Sumber data: CoinGecko untuk market snapshot dan Binance Spot API untuk pasangan serta candle. Sinyal teknikal adalah penyaring, bukan prediksi pasti. Coin kecil memiliki risiko slippage, manipulasi, dan rug pull.")
except Exception as e:
    st.error(f"Gagal mengambil data market: {e}")
    st.info("Periksa koneksi internet, batas API, atau tekan Refresh Market beberapa saat lagi.")
