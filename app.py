@st.cache_data(ttl=3600, show_spinner=False)
def fetch_returns(symbols: tuple, period: str) -> pd.DataFrame:
    yahoo_syms = [to_yahoo(s) for s in symbols]
    sym_map    = dict(zip(yahoo_syms, symbols))
    try:
        raw = yf.download(
            yahoo_syms, period=period,
            progress=False, auto_adjust=True,
            threads=False,
        )
        df = raw["Close"] if "Close" in raw else raw
        if isinstance(df, pd.Series):
            df = df.to_frame(name=symbols[0])
        df.columns = [sym_map.get(str(c), str(c)) for c in df.columns]
        df = df.dropna(how="all").ffill().dropna()
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        return df.pct_change().dropna()
    except Exception as e:
        st.warning(f"Fetch error: {e}")
        return pd.DataFrame()
