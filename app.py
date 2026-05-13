import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Correlation & Buffer Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Symbol map ─────────────────────────────────────────────────────────────────
YAHOO_MAP = {
    "CHFJPY":"CHFJPY=X","EURGBP":"EURGBP=X","USDCAD":"USDCAD=X",
    "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X",
    "AUDUSD":"AUDUSD=X","NZDUSD":"NZDUSD=X","USDCHF":"USDCHF=X",
    "EURJPY":"EURJPY=X","GBPJPY":"GBPJPY=X","CADJPY":"CADJPY=X",
    "XAUUSD":"GC=F","XAGUSD":"SI=F","USOIL":"CL=F","NGAS":"NG=F",
    "BTCUSD":"BTC-USD","ETHUSD":"ETH-USD","XRPUSD":"XRP-USD",
    "BNBUSD":"BNB-USD","SOLUSD":"SOL-USD","DOTUSD":"DOT-USD",
}
def to_yahoo(s): return YAHOO_MAP.get(s.strip().upper(), s.strip())

GRP_COLORS = {
    "FX":      "#3B82F6",
    "STOCK":   "#22C55E",
    "COMMO":   "#F59E0B",
    "CRYPTO":  "#A855F7",
}
DEFAULT_SYMS = {
    "FX":     "CHFJPY, EURGBP, USDCAD",
    "STOCK":  "NVDA, GOOG, TSLA",
    "COMMO":  "XAUUSD, XAGUSD, USOIL",
    "CRYPTO": "BTCUSD, ETHUSD, XRPUSD",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_returns(symbols: tuple, period: str) -> pd.DataFrame:
    frames = {}
    fetch_errors = []
    for sym in symbols:
        ysym = to_yahoo(sym)
        try:
            raw = yf.download(
                ysym, period=period,
                progress=False, auto_adjust=True,
                threads=False, actions=False,
            )
            if raw.empty:
                fetch_errors.append(f"{sym}: no data")
                continue
            # Handle both flat and MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw[("Close", ysym)]
            elif "Close" in raw.columns:
                close = raw["Close"]
            else:
                close = raw.iloc[:, 3]
            close = close.dropna()
            if len(close) > 20:
                frames[sym] = close
            else:
                fetch_errors.append(f"{sym}: only {len(close)} rows")
        except Exception as e:
            fetch_errors.append(f"{sym}: {str(e)[:60]}")

    if fetch_errors:
        st.warning("⚠️ " + "  |  ".join(fetch_errors))

    if len(frames) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(frames).dropna(how="all").ffill().dropna()
    if len(df) < 20:
        return pd.DataFrame()
    return df.pct_change().dropna()


def portfolio_stats(returns: pd.DataFrame, weights: np.ndarray, capital: float, z: float):
    corr = returns.corr().values
    vols = returns.std().values  # daily σ
    cov  = corr * np.outer(vols, vols)
    port_var   = weights @ cov @ weights
    port_sigma = np.sqrt(port_var)
    var_dollar = z * port_sigma * capital
    return port_sigma, var_dollar, vols


def naive_var(vols: np.ndarray, weights: np.ndarray, capital: float, z: float):
    """Assume ρ=1 (all crash together)"""
    sigma = np.dot(weights, vols)
    return sigma, z * sigma * capital


def indep_var(vols: np.ndarray, weights: np.ndarray, capital: float, z: float):
    """Assume ρ=0 (fully independent)"""
    sigma = np.sqrt(np.dot(weights**2, vols**2))
    return sigma, z * sigma * capital


def risk_contribution(returns: pd.DataFrame, weights: np.ndarray, groups: dict) -> dict:
    corr  = returns.corr().values
    vols  = returns.std().values
    cov   = corr * np.outer(vols, vols)
    pvar  = weights @ cov @ weights
    mcv   = cov @ weights
    rc    = weights * mcv / pvar * 100  # % contribution
    syms  = list(returns.columns)
    result = {}
    for grp, gsyms in groups.items():
        idx = [i for i, s in enumerate(syms) if s in gsyms]
        result[grp] = round(float(np.sum(rc[idx])), 1)
    return result

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Grid Port Analyzer")
    st.markdown("---")

    st.markdown("### Symbols per group")
    grp_inputs = {}
    for grp, default in DEFAULT_SYMS.items():
        color = GRP_COLORS[grp]
        st.markdown(
            f'<div style="color:{color};font-weight:600;font-size:13px;'
            f'margin-bottom:2px;">{grp}</div>',
            unsafe_allow_html=True,
        )
        raw = st.text_area(
            f"{grp}_input", value=default,
            height=68, label_visibility="collapsed", key=f"inp_{grp}"
        )
        grp_inputs[grp] = [s.strip().upper() for s in raw.split(",") if s.strip()]

    st.markdown("---")
    st.markdown("### Settings")

    period_map = {"1 ปี":"1y", "3 ปี":"3y", "5 ปี":"5y"}
    period_label = st.radio("ช่วงเวลาข้อมูล", list(period_map.keys()), index=1)
    period = period_map[period_label]

    capital = st.number_input("ทุนในพอร์ต ($)", min_value=100, max_value=10_000_000,
                               value=10_000, step=1000)

    z_map = {"90% (z=1.645)":1.645, "95% (z=1.960)":1.960,
             "99% (z=2.326)":2.326, "99.5% (z=2.576)":2.576}
    z_label = st.selectbox("VaR confidence level", list(z_map.keys()), index=2)
    z_score = z_map[z_label]

    st.markdown("---")
    run = st.button("▶  Calculate", use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# Correlation & Buffer Calculator")
st.markdown(
    "ดึงข้อมูลจาก **Yahoo Finance** อัตโนมัติ · "
    "คำนวณ Correlation Matrix → Portfolio VaR → Leverage Ratio"
)

if not run:
    st.info("กำหนด symbols ในแถบซ้าย แล้วกด **Calculate**")
    st.stop()

all_syms = []
for grp, syms in grp_inputs.items():
    all_syms.extend(syms)
all_syms = list(dict.fromkeys(all_syms))  # dedupe, preserve order

with st.spinner("กำลังดึงข้อมูลจาก Yahoo Finance…"):
    rets = fetch_returns(tuple(all_syms), period)

if rets.empty or len(rets.columns) < 2:
    st.error("ดึงข้อมูลไม่สำเร็จ ลองตรวจสอบชื่อ symbol หรือลองใหม่อีกครั้ง")
    st.stop()

valid_syms = list(rets.columns)
n = len(valid_syms)
weights = np.ones(n) / n

# Build group dict (only valid symbols)
groups_valid = {g: [s for s in syms if s in valid_syms] for g, syms in grp_inputs.items()}
sym_to_grp = {s: g for g, syms in groups_valid.items() for s in syms}

# ── Compute stats ──────────────────────────────────────────────────────────────
port_sigma, var_real, vols = portfolio_stats(rets, weights, capital, z_score)
sigma_naive, var_naive     = naive_var(vols, weights, capital, z_score)
sigma_indep, var_indep     = indep_var(vols, weights, capital, z_score)
rc_dict = risk_contribution(rets, weights, groups_valid)

lev_naive = capital / var_naive
lev_real  = capital / var_real
lev_indep = capital / var_indep

n_days = len(rets)

# ── Summary metrics ────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Symbols loaded", f"{n} / {len(all_syms)}")
c2.metric("Trading days", f"{n_days:,}")
c3.metric("Portfolio σ/day", f"{port_sigma*100:.2f}%")
c4.metric("Max leverage (corr)", f"{lev_real:.1f}x")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗂  Correlation Matrix",
    "💰  Buffer & Leverage",
    "📊  Risk Breakdown",
    "📋  Raw Data",
])

# ══════════════════════════════════════════════════════
# TAB 1 — Correlation Matrix
# ══════════════════════════════════════════════════════
with tab1:
    corr = rets.corr().round(2)

    # Color per group for axis labels
    label_colors = [GRP_COLORS.get(sym_to_grp.get(s, ""), "#888") for s in valid_syms]

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=valid_syms,
        y=valid_syms,
        colorscale=[
            [0.0, "#8B1A1A"], [0.35, "#3D1010"],
            [0.5, "#1A2540"],
            [0.65, "#3D2D08"], [1.0, "#D4A017"],
        ],
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 11},
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", thickness=12, len=0.8),
    ))

    for i, (sym, col) in enumerate(zip(valid_syms, label_colors)):
        fig_corr.add_annotation(
            x=sym, y=-0.6, text=sym, showarrow=False,
            font=dict(color=col, size=11), xref="x", yref="y",
            textangle=-45,
        )

    fig_corr.update_layout(
        title=f"Correlation Matrix — Daily Returns ({period_label})",
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A1628",
        font=dict(color="#C8D8E8"),
        xaxis=dict(tickfont=dict(size=11), side="bottom"),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Download correlation table
    csv_corr = corr.to_csv().encode("utf-8")
    st.download_button("⬇ ดาวน์โหลด Correlation Table (CSV)",
                        csv_corr, "correlation_matrix.csv", "text/csv")

# ══════════════════════════════════════════════════════
# TAB 2 — Buffer & Leverage
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("### เปรียบเทียบ 3 วิธีคำนวณ Buffer")
    st.caption(
        f"ทุน ${capital:,.0f} · {n} symbols · {n_days} trading days · "
        f"VaR {z_label}"
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### กรณีที่ 1 — Worst case")
        st.markdown("สมมติทุก position ล้มพร้อมกัน (ρ = 1)")
        st.metric("Portfolio σ/day", f"{sigma_naive*100:.2f}%")
        st.metric("Buffer ที่ต้องเผื่อ", f"${var_naive:,.0f}")
        st.metric("% ของทุน", f"{var_naive/capital*100:.1f}%")
        st.metric("Max Leverage", f"{lev_naive:.1f}x")

    with col_b:
        st.markdown("#### กรณีที่ 2 — Correlation Matrix ✅")
        st.markdown("คำนวณจาก correlation จริง")
        delta = var_naive - var_real
        st.metric("Portfolio σ/day", f"{port_sigma*100:.2f}%")
        st.metric("Buffer ที่ต้องเผื่อ", f"${var_real:,.0f}",
                  delta=f"-${delta:,.0f} vs worst case", delta_color="inverse")
        st.metric("% ของทุน", f"{var_real/capital*100:.1f}%")
        st.metric("Max Leverage", f"{lev_real:.1f}x",
                  delta=f"+{lev_real-lev_naive:.1f}x vs worst case")

    with col_c:
        st.markdown("#### กรณีที่ 3 — Perfect diversification")
        st.markdown("สมมติทุกตัวอิสระจากกัน (ρ = 0)")
        st.metric("Portfolio σ/day", f"{sigma_indep*100:.2f}%")
        st.metric("Buffer ที่ต้องเผื่อ", f"${var_indep:,.0f}")
        st.metric("% ของทุน", f"{var_indep/capital*100:.1f}%")
        st.metric("Max Leverage", f"{lev_indep:.1f}x")

    st.markdown("---")
    st.markdown("### Buffer comparison (bar chart)")

    scenarios = ["Worst case\n(ρ=1)", "Correlation Matrix\n(actual)", "All independent\n(ρ=0)"]
    buffers   = [var_naive, var_real, var_indep]
    levs      = [lev_naive, lev_real, lev_indep]
    bar_colors = ["#E24B4A", "#378ADD", "#639922"]

    fig_buf = go.Figure()
    fig_buf.add_trace(go.Bar(
        x=scenarios, y=buffers,
        marker_color=bar_colors,
        text=[f"${b:,.0f}" for b in buffers],
        textposition="outside",
        name="Buffer ($)",
    ))
    fig_buf.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A1628", font=dict(color="#C8D8E8"),
        yaxis_title="Buffer needed ($)",
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig_buf.update_yaxes(gridcolor="#1E3A5F")
    fig_buf.update_xaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_buf, use_container_width=True)

    st.markdown("---")
    st.markdown("### Leverage ที่ทำได้ต่อ scenario")

    fig_lev = go.Figure()
    fig_lev.add_trace(go.Bar(
        x=scenarios, y=levs,
        marker_color=bar_colors,
        text=[f"{l:.1f}x" for l in levs],
        textposition="outside",
    ))
    fig_lev.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A1628", font=dict(color="#C8D8E8"),
        yaxis_title="Max Leverage (x)",
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig_lev.update_yaxes(gridcolor="#1E3A5F")
    fig_lev.update_xaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_lev, use_container_width=True)

    # Key insight
    freed = var_naive - var_real
    st.info(
        f"💡 ด้วย Correlation Matrix → ต้องเผื่อ buffer แค่ **${var_real:,.0f}** "
        f"(ไม่ใช่ ${var_naive:,.0f} แบบ worst-case) "
        f"เซฟไปได้ **${freed:,.0f}** → leverage ได้ **{lev_real:.1f}x** "
        f"แทนที่จะได้แค่ {lev_naive:.1f}x"
    )

# ══════════════════════════════════════════════════════
# TAB 3 — Risk Breakdown
# ══════════════════════════════════════════════════════
with tab3:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Risk contribution by group (%)")
        valid_rc = {g: v for g, v in rc_dict.items() if v > 0}
        fig_pie = go.Figure(data=go.Pie(
            labels=list(valid_rc.keys()),
            values=list(valid_rc.values()),
            marker=dict(colors=[GRP_COLORS[g] for g in valid_rc]),
            textinfo="label+percent",
            hole=0.4,
        ))
        fig_pie.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#C8D8E8"),
            legend=dict(font=dict(color="#C8D8E8")),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("### Daily volatility per symbol (%)")
        vol_df = pd.DataFrame({
            "Symbol": valid_syms,
            "Group":  [sym_to_grp.get(s, "—") for s in valid_syms],
            "Daily σ (%)": (vols * 100).round(2),
        }).sort_values("Daily σ (%)", ascending=False)

        fig_vol = go.Figure(go.Bar(
            x=vol_df["Symbol"],
            y=vol_df["Daily σ (%)"],
            marker_color=[GRP_COLORS.get(g, "#888") for g in vol_df["Group"]],
            text=vol_df["Daily σ (%)"].apply(lambda x: f"{x:.2f}%"),
            textposition="outside",
        ))
        fig_vol.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0A1628", font=dict(color="#C8D8E8"),
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_tickangle=-45,
        )
        fig_vol.update_yaxes(gridcolor="#1E3A5F", title="σ/day (%)")
        fig_vol.update_xaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("### Symbol-level summary")
    # Build table
    corr_mat = rets.corr()
    avg_corr = corr_mat.mean()

    summary_df = pd.DataFrame({
        "Symbol":       valid_syms,
        "Group":        [sym_to_grp.get(s, "—") for s in valid_syms],
        "Daily σ (%)":  (vols * 100).round(3),
        "Ann. σ (%)":   (vols * np.sqrt(252) * 100).round(1),
        "Avg Corr":     avg_corr.round(3),
        "Weight":       (weights * 100).round(1),
    })
    summary_df["Individual VaR ($)"] = (z_score * vols * weights * capital).round(0).astype(int)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
# TAB 4 — Raw Data
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown("### Daily Returns (%)")
    display_rets = (rets * 100).round(4)
    st.dataframe(display_rets, use_container_width=True)
    csv_rets = display_rets.to_csv().encode("utf-8")
    st.download_button("⬇ ดาวน์โหลด Returns Data (CSV)",
                        csv_rets, "daily_returns.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: Yahoo Finance · Returns-based Pearson correlation · "
    "VaR assumes normal distribution · ใช้สำหรับประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน"
)
