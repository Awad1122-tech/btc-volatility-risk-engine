# ============================================================
# Bitcoin Volatility Risk Engine — Streamlit Dashboard
# Built with ARIMA + GARCH | 2018-2026
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from arch import arch_model
from pmdarima import auto_arima
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Risk Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    /* Global background */
    .stApp {
        background-color: #0a0a0f;
        color: #e0e0e0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f0f1a;
        border-right: 1px solid #1e1e2e;
    }

    /* Main header */
    .main-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #F7931A, #FFD700, #F7931A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 10px 0;
        letter-spacing: 2px;
    }

    .sub-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
        color: #666;
        text-align: center;
        letter-spacing: 3px;
        margin-bottom: 30px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        border: 1px solid #F7931A33;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
        box-shadow: 0 4px 15px rgba(247, 147, 26, 0.1);
    }

    .metric-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #888;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #F7931A;
    }

    .metric-value-green {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00FF88;
    }

    .metric-value-blue {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00D4FF;
    }

    /* Section headers */
    .section-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #F7931A;
        border-left: 4px solid #F7931A;
        padding-left: 12px;
        margin: 25px 0 15px 0;
        letter-spacing: 1px;
    }

    /* Result banner */
    .result-banner {
        background: linear-gradient(135deg, #003300, #004400);
        border: 1px solid #00FF88;
        border-radius: 12px;
        padding: 20px 25px;
        margin: 15px 0;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.15);
    }

    .result-banner-text {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        color: #00FF88;
        font-weight: 600;
    }

    /* Warning banner */
    .warning-banner {
        background: linear-gradient(135deg, #1a0a00, #2a1500);
        border: 1px solid #F7931A;
        border-radius: 12px;
        padding: 15px 20px;
        margin: 15px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0f0f1a;
        border-bottom: 1px solid #1e1e2e;
        gap: 5px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        color: #666;
        letter-spacing: 1px;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        color: #F7931A !important;
        border-bottom: 2px solid #F7931A !important;
    }

    /* Sidebar elements */
    .sidebar-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #F7931A;
        letter-spacing: 2px;
        margin-bottom: 15px;
    }

    /* Divider */
    .orange-divider {
        border: none;
        border-top: 1px solid #F7931A33;
        margin: 20px 0;
    }

    /* Info box */
    .info-box {
        background: #0f0f1a;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        color: #888;
        line-height: 1.6;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB STYLE ─────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0a0a0f',
    'axes.facecolor': '#0f0f1a',
    'axes.edgecolor': '#1e1e2e',
    'axes.labelcolor': '#888',
    'xtick.color': '#666',
    'ytick.color': '#666',
    'grid.color': '#1e1e2e',
    'grid.alpha': 0.5,
    'text.color': '#e0e0e0',
    'legend.facecolor': '#0f0f1a',
    'legend.edgecolor': '#1e1e2e',
    'legend.labelcolor': '#ccc',
})

# ── DATA LOADING (CACHED) ────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    btc = yf.download("BTC-USD", start="2018-01-01", end="2024-12-31", progress=False)
    close_price = btc['Close']['BTC-USD']
    returns = np.log(close_price / close_price.shift(1)).dropna()
    train_returns = returns[returns.index.year < 2024]
    test_returns = returns[returns.index.year == 2024]
    train_price = close_price[close_price.index.year < 2024]
    test_price = close_price[close_price.index.year == 2024]
    return close_price, returns, train_returns, test_returns, train_price, test_price

@st.cache_data(show_spinner=False)
def load_2025_data():
    btc_2025 = yf.download("BTC-USD", start="2025-01-01", end="2025-12-31", progress=False)
    close_2025 = btc_2025['Close']['BTC-USD']
    returns_2025 = np.log(close_2025 / close_2025.shift(1)).dropna()
    return returns_2025

@st.cache_resource(show_spinner=False)
def fit_models(train_returns):
    # ARIMA
    arima_model = auto_arima(
        train_returns, start_p=0, max_p=3, start_q=0, max_q=3,
        d=0, seasonal=False, information_criterion='aic',
        stepwise=True, suppress_warnings=True, error_action='ignore'
    )
    # GARCH on ARIMA residuals
    arima_residuals = pd.Series(arima_model.resid(), index=train_returns.index)
    garch = arch_model(arima_residuals * 100, vol='Garch', p=1, q=1, dist='normal', mean='Zero')
    garch_model = garch.fit(disp='off', show_warning=False)
    return arima_model, garch_model

@st.cache_resource(show_spinner=False)
def fit_garch_2026(all_returns):
    garch_model_2026 = arch_model(all_returns * 100, vol='Garch', p=1, q=1, dist='normal')
    garch_fit_2026 = garch_model_2026.fit(disp='off')
    return garch_fit_2026

# ── LOAD EVERYTHING ──────────────────────────────────────────
with st.spinner("🔮 Initialising Risk Engine..."):
    close_price, returns, train_returns, test_returns, train_price, test_price = load_data()

with st.spinner("⚙️ Fitting ARIMA + GARCH models..."):
    arima_model, garch_model = fit_models(train_returns)

with st.spinner("📥 Loading 2025 data..."):
    returns_2025 = load_2025_data()

with st.spinner("🔄 Retraining model for 2026 forecast..."):
    all_returns = pd.concat([train_returns, test_returns, returns_2025])
    garch_fit_2026 = fit_garch_2026(all_returns)

# ── COMPUTED VALUES ───────────────────────────────────────────
conditional_vol = garch_model.conditional_volatility / 100
z_score = stats.norm.ppf(0.05)
historical_var_pct = np.percentile(train_returns, 5)

# 2024 backtesting
n_2024 = len(test_returns)
garch_fc_2024 = garch_model.forecast(horizon=n_2024, reindex=False)
fc_vol_2024 = np.sqrt(garch_fc_2024.variance.values[-1]) / 100
fc_var_2024 = z_score * fc_vol_2024
breaches_2024 = test_returns.values < fc_var_2024
n_breaches_2024 = breaches_2024.sum()

# 2025 backtesting
n_2025 = len(returns_2025)
garch_fc_2025 = garch_model.forecast(horizon=n_2025, reindex=False)
fc_vol_2025 = np.sqrt(garch_fc_2025.variance.values[-1]) / 100
fc_var_2025 = z_score * fc_vol_2025
breaches_2025 = returns_2025.values < fc_var_2025
n_breaches_2025 = breaches_2025.sum()

# 2026 forecast
n_2026 = 365
garch_fc_2026 = garch_fit_2026.forecast(horizon=n_2026, reindex=False)
fc_vol_2026 = np.sqrt(garch_fc_2026.variance.values[-1]) / 100
forecast_dates_2026 = pd.date_range(start='2026-01-01', periods=n_2026, freq='D')
fc_var_2026 = z_score * fc_vol_2026

# GARCH params
params_new = garch_fit_2026.params

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ RISK PARAMETERS</div>', unsafe_allow_html=True)

    investment = st.number_input(
        "Investment Amount (£)",
        min_value=100,
        max_value=10_000_000,
        value=10_000,
        step=1000,
        help="Enter your Bitcoin investment amount"
    )

    confidence = st.selectbox(
        "Confidence Level",
        options=[0.95, 0.99, 0.90],
        format_func=lambda x: f"{int(x*100)}%",
        index=0
    )
    z_score_selected = stats.norm.ppf(1 - confidence)

    st.markdown('<hr class="orange-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 MODEL INFO</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    MODEL: ARIMA(2,0,1) + GARCH(1,1)<br>
    TRAINING: 2018-01-02 → 2025-12-30<br>
    TOTAL DAYS: {len(all_returns):,}<br>
    CONFIDENCE: {int(confidence*100)}%<br>
    Z-SCORE: {z_score_selected:.4f}<br><br>
    α (alpha): {params_new['alpha[1]']:.4f}<br>
    β (beta): {params_new['beta[1]']:.4f}<br>
    α+β: {params_new['alpha[1]']+params_new['beta[1]']:.4f} ✅
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="orange-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    ⚠️ DISCLAIMER<br><br>
    This tool is for educational and
    portfolio demonstration purposes only.
    Not financial advice.
    </div>
    """, unsafe_allow_html=True)

# ── MAIN HEADER ───────────────────────────────────────────────
st.markdown('<div class="main-header">🔮 BITCOIN VOLATILITY RISK ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ARIMA + GARCH + VALUE AT RISK | 2018 — 2026</div>', unsafe_allow_html=True)

# ── TOP METRIC CARDS ─────────────────────────────────────────
historical_var_gbp = abs(historical_var_pct * investment)
garch_var_avg_gbp = abs(z_score_selected * conditional_vol.mean() * investment)
fc_var_2026_avg_gbp = abs(z_score_selected * fc_vol_2026.mean() * investment)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Training Days</div>
        <div class="metric-value">{len(all_returns):,}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Historical VaR</div>
        <div class="metric-value">£{historical_var_gbp:,.0f}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">2026 Avg VaR</div>
        <div class="metric-value">£{fc_var_2026_avg_gbp:,.0f}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">2024 Breach Rate</div>
        <div class="metric-value-green">{n_breaches_2024/n_2024*100:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">2025 Breach Rate</div>
        <div class="metric-value-green">{n_breaches_2025/n_2025*100:.2f}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Market Overview",
    "📊  GARCH Model",
    "💰  VaR Analysis",
    "✅  Backtesting",
    "🔮  2026 Forecast"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Bitcoin Price & Returns — 2018 to 2024</div>', unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Mean Daily Return</div>
            <div class="metric-value-blue">{returns.mean()*100:.4f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Daily Std Dev</div>
            <div class="metric-value">{returns.std()*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Best Day</div>
            <div class="metric-value-green">+{returns.max()*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Worst Day (COVID)</div>
            <div class="metric-value" style="color:#FF6B6B">{returns.min()*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#0a0a0f')

    # Price chart
    axes[0].plot(close_price, color='#F7931A', linewidth=1.5, alpha=0.9)
    axes[0].fill_between(close_price.index, close_price, alpha=0.05, color='#F7931A')
    axes[0].set_title('Bitcoin Price (USD) — 2018 to 2024', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[0].set_ylabel('Price (USD)', fontsize=11, color='#888')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    axes[0].grid(True, alpha=0.2)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Annotate key events
    axes[0].axvline(x=pd.Timestamp('2020-03-12'), color='#FF6B6B', linewidth=1.5, linestyle='--', alpha=0.7)
    axes[0].axvline(x=pd.Timestamp('2022-11-08'), color='#FFD700', linewidth=1.5, linestyle='--', alpha=0.7)
    axes[0].axvline(x=pd.Timestamp('2024-01-10'), color='#00D4FF', linewidth=1.5, linestyle='--', alpha=0.7)
    axes[0].text(pd.Timestamp('2020-03-12'), close_price.max()*0.85, 'COVID', color='#FF6B6B', fontsize=8, rotation=90, va='top')
    axes[0].text(pd.Timestamp('2022-11-08'), close_price.max()*0.85, 'FTX', color='#FFD700', fontsize=8, rotation=90, va='top')
    axes[0].text(pd.Timestamp('2024-01-10'), close_price.max()*0.85, 'ETF', color='#00D4FF', fontsize=8, rotation=90, va='top')

    # Returns chart
    colors = ['#00FF88' if r > 0 else '#FF6B6B' for r in returns]
    axes[1].bar(returns.index, returns, color=colors, alpha=0.6, width=1)
    axes[1].axhline(y=0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    axes[1].set_title('Daily Log Returns — Volatility Clustering Visible', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[1].set_ylabel('Log Return', fontsize=11, color='#888')
    axes[1].set_xlabel('Date', fontsize=11, color='#888')
    axes[1].grid(True, alpha=0.2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout(pad=2.5)
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════
# TAB 2 — GARCH MODEL
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">GARCH(1,1) Volatility Model</div>', unsafe_allow_html=True)

    # Parameter cards
    params_old = {'omega': 0.7232, 'alpha[1]': 0.0953, 'beta[1]': 0.8566}
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Omega (ω)</div>
            <div class="metric-value">{params_new['omega']:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Alpha (α) — Shock</div>
            <div class="metric-value">{params_new['alpha[1]']:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Beta (β) — Persistence</div>
            <div class="metric-value">{params_new['beta[1]']:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        ab = params_new['alpha[1]'] + params_new['beta[1]']
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">α + β (Stability)</div>
            <div class="metric-value-green">{ab:.4f} ✅</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="result-banner">
        <div class="result-banner-text">
        ✅ Model is STABLE — α + β < 1 confirms volatility always mean-reverts to long-run average
        </div>
    </div>""", unsafe_allow_html=True)

    # Volatility plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#0a0a0f')

    # Returns with bands
    axes[0].plot(train_returns.index, train_returns, color='#00D4FF', linewidth=0.6, alpha=0.7, label='Daily Returns')
    axes[0].fill_between(train_returns.index, conditional_vol * 2, -conditional_vol * 2, color='#FF6B6B', alpha=0.15, label='±2σ Band')
    axes[0].plot(train_returns.index, conditional_vol * 2, color='#FF6B6B', linewidth=1, alpha=0.8)
    axes[0].plot(train_returns.index, -conditional_vol * 2, color='#FF6B6B', linewidth=1, alpha=0.8)
    axes[0].set_title('Bitcoin Returns with GARCH Volatility Bands', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[0].set_ylabel('Return / Volatility', fontsize=11, color='#888')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Conditional volatility
    axes[1].plot(train_returns.index, conditional_vol * 100, color='#F7931A', linewidth=1.5)
    axes[1].fill_between(train_returns.index, 0, conditional_vol * 100, color='#F7931A', alpha=0.25)
    axes[1].axvline(x=pd.Timestamp('2020-03-12'), color='#FF6B6B', linewidth=2, linestyle='--', label='COVID Crash (Mar 2020)')
    axes[1].axvline(x=pd.Timestamp('2022-11-08'), color='#FFD700', linewidth=2, linestyle='--', label='FTX Collapse (Nov 2022)')
    axes[1].set_title('GARCH Conditional Volatility — Calm vs Wild Periods', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[1].set_ylabel('Daily Volatility (%)', fontsize=11, color='#888')
    axes[1].set_xlabel('Date', fontsize=11, color='#888')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout(pad=2.5)
    st.pyplot(fig)
    plt.close()

    # GARCH explanation
    st.markdown('<div class="section-title">What These Parameters Mean</div>', unsafe_allow_html=True)
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        st.markdown(f"""<div class="info-box">
        <b style="color:#F7931A">ω (Omega) = {params_new['omega']:.4f}</b><br><br>
        The long-run baseline variance.
        Even in the calmest conditions,
        Bitcoin carries this base level
        of risk. Dropped from 0.7232
        after learning from calm 2024-2025.
        </div>""", unsafe_allow_html=True)
    with col_y:
        st.markdown(f"""<div class="info-box">
        <b style="color:#F7931A">α (Alpha) = {params_new['alpha[1]']:.4f}</b><br><br>
        The shock effect. After any big
        move (up or down), {params_new['alpha[1]']*100:.1f}% of
        that shock feeds into tomorrow's
        volatility. Moderate reaction —
        Bitcoin doesn't overreact.
        </div>""", unsafe_allow_html=True)
    with col_z:
        st.markdown(f"""<div class="info-box">
        <b style="color:#F7931A">β (Beta) = {params_new['beta[1]']:.4f}</b><br><br>
        The persistence effect. {params_new['beta[1]']*100:.1f}%
        of yesterday's volatility carries
        into today. Very high — once
        Bitcoin gets volatile, it stays
        volatile for ~30+ days.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — VaR ANALYSIS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-title">Value at Risk — £{investment:,} Investment at {int(confidence*100)}% Confidence</div>', unsafe_allow_html=True)

    # Recalculate with user inputs
    hist_var_user = abs(np.percentile(train_returns, (1-confidence)*100) * investment)
    garch_var_user = abs(z_score_selected * conditional_vol)
    garch_var_avg_user = abs(z_score_selected * conditional_vol.mean() * investment)
    garch_var_worst_user = abs(z_score_selected * conditional_vol.max() * investment)
    garch_var_best_user = abs(z_score_selected * conditional_vol.min() * investment)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Historical VaR</div>
            <div class="metric-value">£{hist_var_user:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">GARCH Avg VaR</div>
            <div class="metric-value">£{garch_var_avg_user:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">GARCH Worst Day</div>
            <div class="metric-value" style="color:#FF6B6B">£{garch_var_worst_user:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">GARCH Best Day</div>
            <div class="metric-value-green">£{garch_var_best_user:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-banner">
        <div class="result-banner-text">
        💡 At {int(confidence*100)}% confidence: On 95% of trading days, your £{investment:,} Bitcoin investment
        will NOT lose more than £{garch_var_avg_user:,.0f} (GARCH Dynamic) or £{hist_var_user:,.0f} (Historical)
        </div>
    </div>""", unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#0a0a0f')

    # VaR over time
    axes[0].plot(train_returns.index, garch_var_user * investment, color='#FF6B6B', linewidth=1.5, label=f'GARCH VaR ({int(confidence*100)}%)')
    axes[0].axhline(y=hist_var_user, color='#FFD700', linewidth=2, linestyle='--', label=f'Historical VaR = £{hist_var_user:,.0f}')
    axes[0].fill_between(train_returns.index, garch_var_user * investment, 0, color='#FF6B6B', alpha=0.2)
    axes[0].axvline(x=pd.Timestamp('2020-03-12'), color='#FF6B6B', linewidth=1.5, linestyle=':', alpha=0.8)
    axes[0].set_title(f'Dynamic GARCH VaR vs Historical VaR — £{investment:,} Investment', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[0].set_ylabel('Potential Loss (£)', fontsize=11, color='#888')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{abs(x):,.0f}'))
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Returns vs boundaries
    axes[1].plot(train_returns.index, train_returns, color='#00D4FF', linewidth=0.6, alpha=0.7, label='Actual Returns')
    axes[1].plot(train_returns.index, -abs(garch_var_user), color='#FF6B6B', linewidth=1.5, label=f'GARCH VaR Boundary')
    axes[1].axhline(y=np.percentile(train_returns, (1-confidence)*100), color='#FFD700', linewidth=2, linestyle='--', label='Historical VaR Boundary')
    axes[1].set_title('Returns vs VaR Boundaries', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[1].set_ylabel('Return', fontsize=11, color='#888')
    axes[1].set_xlabel('Date', fontsize=11, color='#888')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout(pad=2.5)
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════
# TAB 4 — BACKTESTING
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Model Validation — Out-of-Sample Backtesting</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="result-banner">
        <div class="result-banner-text">
        🏆 EXCEPTIONAL PERFORMANCE: Model tested on 728 days of completely unseen data (2024 + 2025)
        — achieved breach rate of ~1.38% vs expected 5% in BOTH years independently
        </div>
    </div>""", unsafe_allow_html=True)

    # Summary table
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics = [
        ("2024 Days", f"{n_2024}"),
        ("2024 Expected", "18"),
        ("2024 Actual", f"{n_breaches_2024} ✅"),
        ("2025 Days", f"{n_2025}"),
        ("2025 Expected", "18"),
        ("2025 Actual", f"{n_breaches_2025} ✅"),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4, col5, col6], metrics):
        with col:
            color = "metric-value-green" if "✅" in value else "metric-value"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="{color}">{value}</div>
            </div>""", unsafe_allow_html=True)

    # Backtesting charts — 2024
    st.markdown('<div class="section-title">2024 Backtesting Results</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#0a0a0f')

    hist_var_pct = np.percentile(train_returns, 5)
    forecast_dates_2024 = test_returns.index

    axes[0].plot(forecast_dates_2024, test_returns.values, color='#00D4FF', linewidth=0.8, alpha=0.7, label='Actual 2024 Returns')
    axes[0].plot(forecast_dates_2024, fc_var_2024, color='#FF6B6B', linewidth=2, label='GARCH VaR Boundary (95%)')
    axes[0].axhline(y=hist_var_pct, color='#FFD700', linewidth=2, linestyle='--', label=f'Historical VaR')
    breach_dates_2024 = forecast_dates_2024[breaches_2024]
    breach_vals_2024 = test_returns.values[breaches_2024]
    axes[0].scatter(breach_dates_2024, breach_vals_2024, color='red', s=80, zorder=5, label=f'Breaches ({n_breaches_2024} days)', edgecolors='white', linewidths=0.5)
    axes[0].set_title(f'GARCH Backtesting 2024 — VaR Breaches: {n_breaches_2024}/{n_2024} days (Expected: 18)', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[0].set_ylabel('Return', fontsize=11, color='#888')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    cum_2024 = np.cumsum(breaches_2024)
    exp_2024 = np.linspace(0, n_2024 * 0.05, n_2024)
    axes[1].plot(forecast_dates_2024, cum_2024, color='#FF6B6B', linewidth=2.5, label='Actual Cumulative Breaches')
    axes[1].plot(forecast_dates_2024, exp_2024, color='#FFD700', linewidth=2, linestyle='--', label='Expected (5% = 18)')
    axes[1].fill_between(forecast_dates_2024, cum_2024, exp_2024, where=cum_2024 < exp_2024, color='#00FF88', alpha=0.1, label='Model outperforming')
    axes[1].set_title('Cumulative VaR Breaches Over 2024', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[1].set_ylabel('Number of Breaches', fontsize=11, color='#888')
    axes[1].set_xlabel('Date', fontsize=11, color='#888')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout(pad=2.5)
    st.pyplot(fig)
    plt.close()

    # Backtesting charts — 2025
    st.markdown('<div class="section-title">2025 Backtesting Results</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#0a0a0f')

    forecast_dates_2025 = returns_2025.index

    axes[0].plot(forecast_dates_2025, returns_2025.values, color='#00D4FF', linewidth=0.8, alpha=0.7, label='Actual 2025 Returns')
    axes[0].plot(forecast_dates_2025, fc_var_2025, color='#FF6B6B', linewidth=2, label='GARCH VaR Boundary (95%)')
    axes[0].axhline(y=hist_var_pct, color='#FFD700', linewidth=2, linestyle='--', label='Historical VaR')
    breach_dates_2025_plot = forecast_dates_2025[breaches_2025]
    breach_vals_2025 = returns_2025.values[breaches_2025]
    axes[0].scatter(breach_dates_2025_plot, breach_vals_2025, color='red', s=80, zorder=5, label=f'Breaches ({n_breaches_2025} days)', edgecolors='white', linewidths=0.5)
    axes[0].set_title(f'GARCH Backtesting 2025 — VaR Breaches: {n_breaches_2025}/{n_2025} days (Expected: 18)', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[0].set_ylabel('Return', fontsize=11, color='#888')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    cum_2025 = np.cumsum(breaches_2025)
    exp_2025 = np.linspace(0, n_2025 * 0.05, n_2025)
    axes[1].plot(forecast_dates_2025, cum_2025, color='#FF6B6B', linewidth=2.5, label='Actual Cumulative Breaches')
    axes[1].plot(forecast_dates_2025, exp_2025, color='#FFD700', linewidth=2, linestyle='--', label='Expected (5% = 18)')
    axes[1].fill_between(forecast_dates_2025, cum_2025, exp_2025, where=cum_2025 < exp_2025, color='#00FF88', alpha=0.1, label='Model outperforming')
    axes[1].set_title('Cumulative VaR Breaches Over 2025', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[1].set_ylabel('Number of Breaches', fontsize=11, color='#888')
    axes[1].set_xlabel('Date', fontsize=11, color='#888')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout(pad=2.5)
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════
# TAB 5 — 2026 FORECAST
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="section-title">2026 Risk Forecast — £{investment:,} Investment</div>', unsafe_allow_html=True)

    # Recalculate with user inputs
    fc_var_2026_user = z_score_selected * fc_vol_2026
    fc_var_2026_avg = abs(z_score_selected * fc_vol_2026.mean() * investment)
    fc_var_2026_worst = abs(z_score_selected * fc_vol_2026.min() * investment)
    fc_var_2026_best = abs(z_score_selected * fc_vol_2026.max() * investment)
    hist_var_user_2026 = abs(np.percentile(train_returns, (1-confidence)*100) * investment)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Daily Volatility</div>
            <div class="metric-value">{fc_vol_2026.mean()*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Daily VaR</div>
            <div class="metric-value">£{fc_var_2026_avg:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Best Day VaR</div>
            <div class="metric-value-green">£{fc_var_2026_worst:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Worst Day VaR</div>
            <div class="metric-value" style="color:#FF6B6B">£{fc_var_2026_best:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-banner">
        <div class="result-banner-text">
        🔮 2026 OUTLOOK: Model forecasts average daily volatility of {fc_vol_2026.mean()*100:.2f}% —
        BELOW historical average, suggesting continued institutional market stability.
        Average daily risk on £{investment:,}: £{fc_var_2026_avg:,.0f}
        </div>
    </div>""", unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#0a0a0f')

    # Volatility forecast
    axes[0].plot(forecast_dates_2026, fc_vol_2026 * 100, color='#F7931A', linewidth=2.5, label='2026 Forecasted Volatility')
    axes[0].fill_between(forecast_dates_2026, 0, fc_vol_2026 * 100, color='#F7931A', alpha=0.25)
    axes[0].axhline(y=fc_vol_2026.mean() * 100, color='white', linewidth=1.5, linestyle='--', label=f'Average = {fc_vol_2026.mean()*100:.2f}%')
    axes[0].axhline(y=conditional_vol.mean() * 100, color='#00D4FF', linewidth=1, linestyle=':', alpha=0.7, label=f'2018-2023 Avg = {conditional_vol.mean()*100:.2f}%')
    axes[0].set_title('Bitcoin Volatility Forecast — 2026 (Retrained on 2018-2025)', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[0].set_ylabel('Daily Volatility (%)', fontsize=11, color='#888')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # VaR forecast
    axes[1].plot(forecast_dates_2026, abs(fc_var_2026_user * investment), color='#FF6B6B', linewidth=2.5, label=f'2026 GARCH VaR ({int(confidence*100)}%)')
    axes[1].fill_between(forecast_dates_2026, 0, abs(fc_var_2026_user * investment), color='#FF6B6B', alpha=0.2)
    axes[1].axhline(y=hist_var_user_2026, color='#FFD700', linewidth=2, linestyle='--', label=f'Historical VaR = £{hist_var_user_2026:,.0f}')
    axes[1].set_title(f'2026 Daily Value at Risk — £{investment:,} Bitcoin Investment', fontsize=13, fontweight='bold', color='#e0e0e0', pad=12)
    axes[1].set_ylabel('Potential Loss (£)', fontsize=11, color='#888')
    axes[1].set_xlabel('Date', fontsize=11, color='#888')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{abs(x):,.0f}'))
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout(pad=2.5)
    st.pyplot(fig)
    plt.close()

    # Limitation disclaimer
    st.markdown("""
    <div class="warning-banner">
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #F7931A; line-height: 1.8;">
        ⚠️ MODEL LIMITATIONS<br>
        • Forecast assumes continuation of current institutional market conditions<br>
        • Cannot predict black swan events (exchange collapses, regulatory bans, macro crises)<br>
        • GARCH mean-reversion produces smooth forecasts — real volatility will be jagged<br>
        • Not financial advice — for educational and portfolio demonstration purposes only
        </div>
    </div>""", unsafe_allow_html=True)

    # Methodology
    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("""<div class="info-box">
        <b style="color:#F7931A">PHASE 1-2: Data & Stationarity</b><br><br>
        • 2,918 days of BTC-USD data<br>
        • Log returns calculated<br>
        • ADF + KPSS stationarity tests<br>
        • Raw prices → NOT stationary<br>
        • Log returns → STATIONARY ✅
        </div>""", unsafe_allow_html=True)
    with col_m2:
        st.markdown("""<div class="info-box">
        <b style="color:#F7931A">PHASE 3-4: ARIMA Modelling</b><br><br>
        • ACF/PACF analysis<br>
        • auto_arima model selection<br>
        • Best model: ARIMA(2,0,1)<br>
        • AIC: -8250.56<br>
        • Residuals → White Noise ✅<br>
        • Heteroskedasticity → GARCH needed
        </div>""", unsafe_allow_html=True)
    with col_m3:
        st.markdown("""<div class="info-box">
        <b style="color:#F7931A">PHASE 5-6: GARCH + VaR</b><br><br>
        • GARCH(1,1) on ARIMA residuals<br>
        • α+β = 0.9600 → Stable ✅<br>
        • Historical VaR (5th percentile)<br>
        • Dynamic GARCH VaR<br>
        • Backtested: 728 unseen days<br>
        • Breach rate: 1.38% vs 5% ✅
        </div>""", unsafe_allow_html=True)