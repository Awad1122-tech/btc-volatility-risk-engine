# Bitcoin Volatility Risk Engine 🪙📊

> **Quantitative risk modelling for Bitcoin using ARIMA + GARCH + Value at Risk**  
> Validated on 728 days of real out-of-sample data - 2024 and 2025

👉 **[Live Interactive Dashboard](https://bitcoin-risk-engine.streamlit.app)** - try it with your own investment amount

---

## Why this project exists

Every serious financial institution has a risk engine. They don't just buy assets and hope - they model the risk, stress test it, and put hard numbers on potential losses.

I built one for Bitcoin.

Using 7 years of real market data (2018–2025), I combined three industry-standard techniques - ARIMA, GARCH, and Value at Risk - into a full end-to-end risk pipeline. Then I validated it on two years of data the model had never seen. The results held up.

---

## The numbers that matter

**For a £10,000 Bitcoin investment in 2026:**

| Risk Metric | Value |
|---|---|
| 💰 Typical daily loss limit (95% confidence) | £582 |
| 📉 Worst case daily loss | £597 |
| 📈 Calm day loss limit | £362 |
| 📊 Average daily volatility forecast | 3.54% |

**Out-of-sample validation - 728 days of real data:**

| Year | Days Tested | Expected Breaches | Actual Breaches | Breach Rate |
|---|---|---|---|---|
| 2024 | 365 | 18 | 5 | 1.37% ✅ |
| 2025 | 363 | 18 | 5 | 1.38% ✅ |

The model produced **identical breach rates across two completely separate years** it had never seen. That's not luck - that's a stable model.

---

## Technical approach

### Data
- 2,918 daily Bitcoin prices downloaded via `yfinance` (2018–2025)
- Daily log returns calculated: `ln(Pt / Pt-1)`
- Covers major market events: COVID crash (-46.47%), 2021 bull run, FTX collapse, 2024 ETF approval

### Stationarity (ADF + KPSS Tests)
- Raw prices: non-stationary (ADF p = 0.92) ❌
- Log returns: stationary (ADF p = 0.00) ✅
- No differencing required — d = 0

### ARIMA Modelling
- Used Box-Jenkins methodology with `auto_arima`
- **Best model: ARIMA(2,0,1)** | AIC = -8250.56
- Ljung-Box p = 1.00 → residuals are white noise ✅
- Heteroskedasticity p = 0.00 → volatility clustering confirmed → GARCH required ✅

### GARCH(1,1) Volatility Modelling
Fitted on ARIMA residuals to capture time-varying volatility:

| Parameter | Value | Interpretation |
|---|---|---|
| omega (ω) | 0.5268 | Long-run baseline volatility |
| alpha (α) | 0.0948 | Sensitivity to market shocks |
| beta (β) | 0.8652 | Volatility persistence (~30 days) |
| α + β | 0.9600 | Model stability confirmed ✅ |

High beta means volatility clusters - once Bitcoin turns wild, it stays wild for roughly a month. The model captures this automatically.

### Value at Risk
Two methods compared:
- **Historical VaR:** £590 (fixed — 5th percentile of returns)
- **GARCH VaR:** dynamic — £382 on calm days, up to £2,417 during COVID

GARCH VaR adapts to current market conditions. Historical VaR cannot.

### Backtesting
At 95% confidence, 5% of days should breach VaR — that's 18 days per year.

- 2024: **5 actual breaches** (1.37%) ✅
- 2025: **5 actual breaches** (1.38%) ✅

The model consistently over-estimates risk - the conservative side, which is exactly where you want to be in risk management.

### 2026 Forecast
Retrained on full 2018–2025 dataset (2,918 days). Omega dropped from 0.7232 to 0.5268 - the model correctly identified that 2024/2025 were calmer years and revised its baseline down.

---

## Dashboard features

Built in Streamlit with a Bloomberg-style dark theme:

- **Market Overview** - BTC price history and return distribution
- **GARCH Model** - conditional volatility with COVID and FTX events marked
- **VaR Analysis** - Historical vs GARCH comparison
- **Backtesting** - 2024 and 2025 breach visualisations
- **2026 Forecast** - full year volatility and VaR forecast
- **Live sidebar** - adjust investment (£1k–£100k) and confidence level, all numbers update instantly

---

## Stack

| Category | Tools |
|---|---|
| Data | `yfinance`, `pandas`, `numpy` |
| Statistics | `statsmodels`, `scipy` |
| Modelling | `pmdarima` (ARIMA), `arch` (GARCH) |
| Evaluation | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Language | Python 3.12 |

---

## Run locally

```bash
git clone https://github.com/awad1122-tech/btc-volatility-risk-engine.git
cd btc-volatility-risk-engine
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## What I learned

- How to apply the **Box-Jenkins methodology** end-to-end on real financial data
- Why GARCH exists and why ARIMA alone isn't enough for volatile assets
- How **VaR backtesting** works in practice and what breach rates tell you
- How to deploy a data science project as a live, interactive web application

---

## About me

I'm **Awad Pervez**, an MSc Data Science student based in the UK, actively looking for graduate roles in **data science**, **quantitative finance**, or **financial modelling**.

I built this project to demonstrate that I can go beyond tutorials - take a real problem, apply rigorous methodology, validate it properly, and deliver something that works.

If you're hiring or want to connect:

📧 awadpervez123@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/awadpervez)  
💻 [GitHub](https://github.com/awad1122-tech)

---

*This project is for educational and portfolio purposes only. Not financial advice.*
