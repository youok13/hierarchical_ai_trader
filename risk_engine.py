import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from hmmlearn.hmm import GaussianHMM

from arch import arch_model
from scipy.stats import norm

from fastapi import FastAPI
import uvicorn

# ============================================================
# DATA DOWNLOAD
# ============================================================

data = yf.download("SPY", start="2010-01-01", auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data["Returns"] = data["Close"].pct_change()
data["Volatility"] = data["Returns"].rolling(20).std()
data["Momentum"] = data["Close"].pct_change(10)

data = data.dropna()

# ============================================================
# REGIME DETECTION (HMM)
# ============================================================

features = data[["Returns", "Volatility"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
hmm.fit(scaled)

data["Regime"] = hmm.predict(scaled)

# ============================================================
# 🔥 REGIME TRANSITION PREDICTION (PATENT)
# ============================================================

data["Next_Regime"] = data["Regime"].shift(-1)
data = data.dropna()

X_regime = data[["Returns", "Volatility", "Momentum"]]
y_regime = data["Next_Regime"]

transition_model = GradientBoostingClassifier()
transition_model.fit(X_regime, y_regime)

regime_probs = transition_model.predict_proba(X_regime)

# Transition risk = uncertainty
data["Transition_Risk"] = 1 - np.max(regime_probs, axis=1)

# ============================================================
# VOLATILITY FORECAST (GARCH)
# ============================================================

garch = arch_model(data["Returns"] * 100, vol="Garch", p=1, q=1)
garch_fit = garch.fit(disp="off")

# ============================================================
# 🔥 ML-BASED CRASH PREDICTION (NEW)
# ============================================================

data["Crash"] = np.where(data["Returns"] < -0.02, 1, 0)

X_crash = data[["Returns", "Volatility", "Momentum"]]
y_crash = data["Crash"]

crash_model = GradientBoostingClassifier()
crash_model.fit(X_crash, y_crash)

data["Crash_Prob_ML"] = crash_model.predict_proba(X_crash)[:,1]

# ============================================================
# RISK COMPUTATION FUNCTION
# ============================================================

def compute_risk_overlay():

    latest = data.iloc[-1]

    # Regime
    latest_regime = latest["Regime"]
    transition_risk = latest["Transition_Risk"]

    # Confidence
    regime_probs = hmm.predict_proba(scaled)[-1]
    confidence = np.max(regime_probs)

    # Volatility Forecast
    forecast = garch_fit.forecast(horizon=1)
    predicted_vol = np.sqrt(forecast.variance.values[-1][0]) / 100

    # Crash Probabilities
    crash_prob_stat = norm.cdf(-0.03, loc=0, scale=predicted_vol)
    crash_prob_ml = latest["Crash_Prob_ML"]

    # 🔥 HYBRID CRASH PROBABILITY
    crash_prob = 0.5 * crash_prob_stat + 0.5 * crash_prob_ml

    # ============================================================
    # 🔥 MULTI-LAYER RISK ENGINE (PATENT CORE)
    # ============================================================

    risk_score = (
        0.4 * predicted_vol +
        0.3 * crash_prob +
        0.3 * transition_risk
    )

    # Adaptive exposure
    exposure = max(0.05, 1 - risk_score * 1.5)

    regime_map = {
        0: "Low Volatility",
        1: "Trending",
        2: "High Volatility"
    }

    return {
        "regime": regime_map.get(int(latest_regime)),
        "confidence": float(confidence),
        "predicted_volatility": float(predicted_vol),
        "crash_probability": float(crash_prob),
        "transition_risk": float(transition_risk),
        "risk_score": float(risk_score),
        "recommended_exposure": float(exposure)
    }

# ============================================================
# FASTAPI DEPLOYMENT
# ============================================================

app = FastAPI()

@app.get("/risk-overlay")
def risk_overlay():
    return compute_risk_overlay()

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)