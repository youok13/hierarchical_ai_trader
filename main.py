# =====================================================
# FINAL ENHANCED HIERARCHICAL AI TRADER (PATENT VERSION)
# =====================================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# MODULE 1 — DATA + REGIME DETECTION
# =====================================================

print("Downloading Data...")
data = yf.download("SPY", start="2015-01-01", end="2023-01-01", auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data["Returns"] = data["Close"].pct_change()
data["Volatility"] = data["Returns"].rolling(20).std()
data = data.dropna()

features = data[["Returns", "Volatility"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
hmm_model.fit(scaled_features)
data["Regime"] = hmm_model.predict(scaled_features)

print("Regime Detection Completed.")

# =====================================================
# MODULE 2 — FEATURE EXPANSION
# =====================================================

data["Momentum"] = data["Close"].pct_change(10)
data["MA_50"] = data["Close"].rolling(50).mean()
data["MA_200"] = data["Close"].rolling(200).mean()
data["MA_Spread"] = data["MA_50"] - data["MA_200"]

data = data.dropna()

# =====================================================
# 🔥 MODULE 2.5 — CRASH PREDICTION
# =====================================================

data["Crash"] = np.where(data["Returns"] < -0.02, 1, 0)

X_crash = data[["Returns", "Volatility", "Momentum"]]
y_crash = data["Crash"]

crash_model = RandomForestClassifier(n_estimators=100)
crash_model.fit(X_crash, y_crash)

data["Crash_Prob"] = crash_model.predict_proba(X_crash)[:,1]

# =====================================================
# 🔥 MODULE 2.6 — REGIME TRANSITION RISK (PATENT IDEA)
# =====================================================

data["Next_Regime"] = data["Regime"].shift(-1)
data = data.dropna()

X_regime = data[["Returns", "Volatility", "Momentum", "MA_Spread"]]
y_regime = data["Next_Regime"]

regime_model = GradientBoostingClassifier()
regime_model.fit(X_regime, y_regime)

regime_probs = regime_model.predict_proba(X_regime)

# 🔥 Transition Risk (uncertainty measure)
data["Transition_Risk"] = 1 - np.max(regime_probs, axis=1)

# =====================================================
# 🔥 COMBINED MULTI-LAYER RISK ENGINE
# =====================================================

data["Risk"] = (
    0.4 * data["Volatility"] +
    0.3 * data["Crash_Prob"] +
    0.3 * data["Transition_Risk"]
)

# =====================================================
# MODULE 3 — RL ENVIRONMENT
# =====================================================

class HierarchicalTradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self.balance = 1.0
        self.max_balance = 1.0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = 1.0
        self.max_balance = 1.0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.loc[self.current_step]
        return np.array([
            row["Returns"],
            row["Volatility"],
            row["Momentum"],
            row["MA_Spread"],
            row["Regime"],
            row["Risk"],
            row["Transition_Risk"]
        ], dtype=np.float32)

    def step(self, action):
        weight = float(action[0])

        # 🔥 Multi-layer risk-aware scaling
        risk = self.df.loc[self.current_step, "Risk"]
        weight = weight * (1 - risk)

        ret = self.df.loc[self.current_step, "Returns"]
        portfolio_return = weight * ret

        self.balance *= (1 + portfolio_return)

        self.max_balance = max(self.max_balance, self.balance)
        drawdown = (self.balance / self.max_balance) - 1

        reward = (
            portfolio_return
            - 0.5 * risk
            - 2.0 * abs(drawdown)
        )

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_obs(), reward, done, False, {}

# =====================================================
# TRAIN PPO
# =====================================================

env = HierarchicalTradingEnv(data)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=15000)

print("Training Completed.")

# =====================================================
# BACKTEST
# =====================================================

obs, _ = env.reset()
balances = []
weights = []

for _ in range(len(data)-1):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    balances.append(env.balance)
    weights.append(action[0])
    if done:
        break

data_bt = data.iloc[:len(balances)].copy()
data_bt["Strategy"] = balances
data_bt["Weight"] = weights
data_bt["BuyHold"] = (1 + data_bt["Returns"]).cumprod()

# =====================================================
# BASELINE STRATEGIES
# =====================================================

data_bt["MA_signal"] = np.where(data_bt["MA_50"] > data_bt["MA_200"], 1, -1)
data_bt["MA_strategy"] = (1 + data_bt["MA_signal"].shift(1) * data_bt["Returns"]).cumprod()

data_bt["Momentum_signal"] = np.where(data_bt["Momentum"] > 0, 1, -1)
data_bt["Momentum_strategy"] = (1 + data_bt["Momentum_signal"].shift(1) * data_bt["Returns"]).cumprod()

# =====================================================
# METRICS
# =====================================================

def sharpe(x):
    x = x.dropna()
    return x.mean() / x.std()

def max_dd(x):
    return (x / x.cummax() - 1).min()

returns_strategy = data_bt["Strategy"].pct_change()

print("\n===== PERFORMANCE =====")
print("Sharpe:", sharpe(returns_strategy))
print("Max Drawdown:", max_dd(data_bt["Strategy"]))

# =====================================================
# VISUALIZATIONS (10 GRAPHS)
# =====================================================

plt.figure(figsize=(12,6))
plt.plot(data_bt["Strategy"], label="AI")
plt.plot(data_bt["BuyHold"], label="BuyHold")
plt.legend(); plt.title("Strategy vs BuyHold"); plt.show()

plt.figure(figsize=(12,6))
plt.plot(data_bt["Weight"]); plt.title("Allocation"); plt.show()

drawdown = data_bt["Strategy"]/data_bt["Strategy"].cummax()-1
plt.figure(figsize=(12,6))
plt.plot(drawdown); plt.title("Drawdown"); plt.show()

plt.figure()
data["Regime"].value_counts().plot(kind="bar")
plt.title("Regime Distribution"); plt.show()

rolling = returns_strategy.rolling(50).mean()/returns_strategy.rolling(50).std()
plt.figure(); plt.plot(rolling); plt.title("Rolling Sharpe"); plt.show()

plt.figure()
plt.scatter(data_bt["Regime"], data_bt["Weight"])
plt.title("Allocation vs Regime"); plt.show()

plt.figure()
plt.plot(data["Crash_Prob"])
plt.title("Crash Probability"); plt.show()

plt.figure()
plt.plot(data["Transition_Risk"])
plt.title("Transition Risk (PATENT FEATURE)"); plt.show()

plt.figure()
returns_strategy.hist(bins=50)
plt.title("Returns Distribution"); plt.show()

plt.figure()
plt.plot(data["Volatility"])
plt.title("Volatility"); plt.show()

print("\n🚀 ALL DONE — PATENT LEVEL SYSTEM READY")