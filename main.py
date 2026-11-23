import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

st.set_page_config(
    page_title="Market Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: "Segoe UI", Arial, sans-serif;
        }

        .stApp {
            background: #f4f7fb;
        }

        [data-testid="stSidebar"] {
            background: #101828;
            border-right: 1px solid #1d2939;
        }

        [data-testid="stSidebar"] * {
            color: #e4e7ec;
        }

        [data-testid="stSidebar"] [data-baseweb="slider"] div {
            color: #667085;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            right: 1rem;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.25rem;
            padding-bottom: 3rem;
        }

        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 1.5rem;
        }

        .app-title {
            color: #101828;
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 700;
            letter-spacing: -0.035em;
            margin: 0;
        }

        .app-subtitle {
            color: #667085;
            font-size: 0.95rem;
            margin-top: 0.45rem;
        }

        .status-pill {
            background: #ecfdf3;
            color: #027a48;
            border: 1px solid #abefc6;
            border-radius: 999px;
            padding: 0.42rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }

        [data-testid="stMetricLabel"] {
            color: #667085;
            font-size: 0.9rem;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #101828;
            font-size: 1.55rem;
            font-weight: 700;
        }

        [data-testid="stMetricDelta"] {
            color: #027a48 !important;
            background: #ecfdf3;
        }

        [data-testid="stPlotlyChart"] {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 14px;
            padding: 0.35rem 0.15rem 0.15rem;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05);
        }

        [data-baseweb="select"] > div {
            background: #1d2939;
            border-color: #344054;
            border-radius: 8px;
        }

        [data-testid="stSlider"] {
            padding-top: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the 25-year S&P 500 dataset.
dataset_dir = Path(__file__).resolve().parent / "new dataset stock"
stocks_path = dataset_dir / "sp500_stocks.csv"
companies_path = dataset_dir / "sp500_companies.csv"


@st.cache_data(ttl=3600, show_spinner="Loading 25 years of market data...")
def load_market_data():
    stocks = pd.read_csv(
        stocks_path,
        parse_dates=["date"],
        usecols=["date", "open", "high", "low", "close", "volume", "symbol"],
    )
    stocks = stocks.sort_values(["symbol", "date"])
    companies = pd.read_csv(companies_path)
    names = dict(zip(companies["symbol"], companies["company"]))
    return stocks, names


stocks, company_names = load_market_data()
symbols = sorted(stocks["symbol"].dropna().unique())
symbol_options = {
    f"{symbol} - {company_names.get(symbol, symbol)}": symbol
    for symbol in symbols
}
default_symbol = "AAPL" if "AAPL" in symbols else symbols[0]

with st.sidebar:
    st.markdown("## Market Forecast")
    st.caption("Machine learning dashboard")
    st.divider()
    selected_label = st.selectbox(
        "Stock",
        options=list(symbol_options),
        index=list(symbol_options).index(
            f"{default_symbol} - {company_names.get(default_symbol, default_symbol)}"
        ),
    )
    target_symbol = symbol_options[selected_label]
    target_name = company_names.get(target_symbol, target_symbol)
    st.divider()
    st.markdown("### Forecast settings")
    future_days = st.slider(
        "Prediction horizon",
        min_value=1,
        max_value=30,
        value=7,
        help="Choose how many days ahead to forecast.",
    )
    st.caption("The model refreshes automatically when settings change.")

symbol_data = (
    stocks.loc[stocks["symbol"] == target_symbol, ["date", "open", "high", "low", "close", "volume"]]
    .dropna()
    .copy()
)
symbol_data = symbol_data.sort_values("date").reset_index(drop=True)

daily_returns = symbol_data["close"].pct_change()
feature_data = pd.DataFrame({
    "target_return": daily_returns,
    "previous_close": symbol_data["close"].shift(1),
    "target_close": symbol_data["close"],
})

return_lags = [1, 2, 3, 5, 10, 20]
for lag in return_lags:
    feature_data[f"return_lag_{lag}"] = daily_returns.shift(lag)

feature_data["momentum_5"] = symbol_data["close"].shift(1) / symbol_data["close"].shift(6) - 1
feature_data["momentum_20"] = symbol_data["close"].shift(1) / symbol_data["close"].shift(21) - 1
feature_data["volatility_5"] = daily_returns.shift(1).rolling(5).std()
feature_data["volatility_20"] = daily_returns.shift(1).rolling(20).std()
feature_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()

features = [
    *(f"return_lag_{lag}" for lag in return_lags),
    "momentum_5",
    "momentum_20",
    "volatility_5",
    "volatility_20",
]
X = feature_data[features]
y = feature_data["target_return"]
split_index = int(len(feature_data) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=max(2, min(20, len(X_train) // 20)),
    max_features=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

validation_previous_close = feature_data["previous_close"].iloc[split_index:]
validation_actual_close = feature_data["target_close"].iloc[split_index:]
validation_returns = model.predict(X_val)
validation_predictions = validation_previous_close * (1 + validation_returns)
mae = mean_absolute_error(validation_actual_close, validation_predictions)
baseline_mae = mean_absolute_error(validation_actual_close, validation_previous_close)

last_known_date = symbol_data["date"].iloc[-1]
future_dates = pd.bdate_range(start=last_known_date + timedelta(days=1), periods=future_days)

history_close = symbol_data["close"].tolist()
last_historical_close = float(history_close[-1])
history_returns = daily_returns.dropna().tolist()
future_predictions = []

for _ in future_dates:
    input_features = {
        f"return_lag_{lag}": history_returns[-lag]
        for lag in return_lags
    }
    input_features["momentum_5"] = history_close[-1] / history_close[-6] - 1
    input_features["momentum_20"] = history_close[-1] / history_close[-21] - 1
    input_features["volatility_5"] = pd.Series(history_returns[-5:]).std()
    input_features["volatility_20"] = pd.Series(history_returns[-20:]).std()
    input_row = pd.DataFrame([input_features], columns=features)

    predicted_return = float(model.predict(input_row)[0])
    predicted_close = history_close[-1] * (1 + predicted_return)
    future_predictions.append(predicted_close)
    history_close.append(predicted_close)
    history_returns.append(predicted_return)

fig = go.Figure()

years_back = 3
start_date = last_known_date - timedelta(days=365 * years_back)
historical_data = symbol_data.loc[
    symbol_data["date"] >= start_date,
    ["date", "close"],
]
fig.add_trace(go.Scatter(
    x=historical_data["date"],
    y=historical_data["close"],
    mode="lines",
    name="Historical",
    line=dict(color="#2563eb", width=2),
    fill="tozeroy",
    fillcolor="rgba(37, 99, 235, 0.06)",
))

fig.add_trace(go.Scatter(
    x=[last_known_date, *future_dates],
    y=[last_historical_close, *future_predictions],
    mode="lines+markers",
    name="Forecast",
    line=dict(color="#b54708", width=3),
    marker=dict(size=7, color="#b54708", line=dict(width=1, color="white")),
))

fig.update_layout(
    title=dict(
        text=f"{target_symbol} price forecast",
        x=0.02,
        xanchor="left",
        font=dict(size=18, color="#101828"),
    ),
    xaxis_title=None,
    yaxis_title="Price (USD)",
    template="plotly_white",
    height=520,
    margin=dict(l=20, r=20, t=72, b=35),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=14, color="#344054"),
    ),
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Segoe UI, Arial, sans-serif", color="#344054", size=14),
)

fig.update_xaxes(
    showgrid=False,
    showline=True,
    linecolor="#d0d5dd",
    tickfont=dict(size=13, color="#475467"),
    title_font=dict(size=14, color="#344054"),
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="#eaecf0",
    zeroline=False,
    tickfont=dict(size=13, color="#475467"),
    title_font=dict(size=14, color="#344054"),
)

st.markdown(
    """
    <div class="app-header">
        <div>
            <h1 class="app-title">Stock performance</h1>
            <div class="app-subtitle">Explore historical trends and model forecasts</div>
        </div>
        <div class="status-pill">● Model ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

latest_price = last_historical_close
forecast_price = float(future_predictions[-1])
forecast_change = ((forecast_price - latest_price) / latest_price) * 100

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Latest close", f"${latest_price:,.2f}")
metric_col2.metric(
    f"{future_days}-day forecast",
    f"${forecast_price:,.2f}",
    f"{forecast_change:+.2f}%",
)
metric_col3.metric("Model 1-day MAE", f"${mae:,.2f}")
metric_col4.metric("No-change MAE", f"${baseline_mae:,.2f}")

if mae >= baseline_mae:
    st.caption("The model did not beat the previous-close baseline in this validation period.")

st.markdown("")
st.plotly_chart(fig, width="stretch", theme=None)
