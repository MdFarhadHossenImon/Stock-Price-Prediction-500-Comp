import streamlit as st
import pandas as pd
from datetime import timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
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
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: "DM Sans", "Segoe UI", sans-serif;
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
            font-size: 0.82rem;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #101828;
            font-size: 1.55rem;
            font-weight: 700;
        }

        [data-testid="stMetricDelta"] {
            color: #027a48;
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

feature_data = symbol_data.copy()
feature_data["lag_1"] = feature_data["close"].shift(1)
feature_data["lag_2"] = feature_data["close"].shift(2)
feature_data["lag_5"] = feature_data["close"].shift(5)
feature_data["lag_20"] = feature_data["close"].shift(20)
feature_data["rolling_mean_5"] = feature_data["close"].shift(1).rolling(5).mean()
feature_data["rolling_mean_20"] = feature_data["close"].shift(1).rolling(20).mean()
feature_data["volume"] = feature_data["volume"].fillna(feature_data["volume"].median())

features = [
    "open", "high", "low", "volume",
    "lag_1", "lag_2", "lag_5", "lag_20",
    "rolling_mean_5", "rolling_mean_20",
]
feature_data = feature_data.dropna(subset=features + ["close"])

X = feature_data[features]
y = feature_data["close"]
split_index = int(len(feature_data) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

val_predictions = model.predict(X_val)
mae = mean_absolute_error(y_val, val_predictions)

last_known_date = symbol_data["date"].iloc[-1]
future_dates = pd.bdate_range(start=last_known_date + timedelta(days=1), periods=future_days)

history_close = symbol_data["close"].tolist()
latest = symbol_data.iloc[-1]
future_predictions = []

for forecast_date in future_dates:
    lag_1 = history_close[-1]
    lag_2 = history_close[-2] if len(history_close) >= 2 else lag_1
    lag_5 = history_close[-5] if len(history_close) >= 5 else lag_1
    lag_20 = history_close[-20] if len(history_close) >= 20 else lag_1

    input_row = pd.DataFrame([{
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "volume": latest["volume"],
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_5": lag_5,
        "lag_20": lag_20,
        "rolling_mean_5": sum(history_close[-5:]) / min(5, len(history_close)),
        "rolling_mean_20": sum(history_close[-20:]) / min(20, len(history_close)),
    }])

    predicted_close = float(model.predict(input_row)[0])
    future_predictions.append(predicted_close)
    history_close.append(predicted_close)

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
    x=future_dates,
    y=future_predictions,
    mode="lines+markers",
    name="Forecast",
    line=dict(color="#12b76a", width=2.5, dash="dot"),
    marker=dict(size=7, color="#12b76a", line=dict(width=1, color="white")),
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
    ),
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="DM Sans, Segoe UI, sans-serif", color="#475467"),
)

fig.update_xaxes(showgrid=False, showline=True, linecolor="#eaecf0")
fig.update_yaxes(showgrid=True, gridcolor="#f2f4f7", zeroline=False)

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

latest_price = float(symbol_data["close"].iloc[-1])
forecast_price = float(future_predictions[-1])
forecast_change = ((forecast_price - latest_price) / latest_price) * 100

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Latest close", f"${latest_price:,.2f}")
metric_col2.metric(
    f"{future_days}-day forecast",
    f"${forecast_price:,.2f}",
    f"{forecast_change:+.2f}%",
)
metric_col3.metric("Validation MAE", f"${mae:,.2f}")

st.markdown("")
st.plotly_chart(fig, width="stretch")
