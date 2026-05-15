
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Afficionado Forecasting Dashboard", layout="wide")

st.markdown(
    '''
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

st.title("☕ Afficionado Coffee Roasters Forecasting Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data/coffee_sales.csv")

    # Generate synthetic timestamps
    base_time = pd.Timestamp("2025-01-01")
    df["datetime"] = [base_time + pd.Timedelta(minutes=i*10) for i in range(len(df))]

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.date
    df["revenue"] = df["transaction_qty"] * df["unit_price"]

    return df

df = load_data()

stores = st.sidebar.multiselect(
    "Select Store Location",
    options=df["store_location"].unique(),
    default=list(df["store_location"].unique())
)

forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 7)

metric_choice = st.sidebar.radio(
    "Forecast Metric",
    ["Revenue", "Quantity"]
)

filtered = df[df["store_location"].isin(stores)]

st.subheader("📊 Key Performance Indicators")

total_revenue = filtered["revenue"].sum()
total_qty = filtered["transaction_qty"].sum()
avg_order = filtered["revenue"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Total Quantity Sold", f"{total_qty:,}")
col3.metric("Average Order Value", f"${avg_order:.2f}")

st.subheader("📈 Daily Revenue Trend")

daily = filtered.groupby("day")["revenue"].sum().reset_index()

fig = px.line(
    daily,
    x="day",
    y="revenue",
    title="Daily Revenue Trend"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 Hourly Demand Heatmap")

heatmap_data = filtered.groupby(["store_location", "hour"])["transaction_qty"].sum().reset_index()

heatmap_fig = px.density_heatmap(
    heatmap_data,
    x="hour",
    y="store_location",
    z="transaction_qty",
    title="Store-wise Hourly Demand"
)

st.plotly_chart(heatmap_fig, use_container_width=True)

st.subheader("🤖 Forecasting Model")

series = daily["revenue"].values

if len(series) > 20:

    train = series[:-forecast_days]
    test = series[-forecast_days:]

    # Exponential Smoothing Forecast
    model = ExponentialSmoothing(train, trend="add").fit()
    forecast = model.forecast(forecast_days)

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))

    forecast_df = pd.DataFrame({
        "Actual": test,
        "Forecast": forecast
    })

    forecast_df["Day"] = range(1, forecast_days + 1)

    forecast_fig = px.line(
        forecast_df,
        x="Day",
        y=["Actual", "Forecast"],
        title="Forecast vs Actual"
    )

    st.plotly_chart(forecast_fig, use_container_width=True)

    col4, col5 = st.columns(2)

    col4.metric("MAE", round(mae, 2))
    col5.metric("RMSE", round(rmse, 2))

else:
    st.warning("Not enough data available for forecasting.")

st.subheader("🏪 Store Performance")

store_perf = filtered.groupby("store_location")["revenue"].sum().reset_index()

store_fig = px.bar(
    store_perf,
    x="store_location",
    y="revenue",
    title="Revenue by Store"
)

st.plotly_chart(store_fig, use_container_width=True)

st.subheader("📌 Business Insights")

peak_hour = filtered.groupby("hour")["transaction_qty"].sum().idxmax()

st.success(f"Peak demand occurs around {peak_hour}:00 hours.")
st.info("Morning hours show the highest transaction volume across stores.")
st.info("Location-specific planning can improve staffing efficiency and reduce waste.")

st.markdown("---")
st.caption("Developed for Afficionado Coffee Roasters - Predictive Retail Intelligence")
