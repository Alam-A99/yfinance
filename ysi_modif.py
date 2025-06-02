import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Stock Analysis", page_icon="📈", layout="wide")
st.title("📈 Yahoo Finance Stock Analyzer")

# Input simbol saham dan rentang waktu
ticker = st.text_input("Masukkan simbol saham (misal: AAPL, GOTO.JK, BBCA.JK)", value="BBCA.JK")
start_date = st.date_input("Tanggal Mulai", pd.to_datetime("2022-01-01"))
end_date = st.date_input("Tanggal Akhir", pd.to_datetime("today"))

if start_date > end_date:
    st.error("❗ Tanggal mulai harus sebelum tanggal akhir.")
    
elif st.button("🔍 Ambil Data & Analisis"):
    try:
        # Unduh data saham dari Yahoo Finance
        df = yf.download(ticker.upper(), start=start_date, end=end_date)

        # Validasi jika data tidak kosong
        if df.empty:
            st.warning(f"⚠️ Tidak ada data untuk simbol '{ticker}' dan rentang waktu tersebut.\nPastikan simbol benar dan pasar aktif dalam rentang waktu itu.")
        else:
            st.success(f"✅ Data untuk simbol '{ticker}' berhasil diambil.")
            
            # Tampilkan preview data
            st.dataframe(df.tail())
            
            # Analisis Grafik Saham
            st.subheader("📊 Analisis Grafik Saham")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name="Candlestick"))
            fig.update_layout(title=f"Grafik Saham {ticker}",
                              xaxis_title="Tanggal",
                              yaxis_title="Harga",
                              template="plotly_dark")
            st.plotly_chart(fig)

            # Analisis Moving Average dan RSI
            st.subheader("📉 Moving Average dan RSI")

            # Moving Average
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()

            # RSI (Relative Strength Index)
            rsi = RSIIndicator(df['Close'], window=14)
            df['RSI'] = rsi.rsi()

            # Plot MA dan RSI
            fig, ax = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot moving averages
            ax[0].plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
            ax[0].plot(df.index, df['MA50'], label='50-Day MA', color='red', alpha=0.6)
            ax[0].plot(df.index, df['MA200'], label='200-Day MA', color='green', alpha=0.6)
            ax[0].set_title(f"Moving Average untuk {ticker}")
            ax[0].legend()

            # Plot RSI
            ax[1].plot(df.index, df['RSI'], label='RSI', color='orange')
            ax[1].axhline(y=70, color='r', linestyle='--', label='Overbought')
            ax[1].axhline(y=30, color='g', linestyle='--', label='Oversold')
            ax[1].set_title(f"RSI untuk {ticker}")
            ax[1].legend()

            st.pyplot(fig)

            # ARIMA Forecasting
            st.subheader("🔮 Forecasting dengan ARIMA")

            # Menggunakan hanya harga penutupan
            df_close = df[['Close']].dropna()

            # Train ARIMA Model
            model = ARIMA(df_close, order=(5, 1, 0))  # Order dapat diubah
            model_fit = model.fit()

            # Forecasting untuk 10 hari ke depan
            forecast_steps = 10
            forecast = model_fit.forecast(steps=forecast_steps)
            forecast_index = pd.date_range(df_close.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
            
            # Visualisasi hasil forecast
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_close.index, df_close['Close'], label='Harga Penutupan Asli', color='blue')
            ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast ARIMA', color='red')
            ax.set_title(f"Forecast Harga Saham {ticker} menggunakan ARIMA")
            ax.legend()
            
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px;'>
        © 2025 | Developed with ❤️ by AlDev - Muhammad Alif<br>
        Feedback and suggestions? Reach me at <a href='https://www.instagram.com/mhdalif.id/' target='_blank'>Instagram</a>
    </div>
    """,
    unsafe_allow_html=True
)
