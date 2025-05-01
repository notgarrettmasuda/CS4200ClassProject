import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Create sequences of data for LSTM
def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # Predict 'Close'
    return np.array(X), np.array(y)

# Train LSTM on a single file
def train_on_file(filepath):
    print(f"\n📁 Processing: {filepath}")
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        if len(df) < 31:
            print(f"⛔ Skipped {filepath} (not enough rows)")
            return

        data = df[['Close', 'Volume']].values

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = create_sequences(scaled_data)
        if len(X) == 0:
            print(f"⛔ Skipped {filepath} (sequence creation failed)")
            return

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

        predicted = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(
            np.concatenate((predicted, X_test[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]
        actual_prices = scaler.inverse_transform(
            np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]

        # Save plot to plots/ folder
        os.makedirs("plots", exist_ok=True)
        symbol = os.path.basename(filepath).replace(".csv", "")
        plot_path = f"plots/{symbol}_plot.png"

        plt.figure()
        plt.plot(actual_prices, label='Actual')
        plt.plot(predicted_prices, label='Predicted')
        plt.title(f"Prediction for {symbol}")
        plt.legend()
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Plot saved: {plot_path}")

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")

# Loop through both stocks and etfs folders
FOLDERS = ["stocks", "etfs"]

#Go over the whole data set
#for folder in FOLDERS:
#    print(f"\n🔎 Entering folder: {folder}")
#    for filename in os.listdir(folder):
#        if filename.endswith(".csv"):
#            full_path = os.path.join(folder, filename)
#    
#        train_on_file(full_path)

#Go over 5 folders
for folder in FOLDERS:
    print(f"\n🔎 Entering folder: {folder}")
    all_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    limited_files = all_files[:5]  # Only take first 5 CSVs
    for filename in limited_files:
        full_path = os.path.join(folder, filename)
        train_on_file(full_path)