import pandas as pd
import numpy as np

# Baca file Excel
file_path = 'C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\metrik eval 30 day.xlsx'
data = pd.read_excel(file_path)

# Ekstrak data aktual dan prediksi
actual = data.iloc[:, :6].values
predicted = data.iloc[:, 6:12].values

# Ganti nilai NaN dengan nilai kecil untuk menghindari masalah
actual = np.nan_to_num(actual, nan=0.0)
predicted = np.nan_to_num(predicted, nan=0.0)

# Hitung MSE per masing-masing input
mse_per_input = np.mean((actual - predicted) ** 2, axis=0)

# Cetak hasil MSE untuk setiap input
input_names = ['N1', 'P1', 'K1', 'N2', 'P2', 'K2']
for name, mse in zip(input_names, mse_per_input):
    print(f'Mean Squared Error (MSE) for {name}: {mse}')

# Hitung MAE per masing-masing input
mae_per_input = np.mean(np.abs(actual - predicted), axis=0)

# Cetak hasil MAE untuk setiap input
mae_results = dict(zip(input_names, mae_per_input))
for name, mae in mae_results.items():
    print(f'Mean Absolute Error (MAE) for {name}: {mae}')

# Hitung MAPE per masing-masing input dengan menghindari pembagian dengan nol
mape_per_input = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual)) * 100, axis=0)

# Cetak hasil MAPE untuk setiap input
mape_results = dict(zip(input_names, mape_per_input))
for name, mape in mape_results.items():
    print(f'Mean Absolute Percentage Error (MAPE) for {name}: {mape}%')
