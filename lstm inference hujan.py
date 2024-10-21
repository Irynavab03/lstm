import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import matplotlib.pyplot as plt

# Fungsi untuk membuat dataset untuk LSTM
def create_dataset(X, time_step=1):
    Xs = []
    for i in range(len(X) - time_step):
        Xs.append(X[i:(i + time_step), :])
    return np.array(Xs)

# Memuat model yang telah disimpan
model = load_model('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\lstm baru\\hujan.h5')

# Memuat scaler yang telah disimpan
input_scaler = joblib.load('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\lstm baru\\input_scaler1.pkl')
output_scaler = joblib.load('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\lstm baru\\output_scaler1.pkl')

# Memuat data uji dari file Excel
test_data = pd.read_excel('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\raw data pengujian.xlsx', sheet_name='pengujian 8 jam 50')

# Menghapus baris yang mengandung nilai NaN
test_data = test_data.dropna()

# Menentukan kolom input dan output
input_columns = ['SM1', 'N1', 'P1', 'K1', 'PH1']
output_columns = ['N1', 'P1', 'K1']
age_column = 'age'  # Kolom usia tanaman

# Mengambil data input dan output
input_test_data = test_data[input_columns].values
output_test_data = test_data[output_columns].values
age_data = test_data[age_column].values

# Normalisasi data menggunakan StandardScaler yang telah disimpan
scaled_input_test_data = input_scaler.transform(input_test_data)
scaled_output_test_data = output_scaler.transform(output_test_data)

# Membuat dataset untuk LSTM
time_step = 90  # Interval waktu harus sama dengan yang digunakan pada pelatihan
X_test = create_dataset(scaled_input_test_data, time_step)

# Membentuk data untuk LSTM [samples, time steps, features]
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Melakukan prediksi menggunakan model yang telah dilatih
test_predict = model.predict(X_test)

# Mengembalikan data ke bentuk asli
test_predict = output_scaler.inverse_transform(test_predict)

# Mengambil data uji yang sesuai untuk evaluasi
y_true = output_test_data[time_step:time_step + len(test_predict)]
ages = age_data[time_step:time_step + len(test_predict)]

# Menghitung metrik evaluasi
rmse = np.sqrt(mean_squared_error(y_true, test_predict))
mse = mean_squared_error(y_true, test_predict)
mae = mean_absolute_error(y_true, test_predict)
mape = mean_absolute_percentage_error(y_true, test_predict)

print(f'Test RMSE: {rmse}')
print(f'Test MSE: {mse}')
print(f'Test MAE: {mae}')
print(f'Test MAPE: {mape * 100}%')

# Menyimpan hasil prediksi ke file Excel
predicted_df = pd.DataFrame(test_predict, columns=[f'Predicted_{col}' for col in output_columns])
#predicted_df.to_excel('predicted_lstm_model.xlsx', index=False)
#predicted_df.to_excel('predicted_kemarau baru.xlsx', index=False)
#predicted_df.to_excel('predicted_hujan baru.xlsx', index=False)

# Plot hasil
plt.figure(figsize=(15, 8))

for i, col in enumerate(output_columns):
    plt.subplot(len(output_columns), 1, i+1)
    plt.plot(ages, y_true[:, i], label=f'Real Data {col}', color='blue')
    plt.plot(ages, test_predict[:len(ages), i], label=f'Test Prediction {col}', color='red')
    plt.xlabel('Plants Age (HST)')
    plt.ylabel(f'PPM {col}')
    plt.title(f'Forecasting Nutrition Level {col}')
    plt.legend()

plt.tight_layout()
plt.show()
