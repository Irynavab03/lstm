import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2, l1, L1L2
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Memuat data dari file Excel
data = pd.read_excel('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\raw data pelatihan.xlsx', sheet_name='pelatihan 8 jam')

# Menghapus baris yang mengandung nilai NaN
data = data.dropna()

# Menentukan kolom input dan output
#input_columns = ['SM1', 'N1', 'P1', 'K1', 'PH1','SM2', 'N2', 'P2', 'K2', 'PH2']
#output_columns = ['N1', 'P1', 'K1','N2', 'P2', 'K2']
input_columns = ['SM2', 'N2', 'P2', 'K2', 'PH2']
output_columns = ['N2', 'P2', 'K2']
#input_columns = ['SM1', 'N1', 'P1', 'K1', 'PH1']
#output_columns = ['N1', 'P1', 'K1']
age_column = 'age'  # Kolom usia tanaman

# Mengambil data input, output, dan usia tanaman
input_data = data[input_columns].values
output_data = data[output_columns].values
age_data = data[age_column].values

# Normalisasi data menggunakan StandardScaler
input_scaler = StandardScaler()
output_scaler = StandardScaler()

scaled_input_data = input_scaler.fit_transform(input_data)
scaled_output_data = output_scaler.fit_transform(output_data)

# Menyimpan StandardScaler
#joblib.dump(input_scaler, 'input_scalerkemarau.pkl')
#joblib.dump(output_scaler, 'output_scalerkemarau.pkl')
#print("Scalers telah disimpan sebagai 'input_scaler.pkl' dan 'output_scaler.pkl'")

# Membuat dataset untuk LSTM
def create_dataset(X, y, age, time_step=1):
    Xs, ys, ages = [], [], []
    for i in range(len(X) - time_step):
        Xs.append(X[i:(i + time_step), :])
        ys.append(y[i + time_step, :])
        ages.append(age[i + time_step])
    return np.array(Xs), np.array(ys), np.array(ages)

time_step = 90  # Interval waktu
X, y, ages = create_dataset(scaled_input_data, scaled_output_data, age_data, time_step)

# Membentuk data untuk LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Membagi data menjadi pelatihan dan pengujian
train_size = int(len(X) * 0.80)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]
ages_train, ages_test = ages[0:train_size], ages[train_size:len(ages)]

# Membangun model LSTM berdasarkan hyperparameter terbaik
best_units_1 = 25
best_units_2 = 35
best_dropout_2 = 0.2
best_learning_rate = 0.001

model = Sequential()
model.add(LSTM(units=best_units_1, return_sequences=True, input_shape=(time_step, X.shape[2]), kernel_regularizer=L1L2(0.01)))
#model.add(Dropout(rate=0.2))
model.add(LSTM(units=best_units_2, return_sequences=False, kernel_regularizer=L1L2(0.01)))
#model.add(Dropout(rate=best_dropout_2))
model.add(Dense(y.shape[1]))
model.compile(optimizer=Adam(learning_rate=best_learning_rate), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Melatih model
history = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

# Menyimpan model yang telah dilatih
#model.save('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\best_hujan baru.h5')
#print("Model telah disimpan sebagai 'best_hujan baru.h5'")
#model.save('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\best_kemarau baru.h5')
#print("Model telah disimpan sebagai 'best_kemarau baru.h5'")

# Plot loss selama pelatihan
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_loss(history)

# Prediksi menggunakan model terbaik
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Mengganti nilai nol dengan nilai kecil pada data asli sebelum invers transformasi
epsilon = 1e-10  # Nilai kecil yang mendekati nol
y_train[y_train == 0] = epsilon
y_test[y_test == 0] = epsilon

# Mengembalikan data ke bentuk asli
train_predict = output_scaler.inverse_transform(train_predict)
test_predict = output_scaler.inverse_transform(test_predict)
y_train = output_scaler.inverse_transform(y_train)
y_test = output_scaler.inverse_transform(y_test)

# Menghitung MAPE dengan mengabaikan nilai nol
def safe_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Menghitung metrik evaluasi
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

train_mse = mean_squared_error(y_train, train_predict)
test_mse = mean_squared_error(y_test, test_predict)

train_mae = mean_absolute_error(y_train, train_predict)
test_mae = mean_absolute_error(y_test, test_predict)

train_mape = safe_mape(y_train, train_predict)
test_mape = safe_mape(y_test, test_predict)

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')

print(f'Train MAE: {train_mae}')
print(f'Test MAE: {test_mae}')

print(f'Train MAPE: {train_mape}%')
print(f'Test MAPE: {test_mape}%')

# Plot hasil
plt.figure(figsize=(15, 8))

for i, col in enumerate(output_columns):
    plt.subplot(len(output_columns), 1, i+1)
    
    # Plot data nyata
    min_len_real = min(len(ages[time_step:]), len(output_data[time_step:, i]))
    plt.plot(ages[time_step:time_step + min_len_real], output_data[time_step:time_step + min_len_real, i], label=f'Real Data {col}', color='blue')
    
    # Plot prediksi pelatihan
    min_len_train = min(len(ages[time_step:time_step + len(train_predict)]), len(train_predict[:, i]))
    plt.plot(ages[time_step:time_step + min_len_train], train_predict[:min_len_train, i], label=f'Train Prediction {col}', color='orange')
    
    # Plot prediksi pengujian
    min_len_test = min(len(ages[time_step + len(train_predict):time_step + len(train_predict) + len(test_predict)]), len(test_predict[:, i]))
    plt.plot(ages[time_step + len(train_predict):time_step + len(train_predict) + min_len_test], test_predict[:min_len_test, i], label=f'Test Prediction {col}', color='red')
    
    plt.xlabel('Plants Age (HST)')
    plt.ylabel(f'PPM ({col})')
    plt.title(f'Forecasting Nutrition Level {col}')
    plt.legend()

plt.tight_layout()
plt.show()