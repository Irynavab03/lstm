import pymongo
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # Tambahkan ini

# Koneksi ke MongoDB
client = pymongo.MongoClient("mongodb+srv://smartfarmingunpad:Zg2btY2zwNddpNsvLrYGNGtgTSZS6xxX@smartfarmingunpad.usves.mongodb.net/?retryWrites=true&w=majority")
db = client['smartfarmingunpad']

# Memuat model yang telah disimpan
model = load_model('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\lstm baru\\best_kemaraubaru.h5')

# Memuat scaler yang telah disimpan
input_scaler = joblib.load('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\lstm baru\\input_scalerkemarau.pkl')
output_scaler = joblib.load('C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\lstm baru\\output_scalerkemarau.pkl')

# Daftar data soil yang ingin diambil dari MongoDB (device_id, index_id)
devices_indices = {
    "SM2": ("Tdr4a4bKp5AzrCe6KGki8bUDF0ynE9l9", "61910bcfd2cd6b06225ee0ca"),
    "PH2": ("lWwWZ7RHI5HToRocg122mLHgmqKsT7F7", "618bce88109f491b98e68b59"),
    "N2": ("gxaZkwZafNVweTq8HycMKpZMz9MvbTyh", "6280a8465a0c89673d266101"),  # NPK Kemarau
    "P2": ("gxaZkwZafNVweTq8HycMKpZMz9MvbTyh", "6280a8275a0c89673d266100"),
    "K2": ("gxaZkwZafNVweTq8HycMKpZMz9MvbTyh", "6280a8505a0c89673d266102"),
}

# Fungsi untuk mengambil 180 data terbaru per device_id dan index_id
def get_latest_data_per_device(collection_name, devices_indices, n=180):
    collection = db[collection_name]
    all_data = []
    
    for device, (device_id, index_id) in devices_indices.items():
        cursor = collection.find(
            {'device_id': device_id, 'index_id': index_id}
        ).sort('_id', -1).limit(n)
        
        data = list(cursor)
        if len(data) < n:
            print(f"Warning: Hanya menemukan {len(data)} data untuk {device} (ID: {device_id}, Index: {index_id})")
        
        df = pd.DataFrame(data)
        df = df.iloc[::-1]  # Membalik urutan data agar urutannya dari yang paling lama ke yang terbaru
        all_data.append(df['value'].values)
    
    return np.array(all_data)

# Fungsi untuk membuat dataset untuk LSTM
def create_dataset(X, time_step=1):
    Xs = []
    for i in range(len(X[0]) - time_step + 1):
        Xs.append(X[:, i:(i + time_step)])
    return np.array(Xs)

# Fungsi utama untuk menjalankan prediksi realtime
def real_time_prediction():
    while True:
        # Ambil 180 data terbaru per device dari MongoDB
        data = get_latest_data_per_device('datasets', devices_indices)
        print("Data retrieved:", data.shape)

        if data.shape[1] < 90:  # Pastikan jumlah data yang ada cukup untuk time_step
            print("Not enough data to form a dataset for prediction. Waiting for more data...")
            time.sleep(900)
            continue

        # Normalisasi data menggunakan StandardScaler yang telah disimpan
        scaled_input_data = input_scaler.transform(data.T)
        print("Data scaled.")

        # Membuat dataset untuk LSTM
        time_step = 90  # Interval waktu harus sama dengan yang digunakan pada pelatihan
        X_new = create_dataset(scaled_input_data.T, time_step)

        # Mengubah urutan dimensi agar sesuai dengan bentuk input yang diharapkan oleh model
        X_new = np.transpose(X_new, (0, 2, 1))

        if X_new.size == 0:
            print("Not enough data to form a dataset for prediction. Waiting for more data...")
            time.sleep(900)
            continue

        # Melakukan prediksi menggunakan model yang telah dilatih
        predictions = model.predict(X_new)

        # Mengembalikan data ke bentuk asli
        predictions = output_scaler.inverse_transform(predictions)

        print(f'Real-time predictions: {predictions}')

        # Generate Timestamps starting from the current time
        timestamps = [pd.Timestamp.now() + pd.Timedelta(hours=8*i) for i in range(len(predictions))]

        # Definisikan input_columns berdasarkan urutan perangkat yang ada
        input_columns = list(devices_indices.keys())

        # Memisahkan nilai prediksi menjadi index yang berbeda dan menyimpan ke MongoDB
        for i, prediction in enumerate(predictions):
            for j, col in enumerate(['N2', 'P2', 'K2']):
                predicted_data = {
                    'index_id': devices_indices[col][1],  # Ambil index_id yang sesuai dengan device
                    'device_id': devices_indices[col][0],  # Ambil device_id yang sesuai dengan device
                    'predicted_value': float(prediction[j]),  # Konversi nilai prediksi menjadi float
                    'timestamp': timestamps[i]  # Gunakan timestamp yang dihasilkan
                }
                db['nutrientforecasts'].insert_one(predicted_data)

        print("Hasil prediksi telah disimpan ke MongoDB di dokumen 'nutrientforecasts'")

        # Plotting hasil prediksi untuk N2, P2, dan K2
        output_columns = ['N2', 'P2', 'K2']
        dates = timestamps  # Menggunakan timestamps sebagai sumbu x
        y_values = predictions.T  # Menggunakan nilai prediksi sebagai sumbu y

        plt.figure(figsize=(15, 8))

        for i, col in enumerate(output_columns):
            plt.subplot(len(output_columns), 1, i + 1)
            plt.plot(dates, y_values[i, :], label=f'Predicted {col}', color='red')
            plt.xlabel('Date')
            plt.ylabel(f'PPM {col}')
            plt.title(f'Forecasting Nutrition Level {col}')
            plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.3f}'))  # Tambahkan ini
            plt.legend()

        plt.tight_layout()
        plt.show()

        # Tunggu beberapa waktu sebelum mengambil data baru (misalnya, 60 detik)
        time.sleep(900)

# Panggil fungsi untuk memulai prediksi waktu nyata
real_time_prediction()
