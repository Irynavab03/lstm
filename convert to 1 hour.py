import pandas as pd

# Path file Excel yang akan diolah
file_path_new = r'C:\\Users\\Windows10\\OneDrive\\Dokumen\\farming Irynaak\\lstm\\raw data pengujian.xlsx'

# Load the data from the first sheet of the Excel file
df_new = pd.read_excel(file_path_new, sheet_name='EMA_Filtered 25')

# Calculate the mean for every 288 rows for each column except 'date'
means_new = df_new.groupby(df_new.index // 40).mean(numeric_only=True)

# Buat kolom 'age' dengan nilai awal 30 dan bertambah 1 setiap 1 baris
means_new['age'] = 120 + (means_new.index // 3)

# Menyimpan hasil ke dalam sheet baru di file Excel yang sama
with pd.ExcelWriter(file_path_new, mode='a', engine='openpyxl') as writer:
    means_new.to_excel(writer, sheet_name='pengujian 8 jam 25 new', index=False)

print(f"Data telah disimpan di sheet 'pelatihan' dalam file:\n{file_path_new}")
