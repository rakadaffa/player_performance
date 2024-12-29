import numpy as np
import pandas as pd

# 1. Memuat Data dari File CSV
PathFile = r"D:\\Algeo\\Makalah\\Cavs_Stats.csv" 
DataFrame = pd.read_csv(PathFile)

# Ambil nama pemain dan statistik
NamaPemain = DataFrame["Player"].values
Statistik = DataFrame.columns[1:]

# Filter data: hanya pemain dengan menit per pertandingan di atas rata-rata
RataRataMenit = DataFrame["Min"].mean()
DataFrameTersaring = DataFrame[DataFrame["Min"] > RataRataMenit]

# Kolom yang digunakan untuk matriks
KolomDipilih = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG%", "3P%", "FT%", "+/-"]
NamaPemain = DataFrameTersaring["Player"].values
Statistik = KolomDipilih
DataTersaring = DataFrameTersaring[KolomDipilih].values

# Membalik tanda untuk statistik buruk (TOV dan +/- negatif jadi positif)
DataTersaring[:, KolomDipilih.index("TOV")] *= -1

# Normalisasi Data (z-score normalization)
RataRataData = np.mean(DataTersaring, axis=0)
StandarDeviasiData = np.std(DataTersaring, axis=0)
DataTernormalisasi = (DataTersaring - RataRataData) / StandarDeviasiData

# Menampilkan data hasil matriks normalisasi
DataFrameNormalisasi = pd.DataFrame(DataTernormalisasi, index=NamaPemain, columns=Statistik)
print("Data setelah normalisasi (z-score):")
print(DataFrameNormalisasi.to_string(float_format="%.2f"))

# Dekomposisi SVD
MatriksU, NilaiSingular, MatriksVT = np.linalg.svd(DataTernormalisasi, full_matrices=False)

# Menampilkan matriks hasil dekomposisi SVD
print("\nMatriks U:")
print(pd.DataFrame(MatriksU, index=NamaPemain, columns=[f"PC{i+1}" for i in range(MatriksU.shape[1])]).to_string(float_format="%.2f"))

print("\nMatriks Sigma:")
print(pd.DataFrame(np.diag(NilaiSingular), columns=[f"PC{i+1}" for i in range(len(NilaiSingular))], index=[f"PC{i+1}" for i in range(len(NilaiSingular))]).to_string(float_format="%.2f"))

print("\nMatriks VT:")
print(pd.DataFrame(MatriksVT, columns=Statistik, index=[f"PC{i+1}" for i in range(MatriksVT.shape[0])]).to_string(float_format="%.2f"))

# Identifikasi komponen utama yang relevan
VariansiDijelaskan = (NilaiSingular ** 2) / np.sum(NilaiSingular ** 2)
VariansiKumulatif = np.cumsum(VariansiDijelaskan)

print("\nVariansi yang dijelaskan oleh setiap komponen utama:")
for i, (Var, VarKum) in enumerate(zip(VariansiDijelaskan, VariansiKumulatif), start=1):
    print(f"Komponen {i}: {Var:.2%} (kumulatif: {VarKum:.2%})")

# Pilih jumlah komponen utama yang relevan
JumlahKomponenRelevan = np.argmax(VariansiKumulatif >= 0.9) + 1
print(f"\nJumlah komponen utama yang relevan: {JumlahKomponenRelevan}")

# Visualisasi kontribusi relatif pemain
MatriksURingkas = MatriksU[:, :JumlahKomponenRelevan]
Kontribusi = np.linalg.norm(MatriksURingkas, axis=1)
KontribusiRelatif = (Kontribusi / np.sum(Kontribusi)) * 100

# Membuat DataFrame untuk tabel kontribusi
TabelKontribusi = pd.DataFrame({
    "Nama Pemain": NamaPemain,
    "Kontribusi (%)": KontribusiRelatif
})

# Menampilkan tabel kontribusi relatif
print("\nTabel Kontribusi Relatif Pemain (Total 100%):")
print(TabelKontribusi.to_string(index=False, float_format="%.2f"))
