# CBAM-ResNet18 pada Imagenette 160 px

Klasifikasi 10 kelas Imagenette menggunakan ResNet-18 dengan modul perhatian
CBAM, dilatih dari awal menggunakan PyTorch.

Dataset yang digunakan adalah **Imagenette versi 160 px**, dengan masukan model
**128 x 128 piksel** sesuai rancangan tugas.

## Menjalankan di Windows

Gunakan Python **3.13 64-bit** dan GPU NVIDIA. Jalankan perintah dari root proyek
menggunakan PowerShell.

**1. Siapkan environment dan dependensi**

```powershell
python3.13.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/pytorch-cu118.txt
.\.venv\Scripts\python.exe -m pip install -r requirements/common.txt
```

Jika `.venv` sudah tersedia, lewati perintah pembuatannya. Jika menggunakan Python
Launcher, perintah pertama dapat diganti dengan `py -3.13 -m venv .venv`.

**2. Siapkan dataset dan periksa GPU**

```powershell
.\.venv\Scripts\python.exe -m utils.download_data
.\.venv\Scripts\python.exe check_env.py --require-cuda
```

Pengunduhan dilewati jika dataset sudah lengkap. Pembagian data: **8.524 gambar
latih**, **948 validasi internal**, dan **3.928 evaluasi akhir**.

**3. Uji pipeline**

Uji ketiga kandidat learning rate dengan satu epoch dan seed 0:

```powershell
.\.venv\Scripts\python.exe tune_lr.py --epochs 1 --device cuda --run-name smoke-tuning
```

Hasil uji singkat ini hanya memeriksa alur, belum menentukan LR eksperimen utama.

**4. Pilih learning rate, lalu jalankan eksperimen utama**

Bandingkan LR **0.01, 0.05, dan 0.1**, masing-masing 30 epoch dengan seed 0:

```powershell
.\.venv\Scripts\python.exe tune_lr.py --device cuda --run-name lr-search-30epoch
```

Tuning hanya memakai data latih dan validasi internal, dengan pembagian data serta
inisialisasi yang sama. LR dipilih dari akurasi validasi terbaik; nilai seri
dipecahkan oleh loss lebih rendah, lalu LR lebih kecil. Perbandingan tersimpan di
`outputs/lr-search-30epoch/results.csv`.

Gunakan LR terpilih untuk pelatihan utama:

```powershell
$selectedLr = (Get-Content outputs/lr-search-30epoch/selected_lr.json -Raw | ConvertFrom-Json).learning_rate
.\.venv\Scripts\python.exe train.py --lr $selectedLr --device cuda --run-name cbam-full-30epoch
```

Konfigurasi bawaan: **30 epoch per seed**, seed **0, 1, 2**, batch **64**, SGD dengan
cosine annealing. Tanpa `--lr`, `train.py` memakai **0.01**. Gunakan nama run baru
untuk setiap percobaan agar hasil sebelumnya tetap tersimpan. Untuk melihat opsi lainnya,
jalankan `.\.venv\Scripts\python.exe train.py --help`.

## Google Colab

Buka [setup_colab.ipynb](notebooks/setup_colab.ipynb), pilih runtime GPU, lalu ikuti
sel secara berurutan. Notebook mendukung Python **3.12-3.13** dan meminta URL
repositori GitHub yang sudah diunggah. Sel tuning dan pelatihan utama ada di bagian
akhir; aktifkan masing-masing saat siap menjalankannya.

Unduh atau simpan folder hasil sebelum runtime Colab berakhir.

## Hasil pelatihan

Hasil disimpan otomatis di `outputs/<run-name>/`:

- `config.json` dan `environment.json`: pengaturan eksperimen dan perangkat.
- `summary.json` dan `summary.csv`: hasil per seed, rata-rata, dan simpangan baku.
- `seed_<n>/best.pt` dan `last.pt`: checkpoint terbaik dan terakhir.
- `seed_<n>/history.csv`: loss, akurasi, learning rate, waktu, dan memori setiap epoch.
- `seed_<n>/metrics.json`: akurasi dan F1-macro evaluasi akhir.
- Folder setiap seed juga berisi grafik pelatihan, confusion matrix, dan prediksi
  per gambar.

Checkpoint terbaik dipilih dari validasi internal, lalu dimuat kembali untuk
evaluasi akhir. Simpangan baku dihitung jika lebih dari satu seed selesai.
Dataset, hasil eksperimen, laporan dalam `docs/`, dan environment tetap lokal
serta diabaikan Git.

## Struktur utama

```text
train.py        Pelatihan dan evaluasi
tune_lr.py      Pencarian learning rate dari validasi internal
check_env.py    Pemeriksaan environment dan GPU
models/         Definisi ResNet-18 dan CBAM
utils/          Pengolahan data dan penyimpanan hasil
notebooks/      Setup Google Colab
requirements/   Daftar dependensi
tests/          Pengujian pipeline
```

## Pengujian

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

12 tes regresi dan uji tuning tiga LR, masing-masing satu epoch pada RTX 3050,
sudah berhasil. Tuning dan eksperimen utama 30 epoch, implementasi ablasi, serta
pengujian langsung di Colab belum dilakukan.

## Referensi

- [Paper CBAM](https://arxiv.org/abs/1807.06521)
- [Dataset Imagenette](https://github.com/fastai/imagenette)
