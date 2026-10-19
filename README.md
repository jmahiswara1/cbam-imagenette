# CBAM-ResNet18 pada Imagenette 160 px

Proyek klasifikasi 10 kelas Imagenette menggunakan ResNet-18 dengan Convolutional
Block Attention Module (CBAM), diimplementasikan dengan PyTorch dari bobot acak.
Nama repositori: `cbam-resnet18-imagenette-160`.

Dataset sumber menggunakan versi **Imagenette 160 px**, sedangkan masukan model
berukuran **128 x 128** sesuai rancangan tugas. Versi 160 px mempertahankan rasio
aspek dengan sisi terpendek 160 piksel; preprocessing menghasilkan tensor persegi.

## Struktur

```text
CBAM/
├── train.py                 # Pelatihan model penuh
├── ablasi.py                # Belum diimplementasikan
├── check_env.py             # Pemeriksaan environment dan satu langkah model
├── models/                 # Definisi ResNet-18 dan CBAM
├── utils/                  # Loader, lokasi data, dan pengunduh dataset
├── data/imagenette2-160/    # Dataset lokal; diabaikan Git
├── docs/                   # Laporan lokal; diabaikan Git
├── notebooks/              # Setup Google Colab
├── requirements/           # Dependensi bersama dan PyTorch CUDA
├── outputs/                # Laporan pemeriksaan dan hasil komputasi lokal
├── .venv/                  # Environment aktif; diabaikan Git
└── env/                    # Cadangan environment lama; diabaikan Git
```

Folder `docs/` tetap lokal sesuai pilihan kelompok. PDF laporan tidak ikut
diunggah ke GitHub, dan tidak tersedia dari hasil clone repositori.

## Penyiapan Windows

Gunakan Python **3.13 64-bit**. Dukungan kode mencakup Python 3.12 dan 3.13.
Jalankan perintah berikut dari root proyek menggunakan PowerShell. Pada laptop
pengembangan, Python tersedia melalui `python3.13.exe`; instalasi dengan Python
Launcher dapat memakai `py -3.13` sebagai pengganti perintah pertama.

```powershell
python3.13.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/pytorch-cu118.txt
.\.venv\Scripts\python.exe -m pip install -r requirements/common.txt
.\.venv\Scripts\python.exe -m pip check
```

Jika `.venv` sudah disiapkan, lewati perintah pembuatannya. Semua perintah di atas
langsung memakai interpreter environment proyek, sehingga aktivasi PowerShell
tidak diperlukan. Folder `env` lama tetap menjadi cadangan dan tidak digunakan.

PyTorch dipatok pada **2.7.1+cu118**, dengan **torchvision 0.22.1+cu118** dari indeks
resmi. Daftar `requirements/common.txt` mengunci dependensi bersama hasil instalasi
bersih. Dependensi platform Linux tambahan ditentukan oleh distribusi PyTorch;
file tersebut bukan salinan seluruh paket bawaan Colab.

## Dataset

Dataset yang sudah ada dapat langsung dipakai. Pada clone baru, unduh dengan:

```powershell
.\.venv\Scripts\python.exe -m utils.download_data
```

Perintah ini memeriksa 10 folder kelas dan jumlah gambar resmi, lalu melewati
pengunduhan jika dataset sudah lengkap. Dataset baru diunduh dan diperiksa dalam
folder sementara sebelum dipindahkan ke tujuan. Folder yang sudah ada tetapi
tidak lengkap tidak ditimpa: pindahkan folder tersebut atau tentukan tujuan baru.
Pemeriksaan kelengkapan ini tidak memverifikasi checksum atau mendekode semua gambar.

```powershell
.\.venv\Scripts\python.exe -m utils.download_data --data-dir "D:\Datasets\imagenette2-160"
```

Pembagian data yang digunakan:

| Sumber / penggunaan | Jumlah gambar |
| --- | ---: |
| `train/` resmi | 9.472 |
| Pelatihan, 90% dari `train/` | 8.524 |
| Validasi internal, 10% terstratifikasi dari `train/` | 948 |
| `val/` resmi, dipakai sebagai evaluasi akhir | 3.928 |

Lokasi dataset bawaan dihitung dari lokasi proyek, sehingga pemanggilan skrip
melalui path absolut juga berfungsi dari direktori terminal lain. Argumen
`data_dir` pada `get_dataloaders` tetap menerima string atau `Path`; path kustom
yang relatif dihitung terhadap direktori terminal. Opsi `num_workers` bernilai
default `2`, sementara pemeriksaan singkat memakai `0`.

## Pemeriksaan kesiapan

```powershell
.\.venv\Scripts\python.exe check_env.py --require-cuda
```

Pemeriksaan mencatat versi Python dan paket, versi driver NVIDIA, CUDA yang
digunakan PyTorch, GPU, jumlah data, dan bentuk tensor. Satu batch kecil berukuran
2 melewati forward, cross-entropy, backward, dan pembaruan parameter. Input harus
berbentuk `[2, 3, 128, 128]` dan output `[2, 10]`, dengan loss dan gradien finite.
Model pemeriksaan sementara tidak disimpan sebagai checkpoint pelatihan.

Hasil tersimpan di `outputs/environment-local.json`. Exit code `0` berarti semua
pemeriksaan berhasil; exit code `1` berarti gagal, dengan alasan di terminal dan
JSON. `--require-cuda` membuat ketiadaan GPU menjadi kegagalan. Untuk pemeriksaan
CPU, hilangkan opsi tersebut pada lingkungan tanpa GPU CUDA.

```powershell
.\.venv\Scripts\python.exe check_env.py --require-cuda --data-dir "D:\Datasets\imagenette2-160" --output outputs/environment-custom.json
```

Waktu dan memori dalam JSON merupakan hasil **pemeriksaan batch kecil**, bukan
pengukuran pelatihan 30 epoch atau hasil eksperimen dalam laporan.

## Google Colab

1. Buka `notebooks/setup_colab.ipynb` di Google Colab.
2. Pilih runtime GPU dan jalankan sel secara berurutan.
3. Masukkan URL HTTPS repositori GitHub yang dapat diakses ketika diminta.
   Repositori perlu diunggah ke GitHub terlebih dahulu untuk jalur clone ini.
4. Notebook membuat `.venv` Linux terpisah dari paket bawaan Colab, memasang
   daftar dependensi proyek, mengunduh dataset jika belum ada, lalu memeriksa GPU.

Notebook memakai Python 3.12 atau 3.13 dari runtime Colab; versi lain dihentikan
dengan pesan penjelasan. Kode arsitektur dan loader diimpor dari repositori,
bukan disalin ke notebook. Laporan tersimpan di `outputs/environment-colab.json`.
Notebook menjalankan pemeriksaan kesiapan; pelatihan penuh dijalankan terpisah.

File runtime Colab dan hasil pada `outputs` perlu disimpan atau diunduh sebelum
runtime dihapus. Notebook ini tidak menghubungkan Google Drive secara otomatis.

## Pelatihan

Setelah pemeriksaan berhasil, jalankan:

```powershell
.\.venv\Scripts\python.exe train.py
```

Di Colab, dari root clone proyek:

```bash
.venv/bin/python train.py
```

Konfigurasi pelatihan dipertahankan: CBAM penuh, 10 kelas, input 128 x 128, batch
64, 30 epoch untuk masing-masing seed `0, 1, 2`, SGD dengan learning rate `0.01`,
momentum `0.9`, weight decay `5e-4`, serta cosine annealing. Skrip memilih CUDA
jika tersedia dan CPU jika tidak.

Tahap penataan ini belum mencakup pelatihan penuh, implementasi ablasi,
checkpoint, grafik, F1-macro, confusion matrix, atau penyuntingan laporan.
`train.py` yang ada masih mencetak hasil ke terminal.

## Status pemeriksaan awal

- Windows: Python 3.13.12, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118,
  scikit-learn 1.9.1; `pip check` berhasil.
- RTX 3050: satu batch gambar nyata berhasil melewati forward, backward, dan
  pembaruan parameter; laporan lengkap berada di `outputs/environment-local.json`.
- Loader berhasil membaca data dari direktori kerja lain, termasuk dengan
  `num_workers=2` pada Windows. Ketiadaan CUDA menghasilkan kegagalan yang jelas
  ketika `--require-cuda` digunakan.
- Sintaks dan urutan sel notebook diperiksa. Resolusi wheel dependensi bersama
  untuk Linux Python 3.12 dan 3.13 berhasil melalui dry-run pip.
- Sesi Google Colab belum dijalankan langsung; hasil dry-run bukan bukti
  keberhasilan runtime Colab atau pelatihan penuh.

## Referensi

- [Paper CBAM](https://arxiv.org/abs/1807.06521)
- [Dataset Imagenette dan definisi ukuran 160 px](https://github.com/fastai/imagenette)
- [Pasangan versi PyTorch 2.7.1 dan torchvision 0.22.1](https://pytorch.org/get-started/previous-versions/#v271)
