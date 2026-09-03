# 🦴 Bone Fracture Detection System

Sistem deteksi fraktur tulang pada citra X-Ray menggunakan model Deep Learning berbasis **DETR** (*Detection Transformer*) dari HuggingFace, dengan antarmuka web interaktif yang dibangun menggunakan **Streamlit**.

> Catatan: repo ini adalah versi bersih (clean rebuild) yang menggunakan checkpoint
> `detr-periodic-epoch299.ckpt` (skenario terbaik: split 70:15:15, learning rate 1×10⁻⁴,
> 300 epoch, AP@[0.50:0.95] = 0,144), yang telah diverifikasi utuh (530/530 bobot
> ter-load, tanpa mismatch) dan bebas dari bug `category_id`.

---

## ✨ Fitur

| Fitur | Deskripsi |
|---|---|
| **Deteksi Otomatis** | Deteksi area fraktur pada citra X-Ray menggunakan model DETR |
| **Confidence Threshold** | Slider untuk mengatur ambang batas minimum confidence score |
| **Non-Maximum Suppression** | Menghilangkan deteksi duplikat yang saling tumpang-tindih (otomatis) |
| **Perbandingan Citra** | Tampilan side-by-side antara citra asli dan hasil deteksi |
| **Metrik Ringkasan** | Kartu metrik: jumlah fraktur, rata-rata confidence, tertinggi, terendah |
| **Metadata Gambar** | Informasi dimensi, ukuran file, format, dan rasio aspek |
| **Unduh Hasil** | Unduh gambar hasil anotasi dalam format PNG atau JPEG |
| **Riwayat Sesi** | Sidebar menampilkan riwayat deteksi selama sesi berjalan |

---

## 🚀 Instalasi & Menjalankan

### Prasyarat
- Python 3.9+
- Koneksi internet (model akan diunduh otomatis dari HuggingFace Hub:
  [`rahmabeee/bone-fracture-detr-v2`](https://huggingface.co/rahmabeee/bone-fracture-detr-v2))

### Langkah Instalasi

```bash
git clone https://github.com/rhmaabl/DETR_BoneFractureDetectionSystem.git
cd DETR_BoneFractureDetectionSystem

python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`.

---

## 🔧 Konfigurasi

Variabel penting di bagian atas `app.py`:

| Variabel | Default | Deskripsi |
|---|---|---|
| `MODEL_CHECKPOINT` | `"rahmabeee/bone-fracture-detr-v2"` | ID model HuggingFace Hub |
| `DEFAULT_CONFIDENCE` | `0.1` | Nilai default slider confidence threshold |
| `NMS_IOU_THRESHOLD` | `0.5` | Ambang batas IoU untuk Non-Maximum Suppression |
| `FRACTURE_LABEL` | `0` | ID label kelas fraktur pada model DETR |

---

## 🛠️ Teknologi

- **Model**: DETR (*DEtection TRansformer*) — arsitektur end-to-end object detection
- **Framework**: PyTorch + HuggingFace Transformers
- **UI**: Streamlit
- **Visualisasi**: Pillow (PIL)
