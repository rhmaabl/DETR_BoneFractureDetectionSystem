"""
Sistem Deteksi Fraktur Tulang pada Citra X-Ray
Menggunakan model Deep Learning berbasis PyTorch dengan arsitektur
Detection Transformer (DETR) dari HuggingFace.

Fitur:
- Deteksi fraktur otomatis pada citra X-Ray
- Pengaturan confidence threshold via sidebar
- Visualisasi bounding box dengan perbandingan Original vs Annotated
- Unduh hasil anotasi sebagai file gambar
- Riwayat deteksi per sesi
- Informasi metadata gambar
"""

import io
import logging
import time
from math import gcd
from typing import List, Dict, Tuple, Optional

import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms as torchvision_nms
from transformers import DetrForObjectDetection, DetrImageProcessor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi halaman Streamlit (wajib dipanggil sebelum elemen UI lainnya)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Fraktur Tulang",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Injeksi CSS untuk tampilan modern dan responsif
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ---- Import font ---- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* ---- Global ---- */
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ---- Latar utama ---- */
        .stApp {
            background: var(--background-color);
            color: var(--text-color);
        }

        /* ---- Judul utama ---- */
        .app-title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #60a5fa 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            margin-bottom: 0.3rem;
            animation: shimmer 3s ease-in-out infinite;
        }
        @keyframes shimmer {
            0%, 100% { background-position: 0% center; }
            50% { background-position: 200% center; }
        }

        /* ---- Sub-judul / deskripsi ---- */
        .subtitle {
            text-align: center;
            color: var(--text-color);
            opacity: 0.8;
            font-size: 1rem;
            margin-bottom: 2rem;
            line-height: 1.7;
            font-weight: 400;
        }

        /* ---- Kartu pembungkus konten ---- */
        .card {
            background: var(--secondary-background-color);
            border-radius: 16px;
            padding: 1.4rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(99, 102, 241, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .card:hover {
            border-color: rgba(99, 102, 241, 0.35);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.08);
        }

        /* ---- Label kecil di atas gambar ---- */
        .section-label {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #6366f1;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        /* ---- Tombol utama (Proses Deteksi) ---- */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #6366f1 100%);
            background-size: 200% auto;
            color: #ffffff;
            border: none;
            border-radius: 14px;
            padding: 0.85rem 2rem;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
            letter-spacing: 0.3px;
        }
        div.stButton > button:hover {
            background-position: right center;
            box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
            transform: translateY(-2px);
        }
        div.stButton > button:active {
            transform: translateY(0);
        }

        /* ---- Download tombol ---- */
        div.stDownloadButton > button {
            background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%) !important;
            border: 1px solid rgba(99, 102, 241, 0.3) !important;
            color: #3730a3 !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.25s ease !important;
        }
        div.stDownloadButton > button:hover {
            border-color: rgba(99, 102, 241, 0.6) !important;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15) !important;
            transform: translateY(-1px) !important;
        }

        /* ---- Gambar & border radius ---- */
        [data-testid="stImage"] > div {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(99, 102, 241, 0.1);
        }

        /* ---- Divider tipis ---- */
        hr {
            border-color: rgba(99, 102, 241, 0.1);
            margin: 1.5rem 0;
        }

        /* ---- Metrik kartu ---- */
        .metric-card {
            background: var(--secondary-background-color);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            padding: 1.3rem 1rem;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6366f1, #a78bfa);
            border-radius: 16px 16px 0 0;
        }
        .metric-card:hover {
            border-color: rgba(99, 102, 241, 0.35);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.1);
            transform: translateY(-2px);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #818cf8, #c4b5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1;
        }
        .metric-label {
            font-size: 0.68rem;
            color: var(--text-color);
            opacity: 0.72;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-top: 0.5rem;
            font-weight: 600;
        }

        /* ---- Tabel deteksi ---- */
        .det-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 0.75rem;
            font-size: 0.88rem;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(99, 102, 241, 0.15);
        }
        .det-table th {
            background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
            color: #4338ca;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 0.8rem 1rem;
            text-align: left;
            border-bottom: 2px solid rgba(99, 102, 241, 0.2);
        }
        .det-table td {
            padding: 0.7rem 1rem;
            border-bottom: 1px solid rgba(99, 102, 241, 0.08);
            color: var(--text-color);
        }
        .det-table tr:hover td {
            background-color: rgba(99, 102, 241, 0.05);
        }
        .det-table tr:last-child td {
            border-bottom: none;
        }

        /* ---- Confidence bar ---- */
        .conf-bar-bg {
            background-color: rgba(99, 102, 241, 0.1);
            border-radius: 6px;
            height: 6px;
            width: 100%;
            overflow: hidden;
        }
        .conf-bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ---- Sidebar styling ---- */
        section[data-testid="stSidebar"] {
            background: var(--secondary-background-color);
            border-right: 1px solid rgba(99, 102, 241, 0.18);
        }

        /* ---- Sidebar header ---- */
        .sidebar-header {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.8px;
            color: #6366f1;
            margin-bottom: 0.5rem;
            margin-top: 1.2rem;
        }

        /* ---- History item ---- */
        .history-item {
            background: var(--background-color);
            border: 1px solid rgba(99, 102, 241, 0.16);
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.5rem;
            font-size: 0.82rem;
            color: var(--text-color);
            transition: border-color 0.2s ease;
        }
        .history-item:hover {
            border-color: rgba(99, 102, 241, 0.3);
        }

        /* ---- Sidebar history buttons (override main button style) ---- */
        section[data-testid="stSidebar"] div.stButton > button {
            background: var(--background-color) !important;
            border: 1px solid rgba(99, 102, 241, 0.16) !important;
            border-radius: 10px !important;
            padding: 0.6rem 0.8rem !important;
            font-size: 0.82rem !important;
            color: var(--text-color) !important;
            text-align: left !important;
            box-shadow: none !important;
            font-weight: 500 !important;
            letter-spacing: 0 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            border-color: rgba(99, 102, 241, 0.3) !important;
            background: #f8faff !important;
            transform: none !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1) !important;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="secondary"][data-active="true"],
        section[data-testid="stSidebar"] div.stButton > button:focus {
            border-color: rgba(99, 102, 241, 0.5) !important;
            background: #eef2ff !important;
        }

        /* ---- Footer ---- */
        .footer {
            text-align: center;
            color: var(--text-color);
            opacity: 0.7;
            font-size: 0.75rem;
            margin-top: 3rem;
            padding: 1.8rem 0;
            border-top: 1px solid rgba(99, 102, 241, 0.1);
            letter-spacing: 0.3px;
        }

        /* ---- Image info badge ---- */
        .img-info {
            display: inline-block;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 8px;
            padding: 0.3rem 0.65rem;
            font-size: 0.76rem;
            color: #4338ca;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            font-weight: 500;
        }

        /* ---- Upload area ---- */
        [data-testid="stFileUploader"] {
            border-radius: 16px;
        }
        [data-testid="stFileUploader"] section {
            border-radius: 16px;
            border: 2px dashed rgba(99, 102, 241, 0.25);
            background: rgba(99, 102, 241, 0.03);
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: rgba(99, 102, 241, 0.4);
            background: rgba(99, 102, 241, 0.06);
        }

        /* ---- Tab styling ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: var(--secondary-background-color);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.88rem;
            padding: 0.5rem 1.2rem;
            color: var(--text-color);
            opacity: 0.8;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Konstanta default
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = "rahmabeee/bone-fracture-detr-v2"  # HuggingFace Hub model ID (checkpoint skenario terbaik, sudah diperbaiki)
DEFAULT_CONFIDENCE = 0.1           # Ambang batas confidence default
NMS_IOU_THRESHOLD = 0.5            # Ambang batas IoU untuk NMS (tidak dapat diubah via UI)
MAX_IMAGE_SIDE = 800               # Panjang sisi terpanjang untuk resize (px)
BOX_COLOR = (99, 102, 241)         # Warna bounding box (indigo/ungu, RGB)
OUTLINE_COLOR = (255, 255, 255)    # Tepi putih tipis agar garis ungu tetap tajam
TEXT_BG_COLOR = (99, 102, 241)     # Warna latar label teks
TEXT_COLOR = (255, 255, 255)       # Warna teks label
BOX_SCALE = 2.5                    # Pembesaran box saat digambar (1.0 = ukuran asli dari model)
MIN_BOX_SIDE_RATIO = 0.10          # Sisi minimum box gambar = 10% sisi terpendek citra

FRACTURE_LABEL = 0                 # Label 0 = fraktur pada model DETR
MIN_BOX_AREA_RATIO = 0.001        # Min rasio luas bbox/gambar (< 0.1% = noise piksel)
MAX_BOX_AREA_RATIO = 0.90         # Max rasio luas bbox/gambar (> 90% = seluruh gambar)
RAW_SCORE_THRESHOLD = 0.01        # Threshold rendah untuk mengumpulkan semua skor mentah
MIN_SUGGEST_GAP = 0.05            # Gap minimum antar skor untuk auto-suggest threshold
MAX_HISTORY_SIZE = 10             # Maksimum jumlah entri riwayat sesi


# ---------------------------------------------------------------------------
# Inisialisasi session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "selected_history" not in st.session_state:
    st.session_state.selected_history: Optional[int] = None


# ===========================================================================
# 1. FUNGSI MUAT MODEL
#    Menggunakan @st.cache_resource agar model hanya di-load sekali.
# ===========================================================================
@st.cache_resource(show_spinner="Memuat model DETR…")
def load_model() -> Tuple[DetrImageProcessor, DetrForObjectDetection]:
    """
    Memuat DetrImageProcessor dan DetrForObjectDetection dari HuggingFace Hub.
    Model di-set ke mode evaluasi untuk inferensi.

    Returns:
        Tuple berisi (processor, model).

    Raises:
        RuntimeError: Jika terjadi kegagalan saat memuat model.
    """
    try:
        logger.info("Memuat model DETR dari HuggingFace Hub: %s", MODEL_CHECKPOINT)

        # Muat processor untuk preprocessing gambar (resize, normalisasi, padding)
        processor = DetrImageProcessor.from_pretrained(MODEL_CHECKPOINT)

        # Muat bobot model DETR dari HuggingFace Hub
        model = DetrForObjectDetection.from_pretrained(MODEL_CHECKPOINT)

        # Set ke mode evaluasi: menonaktifkan dropout dan batch norm tracking
        model.eval()

        logger.info("Model DETR berhasil dimuat.")
        return processor, model

    except Exception as exc:
        logger.error("Gagal memuat model: %s", exc)
        raise


# ===========================================================================
# 2. FUNGSI UTILITAS — Non-Maximum Suppression (NMS)
# ===========================================================================
def apply_nms(
    detections: List[Dict], iou_threshold: float = NMS_IOU_THRESHOLD
) -> List[Dict]:
    """
    Menerapkan Non-Maximum Suppression untuk menyaring deteksi yang
    saling tumpang-tindih, sehingga hanya deteksi terbaik yang tersisa.

    Args:
        detections: List of dict dengan kunci 'box' (x1,y1,x2,y2) dan 'score'.
        iou_threshold: Batas IoU di atas mana deteksi dianggap duplikat.

    Returns:
        List deteksi setelah NMS.
    """
    if len(detections) == 0:
        return detections

    boxes = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)

    keep_indices = torchvision_nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep_indices.tolist()]


# ===========================================================================
# 3. FUNGSI AUTO-SUGGEST THRESHOLD
# ===========================================================================
def suggest_threshold(scores: List[float]) -> Optional[float]:
    """
    Menyarankan threshold optimal berdasarkan distribusi skor.

    Mencari gap terbesar antara skor berurutan untuk menemukan
    titik pemisah alami antara deteksi positif dan negatif.

    Args:
        scores: Daftar skor confidence mentah dari semua deteksi fraktur.

    Returns:
        Threshold yang disarankan, atau None jika tidak ada saran.
    """
    if len(scores) < 2:
        return None

    sorted_scores = sorted(scores)

    max_gap = 0.0
    best_threshold = None
    for i in range(len(sorted_scores) - 1):
        gap = sorted_scores[i + 1] - sorted_scores[i]
        if gap > max_gap:
            max_gap = gap
            best_threshold = (sorted_scores[i] + sorted_scores[i + 1]) / 2

    if max_gap < MIN_SUGGEST_GAP:
        return None

    return round(best_threshold, 2)


# ===========================================================================
# 4. FUNGSI PREPROCESSING & INFERENSI
#    Menangani konversi gambar → tensor, inferensi model, dan post-processing.
# ===========================================================================
def run_inference(
    image: Image.Image,
    processor: DetrImageProcessor,
    model: DetrForObjectDetection,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
) -> Dict:
    """
    Melakukan inferensi DETR pada gambar PIL yang diberikan.

    Args:
        image: Gambar asli dalam format PIL.Image (RGB).
        processor: DetrImageProcessor yang sudah di-load.
        model: DetrForObjectDetection yang sudah di-load.
        confidence_threshold: Ambang batas minimum confidence score.

    Returns:
        Dict dengan kunci:
        - 'detections': List of dict (box, score) terfilter dan terurut.
        - 'all_fracture_scores': Semua skor fraktur mentah (sebelum threshold user).
        - 'inference_time': Waktu inferensi dalam detik.
        - 'label_filtered': Jumlah deteksi yang dibuang karena bukan fraktur.
        - 'bbox_filtered': Jumlah deteksi yang dibuang karena ukuran bbox tidak valid.
    """
    orig_width, orig_height = image.size
    image_area = orig_width * orig_height
    logger.info(
        "Menjalankan inferensi - ukuran gambar: %dx%d, threshold: %.2f",
        orig_width,
        orig_height,
        confidence_threshold,
    )

    # --- Preprocessing ---
    inputs = processor(images=image, return_tensors="pt")

    # --- Inferensi dengan pengukuran waktu ---
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time

    # --- Post-Processing ---
    # Gunakan threshold rendah untuk mengumpulkan semua skor mentah
    target_sizes = torch.tensor([(orig_height, orig_width)])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=RAW_SCORE_THRESHOLD
    )

    result = results[0]
    scores = result["scores"]
    boxes = result["boxes"]
    labels = result["labels"]

    all_fracture_scores: List[float] = []
    detections: List[Dict] = []
    label_filtered = 0
    bbox_filtered = 0
    min_box_area = image_area * MIN_BOX_AREA_RATIO
    max_box_area = image_area * MAX_BOX_AREA_RATIO

    for score, box, label in zip(scores, boxes, labels):
        # --- Filter Label: hanya simpan kelas fraktur (label 0) ---
        if label.item() != FRACTURE_LABEL:
            label_filtered += 1
            continue

        conf = score.item()
        all_fracture_scores.append(conf)

        # --- Filter Threshold: terapkan threshold pengguna ---
        if conf < confidence_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()

        # Klem koordinat agar tidak keluar dari batas gambar
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(orig_width, int(x2))
        y2 = min(orig_height, int(y2))

        # --- Validasi Ukuran Bounding Box ---
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < min_box_area:
            bbox_filtered += 1
            logger.debug("Bbox terlalu kecil dibuang: area=%d, min=%d", box_area, int(min_box_area))
            continue
        if box_area > max_box_area:
            bbox_filtered += 1
            logger.debug("Bbox terlalu besar dibuang: area=%d, max=%d", box_area, int(max_box_area))
            continue

        detections.append({"box": (x1, y1, x2, y2), "score": conf})

    # Terapkan NMS untuk menghilangkan deteksi duplikat
    detections = apply_nms(detections, iou_threshold=NMS_IOU_THRESHOLD)

    # Urutkan berdasarkan score tertinggi
    detections.sort(key=lambda d: d["score"], reverse=True)

    logger.info(
        "Inferensi selesai dalam %.2fs - %d fraktur terdeteksi "
        "(label_filtered=%d, bbox_filtered=%d).",
        inference_time, len(detections), label_filtered, bbox_filtered,
    )

    return {
        "detections": detections,
        "all_fracture_scores": all_fracture_scores,
        "inference_time": inference_time,
        "label_filtered": label_filtered,
        "bbox_filtered": bbox_filtered,
    }


# ===========================================================================
# 5. FUNGSI VISUALISASI
#    Menggambar bounding box dan label pada gambar asli menggunakan PIL.
# ===========================================================================
def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Memuat font TrueType dari berbagai lokasi lintas-platform."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for fp in font_candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except IOError:
            continue
    # Fallback: Pillow >= 10.1 mendukung ukuran pada font bawaan
    try:
        return ImageFont.load_default(font_size)
    except TypeError:
        return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """
    Menggambar bounding box tebal beserta label confidence pada gambar.

    Args:
        image: Gambar asli dalam format PIL.Image.
        detections: List hasil dari run_inference().

    Returns:
        Gambar PIL dengan anotasi bounding box.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    W, H = annotated.size
    short = min(W, H)

    line_width = max(3, int(short * 0.004))     # ketebalan garis box
    font_size = max(18, int(short * 0.028))      # ukuran font label
    halo = 1              # tebal tepi hitam
    pad = max(4, font_size // 4)
    font = _load_font(font_size)

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        conf = det["score"]

        # Perbesar box (khusus tampilan) dari titik tengahnya agar area
        # fraktur terlihat di dalam box. Data deteksi asli tidak diubah.
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw = max((x2 - x1) * BOX_SCALE, short * MIN_BOX_SIDE_RATIO)
        bh = max((y2 - y1) * BOX_SCALE, short * MIN_BOX_SIDE_RATIO)
        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(W - 1, cx + bw / 2))
        y2 = int(min(H - 1, cy + bh / 2))

        # Tepi hitam di luar & dalam garis, lalu garis warna di tengahnya
        draw.rectangle(
            [x1 - halo, y1 - halo, x2 + halo, y2 + halo],
            outline=OUTLINE_COLOR, width=line_width + halo * 2,
        )
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_width)

        # Label: nomor urut + confidence
        label = f"#{idx} Fraktur {conf * 100:.1f}%"
        l, t, r, b = draw.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - t
        box_w, box_h = tw + pad * 2, th + pad * 2

        # Jaga label tetap di dalam gambar
        bx1 = min(max(0, x1), max(0, W - box_w))
        by1 = y1 - box_h - halo
        if by1 < 0:  # tidak muat di atas -> taruh di dalam box
            by1 = y1 + line_width

        draw.rectangle(
            [bx1, by1, bx1 + box_w, by1 + box_h],
            fill=TEXT_BG_COLOR, outline=OUTLINE_COLOR, width=1,
        )
        draw.text((bx1 + pad - l, by1 + pad - t), label, fill=TEXT_COLOR, font=font)

    return annotated


# ===========================================================================
# 6. FUNGSI UTILITAS — Konversi gambar ke bytes untuk unduhan
# ===========================================================================
def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Mengonversi PIL.Image ke bytes buffer untuk unduhan."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def get_file_size_str(size_bytes: int) -> str:
    """Mengonversi ukuran bytes ke string yang mudah dibaca."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_confidence_color(score: float) -> str:
    """Mengembalikan warna hex berdasarkan tingkat confidence."""
    if score >= 0.8:
        return "#22c55e"  # hijau
    if score >= 0.6:
        return "#f59e0b"  # kuning/orange
    return "#ef4444"      # merah


# ===========================================================================
# 7. FUNGSI TAMPILAN HASIL DETEKSI
#    Menampilkan ringkasan metrik, perbandingan citra, dan tombol unduh.
# ===========================================================================
def display_results(
    original_image: Image.Image,
    result_image: Image.Image,
    detections: List[Dict],
    inference_time: float,
    label_filtered: int,
    bbox_filtered: int,
    all_fracture_scores: List[float],
    confidence_threshold: float,
) -> None:
    """Menampilkan hasil deteksi fraktur secara lengkap di area utama."""

    # Informasi waktu inferensi
    st.markdown(
        f'<div class="card" style="text-align:center;">'
        f'⏱️ Waktu inferensi: <b>{inference_time:.2f} detik</b>'
        f' &nbsp;|&nbsp; 🏷️ Filter label (non-fraktur): <b>{label_filtered}</b>'
        f' &nbsp;|&nbsp; 📏 Filter ukuran bbox: <b>{bbox_filtered}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ----- Ringkasan Metrik -----
    avg_conf = (
        sum(d["score"] for d in detections) / len(detections)
        if detections
        else 0.0
    )
    max_conf = max((d["score"] for d in detections), default=0.0)
    min_conf = min((d["score"] for d in detections), default=0.0)

    st.markdown(
        '<p class="section-label">📊 Ringkasan Deteksi</p>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{len(detections)}</div>'
            f'<div class="metric-label">Fraktur Terdeteksi</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{avg_conf:.0%}</div>'
            f'<div class="metric-label">Rata-rata Confidence</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{max_conf:.0%}</div>'
            f'<div class="metric-label">Confidence Tertinggi</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{min_conf:.0%}</div>'
            f'<div class="metric-label">Confidence Terendah</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacing

    # ----- Perbandingan Citra -----
    st.markdown(
        '<p class="section-label">🖼️ Perbandingan Citra</p>',
        unsafe_allow_html=True,
    )
    col_orig, col_result = st.columns(2)
    with col_orig:
        st.markdown(
            '<p class="section-label">Citra Asli</p>',
            unsafe_allow_html=True,
        )
        st.image(original_image, use_container_width=True)
    with col_result:
        st.markdown(
            '<p class="section-label">Hasil Deteksi</p>',
            unsafe_allow_html=True,
        )
        st.image(result_image, use_container_width=True)

    # ----- Auto-Suggest Threshold -----
    suggested = suggest_threshold(all_fracture_scores)
    if suggested is not None and suggested != confidence_threshold:
        st.markdown(
            f'<div class="card" style="border-color: rgba(245,158,11,0.4);">'
            f'💡 <b>Saran Threshold:</b> Berdasarkan distribusi {len(all_fracture_scores)} '
            f'skor fraktur mentah, threshold optimal yang disarankan adalah '
            f'<b>{suggested:.2f}</b> (saat ini: {confidence_threshold:.2f}). '
            f'Atur slider di sidebar untuk menyesuaikan.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ----- Tombol unduh gambar hasil -----
    st.divider()
    dl_col1, dl_col2, _ = st.columns([1, 1, 2])
    with dl_col1:
        st.download_button(
            label="⬇️ Unduh Hasil (PNG)",
            data=image_to_bytes(result_image, "PNG"),
            file_name="hasil_deteksi_fraktur.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="⬇️ Unduh Hasil (JPEG)",
            data=image_to_bytes(result_image, "JPEG"),
            file_name="hasil_deteksi_fraktur.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )


# ===========================================================================
# SIDEBAR — Pengaturan
# ===========================================================================
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")

    st.markdown('<p class="sidebar-header">Parameter Deteksi</p>', unsafe_allow_html=True)

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.01,
        help="Ambang batas minimum confidence score. Deteksi di bawah nilai ini akan diabaikan.",
    )

    st.divider()

    # Riwayat deteksi
    st.markdown('<p class="sidebar-header">Riwayat Sesi</p>', unsafe_allow_html=True)
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), start=1):
            idx = len(st.session_state.history) - i
            count = entry["count"]
            avg = entry["avg_conf"]
            icon = "🔴" if count > 0 else "🟢"
            label = f"{icon} #{idx + 1} — {count} fraktur (avg {avg:.0%})"
            if st.button(label, key=f"history_{idx}", use_container_width=True):
                st.session_state.selected_history = idx
    else:
        st.caption("Belum ada deteksi yang dijalankan.")


# ===========================================================================
# ANTARMUKA PENGGUNA — Konten Utama
# ===========================================================================

# Judul dan deskripsi singkat
st.markdown('<h1 class="app-title">🦴 Deteksi Fraktur Tulang</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    "Sistem cerdas untuk mendeteksi fraktur tulang pada citra X-Ray "
    "menggunakan <b>DETR</b> (<i>Detection Transformer</i>).<br>"
    "Unggah citra &bull; Atur threshold di sidebar &bull; Klik deteksi"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# --- Komponen 1: File Uploader ---
st.markdown('<p class="section-label">📂 Input Citra X-Ray</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="Input Citra X-Ray",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Unggah file citra X-Ray dalam format JPG, JPEG, atau PNG.",
)

# --- Komponen 2: Preview gambar asli + info metadata ---
original_image: Optional[Image.Image] = None

if uploaded_file is not None:
    original_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    img_w, img_h = original_image.size
    file_size = uploaded_file.size

    st.markdown(
        '<p class="section-label">🖼️ Preview Citra & Informasi</p>',
        unsafe_allow_html=True,
    )

    col_img, col_info = st.columns([3, 1])

    with col_img:
        st.image(original_image, use_container_width=True)

    with col_info:
        g = gcd(img_w, img_h)
        st.markdown(
            f"""
            <div class="card">
                <p class="sidebar-header" style="margin-top:0">📋 Metadata Gambar</p>
                <span class="img-info">📐 {img_w} &times; {img_h} px</span><br>
                <span class="img-info">📁 {get_file_size_str(file_size)}</span><br>
                <span class="img-info">🎨 {original_image.mode}</span><br>
                <span class="img-info">📄 {uploaded_file.type or 'N/A'}</span><br>
                <span class="img-info">📏 Rasio {img_w // g}:{img_h // g}</span><br>
                <span class="img-info">🔢 {img_w * img_h:,} piksel</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# --- Komponen 3: Tombol Proses Deteksi ---
process_button = st.button("🔍 Proses Deteksi Fraktur", use_container_width=True)

# ===========================================================================
# LOGIKA BACKEND — dijalankan hanya saat tombol ditekan
# ===========================================================================
if process_button:
    # Validasi: pastikan pengguna sudah mengunggah gambar
    if original_image is None:
        st.warning("⚠️ Harap unggah citra X-Ray terlebih dahulu sebelum memproses.")
    else:
        try:
            # Muat model (di-cache sehingga hanya di-load sekali)
            processor, model = load_model()
        except Exception as exc:
            st.error(f"❌ Gagal memuat model: {exc}")
            st.stop()

        try:
            # Jalankan preprocessing dan inferensi
            with st.spinner("🔄 Menganalisis citra…"):
                inference_result = run_inference(
                    original_image,
                    processor,
                    model,
                    confidence_threshold=confidence_threshold,
                )
                detections = inference_result["detections"]
                all_fracture_scores = inference_result["all_fracture_scores"]
                inference_time = inference_result["inference_time"]
                label_filtered = inference_result["label_filtered"]
                bbox_filtered = inference_result["bbox_filtered"]
        except Exception as exc:
            st.error(f"❌ Terjadi kesalahan saat inferensi: {exc}")
            logger.exception("Inference error")
            st.stop()

        # Ambil hanya 1 deteksi dengan confidence tertinggi
        if detections:
            detections = [detections[0]]

        # Gambar bounding box pada gambar asli
        result_image = draw_detections(original_image, detections)

        # Simpan ke riwayat sesi (termasuk data lengkap untuk ditampilkan ulang)
        avg_conf = (
            sum(d["score"] for d in detections) / len(detections)
            if detections
            else 0.0
        )
        st.session_state.history.append({
            "count": len(detections),
            "avg_conf": avg_conf,
            "original_image": original_image.copy(),
            "result_image": result_image.copy(),
            "detections": detections,
            "inference_time": inference_time,
            "label_filtered": label_filtered,
            "bbox_filtered": bbox_filtered,
            "all_fracture_scores": all_fracture_scores,
            "confidence_threshold": confidence_threshold,
        })

        # Batasi jumlah riwayat untuk menghemat memori
        if len(st.session_state.history) > MAX_HISTORY_SIZE:
            st.session_state.history = st.session_state.history[-MAX_HISTORY_SIZE:]

        # Pilih otomatis entri riwayat terbaru
        st.session_state.selected_history = len(st.session_state.history) - 1

        # Tampilkan hasil deteksi
        display_results(
            original_image=original_image,
            result_image=result_image,
            detections=detections,
            inference_time=inference_time,
            label_filtered=label_filtered,
            bbox_filtered=bbox_filtered,
            all_fracture_scores=all_fracture_scores,
            confidence_threshold=confidence_threshold,
        )

elif st.session_state.selected_history is not None:
    # Tampilkan hasil dari riwayat yang dipilih
    idx = st.session_state.selected_history
    if 0 <= idx < len(st.session_state.history):
        entry = st.session_state.history[idx]
        st.info(f"📂 Menampilkan riwayat deteksi **#{idx + 1}**")
        display_results(
            original_image=entry["original_image"],
            result_image=entry["result_image"],
            detections=entry["detections"],
            inference_time=entry["inference_time"],
            label_filtered=entry["label_filtered"],
            bbox_filtered=entry["bbox_filtered"],
            all_fracture_scores=entry["all_fracture_scores"],
            confidence_threshold=entry["confidence_threshold"],
        )
    else:
        st.session_state.selected_history = None

# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown(
    '<div class="footer">'
    "🦴 Bone Fracture Detection System &mdash; "
    "Dibangun dengan Streamlit, PyTorch, dan HuggingFace Transformers"
    "</div>",
    unsafe_allow_html=True,
)
