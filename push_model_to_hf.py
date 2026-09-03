"""
Script sekali-jalan untuk push model hasil export (folder 'model_export/')
ke HuggingFace Hub, supaya app.py bisa memuatnya via MODEL_CHECKPOINT.

Cara pakai:
    1. Login dulu: huggingface-cli login   (atau isi token di bawah)
    2. Pastikan folder 'model_export/' berisi config.json, model.safetensors,
       preprocessor_config.json (ini adalah isi dari detr_fracatlas_fixed_hf.zip
       yang sudah dibuat sebelumnya -- ekstrak dulu ke folder ini).
    3. Jalankan: python push_model_to_hf.py
"""

from transformers import DetrForObjectDetection, DetrImageProcessor

MODEL_DIR = "model_export"          # folder hasil ekstrak detr_fracatlas_fixed_hf.zip
HF_REPO_ID = "nantarach/bone-fracture-detr-v2"  # ganti sesuai username & nama repo HF kamu

model = DetrForObjectDetection.from_pretrained(MODEL_DIR)
processor = DetrImageProcessor.from_pretrained(MODEL_DIR)

model.push_to_hub(HF_REPO_ID)
processor.push_to_hub(HF_REPO_ID)

print(f"Selesai. Model tersedia di: https://huggingface.co/{HF_REPO_ID}")
