import os
import sys
import io
from pathlib import Path
from typing import Tuple, Optional

from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def get_app_dir() -> Path:
    """
    Retorna a pasta 'real' onde o app está:
    - Rodando como .py: pasta do arquivo app.py
    - Rodando como .exe (PyInstaller): pasta do executável
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

APP_DIR = get_app_dir()

DEFAULT_INPUT = APP_DIR / "Imagens originais"
DEFAULT_OUTPUT = APP_DIR / "Novas imagens"

INPUT_DIR = DEFAULT_INPUT
OUTPUT_DIR = DEFAULT_OUTPUT

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Template folder: em .exe, os arquivos são extraídos para sys._MEIPASS
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    TEMPLATE_DIR = Path(sys._MEIPASS) / "templates"
else:
    TEMPLATE_DIR = APP_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# =========================
# Presets
# =========================
PRESETS = {
    "youtube_thumb_1280x720_1_5mb": {
        "label": "YouTube Thumbnail (1280x720, 1,0–1,5MB)",
        "size": (1280, 720),
        "min_mb": 1.0,
        "max_mb": 1.5,
        "format": "JPEG",
        "fit": "cover",
    },
    "instagram_square_1080": {
        "label": "Instagram Quadrado (1080x1080)",
        "size": (1080, 1080),
        "min_mb": None,
        "max_mb": None,
        "format": "JPEG",
        "fit": "cover",
    },
    "instagram_story_1080x1920": {
        "label": "Instagram Story/Reels (1080x1920)",
        "size": (1080, 1920),
        "min_mb": None,
        "max_mb": None,
        "format": "JPEG",
        "fit": "cover",
    },
    "facebook_cover_820x312": {
        "label": "Facebook Capa (820x312)",
        "size": (820, 312),
        "min_mb": None,
        "max_mb": None,
        "format": "JPEG",
        "fit": "cover",
    },
    "custom": {
        "label": "Personalizado (defina largura/altura)",
        "size": None,
        "min_mb": None,
        "max_mb": None,
        "format": "JPEG",
        "fit": "cover",
    }
}


def safe_open_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    # Corrige rotação (câmera/celular)
    img = ImageOps.exif_transpose(img)
    # Padroniza para JPEG
    return img.convert("RGB")


def resize_image(img: Image.Image, size: Tuple[int, int], fit: str) -> Image.Image:
    """
    Resize to exact target size.

    - cover  : fills entire canvas (may crop) ✅ melhor para thumbnails
    - contain: mantém 100% da imagem e preenche o resto com blur (evita barras pretas)
    """
    if fit == "contain":
        fg = ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)

        bg = ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=18))
        bg = ImageEnhance.Brightness(bg).enhance(0.75)

        x = (size[0] - fg.size[0]) // 2
        y = (size[1] - fg.size[1]) // 2
        bg.paste(fg, (x, y))
        return bg

    return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def encode_jpeg_to_range(img: Image.Image, min_bytes: Optional[int], max_bytes: Optional[int]):
    """
    Tenta salvar JPEG dentro da faixa [min_bytes, max_bytes] ajustando a qualidade.
    Retorna (bytes, qualidade, status).

    status:
      - "ok"        (ficou na faixa)
      - "below_min" (mesmo no máximo ficou pequeno)
      - "above_max" (mesmo no mínimo ficou grande)
    """
    def save_with_quality(q: int) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        return buf.getvalue()

    if min_bytes is None and max_bytes is None:
        data = save_with_quality(90)
        return data, 90, "ok"

    q_min, q_max = 30, 95

    data_low = save_with_quality(q_min)
    data_high = save_with_quality(q_max)

    if max_bytes is not None and len(data_low) > max_bytes:
        return data_low, q_min, "above_max"

    if min_bytes is not None and len(data_high) < min_bytes:
        return data_high, q_max, "below_min"

    if min_bytes is not None and max_bytes is not None:
        target = (min_bytes + max_bytes) // 2
    elif max_bytes is not None:
        target = max_bytes
    else:
        target = min_bytes

    best = None  # (diff, data, q)
    lo, hi = q_min, q_max

    for _ in range(12):
        mid = (lo + hi) // 2
        data = save_with_quality(mid)
        size = len(data)

        in_range = True
        if min_bytes is not None and size < min_bytes:
            in_range = False
        if max_bytes is not None and size > max_bytes:
            in_range = False

        if in_range:
            diff = abs(size - target)
            if best is None or diff < best[0]:
                best = (diff, data, mid)

            if size < target:
                lo = mid + 1
            else:
                hi = mid - 1
        else:
            if max_bytes is not None and size > max_bytes:
                hi = mid - 1
            else:
                lo = mid + 1

    if best is not None:
        _, data, q = best
        return data, q, "ok"

    data = save_with_quality(90)
    status = "ok"
    if min_bytes is not None and len(data) < min_bytes:
        status = "below_min"
    if max_bytes is not None and len(data) > max_bytes:
        status = "above_max"
    return data, 90, status


def process_one_image_bytes(
    file_bytes: bytes,
    out_path: Path,
    size: Tuple[int, int],
    fit: str,
    min_mb: Optional[float],
    max_mb: Optional[float],
):
    img = safe_open_image(file_bytes)
    img2 = resize_image(img, size=size, fit=fit)

    min_bytes = None if min_mb is None else int(float(min_mb) * 1024 * 1024)
    max_bytes = None if max_mb is None else int(float(max_mb) * 1024 * 1024)

    data, q, status = encode_jpeg_to_range(img2, min_bytes=min_bytes, max_bytes=max_bytes)
    out_path.write_bytes(data)

    return {
        "output": str(out_path),
        "bytes": len(data),
        "mb": round(len(data) / (1024 * 1024), 3),
        "quality": q,
        "status": status,
    }


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@app.route("/", methods=["GET"])
def index():
    presets_for_ui = {k: v["label"] for k, v in PRESETS.items()}
    return render_template("index.html", presets=presets_for_ui)


@app.route("/batch", methods=["POST"])
def batch_process():
    payload = request.get_json(force=True)

    preset_key = payload.get("preset")
    fit = payload.get("fit", "cover")

    if preset_key not in PRESETS:
        return jsonify({"ok": False, "error": "Preset inválido"}), 400

    preset = PRESETS[preset_key]

    if preset_key == "custom":
        w = int(payload.get("width", 0))
        h = int(payload.get("height", 0))
        if w <= 0 or h <= 0:
            return jsonify({"ok": False, "error": "Largura/altura inválidas"}), 400

        size = (w, h)
        min_mb = payload.get("min_mb")
        max_mb = payload.get("max_mb")

        min_mb = None if min_mb in [None, "", 0, "0"] else float(min_mb)
        max_mb = None if max_mb in [None, "", 0, "0"] else float(max_mb)

    else:
        size = preset["size"]
        min_mb = preset.get("min_mb")
        max_mb = preset.get("max_mb")
        fit = payload.get("fit", preset.get("fit", "cover"))

    files = [p for p in INPUT_DIR.iterdir() if p.is_file() and is_image_file(p)]
    if not files:
        return jsonify({"ok": False, "error": f"Nenhuma imagem encontrada em: {INPUT_DIR}"}), 200

    results = []
    for p in files:
        out_name = f"{p.stem}__{size[0]}x{size[1]}.jpg"
        out_path = OUTPUT_DIR / out_name
        try:
            res = process_one_image_bytes(
                p.read_bytes(),
                out_path,
                size=size,
                fit=fit,
                min_mb=min_mb,
                max_mb=max_mb,
            )
            res["input"] = str(p)
            results.append({"ok": True, **res})
        except Exception as e:
            results.append({"ok": False, "input": str(p), "error": str(e)})

    return jsonify({
        "ok": True,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "count": len(results),
        "results": results
    })


@app.route("/upload", methods=["POST"])
def upload_process():
    preset_key = request.form.get("preset", "")
    fit = request.form.get("fit", "cover")

    if preset_key not in PRESETS:
        return jsonify({"ok": False, "error": "Preset inválido"}), 400

    preset = PRESETS[preset_key]

    if preset_key == "custom":
        w = int(request.form.get("width", 0))
        h = int(request.form.get("height", 0))
        if w <= 0 or h <= 0:
            return jsonify({"ok": False, "error": "Largura/altura inválidas"}), 400
        size = (w, h)

        min_mb = request.form.get("min_mb")
        max_mb = request.form.get("max_mb")

        min_mb = None if min_mb in [None, "", "0", 0] else float(min_mb)
        max_mb = None if max_mb in [None, "", "0", 0] else float(max_mb)

    else:
        size = preset["size"]
        min_mb = preset.get("min_mb")
        max_mb = preset.get("max_mb")
        fit = request.form.get("fit", preset.get("fit", "cover"))

    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "error": "Nenhum arquivo enviado"}), 400

    results = []
    for f in files:
        try:
            raw = f.read()
            stem = Path(f.filename).stem
            out_name = f"{stem}__{size[0]}x{size[1]}.jpg"
            out_path = OUTPUT_DIR / out_name

            res = process_one_image_bytes(
                raw,
                out_path,
                size=size,
                fit=fit,
                min_mb=min_mb,
                max_mb=max_mb,
            )

            res["input"] = f.filename
            results.append({"ok": True, **res})
        except Exception as e:
            results.append({"ok": False, "input": f.filename, "error": str(e)})

    return jsonify({
        "ok": True,
        "output_dir": str(OUTPUT_DIR),
        "count": len(results),
        "results": results
    })


if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    url = "http://127.0.0.1:5000"
    Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=5000, debug=False)
