import sys
import json
import math

import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import transformers.dynamic_module_utils
import ollama

# --- HACK TO BYPASS FLASH_ATTN STRICT IMPORT CHECK ---
original_check_imports = transformers.dynamic_module_utils.check_imports


def custom_check_imports(filename):
    try:
        return original_check_imports(filename)
    except ImportError as e:
        if "flash_attn" in str(e):
            return []
        raise e


transformers.dynamic_module_utils.check_imports = custom_check_imports
# -----------------------------------------------------

FLORENCE_ID = "microsoft/Florence-2-large"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Set to True to show the raw Florence OCR text in a debug textbox in the UI.
DEBUG = False

# ---------------------------------------------------------------------------
# STAGE 1 — Florence-2 OCR (loaded once, runs on your local GPU)
# ---------------------------------------------------------------------------

_FLOR = None
_PROC = None


def load_florence():
    global _FLOR, _PROC
    if _FLOR is None:
        print("Loading Florence-2 ...", file=sys.stderr)
        _FLOR = AutoModelForCausalLM.from_pretrained(
            FLORENCE_ID, torch_dtype=DTYPE, trust_remote_code=True
        ).to(DEVICE)
        _PROC = AutoProcessor.from_pretrained(
            FLORENCE_ID, trust_remote_code=True)
    return _FLOR, _PROC


def run_florence(image_path):
    model, processor = load_florence()
    image = Image.open(image_path).convert("RGB")

    prompt = "<OCR_WITH_REGION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE, DTYPE) if v.dtype == torch.float32 else v.to(DEVICE)
              for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=prompt, image_size=(image.width, image.height))

    region_data = parsed.get("<OCR_WITH_REGION>", {})
    labels = region_data.get("labels", [])
    quad_boxes = region_data.get("quad_boxes", [])

    parsed_lines = []
    for i, (label, coords) in enumerate(zip(labels, quad_boxes)):
        if len(coords) == 8:
            x_center = sum(coords[0::2]) / 4
            y_center = sum(coords[1::2]) / 4
        elif len(coords) == 4:
            x_center = (coords[0] + coords[2]) / 2
            y_center = (coords[1] + coords[3]) / 2
        else:
            x_center, y_center = 0, 0

        x_int, y_int = int(x_center), int(y_center)
        parsed_lines.append({
            "id": i,
            "x": x_int,
            "y": y_int,
            "text": label,
            "raw_line": f"Line {i} | [X:{x_int:04d}, Y:{y_int:04d}] {label}",
        })

    return parsed_lines

# ---------------------------------------------------------------------------
# STAGE 2 — Qwen via local Ollama
# ---------------------------------------------------------------------------

EXPIRY_PROMPT = """You are given numbered OCR text lines from a photo of a medicine box.
Find the EXPIRY DATE. It may be labelled in several languages, including:
EXP, EXPIRY, USE BY, BEST BEFORE, PER, PER:, PÉR, PÉREMPTION, CAD, CADUCITE, VENC, VENCIMENTO,
or it may just be a standalone date like MM/YY or MM-YYYY.
Do NOT confuse it with other labels such as PPV (price) or LOT (batch number).

If MULTIPLE dates are present (e.g. a manufacturing date and an expiry date), the EXPIRY DATE is
ALWAYS the LATEST (largest/furthest-in-the-future) date. Compare all dates and pick the maximum.

OCR TEXT:
\"\"\"
{ocr_text}
\"\"\"

Respond with STRICT JSON ONLY. Extract the date and the Line ID(s) where it was found.
{{"expiry_date": "<date>", "line_ids": [<int>]}}

Return the expiry date EXACTLY as it appears in the OCR text, character for character.
Do NOT reformat, convert, or expand it (e.g. keep "10/27" as "10/27", not "2027-10").
Only strip the surrounding label (like "PER:" or "EXP:") and any extra whitespace."""

LOT_PROMPT = """You are given a list of OCR text lines found physically near the expiry date.
They are sorted from CLOSEST to FURTHEST.

Find the LOT NUMBER (batch/lot code).
- Look for a short alphanumeric token, typically 4-12 characters.
- Heavily favor lines with the SMALLEST distance in pixels (the ones at the top of the list).

OCR TEXT (Sorted by Proximity):
\"\"\"
{ocr_text}
\"\"\"

Respond with STRICT JSON ONLY.
{{"lot_number": "<string>"}}"""


def run_qwen_json(prompt_text):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        options={"temperature": 0},
        format="json",
    )
    content = response["message"]["content"].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}

# ---------------------------------------------------------------------------
# Pipeline (your original main(), adapted to take an image and return values)
# ---------------------------------------------------------------------------


def process(image_path):
    if not image_path:
        return ({"error": "Please upload an image."}, "") if DEBUG else {"error": "Please upload an image."}

    parsed_lines = run_florence(image_path)
    debug_text = "\n".join(l["raw_line"] for l in parsed_lines)

    # Pass 1: expiry date
    full_ocr_text = "\n".join(l["raw_line"] for l in parsed_lines)
    expiry_result = run_qwen_json(EXPIRY_PROMPT.format(ocr_text=full_ocr_text))
    expiry_date = expiry_result.get("expiry_date", "")
    expiry_line_ids = expiry_result.get("line_ids", [])

    # Anchor coordinates from expiry lines
    exp_x, exp_y = 0, 0
    if expiry_line_ids:
        used = [l for l in parsed_lines if l["id"] in expiry_line_ids]
        if used:
            exp_x = sum(l["x"] for l in used) // len(used)
            exp_y = sum(l["y"] for l in used) // len(used)

    # Filter + sort candidate lot lines by proximity
    valid_lot_lines = []
    for line in parsed_lines:
        if line["id"] in expiry_line_ids:
            continue
        text_clean = line["text"].replace(" ", "").replace("</s>", "")
        if text_clean.isdigit() and len(text_clean) >= 13:  # drop barcodes
            continue
        dist = math.sqrt((line["x"] - exp_x) ** 2 + (line["y"] - exp_y) ** 2)
        line["distance"] = round(dist)
        valid_lot_lines.append(line)

    valid_lot_lines.sort(key=lambda l: l["distance"])
    filtered_ocr_text = "\n".join(
        f"Distance: {l['distance']} pixels away | {l['text']}"
        for l in valid_lot_lines
    )

    # Pass 2: lot number
    lot_result = run_qwen_json(LOT_PROMPT.format(ocr_text=filtered_ocr_text))
    lot_number = lot_result.get("lot_number", "")

    result = {"lot_number": lot_number, "expiry_date": expiry_date}
    return (result, debug_text) if DEBUG else result

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

demo = gr.Interface(
    fn=process,
    inputs=gr.Image(type="filepath", label="Medicine box photo"),
    outputs=(
        [gr.JSON(label="Result"), gr.Textbox(label="", lines=14)]
        if DEBUG
        else gr.JSON(label="Result")
    ),
    title="Medicine Box — LOT & Expiry Extractor",
    description="Upload a photo",
    flagging_mode="never",
)

# Serialize requests: you have one GPU, so handle one image at a time.
demo.queue(max_size=20, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(
        share=True,            # public *.gradio.live link (expires after 1 week)
        # auth=("user", "pass"),  # uncomment to password-protect the public link
    )