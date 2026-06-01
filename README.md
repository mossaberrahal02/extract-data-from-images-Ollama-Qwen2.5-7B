# Medicine Box — LOT & Expiry Extractor

A web application that extracts the **lot number** and **expiry date** from a
photo of a medicine box. It combines on-device OCR with a local language model
to read and interpret the printed text, and exposes the result through a simple
Gradio interface.

## How It Works

The pipeline runs in three stages:

1. **OCR (Florence-2).** The uploaded image is passed through Microsoft's
   `Florence-2-large` model using the `<OCR_WITH_REGION>` task. This returns the
   text lines found on the box along with the pixel coordinates of each line.

2. **Expiry extraction (Qwen).** The full OCR text is sent to a local Qwen model
   (via Ollama) with a prompt that locates the expiry date. It recognises
   multilingual labels (EXP, PER / péremption, CAD, VENC, etc.) and returns the
   date exactly as printed, without reformatting.

3. **Lot extraction (Qwen).** Using the coordinates of the expiry date as an
   anchor, the remaining text lines are sorted by physical proximity. Barcodes
   are filtered out, and the closest candidates are sent to the model to identify
   the lot/batch number.

The final result is returned as JSON:

```json
{
  "lot_number": "N3661",
  "expiry_date": "10/27"
}
```

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA is recommended for Florence-2. The code falls back to
  CPU automatically, but inference will be significantly slower.
- [Ollama](https://ollama.com) installed and running locally.

## Setup

1. **Clone the repository and enter the directory.**

   ```bash
   git clone <repository-url>
   cd ocr-queen
   ```

2. **Create a virtual environment and install dependencies.**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install and start Ollama, then pull the model.**

   ```bash
   ollama pull qwen2.5:7b-instruct
   ```

   Make sure the Ollama service is running before launching the app.

   The Florence-2 model is downloaded automatically from Hugging Face on first
   run.

## Running

```bash
python app.py
```

This launches the Gradio interface and prints a local URL (and, by default, a
temporary public `*.gradio.live` link). Open the URL, upload a photo of a
medicine box, and the extracted lot number and expiry date are returned.

## Configuration

The following options are set near the top of `app.py`:

| Setting        | Description                                                        | Default                  |
| -------------- | ----------------------------------------------------------------- | ------------------------ |
| `FLORENCE_ID`  | Hugging Face model used for OCR.                                   | `microsoft/Florence-2-large` |
| `OLLAMA_MODEL` | Ollama model used for extraction.                                 | `qwen2.5:7b-instruct`    |
| `DEBUG`        | When `True`, adds a textbox to the UI showing the raw OCR output. | `False`                  |

The public share link can be disabled, and basic auth enabled, in the
`demo.launch(...)` call at the bottom of `app.py`.

## Notes

- Requests are serialised (one image at a time) since the pipeline is designed
  for a single GPU.
- The expiry date is returned verbatim from the OCR text. No date-format
  conversion is applied, to avoid errors when expanding short formats such as
  `MM/YY`.
