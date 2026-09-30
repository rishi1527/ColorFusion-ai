from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import onnxruntime as ort
import requests

from io import BytesIO
from pathlib import Path
import time
import os


# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "*"
        }
    }
)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "ddcolor-tiny.onnx"

MODEL_TEMP_PATH = MODEL_DIR / "ddcolor-tiny.onnx.download"


# =========================================================
# MODEL DOWNLOAD CONFIG
# =========================================================

MODEL_URL = (
    "https://github.com/rishi1527/ColorFusion-ai/"
    "releases/download/v1.0.0/ddcolor-tiny.onnx"
)


# =========================================================
# DOWNLOAD DDColor MODEL
# =========================================================

def download_model():
    """
    Download DDColor Tiny ONNX model from GitHub Release
    if it does not already exist locally.
    """

    if MODEL_PATH.exists():
        model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

        print()
        print("DDColor model already exists.")
        print(f"Model: {MODEL_PATH}")
        print(f"Size : {model_size_mb:.2f} MB")

        return

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("DOWNLOADING DDColor TINY ONNX MODEL")
    print("=" * 60)
    print()
    print(f"Source: {MODEL_URL}")
    print(f"Target: {MODEL_PATH}")
    print()

    try:

        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=(30, 600),
            headers={
                "User-Agent": "ColorFusion-AI"
            }
        )

        response.raise_for_status()

        total_size = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        downloaded = 0

        with open(
            MODEL_TEMP_PATH,
            "wb"
        ) as model_file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                model_file.write(chunk)

                downloaded += len(chunk)

                if total_size:

                    percent = (
                        downloaded / total_size
                    ) * 100

                    downloaded_mb = (
                        downloaded /
                        (1024 * 1024)
                    )

                    total_mb = (
                        total_size /
                        (1024 * 1024)
                    )

                    print(
                        f"\rDownloading: "
                        f"{percent:6.2f}% "
                        f"({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                        end=""
                    )

        print()
        print()

        # -------------------------------------------------
        # Validate downloaded file
        # -------------------------------------------------

        if not MODEL_TEMP_PATH.exists():
            raise RuntimeError(
                "Model download completed but "
                "temporary file was not created."
            )

        downloaded_size = MODEL_TEMP_PATH.stat().st_size

        if downloaded_size < 100 * 1024 * 1024:
            raise RuntimeError(
                "Downloaded model appears to be invalid "
                f"or incomplete. Size: {downloaded_size} bytes."
            )

        # -------------------------------------------------
        # Move temporary file to final location
        # -------------------------------------------------

        MODEL_TEMP_PATH.replace(
            MODEL_PATH
        )

        model_size_mb = (
            MODEL_PATH.stat().st_size /
            (1024 * 1024)
        )

        print(
            f"✓ DDColor model downloaded successfully!"
        )

        print(
            f"✓ Model size: {model_size_mb:.2f} MB"
        )

        print(
            f"✓ Saved to: {MODEL_PATH}"
        )

        print()
        print("=" * 60)
        print()

    except Exception as error:

        # Remove incomplete download
        if MODEL_TEMP_PATH.exists():
            try:
                MODEL_TEMP_PATH.unlink()
            except Exception:
                pass

        print()
        print("MODEL DOWNLOAD ERROR:")
        print(error)
        print()

        raise RuntimeError(
            "Could not download DDColor Tiny ONNX model."
        ) from error


# =========================================================
# APP STARTUP
# =========================================================

print()
print("=" * 60)
print("                 COLORFUSION AI")
print("=" * 60)

print()
print("Checking DDColor Tiny ONNX model...")

download_model()


# =========================================================
# LOAD DDColor ONNX MODEL
# =========================================================

print()
print("Loading DDColor Tiny ONNX...")
print(f"Model: {MODEL_PATH}")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[
        "CPUExecutionProvider"
    ]
)

INPUT_NAME = session.get_inputs()[0].name

OUTPUT_NAME = session.get_outputs()[0].name

INPUT_SHAPE = session.get_inputs()[0].shape

OUTPUT_SHAPE = session.get_outputs()[0].shape


print()
print("✓ DDColor model loaded successfully!")

print(
    f"Input : {INPUT_NAME} {INPUT_SHAPE}"
)

print(
    f"Output: {OUTPUT_NAME} {OUTPUT_SHAPE}"
)

print()
print("Runtime: ONNX Runtime CPU")

print()
print("=" * 60)
print()


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "ColorFusion AI",
        "model": "DDColor Tiny ONNX",
        "runtime": "ONNX Runtime CPU",
        "input": INPUT_SHAPE,
        "output": OUTPUT_SHAPE
    })


# =========================================================
# DDColor COLORIZATION
# =========================================================

def colorize_image(img_bgr):
    """
    DDColor-style preprocessing/postprocessing.

    Input:
        BGR uint8 image

    Output:
        BGR uint8 colorized image
    """

    height, width = img_bgr.shape[:2]

    # -----------------------------------------------------
    # Normalize BGR image
    # -----------------------------------------------------

    img = (
        img_bgr.astype(np.float32) / 255.0
    )

    # -----------------------------------------------------
    # Preserve original L channel
    # -----------------------------------------------------

    orig_lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2Lab
    )

    orig_l = orig_lab[:, :, :1]

    # -----------------------------------------------------
    # Resize for DDColor
    # -----------------------------------------------------

    img_resized = cv2.resize(
        img,
        (512, 512),
        interpolation=cv2.INTER_LINEAR
    )

    # -----------------------------------------------------
    # Convert resized image to Lab
    # -----------------------------------------------------

    img_lab = cv2.cvtColor(
        img_resized,
        cv2.COLOR_BGR2Lab
    )

    img_l = img_lab[:, :, :1]

    # -----------------------------------------------------
    # Create grayscale Lab image
    # -----------------------------------------------------

    img_gray_lab = np.concatenate(
        (
            img_l,
            np.zeros_like(img_l),
            np.zeros_like(img_l)
        ),
        axis=-1
    )

    # -----------------------------------------------------
    # Lab → RGB
    # -----------------------------------------------------

    img_gray_rgb = cv2.cvtColor(
        img_gray_lab,
        cv2.COLOR_LAB2RGB
    )

    # -----------------------------------------------------
    # HWC → CHW
    # -----------------------------------------------------

    input_tensor = (
        img_gray_rgb
        .transpose((2, 0, 1))
        .astype(np.float32)
    )

    # -----------------------------------------------------
    # Add batch dimension
    # -----------------------------------------------------

    input_tensor = np.expand_dims(
        input_tensor,
        axis=0
    )

    # -----------------------------------------------------
    # Run DDColor
    # -----------------------------------------------------

    output_ab = session.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: input_tensor
        }
    )[0]

    # -----------------------------------------------------
    # Resize predicted AB channels
    # -----------------------------------------------------

    output_ab = output_ab[0]

    output_ab_resized = cv2.resize(
        output_ab.transpose(
            1,
            2,
            0
        ),
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    # -----------------------------------------------------
    # Combine original L + predicted AB
    # -----------------------------------------------------

    output_lab = np.concatenate(
        (
            orig_l,
            output_ab_resized
        ),
        axis=-1
    )

    # -----------------------------------------------------
    # Lab → BGR
    # -----------------------------------------------------

    output_bgr = cv2.cvtColor(
        output_lab,
        cv2.COLOR_Lab2BGR
    )

    # -----------------------------------------------------
    # Float [0,1] → uint8 [0,255]
    # -----------------------------------------------------

    output_img = (
        np.clip(
            output_bgr,
            0.0,
            1.0
        ) * 255.0
    ).round().astype(np.uint8)

    return output_img


# =========================================================
# COLORIZE API
# =========================================================

@app.post("/api/colorize")
def colorize():

    start_time = time.time()

    print()
    print("=" * 60)
    print("NEW COLORIZATION REQUEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Check upload
    # -----------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error": (
                "No image uploaded. "
                "Use form field 'image'."
            )
        }), 400

    file = request.files["image"]

    if not file.filename:

        return jsonify({
            "error": "No filename provided."
        }), 400

    print(
        f"Image: {file.filename}"
    )

    # -----------------------------------------------------
    # Read image bytes
    # -----------------------------------------------------

    image_bytes = file.read()

    if not image_bytes:

        return jsonify({
            "error": "Uploaded image is empty."
        }), 400

    # -----------------------------------------------------
    # Decode image
    # -----------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    img = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if img is None:

        return jsonify({
            "error": (
                "Could not decode image. "
                "Please upload JPG, JPEG or PNG."
            )
        }), 400

    height, width = img.shape[:2]

    print(
        f"Size: {width} x {height}"
    )

    # -----------------------------------------------------
    # Run AI
    # -----------------------------------------------------

    print(
        "Running DDColor AI..."
    )

    try:

        result = colorize_image(
            img
        )

    except Exception as error:

        print()
        print("COLORIZATION ERROR:")
        print(error)

        return jsonify({
            "error": "AI colorization failed.",
            "details": str(error)
        }), 500

    # -----------------------------------------------------
    # Encode JPEG
    # -----------------------------------------------------

    success, encoded = cv2.imencode(
        ".jpg",
        result,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not success:

        return jsonify({
            "error": (
                "Could not encode "
                "colorized image."
            )
        }), 500

    output_bytes = encoded.tobytes()

    elapsed = (
        time.time() -
        start_time
    )

    print(
        f"✓ Colorization complete "
        f"in {elapsed:.2f} seconds"
    )

    print(
        "=" * 60
    )

    print()

    # -----------------------------------------------------
    # Return image
    # -----------------------------------------------------

    return send_file(
        BytesIO(output_bytes),
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="colorfusion-result.jpg"
    )


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": (
            "Image is too large. "
            "Maximum size is 15 MB."
        )
    }), 413


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )