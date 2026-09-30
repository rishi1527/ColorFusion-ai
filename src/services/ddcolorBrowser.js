import * as ort from "onnxruntime-web";

const MODEL_URL =
  "https://huggingface.co/rishi1527/colorfusion-ddcolor/resolve/main/ddcolor-tiny.onnx";

const MODEL_SIZE = 512;
const MAX_OUTPUT_SIZE = 2048;

let sessionPromise = null;

/* =========================================================
   COLOR CONVERSION
   OpenCV-style float Lab
   L: 0..100
   a/b: approximately -127..127
   ========================================================= */

const D65_X = 0.950456;
const D65_Y = 1.0;
const D65_Z = 1.088754;

const EPSILON = 216 / 24389;
const KAPPA = 24389 / 27;

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function linearToSrgb(c) {
  c = Math.max(0, Math.min(1, c));

  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

function labF(t) {
  return t > EPSILON ? Math.cbrt(t) : (KAPPA * t + 16) / 116;
}

function labFInverse(t) {
  const cube = t * t * t;

  return cube > EPSILON ? cube : (116 * t - 16) / KAPPA;
}

/* =========================================================
   RGB -> LAB
   ========================================================= */

function rgbToLab(r, g, b) {
  const R = srgbToLinear(r);
  const G = srgbToLinear(g);
  const B = srgbToLinear(b);

  const X = (0.412453 * R + 0.35758 * G + 0.180423 * B) / D65_X;

  const Y = 0.212671 * R + 0.71516 * G + 0.072169 * B;

  const Z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / D65_Z;

  const fx = labF(X);
  const fy = labF(Y);
  const fz = labF(Z);

  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/* =========================================================
   LAB -> RGB
   ========================================================= */

function labToRgb(L, a, b) {
  const fy = (L + 16) / 116;
  const fx = fy + a / 500;
  const fz = fy - b / 200;

  const X = D65_X * labFInverse(fx);
  const Y = D65_Y * labFInverse(fy);
  const Z = D65_Z * labFInverse(fz);

  let R = 3.240479 * X - 1.53715 * Y - 0.498535 * Z;

  let G = -0.969256 * X + 1.875992 * Y + 0.041556 * Z;

  let B = 0.055648 * X - 0.204043 * Y + 1.057311 * Z;

  R = linearToSrgb(R);
  G = linearToSrgb(G);
  B = linearToSrgb(B);

  return [R, G, B];
}

/* =========================================================
   LOAD IMAGE
   ========================================================= */

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();

    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };

    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Unable to read the selected image."));
    };

    image.src = url;
  });
}

/* =========================================================
   OUTPUT SIZE
   ========================================================= */

function getOutputSize(width, height) {
  const largestSide = Math.max(width, height);

  if (largestSide <= MAX_OUTPUT_SIZE) {
    return {
      width,
      height,
    };
  }

  const scale = MAX_OUTPUT_SIZE / largestSide;

  return {
    width: Math.round(width * scale),
    height: Math.round(height * scale),
  };
}

/* =========================================================
   DRAW IMAGE TO CANVAS
   ========================================================= */

function drawImageToCanvas(image, width, height) {
  const canvas = document.createElement("canvas");

  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d", {
    willReadFrequently: true,
  });

  if (!context) {
    throw new Error("Canvas 2D context is not available.");
  }

  context.drawImage(image, 0, 0, width, height);

  return {
    canvas,
    context,
    imageData: context.getImageData(0, 0, width, height),
  };
}

/* =========================================================
   ORIGINAL L CHANNEL
   ========================================================= */

function extractOriginalL(imageData) {
  const { data, width, height } = imageData;

  const pixelCount = width * height;

  const L = new Float32Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    const p = i * 4;

    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;

    const lab = rgbToLab(r, g, b);

    L[i] = lab[0];
  }

  return L;
}

/* =========================================================
   CREATE DDColor INPUT
   ========================================================= */

function createModelInput(imageData) {
  const { data, width, height } = imageData;

  const pixelCount = width * height;

  /*
   * CHW:
   *
   * [all R]
   * [all G]
   * [all B]
   */

  const input = new Float32Array(pixelCount * 3);

  for (let i = 0; i < pixelCount; i++) {
    const p = i * 4;

    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;

    const [L] = rgbToLab(r, g, b);

    /*
     * Grayscale Lab:
     *
     * L = original
     * a = 0
     * b = 0
     */

    const [grayR, grayG, grayB] = labToRgb(L, 0, 0);

    input[i] = grayR;

    input[pixelCount + i] = grayG;

    input[pixelCount * 2 + i] = grayB;
  }

  return input;
}

/* =========================================================
   RESIZE DDColor AB OUTPUT
   =========================================================
   
   ONNX output:

   [1, 2, 512, 512]

   Memory:

   [A channel: 512*512]
   [B channel: 512*512]

   We convert it to:

   [A,B]
   [A,B]
   [A,B]
   ...
   ========================================================= */

function resizeAB(
  outputAB,
  sourceWidth,
  sourceHeight,
  targetWidth,
  targetHeight,
) {
  const result = new Float32Array(targetWidth * targetHeight * 2);

  const sourcePlaneSize = sourceWidth * sourceHeight;

  const scaleX = sourceWidth / targetWidth;

  const scaleY = sourceHeight / targetHeight;

  for (let y = 0; y < targetHeight; y++) {
    const sourceY = (y + 0.5) * scaleY - 0.5;

    let y0 = Math.floor(sourceY);

    y0 = Math.max(0, Math.min(sourceHeight - 1, y0));

    const y1 = Math.min(sourceHeight - 1, y0 + 1);

    const fy = Math.max(0, Math.min(1, sourceY - y0));

    for (let x = 0; x < targetWidth; x++) {
      const sourceX = (x + 0.5) * scaleX - 0.5;

      let x0 = Math.floor(sourceX);

      x0 = Math.max(0, Math.min(sourceWidth - 1, x0));

      const x1 = Math.min(sourceWidth - 1, x0 + 1);

      const fx = Math.max(0, Math.min(1, sourceX - x0));

      /*
       * A channel
       */

      const a00 = outputAB[y0 * sourceWidth + x0];

      const a01 = outputAB[y0 * sourceWidth + x1];

      const a10 = outputAB[y1 * sourceWidth + x0];

      const a11 = outputAB[y1 * sourceWidth + x1];

      /*
       * B channel
       */

      const b00 = outputAB[sourcePlaneSize + y0 * sourceWidth + x0];

      const b01 = outputAB[sourcePlaneSize + y0 * sourceWidth + x1];

      const b10 = outputAB[sourcePlaneSize + y1 * sourceWidth + x0];

      const b11 = outputAB[sourcePlaneSize + y1 * sourceWidth + x1];

      /*
       * Bilinear interpolation
       */

      const topA = a00 * (1 - fx) + a01 * fx;

      const bottomA = a10 * (1 - fx) + a11 * fx;

      const topB = b00 * (1 - fx) + b01 * fx;

      const bottomB = b10 * (1 - fx) + b11 * fx;

      const finalA = topA * (1 - fy) + bottomA * fy;

      const finalB = topB * (1 - fy) + bottomB * fy;

      const outputIndex = (y * targetWidth + x) * 2;

      result[outputIndex] = finalA;

      result[outputIndex + 1] = finalB;
    }
  }

  return result;
}

/* =========================================================
   CREATE FINAL IMAGE
   ========================================================= */

async function createColorizedBlob(L, AB, width, height) {
  const pixelCount = width * height;

  const output = new Uint8ClampedArray(pixelCount * 4);

  for (let i = 0; i < pixelCount; i++) {
    const LValue = L[i];

    const aValue = AB[i * 2];

    const bValue = AB[i * 2 + 1];

    const [r, g, b] = labToRgb(LValue, aValue, bValue);

    const p = i * 4;

    output[p] = Math.round(Math.max(0, Math.min(1, r)) * 255);

    output[p + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);

    output[p + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);

    output[p + 3] = 255;
  }

  const canvas = document.createElement("canvas");

  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("Canvas 2D context is not available.");
  }

  const imageData = new ImageData(output, width, height);

  context.putImageData(imageData, 0, 0);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Failed to create colorized image."));
          return;
        }

        resolve(blob);
      },
      "image/webp",
      0.92,
    );
  });
}

/* =========================================================
   LOAD MODEL
   ========================================================= */

export async function loadDDColorModel() {
  if (!sessionPromise) {
    console.log("Loading DDColor model from Hugging Face...");

    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }

  const session = await sessionPromise;

  console.log("DDColor model loaded successfully!");

  console.log("Input:", session.inputNames);

  console.log("Output:", session.outputNames);

  return session;
}

/* =========================================================
   MAIN COLORIZATION
   ========================================================= */

export async function colorizeImage(file, onProgress) {
  if (!file) {
    throw new Error("No image selected.");
  }

  onProgress?.("Loading image...");

  const image = await loadImage(file);

  const outputSize = getOutputSize(image.naturalWidth, image.naturalHeight);

  /* -----------------------------------------
     Original image
     ----------------------------------------- */

  const original = drawImageToCanvas(
    image,
    outputSize.width,
    outputSize.height,
  );

  /* -----------------------------------------
     Preserve original L
     ----------------------------------------- */

  const originalL = extractOriginalL(original.imageData);

  onProgress?.("Preparing AI input...");

  /* -----------------------------------------
     512 × 512 model image
     ----------------------------------------- */

  const modelCanvas = drawImageToCanvas(image, MODEL_SIZE, MODEL_SIZE);

  const modelInput = createModelInput(modelCanvas.imageData);

  /* -----------------------------------------
     Load model
     ----------------------------------------- */

  const session = await loadDDColorModel();

  const tensor = new ort.Tensor("float32", modelInput, [
    1,
    3,
    MODEL_SIZE,
    MODEL_SIZE,
  ]);

  onProgress?.("Running DDColor AI...");

  /* -----------------------------------------
     Run ONNX
     ----------------------------------------- */

  const result = await session.run({
    [session.inputNames[0]]: tensor,
  });

  const outputTensor = result[session.outputNames[0]];

  /*
   * Safety checks
   */

  if (!outputTensor || !outputTensor.data) {
    throw new Error("DDColor returned an invalid output.");
  }

  console.log("DDColor output shape:", outputTensor.dims);

  console.log("DDColor output type:", outputTensor.type);

  if (
    outputTensor.dims.length !== 4 ||
    outputTensor.dims[0] !== 1 ||
    outputTensor.dims[1] !== 2 ||
    outputTensor.dims[2] !== MODEL_SIZE ||
    outputTensor.dims[3] !== MODEL_SIZE
  ) {
    throw new Error(
      `Unexpected DDColor output shape: ${outputTensor.dims.join(" × ")}`,
    );
  }

  const outputAB = new Float32Array(outputTensor.data);

  console.log(
    "DDColor output first values:",
    Array.from(outputAB.slice(0, 20)),
  );

  /* -----------------------------------------
     IMPORTANT:
     Model output is CHW:
     
     [A plane][B plane]
     
     Convert it to HWC:
     
     [A,B]
     ----------------------------------------- */

  onProgress?.("Resizing AI color channels...");

  const resizedAB = resizeAB(
    outputAB,
    MODEL_SIZE,
    MODEL_SIZE,
    outputSize.width,
    outputSize.height,
  );

  /* -----------------------------------------
     L + AB -> RGB
     ----------------------------------------- */

  onProgress?.("Creating colorized image...");

  const blob = await createColorizedBlob(
    originalL,
    resizedAB,
    outputSize.width,
    outputSize.height,
  );

  return {
    blob,
    width: outputSize.width,
    height: outputSize.height,
  };
}
