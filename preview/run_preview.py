"""Runs the same steps as the phone app (with the real AI models) on test photos,
so the result can be checked before installing. Output goes to preview/results/."""
import os, sys, time
import cv2, numpy as np, onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from proto import clean, grade, crop_box

M = "app/src/main/assets/models"
OUT = "preview/results"
os.makedirs(OUT, exist_ok=True)
col = ort.InferenceSession(f"{M}/colorize.onnx", providers=["CPUExecutionProvider"])
up = ort.InferenceSession(f"{M}/upscale4x.onnx", providers=["CPUExecutionProvider"])


def colorize(gray8):
    s = 512
    small = cv2.resize(gray8, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = np.repeat(small[None, None], 3, axis=1)
    ab = col.run(None, {"input": x})[0][0]
    h, w = gray8.shape
    a = cv2.resize(ab[0], (w, h), interpolation=cv2.INTER_CUBIC)
    b = cv2.resize(ab[1], (w, h), interpolation=cv2.INTER_CUBIC)
    L = cv2.cvtColor(cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[..., 0]
    bgr = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_Lab2BGR)
    return np.clip(bgr * 255 + 0.5, 0, 255).astype(np.uint8)


def upscale(bgr8, tile=192, pad=10):
    rgb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    h, w = rgb.shape[:2]
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
            px0, py0, px1, py1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)
            t = rgb[py0:py1, px0:px1].transpose(2, 0, 1)[None]
            r = up.run(None, {"input": np.ascontiguousarray(t)})[0][0].transpose(1, 2, 0)
            oy, ox = (y0 - py0) * 4, (x0 - px0) * 4
            out[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = r[oy:oy + (y1 - y0) * 4, ox:ox + (x1 - x0) * 4]
    return cv2.cvtColor(np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


for name in ["test_bw_damaged.png", "test_bw_clean.jpg"]:
    img = cv2.imread(f"preview/{name}", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    t0 = time.time()
    x, y, cw, ch = crop_box(w, h)
    work = cv2.resize(img[y:y + ch, x:x + cw], (432, 960),
                      interpolation=cv2.INTER_AREA if cw > 432 else cv2.INTER_LANCZOS4)
    c, sigma, masked = clean(work)
    t1 = time.time(); colour = colorize(c)
    t2 = time.time(); graded = grade(colour)
    t3 = time.time(); big = upscale(graded)
    t4 = time.time()
    stem = os.path.splitext(name)[0]
    cv2.imwrite(f"{OUT}/{stem}_4k.jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 93])
    strip = np.hstack([cv2.cvtColor(work, cv2.COLOR_GRAY2BGR), cv2.cvtColor(c, cv2.COLOR_GRAY2BGR), colour, graded,
                       cv2.resize(big, (432, 960), interpolation=cv2.INTER_AREA)])
    cv2.imwrite(f"{OUT}/{stem}_steps.jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
    crop = big[700:1500, 500:1300]   # 1:1 detail crop of the 4K result
    cv2.imwrite(f"{OUT}/{stem}_4k_detail.png", crop)
    print(f"{name}: clean {t1 - t0:.1f}s colour {t2 - t1:.1f}s grade {t3 - t2:.1f}s upscale {t4 - t3:.1f}s -> {big.shape[1]}x{big.shape[0]}")
