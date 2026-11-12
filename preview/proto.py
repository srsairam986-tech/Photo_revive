"""Python mirror of the phone pipeline (non-AI steps) for tuning.
Same maths is ported to Kotlin in the app."""
import cv2, numpy as np, sys, time

TARGET_W, TARGET_H = 1729, 3840          # 922x2048 ratio, 4K long edge

# ---------- 1. noise estimate (Immerkaer) ----------
def noise_sigma(L8):
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(L8.astype(np.float32), -1, M)
    h, w = L8.shape
    return float(np.sqrt(np.pi / 2) * np.abs(r[1:-1, 1:-1]).sum() / (6 * (w - 2) * (h - 2)))

# ---------- 2. scratch / dust mask ----------
def scratch_mask(L8, sigma, strong=False):
    # detect on a lightly smoothed copy so film grain doesn't hide the lines
    sm = cv2.GaussianBlur(L8, (0, 0), 0.8 + 0.08 * sigma)
    med = cv2.medianBlur(sm, 9)                      # removes lines up to ~4 px wide
    d = sm.astype(np.int16) - med.astype(np.int16)
    thr = max(16.0, 1.6 * sigma)
    cand = (np.abs(d) > thr).astype(np.uint8)
    # bright and dark defects are found separately so they don't merge with texture
    mask = np.zeros_like(L8)
    for sign in (1, -1):
        c = ((sign * d) > thr).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(c, connectivity=8)
        keep = np.zeros(n, np.uint8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            length = float(np.hypot(w, h))
            thick = area / max(length, 1.0)
            fill = area / float(w * h)
            if length >= 30 and thick <= 4.5 and (fill < 0.35 or min(w, h) <= 5):
                keep[i] = 1                                      # long thin line = scratch
            elif strong and 2 <= area <= 30 and max(w, h) <= 7:
                keep[i] = 1                                      # dust speck
        mask |= keep[lab]
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask * 255

def second_pass(den, strong):
    med = cv2.medianBlur(den, 7); d = den.astype(np.int16) - med.astype(np.int16)
    out = np.zeros_like(den)
    for sign in (1, -1):
        c = ((sign * d) > 20).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(c, connectivity=8)
        keep = np.zeros(n, np.uint8)
        for i in range(1, n):
            x, y, w, h, area = st[i]; length = np.hypot(w, h)
            if length >= 60 and max(w, h) >= 6 * max(min(w, h), 1) and area / length <= 3: keep[i] = 1
            elif strong and area <= 20 and max(w, h) <= 6 and np.abs(d[y:y + h, x:x + w]).max() > 45: keep[i] = 1
        out |= keep[lab]
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) * 255

def clean(L8, strong=False):
    s = noise_sigma(L8)
    m = scratch_mask(L8, s, strong)
    if m.any():
        L8 = cv2.inpaint(L8, m, 4, cv2.INPAINT_TELEA)
    h = float(np.clip(1.15 * s, 3, 16))
    L8 = cv2.fastNlMeansDenoising(L8, None, h, 7, 15)
    m2 = second_pass(L8, strong)
    if m2.any():
        L8 = cv2.inpaint(L8, m2, 4, cv2.INPAINT_TELEA)
    blur = cv2.GaussianBlur(L8, (0, 0), 1.6)
    L8 = cv2.addWeighted(L8, 1.45, blur, -0.45, 0)
    return L8, s, int(m.astype(bool).sum() + m2.astype(bool).sum())

# ---------- 3. natural window-light grade (from user's reference photo) ----------

REF = dict(L_lo=4.0, L_hi=94.0, L_med=53.5, chroma_med=13.0, hi_a=0.5, hi_b=3.0)
def grade(bgr8, strength=1.0):
    lab = cv2.cvtColor(bgr8.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    lo, hi = np.percentile(L, [1, 99])
    Ln = np.clip((L - lo) / max(hi - lo, 1e-3), 0, 1)
    med = float(np.median(Ln))
    tmed = (REF['L_med'] - REF['L_lo']) / (REF['L_hi'] - REF['L_lo'])
    g = np.clip(np.log(tmed) / np.log(min(max(med, 0.01), 0.99)), 0.8, 1.25)
    Lt = REF['L_lo'] + Ln ** g * (REF['L_hi'] - REF['L_lo'])
    L2 = L + strength * (Lt - L)
    # neutral, warm-white highlights (removes pink/blue casts on white clothes, walls)
    hm = L2 > 72
    if hm.sum() > 200:
        da, db = REF['hi_a'] - a[hm].mean(), REF['hi_b'] - b[hm].mean()
        w = np.clip((L2 - 35) / 45, 0, 1) * 0.8 * strength
        a = a + da * w; b = b + db * w
    # gentle vibrance: only lifts dull colour, never pushes strong colour further
    C = np.sqrt(a * a + b * b)
    mid = (L2 > 20) & (L2 < 88)
    cm = float(np.median(C[mid])) if mid.any() else 10
    k = np.clip(REF['chroma_med'] / max(cm, 1e-3), 0.9, 1.35)
    k = 1 + (k - 1) * strength
    boost = 1 + (k - 1) * np.clip(1 - C / 30, 0, 1)
    a, b = a * boost, b * boost
    # sunlit warmth on skin tones only (hue 25-75 deg, real colour), tiny warmth elsewhere
    C = np.sqrt(a * a + b * b); hue = np.degrees(np.arctan2(b, a))
    skin = ((hue > 25) & (hue < 75) & (C > 5) & (C < 40)).astype(np.float32)
    skin = cv2.GaussianBlur(skin, (0, 0), 2)
    wmid = np.exp(-((L2 - 60) / 22) ** 2) * strength
    a = a + wmid * (0.3 + 1.5 * skin)
    b = b + wmid * (0.8 + 3.0 * skin)
    out = cv2.cvtColor(cv2.merge([np.clip(L2, 0, 100), a, b]).astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)

# ---------- 4. 922x2048 framing ----------
def crop_box(w, h, faces=None):
    """Largest 922:2048 box inside the image, centred on faces if any."""
    r = 922 / 2048
    if w / h > r:      # too wide -> crop width
        cw, ch = int(round(h * r)), h
        cx = w / 2 if not faces else np.mean([fx + fw / 2 for fx, fy, fw, fh in faces])
        x = int(np.clip(cx - cw / 2, 0, w - cw)); y = 0
    else:              # too tall -> crop height
        cw, ch = w, int(round(w / r))
        cy = h / 2 if not faces else np.mean([fy + fh / 2 for fx, fy, fw, fh in faces]) + ch * 0.15
        y = int(np.clip(cy - ch / 2, 0, h - ch)); x = 0
    return x, y, cw, ch

def fit_blur(img, W, H):
    """Whole image fitted inside W x H, blurred enlarged copy behind it."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    fw, fh = int(round(w * s)), int(round(h * s))
    sb = max(W / w, H / h)
    small = cv2.resize(img, (max(1, int(w * sb / 8)), max(1, int(h * sb / 8))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6)
    bg = cv2.resize(small, (int(w * sb) + 1, int(h * sb) + 1), interpolation=cv2.INTER_LINEAR)
    bx, by = (bg.shape[1] - W) // 2, (bg.shape[0] - H) // 2
    bg = (bg[by:by + H, bx:bx + W] * 0.55).astype(np.uint8)
    fg = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    x, y = (W - fw) // 2, (H - fh) // 2
    bg[y:y + fh, x:x + fw] = fg
    return bg
