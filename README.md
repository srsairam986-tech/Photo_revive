# Photo Revive

Offline Android app that turns old black & white photos into clean, natural-colour 4K pictures
in the 922 × 2048 phone shape. Free open-source AI only — no internet, no subscription, no per-photo cost.

## What it does (all on the phone)

1. **Frame to 922:2048** — *Fill screen* crops around the faces; *Fit whole* keeps the full photo on a soft blurred background.
2. **Clean** — removes scratches, film grain and noise; *Strong* also removes dust specks (may soften tiny jewellery).
3. **Colourise** — DDColor tiny AI model (Apache-2.0).
4. **Natural light** — bright clean whites, soft contrast, warm sunlit skin (tuned from a reference window-light photo).
5. **AI upscale ×4** — Real-ESRGAN general x4v3 (BSD-3); also sharpens mild camera shake.
   - 4K = **1728 × 3840**, 8K = **3456 × 7680** (8K is slow).
6. **Viewer** — full-screen 922×2048, pinch / double-tap zoom, *Hold* to compare with the original, *Save* to `Pictures/PhotoRevive`.

Colour photos skip step 3 and get cleaned, relit and upscaled.

## Build (GitHub, no PC needed)

1. Create a new **private** GitHub repo (e.g. `photo-revive`).
2. Upload everything in this folder, keeping the folders (`.github`, `app`, `tools`, …).
3. Open the repo's **Actions** tab → *Build Photo Revive APK* runs automatically
   (first build ~15–25 min: it downloads and converts the free AI models; later builds reuse them).
4. Open the finished run → **Artifacts** → download `PhotoRevive-apk` → unzip → install the `.apk`.

## Expected speed on a vivo Y20

- Open / clean / colourise: about 15–40 seconds
- 4K upscale: roughly 2–5 minutes
- 8K upscale: roughly 10–20 minutes (needs free memory; close other apps)

## Limits to know

- AI colours are a best guess — the model cannot know a saree's or a wall's real colour.
- Heavy motion blur can only be reduced, not fully fixed.
- Very small originals (e.g. 300 px wide) will look smooth/painterly at 4K: the AI invents detail it cannot see.
- Very blurry or badly damaged photos can come out over-coloured; the app tones this down automatically.
- Very faint hairline scratches on white areas may remain.
