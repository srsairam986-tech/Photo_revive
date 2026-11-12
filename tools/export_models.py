"""
Downloads the free, open-source AI models once (at build time on GitHub)
and converts them to ONNX so the Android app can run them fully offline.

  * DDColor tiny (Apache-2.0)          -> colorize.onnx   (B&W -> colour)
  * Real-ESRGAN general x4v3 (BSD-3)   -> upscale4x.onnx  (x4 upscale, de-blur, de-noise)

Usage:  python tools/export_models.py --ddcolor-repo DDColor --out app/src/main/assets/models
Add --random-weights to test the export without downloading anything.
"""
import argparse, os, sys, urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ESRGAN_URLS = {
    "general": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "wdn": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
}
DDCOLOR_INPUT = 512


# ---------------------------------------------------------------- Real-ESRGAN
class SRVGGNetCompact(nn.Module):
    """Same architecture as realesrgan/archs/srvgg_arch.py (realesr-general-x4v3)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def fetch(url, cache):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, os.path.basename(url))
    if not os.path.exists(path):
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    return path


def export_upscaler(out_dir, cache, random_weights, denoise=0.5):
    net = SRVGGNetCompact()
    if not random_weights:
        # Blend the normal and the "strong denoise" weights, exactly like Real-ESRGAN's
        # own --denoise_strength option (0.5 = its default). Good for grainy old prints.
        a = torch.load(fetch(ESRGAN_URLS["general"], cache), map_location="cpu")
        b = torch.load(fetch(ESRGAN_URLS["wdn"], cache), map_location="cpu")
        a = a.get("params_ema", a.get("params", a))
        b = b.get("params_ema", b.get("params", b))
        net.load_state_dict({k: (1 - denoise) * a[k] + denoise * b[k] for k in a}, strict=True)
    net.eval()
    path = os.path.join(out_dir, "upscale4x.onnx")
    dummy = torch.rand(1, 3, 64, 64)
    torch.onnx.export(net, dummy, path, opset_version=17, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h4", 3: "w4"}})
    return path


# ---------------------------------------------------------------- DDColor
def export_colorizer(out_dir, repo, random_weights):
    sys.path.insert(0, os.path.abspath(repo))
    from ddcolor import DDColor  # noqa: E402

    kwargs = dict(encoder_name="convnext-t", decoder_name="MultiScaleColorDecoder",
                  input_size=[DDCOLOR_INPUT, DDCOLOR_INPUT], num_output_channels=2,
                  last_norm="Spectral", do_normalize=False, num_queries=100, num_scales=3, dec_layers=9)
    if random_weights:
        model = DDColor(**kwargs)
    else:
        from huggingface_hub import PyTorchModelHubMixin

        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kw):
                if isinstance(config, dict):
                    kw = {**config, **kw}
                super().__init__(**kw)

        model = DDColorHF.from_pretrained("piddnad/ddcolor_paper_tiny")
    model.eval()
    path = os.path.join(out_dir, "colorize.onnx")
    dummy = torch.rand(1, 3, DDCOLOR_INPUT, DDCOLOR_INPUT)
    with torch.no_grad():
        torch.onnx.export(model, dummy, path, opset_version=17, input_names=["input"],
                          output_names=["output"])
    return path


def verify(path, shape, expect):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": np.random.rand(*shape).astype(np.float32)})[0]
    assert tuple(out.shape) == expect, (path, out.shape, expect)
    assert np.isfinite(out).all(), path
    print(f"OK {os.path.basename(path)}: {os.path.getsize(path) / 1e6:.1f} MB, output {out.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddcolor-repo", default="DDColor")
    ap.add_argument("--out", default="app/src/main/assets/models")
    ap.add_argument("--cache", default=".model-cache")
    ap.add_argument("--random-weights", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    up = export_upscaler(args.out, args.cache, args.random_weights)
    verify(up, (1, 3, 72, 40), (1, 3, 288, 160))
    col = export_colorizer(args.out, args.ddcolor_repo, args.random_weights)
    verify(col, (1, 3, DDCOLOR_INPUT, DDCOLOR_INPUT), (1, 2, DDCOLOR_INPUT, DDCOLOR_INPUT))
