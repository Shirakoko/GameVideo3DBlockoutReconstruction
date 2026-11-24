"""Visual resemblance is supplementary, never a geometric accuracy gate."""
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter


def _images(predicted, reference):
    p, g = np.asarray(predicted), np.asarray(reference)
    if p.shape != g.shape or p.ndim != 3 or p.shape[2] != 3 or p.dtype != np.uint8 or g.dtype != np.uint8:
        raise ValueError("Equal-sized native RGB uint8 images required; no implicit resize/alignment")
    return p.astype(float)/255., g.astype(float)/255.


def ssim(predicted, reference, *, window=7):
    """RGB channel-mean SSIM, uniform local window, population covariance."""
    p, g = _images(predicted, reference)
    if isinstance(window, bool) or not isinstance(window, int) or window < 3 or window % 2 != 1 or min(p.shape[:2]) < window:
        raise ValueError("Odd SSIM window >=3 must fit the raster")
    size = (window, window, 1)
    mp, mg = uniform_filter(p, size), uniform_filter(g, size)
    vp, vg = uniform_filter(p*p, size)-mp*mp, uniform_filter(g*g, size)-mg*mg
    covariance = uniform_filter(p*g, size)-mp*mg
    score = ((2*mp*mg+0.01**2)*(2*covariance+0.03**2))/((mp*mp+mg*mg+0.01**2)*(vp+vg+0.03**2))
    border = window//2
    return float(score[border:-border, border:-border].mean())


def lpips(predicted, reference, *, backbone_weights, linear_weights):
    """Optional local-only AlexNet LPIPS; never auto-download torchvision weights.

    Explicit files for BOTH the ImageNet backbone and LPIPS linear layers are
    required. This keeps restricted video frames entirely on the local host.
    """
    p, g = _images(predicted, reference)
    if min(p.shape[:2]) < 64:
        raise ValueError("AlexNet LPIPS requires a raster of at least 64x64")
    if not Path(backbone_weights).is_file() or not Path(linear_weights).is_file():
        raise ValueError("LPIPS requires explicit existing local backbone and linear weight files")
    import torch
    import lpips as package
    model = package.LPIPS(net="alex", pnet_rand=True, model_path=str(linear_weights), verbose=False)
    state = torch.load(backbone_weights, map_location="cpu", weights_only=True)
    # torchvision feature keys -> LPIPS slice keys, without pretrained=True.
    mapping = {0: "slice1.0", 3: "slice2.3", 6: "slice3.6", 8: "slice4.8", 10: "slice5.10"}
    translated = {f"{target}.{kind}": state[f"features.{index}.{kind}"]
                  for index, target in mapping.items() for kind in ("weight", "bias")}
    model.net.load_state_dict(translated, strict=True)
    model.eval()
    tensors = [torch.from_numpy(x.transpose(2, 0, 1).copy()).float().unsqueeze(0)*2-1 for x in (p, g)]
    with torch.no_grad():
        return float(model(*tensors).item())
