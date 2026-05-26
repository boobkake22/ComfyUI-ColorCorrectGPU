from __future__ import annotations

from typing import Tuple

import torch

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None


_TARGET_PIXELS_PER_CHUNK = 16_000_000


def _comfy_torch_device() -> torch.device:
    if model_management is not None:
        try:
            return torch.device(model_management.get_torch_device())
        except Exception:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _comfy_intermediate_device(fallback: torch.device) -> torch.device:
    if model_management is not None:
        try:
            return torch.device(model_management.intermediate_device())
        except Exception:
            pass
    return fallback


def _work_device(image: torch.Tensor) -> torch.device:
    if image.device.type != "cpu":
        return image.device
    return _comfy_torch_device()


def _validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Color Correct GPU expected an IMAGE tensor.")
    if image.ndim != 4:
        raise ValueError("Color Correct GPU expected IMAGE shape [B, H, W, C].")
    if image.shape[-1] < 3:
        raise ValueError("Color Correct GPU expected at least 3 image channels.")


def _chunk_size(batch_size: int, height: int, width: int) -> int:
    pixels_per_frame = max(1, height * width)
    chunk = max(1, _TARGET_PIXELS_PER_CHUNK // pixels_per_frame)
    return min(batch_size, chunk)


def _rgb_to_hsl(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    red, green, blue = rgb.unbind(dim=-1)
    maxc = rgb.amax(dim=-1)
    minc = rgb.amin(dim=-1)
    delta = maxc - minc

    lightness = (maxc + minc) * 0.5
    saturation = torch.zeros_like(lightness)
    non_gray = delta > 1.0e-10

    low_lightness = lightness <= 0.5
    low_denominator = (maxc + minc).clamp_min(1.0e-10)
    high_denominator = (2.0 - maxc - minc).clamp_min(1.0e-10)
    saturation = torch.where(
        non_gray & low_lightness,
        delta / low_denominator,
        saturation,
    )
    saturation = torch.where(
        non_gray & ~low_lightness,
        delta / high_denominator,
        saturation,
    )

    safe_delta = delta.clamp_min(1.0e-10)
    red_is_max = (maxc == red) & non_gray
    green_is_max = (maxc == green) & non_gray
    blue_is_max = (maxc == blue) & non_gray

    hue = torch.zeros_like(lightness)
    hue = torch.where(red_is_max, torch.remainder((green - blue) / safe_delta, 6.0), hue)
    hue = torch.where(green_is_max, ((blue - red) / safe_delta) + 2.0, hue)
    hue = torch.where(blue_is_max, ((red - green) / safe_delta) + 4.0, hue)
    hue = torch.remainder(hue / 6.0, 1.0)

    return hue, lightness, saturation.clamp(0.0, 1.0)


def _hue_to_rgb(p: torch.Tensor, q: torch.Tensor, hue: torch.Tensor) -> torch.Tensor:
    hue = torch.remainder(hue, 1.0)
    return torch.where(
        hue < 1.0 / 6.0,
        p + (q - p) * 6.0 * hue,
        torch.where(
            hue < 0.5,
            q,
            torch.where(
                hue < 2.0 / 3.0,
                p + (q - p) * (2.0 / 3.0 - hue) * 6.0,
                p,
            ),
        ),
    )


def _hsl_to_rgb(hue: torch.Tensor, lightness: torch.Tensor, saturation: torch.Tensor) -> torch.Tensor:
    q = torch.where(
        lightness < 0.5,
        lightness * (1.0 + saturation),
        lightness + saturation - lightness * saturation,
    )
    p = 2.0 * lightness - q

    red = _hue_to_rgb(p, q, hue + 1.0 / 3.0)
    green = _hue_to_rgb(p, q, hue)
    blue = _hue_to_rgb(p, q, hue - 1.0 / 3.0)

    gray = saturation <= 1.0e-10
    red = torch.where(gray, lightness, red)
    green = torch.where(gray, lightness, green)
    blue = torch.where(gray, lightness, blue)

    return torch.stack((red, green, blue), dim=-1).clamp(0.0, 1.0)


def _rgb_to_hsv(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    red, green, blue = rgb.unbind(dim=-1)
    maxc = rgb.amax(dim=-1)
    minc = rgb.amin(dim=-1)
    delta = maxc - minc
    safe_delta = delta.clamp_min(1.0e-10)

    hue = torch.zeros_like(maxc)
    non_gray = delta > 1.0e-10
    red_is_max = (maxc == red) & non_gray
    green_is_max = (maxc == green) & non_gray
    blue_is_max = (maxc == blue) & non_gray

    hue = torch.where(red_is_max, torch.remainder((green - blue) / safe_delta, 6.0), hue)
    hue = torch.where(green_is_max, ((blue - red) / safe_delta) + 2.0, hue)
    hue = torch.where(blue_is_max, ((red - green) / safe_delta) + 4.0, hue)
    hue = torch.remainder(hue / 6.0, 1.0)

    saturation = torch.where(maxc > 1.0e-10, delta / maxc.clamp_min(1.0e-10), torch.zeros_like(maxc))
    value = maxc
    return hue, saturation.clamp(0.0, 1.0), value


def _hsv_to_rgb(hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    hue6 = torch.remainder(hue, 1.0) * 6.0
    sector = torch.floor(hue6)
    fraction = hue6 - sector

    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector = torch.remainder(sector, 6.0)

    red = torch.where(
        sector == 0,
        value,
        torch.where(
            sector == 1,
            q,
            torch.where(sector == 4, t, torch.where(sector == 5, value, p)),
        ),
    )
    green = torch.where(
        sector == 0,
        t,
        torch.where(
            (sector == 1) | (sector == 2),
            value,
            torch.where(sector == 3, q, p),
        ),
    )
    blue = torch.where(
        (sector == 0) | (sector == 1),
        p,
        torch.where(
            sector == 2,
            t,
            torch.where((sector == 3) | (sector == 4), value, q),
        ),
    )

    return torch.stack((red, green, blue), dim=-1)


def _as_uint8_float(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb.clamp(0.0, 1.0) * 255.0).floor().clamp(0.0, 255.0)


def _pil_luma_uint8(rgb255: torch.Tensor) -> torch.Tensor:
    weights = rgb255.new_tensor((0.299, 0.587, 0.114))
    return (rgb255 * weights).sum(dim=-1).add(0.5).floor().clamp(0.0, 255.0)


def _apply_color_correct(
    rgb: torch.Tensor,
    temperature: float,
    hue: float,
    brightness: float,
    contrast: float,
    saturation: float,
    gamma: float,
) -> torch.Tensor:
    brightness_factor = 1.0 + brightness / 100.0
    contrast_factor = 1.0 + contrast / 100.0
    saturation_factor = 1.0 + saturation / 100.0
    temperature_factor = temperature / 100.0

    rgb255 = _as_uint8_float(rgb)

    if brightness_factor != 1.0:
        rgb255 = (rgb255 * brightness_factor).clamp(0.0, 255.0).floor()

    if contrast_factor != 1.0:
        mean_luma = _pil_luma_uint8(rgb255).mean(dim=(1, 2), keepdim=True)
        mean_luma = mean_luma.add(0.5).floor().unsqueeze(-1)
        rgb255 = (mean_luma + (rgb255 - mean_luma) * contrast_factor).clamp(0.0, 255.0).floor()

    if temperature_factor != 0.0:
        factors = rgb255.new_ones((3,))
        if temperature_factor > 0.0:
            factors[0] = 1.0 + temperature_factor
            factors[1] = 1.0 + temperature_factor * 0.4
        else:
            factors[2] = 1.0 - temperature_factor
        rgb255 = (rgb255 * factors).clamp(0.0, 255.0)

    rgb = rgb255 / 255.0

    if gamma != 1.0:
        rgb = rgb.clamp(0.0, 1.0).pow(gamma).clamp(0.0, 1.0)

    if saturation_factor != 1.0:
        hsl_hue, lightness, hsl_saturation = _rgb_to_hsl(rgb)
        hsl_saturation = (hsl_saturation * saturation_factor).clamp(0.0, 1.0)
        rgb = _hsl_to_rgb(hsl_hue, lightness, hsl_saturation)

    if hue != 0.0:
        rgb255 = rgb * 255.0
        hsv_hue, hsv_saturation, value = _rgb_to_hsv(rgb255)
        hsv_hue = torch.remainder(hsv_hue + hue / 360.0, 1.0)
        rgb = _hsv_to_rgb(hsv_hue, hsv_saturation, value) / 255.0

    return (rgb.clamp(0.0, 1.0) * 255.0).floor().clamp(0.0, 255.0) / 255.0


class ColorCorrectGPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "hue": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": 5}),
                "brightness": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "gamma": ("FLOAT", {"default": 1, "min": 0.2, "max": 2.2, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_correct"
    CATEGORY = "ArtVenture/Post Processing"
    DESCRIPTION = "Compatibility-focused GPU torch implementation of Art Venture's Color Correct node."
    SEARCH_ALIASES = ["color correct gpu", "art venture color correct gpu"]

    def color_correct(
        self,
        image: torch.Tensor,
        temperature: float,
        hue: float,
        brightness: float,
        contrast: float,
        saturation: float,
        gamma: float,
    ):
        _validate_image(image)

        original_device = image.device
        original_dtype = image.dtype if image.is_floating_point() else torch.float32
        device = _work_device(image)
        output_device = _comfy_intermediate_device(original_device)

        work = image.to(device=device, dtype=torch.float32)
        result = work.clone()

        batch_size, height, width, _ = result.shape
        chunk_size = _chunk_size(batch_size, height, width)
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            result[start:end, ..., :3] = _apply_color_correct(
                work[start:end, ..., :3],
                temperature,
                hue,
                brightness,
                contrast,
                saturation,
                gamma,
            )

        return (result.to(device=output_device, dtype=original_dtype),)
