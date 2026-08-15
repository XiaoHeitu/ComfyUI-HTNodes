import os
from inspect import cleandoc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

try:
    from comfy.comfy_types.node_typing import IO, ComfyNodeABC
except Exception:
    class IO:
        STRING = "STRING"
        IMAGE = "IMAGE"
        LATENT = "LATENT"

    ComfyNodeABC = object


def _normalize_path(path: str) -> str:
    if not path:
        raise ValueError("HTNodes: 文件路径不能为空。")
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _file_mtime(path: str):
    try:
        resolved = _normalize_path(path)
        if os.path.exists(resolved):
            stat = os.stat(resolved)
            return (stat.st_mtime_ns, stat.st_size)
    except Exception:
        pass
    return float("NaN")


def _load_image_tensor(path: str) -> torch.Tensor:
    resolved = _normalize_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"HTNodes: 图像文件不存在: {resolved}")

    image = Image.open(resolved)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).unsqueeze(0)


def _load_text_content(path: str) -> str:
    resolved = _normalize_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"HTNodes: 文本文件不存在: {resolved}")

    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            with open(resolved, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("unknown", b"", 0, 1, f"HTNodes: 无法解码文本文件: {resolved}")


def _as_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return None


def _coerce_latent_dict(payload: Any) -> dict[str, Any]:
    tensor = _as_tensor(payload)
    if tensor is not None:
        return {"samples": tensor}

    if not isinstance(payload, dict):
        raise TypeError("HTNodes: latent 文件内容不是 Tensor 或 dict，无法转换为 LATENT。")

    latent: dict[str, Any] = {}
    for key, value in payload.items():
        tensor_value = _as_tensor(value)
        latent[key] = tensor_value if tensor_value is not None else value

    if "samples" in latent and isinstance(latent["samples"], torch.Tensor):
        return latent

    tensor_items = [value for value in latent.values() if isinstance(value, torch.Tensor)]
    if len(tensor_items) == 1:
        return {"samples": tensor_items[0]}

    raise ValueError("HTNodes: 未在文件中找到可用的 latent samples。")


def _coerce_comfyui_latent(payload: dict[str, Any]) -> dict[str, Any]:
    if "latent_tensor" in payload:
        multiplier = 1.0
        if "latent_format_version_0" not in payload:
            multiplier = 1.0 / 0.18215

        samples = _as_tensor(payload["latent_tensor"])
        if samples is None:
            raise ValueError("HTNodes: ComfyUI latent 文件中的 latent_tensor 无法转换为 Tensor。")

        output = {"samples": samples.float() * multiplier}
        noise_mask = _as_tensor(payload.get("noise_mask"))
        if noise_mask is not None:
            output["noise_mask"] = noise_mask
        return output

    return _coerce_latent_dict(payload)


def _load_latent(path: str) -> dict[str, Any]:
    resolved = _normalize_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"HTNodes: latent 文件不存在: {resolved}")

    suffix = Path(resolved).suffix.lower()

    if suffix in {".latent", ".safetensors", ".sft"}:
        from safetensors.torch import load_file

        payload = load_file(resolved, device="cpu")
        return _coerce_comfyui_latent(payload)

    if suffix == ".npy":
        payload = np.load(resolved, allow_pickle=True)
        return _coerce_latent_dict(payload)

    if suffix == ".npz":
        payload = dict(np.load(resolved, allow_pickle=True))
        return _coerce_latent_dict(payload)

    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    return _coerce_latent_dict(payload)


class HTLoadImageFromFile(ComfyNodeABC):
    """
    从任意文件路径加载图像并输出为 ComfyUI IMAGE 类型。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (IO.STRING, {"default": ""}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "HTNodes/Loaders"
    DESCRIPTION = cleandoc(__doc__ or "")

    @classmethod
    def IS_CHANGED(cls, path: str):
        return _file_mtime(path)

    def load_image(self, path: str):
        return (_load_image_tensor(path),)


class HTLoadTextFromFile(ComfyNodeABC):
    """
    从任意文件路径加载文本并输出为 ComfyUI STRING 类型。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (IO.STRING, {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_text"
    CATEGORY = "HTNodes/Loaders"
    DESCRIPTION = cleandoc(__doc__ or "")

    @classmethod
    def IS_CHANGED(cls, path: str):
        return _file_mtime(path)

    def load_text(self, path: str):
        return (_load_text_content(path),)


class HTLoadLatentFromFile(ComfyNodeABC):
    """
    从文件路径加载 latent，并输出为 ComfyUI LATENT 类型。
    优先按 ComfyUI 官方 .latent 保存格式读取，也兼容 .safetensors / .npy / .npz / .pt / .pth / .bin / .ckpt。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (IO.STRING, {"default": ""}),
            }
        }

    RETURN_TYPES = (IO.LATENT,)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load_latent"
    CATEGORY = "HTNodes/Loaders"
    DESCRIPTION = cleandoc(__doc__ or "")

    @classmethod
    def IS_CHANGED(cls, path: str):
        return _file_mtime(path)

    def load_latent(self, path: str):
        return (_load_latent(path),)


NODE_CLASS_MAPPINGS = {
    "HTLoadImageFromFile": HTLoadImageFromFile,
    "HTLoadTextFromFile": HTLoadTextFromFile,
    "HTLoadLatentFromFile": HTLoadLatentFromFile,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "HTLoadImageFromFile": "HT 从文件加载图像",
    "HTLoadTextFromFile": "HT 从文件加载文本",
    "HTLoadLatentFromFile": "HT 从文件加载 Latent",
}
