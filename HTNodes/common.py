import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import comfy.utils as comfy_utils
except Exception:
    comfy_utils = None

try:
    from comfy.cli_args import args
except Exception:
    class _Args:
        disable_metadata = False

    args = _Args()

try:
    from comfy.comfy_types.node_typing import IO, ComfyNodeABC
except Exception:
    class IO:
        STRING = "STRING"
        IMAGE = "IMAGE"
        LATENT = "LATENT"

    ComfyNodeABC = object


def normalize_path(path: str) -> str:
    if not path:
        raise ValueError("HTNodes: 文件路径不能为空。")
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def get_default_output_dir() -> str:
    if folder_paths is not None:
        try:
            return folder_paths.get_output_directory()
        except Exception:
            pass
    return os.getcwd()


def file_mtime(path: str):
    try:
        resolved = normalize_path(path)
        if os.path.exists(resolved):
            stat = os.stat(resolved)
            return (stat.st_mtime_ns, stat.st_size)
    except Exception:
        pass
    return float("NaN")


def load_image_tensor(path: str) -> torch.Tensor:
    resolved = normalize_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"HTNodes: 图像文件不存在: {resolved}")

    image = Image.open(resolved)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).unsqueeze(0)


def load_text_content(path: str) -> str:
    resolved = normalize_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"HTNodes: 文本文件不存在: {resolved}")

    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            with open(resolved, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("unknown", b"", 0, 1, f"HTNodes: 无法解码文本文件: {resolved}")


def resolve_save_directory(directory: str) -> str:
    if not directory:
        resolved = get_default_output_dir()
    else:
        expanded = os.path.expanduser(os.path.expandvars(directory))
        if os.path.isabs(expanded):
            resolved = os.path.abspath(expanded)
        else:
            resolved = os.path.abspath(os.path.join(get_default_output_dir(), expanded))

    os.makedirs(resolved, exist_ok=True)
    return resolved


def normalize_file_stem(filename: str) -> str:
    if not filename:
        raise ValueError("HTNodes: 文件名不能为空。")

    basename = os.path.basename(filename.strip())
    stem = Path(basename).stem
    if not stem:
        raise ValueError("HTNodes: 文件名不能为空。")
    return stem


def build_save_path(directory: str, filename: str, extension: str, allow_overwrite: bool) -> tuple[str, str]:
    resolved_dir = resolve_save_directory(directory)
    stem = normalize_file_stem(filename)
    final_name = f"{stem}{extension}"
    final_path = os.path.join(resolved_dir, final_name)

    if allow_overwrite or not os.path.exists(final_path):
        return final_path, final_name

    counter = 1
    while True:
        candidate_name = f"{stem}_{counter:05}{extension}"
        candidate_path = os.path.join(resolved_dir, candidate_name)
        if not os.path.exists(candidate_path):
            return candidate_path, candidate_name
        counter += 1


def build_batch_save_path(
    directory: str,
    filename: str,
    extension: str,
    batch_number: int,
    allow_overwrite: bool,
    multi_batch: bool,
) -> tuple[str, str]:
    stem = normalize_file_stem(filename)
    batch_stem = f"{stem}_{batch_number:05}" if multi_batch else stem
    return build_save_path(directory, batch_stem, extension, allow_overwrite)


def as_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return None


def coerce_latent_dict(payload: Any) -> dict[str, Any]:
    tensor = as_tensor(payload)
    if tensor is not None:
        return {"samples": tensor}

    if not isinstance(payload, dict):
        raise TypeError("HTNodes: latent 文件内容不是 Tensor 或 dict，无法转换为 LATENT。")

    latent: dict[str, Any] = {}
    for key, value in payload.items():
        tensor_value = as_tensor(value)
        latent[key] = tensor_value if tensor_value is not None else value

    if "samples" in latent and isinstance(latent["samples"], torch.Tensor):
        return latent

    tensor_items = [value for value in latent.values() if isinstance(value, torch.Tensor)]
    if len(tensor_items) == 1:
        return {"samples": tensor_items[0]}

    raise ValueError("HTNodes: 未在文件中找到可用的 latent samples。")


def coerce_comfyui_latent(payload: dict[str, Any]) -> dict[str, Any]:
    if "latent_tensor" in payload:
        multiplier = 1.0
        if "latent_format_version_0" not in payload:
            multiplier = 1.0 / 0.18215

        samples = as_tensor(payload["latent_tensor"])
        if samples is None:
            raise ValueError("HTNodes: ComfyUI latent 文件中的 latent_tensor 无法转换为 Tensor。")

        output = {"samples": samples.float() * multiplier}
        noise_mask = as_tensor(payload.get("noise_mask"))
        if noise_mask is not None:
            output["noise_mask"] = noise_mask
        return output

    return coerce_latent_dict(payload)


def load_latent(path: str) -> dict[str, Any]:
    resolved = normalize_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"HTNodes: latent 文件不存在: {resolved}")

    suffix = Path(resolved).suffix.lower()

    if suffix in {".latent", ".safetensors", ".sft"}:
        from safetensors.torch import load_file

        payload = load_file(resolved, device="cpu")
        return coerce_comfyui_latent(payload)

    if suffix == ".npy":
        payload = np.load(resolved, allow_pickle=True)
        return coerce_latent_dict(payload)

    if suffix == ".npz":
        payload = dict(np.load(resolved, allow_pickle=True))
        return coerce_latent_dict(payload)

    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    return coerce_latent_dict(payload)


def build_png_metadata(prompt=None, extra_pnginfo=None):
    if args.disable_metadata:
        return None

    metadata = PngInfo()
    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo is not None:
        for key, value in extra_pnginfo.items():
            metadata.add_text(key, json.dumps(value))
    return metadata


def build_latent_metadata(prompt=None, extra_pnginfo=None):
    if args.disable_metadata:
        return None

    metadata = {"prompt": json.dumps(prompt) if prompt is not None else ""}
    if extra_pnginfo is not None:
        for key, value in extra_pnginfo.items():
            metadata[key] = json.dumps(value)
    return metadata
