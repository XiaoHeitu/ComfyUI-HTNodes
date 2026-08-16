import os
from inspect import cleandoc

import numpy as np
import torch
from PIL import Image

from .common import (
    ComfyNodeABC,
    IO,
    build_video_metadata,
    build_batch_save_path,
    build_latent_metadata,
    build_png_metadata,
    build_save_path,
    comfy_utils,
    get_video_codec_options,
    get_video_container_options,
    get_video_extension,
    save_audio_file,
)


class HTSaveImageToFile(ComfyNodeABC):
    """
    按指定目录和文件名保存图像，扩展名与 ComfyUI 官方保存图像节点一致，默认为 .png。
    """

    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"tooltip": "要保存的图像。"}),
                "directory": (IO.STRING, {"default": ""}),
                "filename": (IO.STRING, {"default": "ComfyUI"}),
                "allow_overwrite": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "HTNodes/Savers"
    DESCRIPTION = cleandoc(__doc__ or "")

    def save_images(self, images, directory, filename, allow_overwrite, prompt=None, extra_pnginfo=None):
        metadata = build_png_metadata(prompt=prompt, extra_pnginfo=extra_pnginfo)
        results = []
        multi_batch = len(images) > 1

        for batch_number, image in enumerate(images):
            image_data = 255.0 * image.cpu().numpy()
            pil_image = Image.fromarray(np.clip(image_data, 0, 255).astype(np.uint8))
            full_path, file_name = build_batch_save_path(
                directory=directory,
                filename=filename,
                extension=".png",
                batch_number=batch_number,
                allow_overwrite=allow_overwrite,
                multi_batch=multi_batch,
            )
            pil_image.save(full_path, pnginfo=metadata, compress_level=self.compress_level)
            results.append(
                {
                    "filename": file_name,
                    "subfolder": os.path.dirname(full_path),
                    "type": "output",
                }
            )

        return {"ui": {"images": results}, "result": (images,)}


class HTSaveTextToFile(ComfyNodeABC):
    """
    按指定目录和文件名保存文本，默认保存为 .txt。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (IO.STRING, {"default": "", "multiline": True}),
                "directory": (IO.STRING, {"default": ""}),
                "filename": (IO.STRING, {"default": "ComfyUI"}),
                "allow_overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("text",)
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "HTNodes/Savers"
    DESCRIPTION = cleandoc(__doc__ or "")

    def save_text(self, text, directory, filename, allow_overwrite):
        full_path, _ = build_save_path(directory, filename, ".txt", allow_overwrite)
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(text)
        return (text,)


class HTSaveLatentToFile(ComfyNodeABC):
    """
    按指定目录和文件名保存 latent，扩展名与 ComfyUI 官方保存 latent 节点一致，默认为 .latent。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (IO.LATENT, {"tooltip": "要保存的 latent。"}),
                "directory": (IO.STRING, {"default": ""}),
                "filename": (IO.STRING, {"default": "ComfyUI"}),
                "allow_overwrite": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (IO.LATENT,)
    RETURN_NAMES = ("samples",)
    FUNCTION = "save_latent"
    OUTPUT_NODE = True
    CATEGORY = "HTNodes/Savers"
    DESCRIPTION = cleandoc(__doc__ or "")

    def save_latent(self, samples, directory, filename, allow_overwrite, prompt=None, extra_pnginfo=None):
        if comfy_utils is None:
            raise RuntimeError("HTNodes: 当前环境无法导入 comfy.utils，不能保存 latent。")

        full_path, file_name = build_save_path(directory, filename, ".latent", allow_overwrite)
        output = {
            "latent_tensor": samples["samples"].contiguous(),
            "latent_format_version_0": torch.tensor([]),
        }

        if "noise_mask" in samples and isinstance(samples["noise_mask"], torch.Tensor):
            output["noise_mask"] = samples["noise_mask"].contiguous()

        metadata = build_latent_metadata(prompt=prompt, extra_pnginfo=extra_pnginfo)
        comfy_utils.save_torch_file(output, full_path, metadata=metadata)
        return {
            "ui": {
                "latents": [
                    {
                        "filename": file_name,
                        "subfolder": os.path.dirname(full_path),
                        "type": "output",
                    }
                ]
            },
            "result": (samples,),
        }


class HTSaveAudioToFile(ComfyNodeABC):
    """
    按指定目录和文件名保存音频，保留 ComfyUI 原版高级音频保存节点的格式和质量选择。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (IO.AUDIO, {"tooltip": "要保存的音频。"}),
                "directory": (IO.STRING, {"default": ""}),
                "filename": (IO.STRING, {"default": "audio"}),
                "format": (["flac", "mp3", "opus"], {"default": "flac"}),
                "mp3_quality": (["V0", "128k", "320k"], {"default": "V0"}),
                "opus_quality": (["64k", "96k", "128k", "192k", "320k"], {"default": "128k"}),
                "allow_overwrite": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (IO.AUDIO,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "HTNodes/Savers"
    DESCRIPTION = cleandoc(__doc__ or "")

    def save_audio(
        self,
        audio,
        directory,
        filename,
        format,
        mp3_quality,
        opus_quality,
        allow_overwrite,
        prompt=None,
        extra_pnginfo=None,
    ):
        if audio is None:
            raise ValueError("HTNodes: 输入音频为空。")

        waveform_batch = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0))
        if waveform_batch is None or not isinstance(waveform_batch, torch.Tensor):
            raise ValueError("HTNodes: 音频输入缺少 waveform 张量。")
        if waveform_batch.ndim != 3 or waveform_batch.shape[0] < 1:
            raise ValueError("HTNodes: 音频 waveform 形状必须为 [B, C, T]。")

        quality = None
        if format == "mp3":
            quality = mp3_quality
        elif format == "opus":
            quality = opus_quality

        results = []
        multi_batch = waveform_batch.shape[0] > 1
        for batch_number, waveform in enumerate(waveform_batch.detach().cpu()):
            full_path, file_name = build_batch_save_path(
                directory=directory,
                filename=filename,
                extension=f".{format}",
                batch_number=batch_number,
                allow_overwrite=allow_overwrite,
                multi_batch=multi_batch,
            )
            save_audio_file(
                waveform=waveform,
                sample_rate=sample_rate,
                full_path=full_path,
                file_format=format,
                quality=quality,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )
            results.append(
                {
                    "filename": file_name,
                    "subfolder": os.path.dirname(full_path),
                    "type": "output",
                }
            )

        return {
            "ui": {"audio": results},
            "result": (audio,),
        }


class HTSaveVideoToFile(ComfyNodeABC):
    """
    按指定目录和文件名保存视频，保留 ComfyUI 原版视频保存节点的格式和编码选择。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "要保存的视频。"}),
                "directory": (IO.STRING, {"default": ""}),
                "filename": (IO.STRING, {"default": "video"}),
                "format": (get_video_container_options(), {"default": "auto"}),
                "codec": (get_video_codec_options(), {"default": "auto"}),
                "encoding_mode": (["auto", "re-encode"], {"default": "auto"}),
                "crf": ("FLOAT", {"default": 23.0, "min": 0.0, "max": 51.0, "step": 1.0}),
                "allow_overwrite": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (IO.VIDEO,)
    RETURN_NAMES = ("video",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "HTNodes/Savers"
    DESCRIPTION = cleandoc(__doc__ or "")

    def save_video(
        self,
        video,
        directory,
        filename,
        format,
        codec,
        encoding_mode,
        crf,
        allow_overwrite,
        prompt=None,
        extra_pnginfo=None,
    ):
        extension = get_video_extension(format)
        full_path, file_name = build_save_path(directory, filename, extension, allow_overwrite)

        metadata = build_video_metadata(prompt=prompt, extra_pnginfo=extra_pnginfo)
        crf_value = crf if codec == "h264" and encoding_mode == "re-encode" else None
        video.save_to(
            full_path,
            format=format,
            codec=codec,
            metadata=metadata,
            crf=crf_value,
        )

        return {
            "ui": {
                "images": [
                    {
                        "filename": file_name,
                        "subfolder": os.path.dirname(full_path),
                        "type": "output",
                    }
                ],
                "animated": (True,),
            },
            "result": (video,),
        }


NODE_CLASS_MAPPINGS = {
    "HTSaveImageToFile": HTSaveImageToFile,
    "HTSaveTextToFile": HTSaveTextToFile,
    "HTSaveLatentToFile": HTSaveLatentToFile,
    "HTSaveAudioToFile": HTSaveAudioToFile,
    "HTSaveVideoToFile": HTSaveVideoToFile,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "HTSaveImageToFile": "HT 保存图像到文件",
    "HTSaveTextToFile": "HT 保存文本到文件",
    "HTSaveLatentToFile": "HT 保存 Latent 到文件",
    "HTSaveAudioToFile": "HT 保存音频到文件",
    "HTSaveVideoToFile": "HT 保存视频到文件",
}
