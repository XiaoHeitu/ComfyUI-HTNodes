import os
from typing_extensions import override

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import ComfyExtension, Types, io, ui

from .common import (
    build_batch_save_path,
    build_latent_metadata,
    build_png_metadata,
    build_save_path,
    build_video_metadata,
    comfy_utils,
    file_mtime,
    get_video_codec_options,
    get_video_container_options,
    get_video_extension,
    load_audio_file,
    load_image_tensor,
    load_latent,
    load_text_content,
    load_video_file,
    save_audio_file,
)


class HTLoadImageFromFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTLoadImageFromFile",
            display_name="HT 从文件加载图像",
            category="HTNodes/Loaders",
            inputs=[io.String.Input("path", default="")],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, path):
        return io.NodeOutput(load_image_tensor(path))

    @classmethod
    def fingerprint_inputs(cls, path):
        return file_mtime(path)


class HTLoadTextFromFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTLoadTextFromFile",
            display_name="HT 从文件加载文本",
            category="HTNodes/Loaders",
            inputs=[io.String.Input("path", default="", multiline=False)],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, path):
        return io.NodeOutput(load_text_content(path))

    @classmethod
    def fingerprint_inputs(cls, path):
        return file_mtime(path)


class HTLoadLatentFromFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTLoadLatentFromFile",
            display_name="HT 从文件加载 Latent",
            category="HTNodes/Loaders",
            inputs=[io.String.Input("path", default="")],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, path):
        return io.NodeOutput(load_latent(path))

    @classmethod
    def fingerprint_inputs(cls, path):
        return file_mtime(path)


class HTLoadAudioFromFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTLoadAudioFromFile",
            display_name="HT 从文件加载音频",
            category="HTNodes/Loaders",
            inputs=[io.String.Input("path", default="")],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, path):
        return io.NodeOutput(load_audio_file(path))

    @classmethod
    def fingerprint_inputs(cls, path):
        return file_mtime(path)


class HTLoadVideoFromFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTLoadVideoFromFile",
            display_name="HT 从文件加载视频",
            category="HTNodes/Loaders",
            inputs=[io.String.Input("path", default="")],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, path):
        return io.NodeOutput(load_video_file(path))

    @classmethod
    def fingerprint_inputs(cls, path):
        return file_mtime(path)


class HTSaveImageToFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTSaveImageToFile",
            display_name="HT 保存图像到文件",
            category="HTNodes/Savers",
            inputs=[
                io.Image.Input("images", tooltip="要保存的图像。"),
                io.String.Input("directory", default=""),
                io.String.Input("filename", default="ComfyUI"),
                io.Boolean.Input("allow_overwrite", default=False),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, directory, filename, allow_overwrite):
        metadata = build_png_metadata(prompt=cls.hidden.prompt, extra_pnginfo=cls.hidden.extra_pnginfo)
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
            pil_image.save(full_path, pnginfo=metadata, compress_level=4)
            results.append(ui.SavedResult(file_name, os.path.dirname(full_path), io.FolderType.output))

        return io.NodeOutput(images, ui={"images": results})


class HTSaveTextToFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTSaveTextToFile",
            display_name="HT 保存文本到文件",
            category="HTNodes/Savers",
            inputs=[
                io.String.Input("text", default="", multiline=True),
                io.String.Input("directory", default=""),
                io.String.Input("filename", default="ComfyUI"),
                io.Boolean.Input("allow_overwrite", default=False),
            ],
            is_output_node=True,
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, text, directory, filename, allow_overwrite):
        full_path, file_name = build_save_path(directory, filename, ".txt", allow_overwrite)
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(text)

        return io.NodeOutput(
            text,
            ui={
                "text": (text,),
                "files": [ui.SavedResult(file_name, os.path.dirname(full_path), io.FolderType.output)],
            },
        )


class HTSaveLatentToFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTSaveLatentToFile",
            display_name="HT 保存 Latent 到文件",
            category="HTNodes/Savers",
            inputs=[
                io.Latent.Input("samples", tooltip="要保存的 latent。"),
                io.String.Input("directory", default=""),
                io.String.Input("filename", default="ComfyUI"),
                io.Boolean.Input("allow_overwrite", default=False),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Latent.Output(display_name="samples")],
        )

    @classmethod
    def execute(cls, samples, directory, filename, allow_overwrite):
        if comfy_utils is None:
            raise RuntimeError("HTNodes: 当前环境无法导入 comfy.utils，不能保存 latent。")

        full_path, file_name = build_save_path(directory, filename, ".latent", allow_overwrite)
        output = {
            "latent_tensor": samples["samples"].contiguous(),
            "latent_format_version_0": torch.tensor([]),
        }
        if "noise_mask" in samples and isinstance(samples["noise_mask"], torch.Tensor):
            output["noise_mask"] = samples["noise_mask"].contiguous()

        metadata = build_latent_metadata(prompt=cls.hidden.prompt, extra_pnginfo=cls.hidden.extra_pnginfo)
        comfy_utils.save_torch_file(output, full_path, metadata=metadata)
        return io.NodeOutput(
            samples,
            ui={
                "latents": [ui.SavedResult(file_name, os.path.dirname(full_path), io.FolderType.output)],
            },
        )


class HTSaveAudioToFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTSaveAudioToFile",
            display_name="HT 保存音频到文件",
            category="HTNodes/Savers",
            inputs=[
                io.Audio.Input("audio", tooltip="要保存的音频。"),
                io.String.Input("directory", default=""),
                io.String.Input("filename", default="audio"),
                io.DynamicCombo.Input(
                    "format",
                    options=[
                        io.DynamicCombo.Option("flac", []),
                        io.DynamicCombo.Option(
                            "mp3",
                            [io.Combo.Input("quality", options=["V0", "128k", "320k"], default="V0")],
                        ),
                        io.DynamicCombo.Option(
                            "opus",
                            [io.Combo.Input("quality", options=["64k", "96k", "128k", "192k", "320k"], default="128k")],
                        ),
                    ],
                    tooltip="音频保存格式与编码质量。",
                ),
                io.Boolean.Input("allow_overwrite", default=False),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, audio, directory, filename, format, allow_overwrite):
        if audio is None:
            raise ValueError("HTNodes: 输入音频为空。")

        waveform_batch = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0))
        if waveform_batch is None or not isinstance(waveform_batch, torch.Tensor):
            raise ValueError("HTNodes: 音频输入缺少 waveform 张量。")
        if waveform_batch.ndim != 3 or waveform_batch.shape[0] < 1:
            raise ValueError("HTNodes: 音频 waveform 形状必须为 [B, C, T]。")

        # 与官方 SaveAudioAdvanced 保持一致：
        # - format 未显式传入时按 flac 处理
        # - mp3 的默认质量为 V0
        # - opus 的默认质量为 128k
        file_format = format.get("format", "flac")
        quality = format.get("quality")
        if file_format == "mp3" and not quality:
            quality = "V0"
        elif file_format == "opus" and not quality:
            quality = "128k"
        results = []
        multi_batch = waveform_batch.shape[0] > 1

        for batch_number, waveform in enumerate(waveform_batch.detach().cpu()):
            full_path, file_name = build_batch_save_path(
                directory=directory,
                filename=filename,
                extension=f".{file_format}",
                batch_number=batch_number,
                allow_overwrite=allow_overwrite,
                multi_batch=multi_batch,
            )
            save_audio_file(
                waveform=waveform,
                sample_rate=sample_rate,
                full_path=full_path,
                file_format=file_format,
                quality=quality,
                prompt=cls.hidden.prompt,
                extra_pnginfo=cls.hidden.extra_pnginfo,
            )
            results.append(ui.SavedResult(file_name, os.path.dirname(full_path), io.FolderType.output))

        return io.NodeOutput(audio, ui={"audio": results})


class HTSaveVideoToFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HTSaveVideoToFile",
            display_name="HT 保存视频到文件",
            category="HTNodes/Savers",
            inputs=[
                io.Video.Input("video", tooltip="要保存的视频。"),
                io.String.Input("directory", default=""),
                io.String.Input("filename", default="video"),
                io.Combo.Input("format", options=get_video_container_options(), default="auto"),
                io.DynamicCombo.Input(
                    "codec",
                    options=[
                        io.DynamicCombo.Option("auto", []),
                        io.DynamicCombo.Option(
                            "h264",
                            [
                                io.DynamicCombo.Input(
                                    "encoding",
                                    display_name="encoding mode",
                                    options=[
                                        io.DynamicCombo.Option("auto", []),
                                        io.DynamicCombo.Option(
                                            "re-encode",
                                            [
                                                io.Float.Input(
                                                    "crf",
                                                    default=23.0,
                                                    min=0.0,
                                                    max=51.0,
                                                    step=1.0,
                                                    tooltip="Lower values produce higher quality and larger files.",
                                                )
                                            ],
                                        ),
                                    ],
                                    optional=True,
                                    tooltip="Automatic preserves compatible H.264 streams. Re-encode applies a custom CRF.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="视频编码方式。",
                ),
                io.Boolean.Input("allow_overwrite", default=False),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, video, directory, filename, format, codec, allow_overwrite):
        # 与官方 SaveVideo 保持一致：
        # - format 默认 auto
        # - codec 默认 auto
        # - h264 时 encoding 缺失视为自动保留兼容流
        # - re-encode 时 crf 默认 23
        selected_format = format or "auto"
        codec_name = codec.get("codec", "auto")
        encoding = codec.get("encoding") or {}
        crf = encoding.get("crf")
        if codec_name == "h264" and encoding.get("encoding") == "re-encode" and crf is None:
            crf = 23.0

        extension = get_video_extension(selected_format)
        full_path, file_name = build_save_path(directory, filename, extension, allow_overwrite)

        metadata = build_video_metadata(prompt=cls.hidden.prompt, extra_pnginfo=cls.hidden.extra_pnginfo)
        video.save_to(
            full_path,
            format=Types.VideoContainer(selected_format),
            codec=codec_name,
            metadata=metadata,
            crf=crf,
        )
        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file_name, os.path.dirname(full_path), io.FolderType.output)]))


class HTNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HTLoadImageFromFile,
            HTLoadTextFromFile,
            HTLoadLatentFromFile,
            HTLoadAudioFromFile,
            HTLoadVideoFromFile,
            HTSaveImageToFile,
            HTSaveTextToFile,
            HTSaveLatentToFile,
            HTSaveAudioToFile,
            HTSaveVideoToFile,
        ]


async def comfy_entrypoint() -> HTNodesExtension:
    return HTNodesExtension()
