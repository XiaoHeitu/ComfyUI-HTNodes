from inspect import cleandoc

from .common import (
    ComfyNodeABC,
    IO,
    file_mtime,
    load_audio_file,
    load_image_tensor,
    load_latent,
    load_text_content,
    load_video_file,
)


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
        return file_mtime(path)

    def load_image(self, path: str):
        return (load_image_tensor(path),)


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
        return file_mtime(path)

    def load_text(self, path: str):
        return (load_text_content(path),)


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
        return file_mtime(path)

    def load_latent(self, path: str):
        return (load_latent(path),)


class HTLoadAudioFromFile(ComfyNodeABC):
    """
    从任意文件路径加载音频并输出为 ComfyUI AUDIO 类型。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (IO.STRING, {"default": ""}),
            }
        }

    RETURN_TYPES = (IO.AUDIO,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"
    CATEGORY = "HTNodes/Loaders"
    DESCRIPTION = cleandoc(__doc__ or "")

    @classmethod
    def IS_CHANGED(cls, path: str):
        return file_mtime(path)

    def load_audio(self, path: str):
        return (load_audio_file(path),)


class HTLoadVideoFromFile(ComfyNodeABC):
    """
    从任意文件路径加载视频并输出为 ComfyUI VIDEO 类型。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (IO.STRING, {"default": ""}),
            }
        }

    RETURN_TYPES = (IO.VIDEO,)
    RETURN_NAMES = ("video",)
    FUNCTION = "load_video"
    CATEGORY = "HTNodes/Loaders"
    DESCRIPTION = cleandoc(__doc__ or "")

    @classmethod
    def IS_CHANGED(cls, path: str):
        return file_mtime(path)

    def load_video(self, path: str):
        return (load_video_file(path),)


NODE_CLASS_MAPPINGS = {
    "HTLoadImageFromFile": HTLoadImageFromFile,
    "HTLoadTextFromFile": HTLoadTextFromFile,
    "HTLoadLatentFromFile": HTLoadLatentFromFile,
    "HTLoadAudioFromFile": HTLoadAudioFromFile,
    "HTLoadVideoFromFile": HTLoadVideoFromFile,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "HTLoadImageFromFile": "HT 从文件加载图像",
    "HTLoadTextFromFile": "HT 从文件加载文本",
    "HTLoadLatentFromFile": "HT 从文件加载 Latent",
    "HTLoadAudioFromFile": "HT 从文件加载音频",
    "HTLoadVideoFromFile": "HT 从文件加载视频",
}
