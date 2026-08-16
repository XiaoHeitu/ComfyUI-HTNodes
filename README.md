# ComfyUI-HTNodes

`ComfyUI-HTNodes` 是一个用于 `ComfyUI` 的自定义节点包，当前提供一组“按路径加载”和“按路径保存”的基础文件节点。

## 当前包含的节点

### `HTNodes/Loaders`

- `HT 从文件加载图像`
- `HT 从文件加载文本`
- `HT 从文件加载 Latent`
- `HT 从文件加载音频`
- `HT 从文件加载视频`

### `HTNodes/Savers`

- `HT 保存图像到文件`
- `HT 保存文本到文件`
- `HT 保存 Latent 到文件`
- `HT 保存音频到文件`
- `HT 保存视频到文件`

## 节点说明

### 1. HT 从文件加载图像

- 输入：`path` (`STRING`)
- 输出：`image` (`IMAGE`)
- 说明：
  - 通过任意字符串路径加载图像文件
  - 自动处理 EXIF 方向
  - 输出为 ComfyUI 标准 `IMAGE` 类型

### 2. HT 从文件加载文本

- 输入：`path` (`STRING`)
- 输出：`text` (`STRING`)
- 说明：
  - 通过任意字符串路径加载文本文件
  - 当前支持 `utf-8`、`utf-8-sig`、`gb18030`

### 3. HT 从文件加载 Latent

- 输入：`path` (`STRING`)
- 输出：`latent` (`LATENT`)
- 说明：
  - 优先按 ComfyUI 官方 latent 保存方式读取
  - 支持 ComfyUI 官方 `SaveLatent` 导出的 `.latent`
  - 兼容 `.safetensors`、`.sft`、`.npy`、`.npz`、`.pt`、`.pth`、`.bin`、`.ckpt`

### 4. HT 保存图像到文件

- 输入：
  - `images` (`IMAGE`)
  - `directory` (`STRING`)
  - `filename` (`STRING`)
  - `allow_overwrite` (`BOOLEAN`)
- 输出：`images` (`IMAGE`)
- 说明：
  - 按“目录 + 文件名”的方式保存图像
  - 扩展名与 ComfyUI 官方 `SaveImage` 一致，固定为 `.png`
  - 支持保留官方 PNG 元数据
  - 当输入是 batch 图像时，会自动按批次编号保存

### 5. HT 保存文本到文件

- 输入：
  - `text` (`STRING`)
  - `directory` (`STRING`)
  - `filename` (`STRING`)
  - `allow_overwrite` (`BOOLEAN`)
- 输出：`text` (`STRING`)
- 说明：
  - 按“目录 + 文件名”的方式保存文本
  - 默认保存为 `.txt`
  - 使用 `utf-8` 编码写入

### 6. HT 保存 Latent 到文件

- 输入：
  - `samples` (`LATENT`)
  - `directory` (`STRING`)
  - `filename` (`STRING`)
  - `allow_overwrite` (`BOOLEAN`)
- 输出：`samples` (`LATENT`)
- 说明：
  - 按“目录 + 文件名”的方式保存 latent
  - 扩展名与 ComfyUI 官方 `SaveLatent` 一致，固定为 `.latent`
  - 保存格式对齐 ComfyUI 官方 latent 保存方式
  - 如果存在 `noise_mask`，也会一起写入

### 7. HT 从文件加载音频

- 输入：`path` (`STRING`)
- 输出：`audio` (`AUDIO`)
- 说明：
  - 通过任意字符串路径加载音频文件
  - 输出结构对齐 ComfyUI 官方 `LoadAudio`
  - 支持 ComfyUI / FFmpeg 可解码的常见音频格式，也可从带音轨的视频中提取音频

### 8. HT 从文件加载视频

- 输入：`path` (`STRING`)
- 输出：`video` (`VIDEO`)
- 说明：
  - 通过任意字符串路径加载视频文件
  - 输出类型对齐 ComfyUI 官方 `LoadVideo`
  - 内部通过 ComfyUI 官方 `VideoFromFile` 读取

### 9. HT 保存音频到文件

- 输入：
  - `audio` (`AUDIO`)
  - `directory` (`STRING`)
  - `filename` (`STRING`)
  - `format` (`flac` / `mp3` / `opus`)
  - `mp3_quality` (`V0` / `128k` / `320k`)
  - `opus_quality` (`64k` / `96k` / `128k` / `192k` / `320k`)
  - `allow_overwrite` (`BOOLEAN`)
- 输出：`audio` (`AUDIO`)
- 说明：
  - 按“目录 + 文件名”的方式保存音频
  - 格式和质量选项对齐 ComfyUI 官方 `Save Audio (Advanced)`
  - `format=mp3` 时使用 `mp3_quality`
  - `format=opus` 时使用 `opus_quality`
  - `format=flac` 时忽略质量选项
  - 如果输入是 batch 音频，会按批次编号分别保存

### 10. HT 保存视频到文件

- 输入：
  - `video` (`VIDEO`)
  - `directory` (`STRING`)
  - `filename` (`STRING`)
  - `format` (`auto` / `mp4`)
  - `codec` (`auto` / `h264`)
  - `encoding_mode` (`auto` / `re-encode`)
  - `crf` (`FLOAT`)
  - `allow_overwrite` (`BOOLEAN`)
- 输出：`video` (`VIDEO`)
- 说明：
  - 按“目录 + 文件名”的方式保存视频
  - 格式与编码选项对齐 ComfyUI 官方 `SaveVideo`
  - 当 `codec=h264` 且 `encoding_mode=re-encode` 时，会使用 `crf`
  - 其他情况下会尽量保持官方自动复用/自动转码行为

## 路径与命名规则

所有保存节点都遵循下面这套规则：

- `directory` 为空时，默认保存到 ComfyUI 的 `output` 目录
- `directory` 为相对路径时，会以 ComfyUI 的 `output` 目录为基准
- `directory` 为绝对路径时，直接保存到指定位置
- `filename` 只需要填写文件名本体，不需要带扩展名
- 如果 `filename` 里误带了扩展名，节点会自动去掉，再使用节点固定扩展名

例如：

```text
directory = my_outputs/test
filename = demo_file
```

图像会保存为：

```text
<ComfyUI output>/my_outputs/test/demo_file.png
```

Latent 会保存为：

```text
<ComfyUI output>/my_outputs/test/demo_file.latent
```

音频会保存为：

```text
<ComfyUI output>/my_outputs/test/demo_file.flac
```

视频会保存为：

```text
<ComfyUI output>/my_outputs/test/demo_file.mp4
```

## 允许覆盖规则

所有保存节点都提供 `allow_overwrite` 选项：

- `True`：如果目标文件已存在，直接覆盖
- `False`：如果目标文件已存在，自动避让重名

自动避让重名时，文件名会追加编号，例如：

```text
demo_file.png
demo_file_00001.png
demo_file_00002.png
```

如果图像输入本身是 batch，多张图也会自动附加批次编号。

## Latent 读取与保存说明

本项目中的 `HT 从文件加载 Latent` 和 `HT 保存 Latent 到文件` 已对齐 ComfyUI 官方 latent 的保存/读取方式。

对于 ComfyUI 官方 `SaveLatent` 导出的文件：

- 扩展名通常为 `.latent`
- 文件内容实际按 `safetensors` 格式保存
- 内部核心键通常为：
  - `latent_tensor`
  - `latent_format_version_0`

节点会自动将其转换为 ComfyUI 可继续传递的：

```python
{"samples": ...}
```

如果文件中存在 `noise_mask`，也会一并保留和保存。

## 安装方式

将本目录放入 ComfyUI 的 `custom_nodes` 目录下，例如：

```text
ComfyUI/custom_nodes/ComfyUI-HTNodes
```

然后重启 ComfyUI。

## 目录结构

```text
ComfyUI-HTNodes/
├─ __init__.py
├─ .gitignore
├─ README.md
└─ HTNodes/
   ├─ __init__.py
   ├─ common.py
   ├─ file_loaders.py
   └─ file_savers.py
```

## 使用建议

- 加载节点适合与路径拼接、文本生成、批处理类节点联动使用
- 保存节点适合在流程中直接把结果落盘到固定位置
- `Latent` 节点更适合读取和保存 ComfyUI 官方导出的 latent 文件
- 音频/视频节点尽量复用 ComfyUI 官方媒体读写行为，适合与原版 `AUDIO` / `VIDEO` 工作流直接互通
- 对于来源不明的 `.pt/.pth/.ckpt` 文件，建议先确认文件来源可信

## 后续可扩展方向

- 从文件加载 `MASK`
- 从文件加载 `AUDIO`
- 从文件加载 `JSON`
- 保存 `MASK`
- 自定义图像扩展名
- 批量路径处理节点
