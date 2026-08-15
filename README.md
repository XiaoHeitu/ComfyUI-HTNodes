# ComfyUI-HTNodes

`ComfyUI-HTNodes` 是一个用于 `ComfyUI` 的自定义节点包，当前提供按文件路径直接加载资源的节点。

目前包含以下节点：

- `HT 从文件加载图像`
- `HT 从文件加载文本`
- `HT 从文件加载 Latent`

节点分组：

- `HTNodes/Loaders`

## 功能说明

### 1. 从文件加载图像

- 输入：`path` (`STRING`)
- 输出：`image` (`IMAGE`)
- 说明：
  - 通过任意字符串路径加载图像文件
  - 自动处理 EXIF 方向
  - 输出为 ComfyUI 标准 `IMAGE` 类型

### 2. 从文件加载文本

- 输入：`path` (`STRING`)
- 输出：`text` (`STRING`)
- 说明：
  - 通过任意字符串路径加载文本文件
  - 当前支持 `utf-8`、`utf-8-sig`、`gb18030`

### 3. 从文件加载 Latent

- 输入：`path` (`STRING`)
- 输出：`latent` (`LATENT`)
- 说明：
  - 优先按 ComfyUI 官方 latent 保存方式读取
  - 支持 ComfyUI 官方 `SaveLatent` 导出的 `.latent`
  - 兼容 `.safetensors`、`.sft`、`.npy`、`.npz`、`.pt`、`.pth`、`.bin`、`.ckpt`

## Latent 读取说明

本项目里的 `从文件加载 Latent` 已对齐 ComfyUI 官方 latent 的保存/读取方式。

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

如果文件中存在 `noise_mask`，也会一并保留。

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
├─ README.md
└─ HTNodes/
   ├─ __init__.py
   └─ file_loaders.py
```

## 使用建议

- 图像、文本节点适合与路径生成类节点联动使用
- `Latent` 节点更适合读取 ComfyUI 官方导出的 latent 文件
- 对于来源不明的 `.pt/.pth/.ckpt` 文件，建议先确认文件来源可信

## 当前状态

当前仓库是一个精简版节点包，核心目标是提供简单直接的“从文件加载资源”能力。  
后续如果需要，可以继续扩展：

- 从文件加载 `MASK`
- 从文件加载 `AUDIO`
- 从文件加载 `JSON`
- 路径校验/批量加载类节点
