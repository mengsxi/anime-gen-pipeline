# Anime Gen Pipeline

<div align="center">

**2D 动漫可控图像生成 Pipeline · 低配 PC 优化版**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.27%2B-orange)](https://github.com/huggingface/diffusers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

*让 4GB 显存的显卡也能愉快地生成动漫图片*

</div>

---

## 特性

- **多种生成模式**：文生图 / 图生图 / 局部重绘 / ControlNet 可控生成
- **6 种 ControlNet 控制**：OpenPose 骨骼 · 线稿 · 涂鸦 · Canny 边缘 · 深度 · 软边缘
- **低配优化**：为 4-8GB VRAM 设备深度优化, 支持 CPU offload
- **LoRA 支持**：加载角色/风格 LoRA 模型
- **双界面**：Gradio WebUI + 命令行 CLI
- **PNG 元数据**：生成参数自动嵌入 PNG, 兼容 A1111
- **纯 Python**：基于 HuggingFace diffusers, 无依赖黑洞

## 快速开始

### 环境要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| VRAM | 4 GB | 8+ GB |
| RAM | 8 GB | 16 GB |
| Python | 3.10 | 3.11 |
| OS | Windows / Linux / macOS | Windows / Linux |

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/anime-gen-pipeline.git
cd anime-gen-pipeline

# 创建虚拟环境
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 安装
pip install -e .
```

### 下载模型

```bash
# 一键下载全部模型 (基础模型 + ControlNet)
anime-gen-download

# 仅下载基础模型
anime-gen-download --base-only

# 仅下载指定 ControlNet
anime-gen-download --controlnet pose,lineart,canny
```

### 启动 WebUI

```bash
anime-gen-webui
# 浏览器打开 http://127.0.0.1:7860
```

### 命令行使用

```bash
# 文生图
anime-gen --prompt "1girl, sunset, masterpiece, best quality" -o outputs/

# 图生图
anime-gen --prompt "anime style, detailed" --init-image photo.jpg --strength 0.7

# ControlNet 线稿上色
anime-gen --prompt "1girl, blue hair" --control-image sketch.png --control-mode lineart

# 批量 + 固定种子
anime-gen --prompt "1girl, beach" --batch 4 --seed 42 --steps 25

# 使用 LoRA
anime-gen --prompt "1girl" --lora ./models/lora/my_character.safetensors --lora-scale 0.8
```

## Python API

```python
from anime_gen import AnimePipeline, GenParams, ControlMode

# 初始化
pipe = AnimePipeline()

# 文生图
result = pipe.txt2img(GenParams(
    prompt="1girl, solo, long hair, looking at viewer, masterpiece",
    width=512, height=768, steps=20,
))
result.images[0].save("output.png")

# ControlNet — 线稿上色
result = pipe.generate_with_control(
    GenParams(prompt="1girl, blue hair, school uniform"),
    control_image="my_sketch.png",
    control_mode=ControlMode.LINEART,
)

# 清理
pipe.cleanup()
```

## 项目结构

```
anime-gen-pipeline/
├── configs/
│   └── default.yaml          # 默认配置 (模型路径/优化选项/WebUI设置)
├── src/anime_gen/
│   ├── __init__.py           # 包入口
│   ├── config.py             # 配置管理
│   ├── engine.py             # 模型引擎 (加载/卸载/低配优化)
│   ├── pipeline.py           # 核心生成 Pipeline
│   ├── controlnet.py         # ControlNet 预处理器
│   ├── webui.py              # Gradio WebUI
│   ├── cli.py                # 命令行入口
│   └── download.py           # 模型下载工具
├── pyproject.toml            # 项目元数据
├── requirements.txt          # 依赖清单
├── LICENSE                   # MIT 开源协议
└── README.md
```

## 低配优化策略

| 优化项 | VRAM 节省 | 性能影响 | 说明 |
|--------|-----------|----------|------|
| Attention Slicing | ~30% | 略慢 | 切片计算注意力 |
| VAE Slicing | ~40% | 略慢 | 分块 VAE 解码 |
| VAE Tiling | 大幅节省 | 中等 | 大图必须开启 |
| CPU Offload | 极多 | 明显变慢 | 4GB 显存首选 |
| xformers | ~20% | 无影响 | 高效注意力实现 |
| ToMe | ~15% | 质量略降 | Token 合并加速 |

配置文件 `configs/default.yaml` 中可自由组合这些优化:

```yaml
low_vram:
  enable_attention_slicing: true
  enable_vae_slicing: true
  enable_vae_tiling: true
  enable_model_cpu_offload: false   # 4GB显存时开启
  enable_xformers: true
```

## ControlNet 模式速查

| 模式 | 用途 | 输入 | 示例场景 |
|------|------|------|----------|
| `pose` | 骨骼姿态 | 人物照片 | "让 AI 画的角色摆这个动作" |
| `lineart` | 线稿上色 | 线稿图 | 给草稿上色 |
| `scribble` | 涂鸦引导 | 随手涂鸦 | 快速构图 |
| `canny` | 边缘轮廓 | 任意图片 | 保持原图轮廓 |
| `depth` | 空间结构 | 场景照片 | 保持空间关系 |
| `softedge` | 柔和引导 | 任意图片 | 参考构图但更自由 |

## 推荐模型

### 基础模型 (SD 1.5 动漫系)
- [Anything V5](https://huggingface.co/andite/anything-v5.0) — 通用动漫 (默认)
- [Counterfeit V3.0](https://civitai.com/models/4468/counterfeit-v30) — 精致风格
- [AOM3 (AbyssOrangeMix3)](https://huggingface.co/WarriorMama777/OrangeMixs) — 色彩丰富
- [MeinaMix](https://civitai.com/models/7240/meinamix) — 东亚风格

### LoRA
- 在 [CivitAI](https://civitai.com/) 搜索角色/风格 LoRA, 放入 `./models/lora/`

## 常见问题

### Q: 4GB 显存能跑吗?
A: 可以。在 `configs/default.yaml` 中开启 `enable_model_cpu_offload: true`, 虽然慢一些, 但能正常生成。

### Q: 下载模型太慢?
A: 可以设置 HuggingFace 镜像:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
或在 `download.py` 中取消注释镜像地址。

### Q: 支持 SDXL 吗?
A: 当前版本专注于 SD 1.5 (兼容性最好, VRAM 消耗最低)。SDXL 支持在 Roadmap 中。

### Q: 生成出来的图能商用吗?
A: 取决于你使用的模型许可。本项目代码是 MIT 协议, 但模型有各自的许可, 请查看模型页面。

## Roadmap

- [ ] SDXL 支持
- [ ] LCM/Turbo 快速采样 (2-4步出图)
- [ ] IP-Adapter (图像风格参考)
- [ ] 批量队列 + 进度恢复
- [ ] ControlNet 多条件融合
- [ ] GGUF/ONNX 量化推理
- [ ] 一键打包 Windows exe

## 贡献

欢迎 PR! 请确保:

1. 代码通过 `ruff check` 和 `black`
2. 新功能有对应的测试
3. README 有相应更新

## 致谢

- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Gradio](https://github.com/gradio-app/gradio)

## License

MIT License — 详见 [LICENSE](LICENSE)

---

<div align="center">
  <sub>Made with ❤️ for the anime AI community</sub>
</div>
