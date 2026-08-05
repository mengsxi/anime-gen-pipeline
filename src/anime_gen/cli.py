"""
命令行入口

用法:
    # 文生图
    anime-gen --prompt "1girl, masterpiece" -o output.png

    # 图生图
    anime-gen --prompt "anime style" --init-image photo.jpg --strength 0.7

    # ControlNet
    anime-gen --prompt "1girl" --control-image sketch.png --control-mode lineart

    # 批量
    anime-gen --prompt "1girl, beach" --batch 4 --seed 42

    # 下载模型
    anime-gen-download
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .controlnet import ControlMode
from .pipeline import AnimePipeline, GenParams


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anime-gen",
        description="2D 动漫可控图像生成 Pipeline",
    )

    # 生成参数
    p.add_argument("--prompt", "-p", type=str, required=True, help="正向提示词")
    p.add_argument("--negative-prompt", "-n", type=str, default=None, help="负面提示词")
    p.add_argument("--steps", type=int, default=20, help="推理步数 (默认 20)")
    p.add_argument("--cfg", type=float, default=7.0, help="CFG scale (默认 7.0)")
    p.add_argument("--seed", type=int, default=-1, help="随机种子, -1 为随机")
    p.add_argument("--width", "-W", type=int, default=512, help="宽度 (默认 512)")
    p.add_argument("--height", "-H", type=int, default=768, help="高度 (默认 768)")
    p.add_argument("--batch", "-b", type=int, default=1, help="单次生成数量")

    # 模式
    p.add_argument("--init-image", "-i", type=str, default=None, help="img2img 初始图片路径")
    p.add_argument("--strength", type=float, default=0.75, help="img2img 重绘强度")
    p.add_argument("--mask-image", "-m", type=str, default=None, help="inpaint 蒙版图片路径")
    p.add_argument("--control-image", "-c", type=str, default=None, help="ControlNet 条件图路径")
    p.add_argument("--control-mode", type=str, default=None,
                   choices=[m.value for m in ControlMode],
                   help="ControlNet 模式")
    p.add_argument("--control-weight", type=float, default=0.8, help="ControlNet 权重")

    # LoRA
    p.add_argument("--lora", type=str, default=None, help="LoRA 权重文件路径")
    p.add_argument("--lora-scale", type=float, default=0.75, help="LoRA 强度")

    # 输出
    p.add_argument("--output", "-o", type=str, default="./outputs", help="输出目录")
    p.add_argument("--prefix", type=str, default="anime", help="文件名前缀")
    p.add_argument("--no-metadata", action="store_true", help="不嵌入生成参数到 PNG")

    # 配置
    p.add_argument("--config", type=str, default=None, help="配置文件路径")
    p.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 加载配置
    config = load_config(args.config)

    # 构造参数
    params = GenParams(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or GenParams.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        batch_size=args.batch,
        strength=args.strength,
        lora_path=args.lora,
        lora_scale=args.lora_scale,
        output_dir=args.output,
        filename_prefix=args.prefix,
        save_metadata=not args.no_metadata,
    )

    # 加载图片
    if args.init_image:
        from PIL import Image
        params.init_image = Image.open(args.init_image).convert("RGB")

    if args.mask_image:
        from PIL import Image
        params.mask_image = Image.open(args.mask_image).convert("L")

    if args.control_image:
        from PIL import Image
        control_img = Image.open(args.control_image).convert("RGB")

    # 初始化 Pipeline
    pipe = AnimePipeline(config)

    try:
        # 选择模式
        if args.mask_image:
            result = pipe.inpaint(params)
        elif args.init_image and not args.control_image:
            result = pipe.img2img(params)
        elif args.control_image and args.control_mode:
            result = pipe.generate_with_control(
                params,
                control_image=control_img,
                control_mode=args.control_mode,
            )
        else:
            result = pipe.txt2img(params)

        # 报告结果
        print(f"\n✓ 生成完成!")
        print(f"  Seed: {result.seed_used}")
        print(f"  耗时: {result.elapsed:.1f}s")
        print(f"  输出: {len(result.images)} 张图片")

        for path in result.metadata.get("saved_paths", []):
            print(f"    → {path}")

    finally:
        pipe.cleanup()


if __name__ == "__main__":
    main()
