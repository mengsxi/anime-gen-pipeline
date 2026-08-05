"""Gradio WebUI。"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import AppConfig, load_config
from .controlnet import ControlMode
from .pipeline import DEFAULT_NEGATIVE_PROMPT, AnimePipeline, GenParams

logger = logging.getLogger(__name__)


def create_ui(config: AppConfig | None = None):
    """构建 WebUI；所有模型事件共享一个串行队列。"""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            'WebUI 依赖未安装，请运行: pip install -e ".[webui]"'
        ) from exc

    config = config or load_config()
    pipe_holder = {"pipeline": None}
    gpu_event = {"concurrency_id": "anime_gpu", "concurrency_limit": 1}
    lora_dir = Path(config.models.lora_dir)
    lora_choices = (
        [str(path) for path in sorted(lora_dir.glob("*.safetensors"))]
        if lora_dir.exists()
        else []
    )

    def get_pipeline() -> AnimePipeline:
        if pipe_holder["pipeline"] is None:
            pipe_holder["pipeline"] = AnimePipeline(config)
        return pipe_holder["pipeline"]

    def make_progress_callback(progress):
        def callback(current: int, total: int):
            progress(
                min(0.95, 0.2 + 0.75 * current / max(total, 1)),
                desc=f"采样 {current}/{total}",
            )

        return callback

    def common_params(
        prompt, negative_prompt, width, height, steps, cfg, seed, clip_skip
    ):
        return GenParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            cfg_scale=float(cfg),
            seed=int(seed),
            clip_skip=int(clip_skip) if clip_skip else None,
        )

    def on_generate_txt2img(
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg,
        seed,
        clip_skip,
        batch_size,
        lora_path,
        lora_scale,
        progress=gr.Progress(),
    ):
        pipeline = get_pipeline()
        params = common_params(
            prompt, negative_prompt, width, height, steps, cfg, seed, clip_skip
        )
        params.batch_size = int(batch_size)
        params.lora_path = (
            lora_path.strip() if lora_path and lora_path.strip() else None
        )
        params.lora_scale = float(lora_scale)
        progress(0.1, desc="准备模型...")
        result = pipeline.txt2img(params, make_progress_callback(progress))
        progress(1.0, desc=f"完成 ({result.elapsed:.1f}s)")
        return result.images, pipeline.engine.get_vram_info(), result.seed_used

    def on_generate_img2img(
        prompt,
        negative_prompt,
        init_image,
        strength,
        width,
        height,
        steps,
        cfg,
        seed,
        clip_skip,
        progress=gr.Progress(),
    ):
        if init_image is None:
            raise gr.Error("请上传初始图片")
        pipeline = get_pipeline()
        params = common_params(
            prompt, negative_prompt, width, height, steps, cfg, seed, clip_skip
        )
        params.init_image = init_image
        params.strength = float(strength)
        progress(0.1, desc="准备模型...")
        result = pipeline.img2img(params, make_progress_callback(progress))
        progress(1.0, desc=f"完成 ({result.elapsed:.1f}s)")
        return result.images[0], pipeline.engine.get_vram_info(), result.seed_used

    def on_generate_inpaint(
        prompt,
        negative_prompt,
        init_image,
        mask_image,
        strength,
        width,
        height,
        steps,
        cfg,
        seed,
        clip_skip,
        progress=gr.Progress(),
    ):
        if init_image is None or mask_image is None:
            raise gr.Error("请同时上传原图和蒙版")
        pipeline = get_pipeline()
        params = common_params(
            prompt, negative_prompt, width, height, steps, cfg, seed, clip_skip
        )
        params.init_image = init_image
        params.mask_image = mask_image.convert("L")
        params.strength = float(strength)
        progress(0.1, desc="准备模型...")
        result = pipeline.inpaint(params, make_progress_callback(progress))
        progress(1.0, desc=f"完成 ({result.elapsed:.1f}s)")
        return result.images[0], pipeline.engine.get_vram_info(), result.seed_used

    def on_generate_control(
        prompt,
        negative_prompt,
        control_image,
        control_mode,
        control_weight,
        width,
        height,
        steps,
        cfg,
        seed,
        clip_skip,
        progress=gr.Progress(),
    ):
        if control_image is None:
            raise gr.Error("请上传 ControlNet 条件图")
        pipeline = get_pipeline()
        params = common_params(
            prompt, negative_prompt, width, height, steps, cfg, seed, clip_skip
        )
        params.control_weight = float(control_weight)
        progress(0.05, desc="预处理条件图...")
        result = pipeline.generate_with_control(
            params,
            control_image=control_image,
            control_mode=control_mode,
            progress_callback=make_progress_callback(progress),
        )
        progress(1.0, desc=f"完成 ({result.elapsed:.1f}s)")
        return result.images[0], pipeline.engine.get_vram_info(), result.seed_used

    def on_unload():
        if pipe_holder["pipeline"] is not None:
            pipe_holder["pipeline"].cleanup()
            pipe_holder["pipeline"] = None
        return "模型已卸载", "VRAM 已释放"

    css = """
    .anime-title {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .anime-subtitle { text-align: center; color: #888; margin-bottom: 20px; }
    """

    with gr.Blocks(
        css=css,
        title="Anime Gen Pipeline",
        theme=config.webui.theme,
    ) as demo:
        gr.HTML('<div class="anime-title">Anime Gen Pipeline</div>')
        gr.HTML(
            '<div class="anime-subtitle">2D 动漫可控图像生成 · 低配 PC 优化版</div>'
        )
        vram_display = gr.Textbox(
            label="设备状态", value="模型未加载", interactive=False
        )

        with gr.Accordion("全局生成设置", open=False):
            with gr.Row():
                width = gr.Slider(
                    256,
                    1024,
                    value=config.inference.default_width,
                    step=64,
                    label="宽度",
                )
                height = gr.Slider(
                    256,
                    1024,
                    value=config.inference.default_height,
                    step=64,
                    label="高度",
                )
            with gr.Row():
                steps = gr.Slider(
                    1,
                    50,
                    value=config.inference.default_steps,
                    step=1,
                    label="采样步数",
                )
                cfg = gr.Slider(
                    0.0, 15.0, value=config.inference.default_cfg, step=0.5, label="CFG"
                )
                clip_skip = gr.Number(
                    value=config.inference.clip_skip or 0,
                    precision=0,
                    label="Clip Skip (0=关闭)",
                )

        common_generation_inputs = [width, height, steps, cfg]

        with gr.Tabs():
            with gr.TabItem("文生图"):
                with gr.Row():
                    with gr.Column(scale=2):
                        txt_prompt = gr.Textbox(label="提示词", lines=3)
                        txt_negative = gr.Textbox(
                            label="负面提示词", value=DEFAULT_NEGATIVE_PROMPT, lines=2
                        )
                        with gr.Row():
                            txt_seed = gr.Number(
                                label="Seed (-1=随机)",
                                value=config.inference.default_seed,
                                precision=0,
                            )
                            txt_batch = gr.Slider(1, 4, value=1, step=1, label="批量")
                        with gr.Accordion("LoRA", open=False):
                            txt_lora = gr.Dropdown(
                                choices=lora_choices,
                                value=None,
                                label="LoRA（来自配置目录）",
                            )
                            txt_lora_scale = gr.Slider(
                                0.0, 1.5, value=0.75, step=0.05, label="LoRA 强度"
                            )
                        txt_button = gr.Button("生成", variant="primary")
                        txt_seed_out = gr.Number(label="实际 Seed", interactive=False)
                    with gr.Column(scale=1):
                        txt_output = gr.Gallery(
                            label="生成结果", type="pil", columns=2, height=500
                        )
                txt_button.click(
                    fn=on_generate_txt2img,
                    inputs=[
                        txt_prompt,
                        txt_negative,
                        *common_generation_inputs,
                        txt_seed,
                        clip_skip,
                        txt_batch,
                        txt_lora,
                        txt_lora_scale,
                    ],
                    outputs=[txt_output, vram_display, txt_seed_out],
                    **gpu_event,
                )

            with gr.TabItem("图生图"):
                img_prompt = gr.Textbox(label="提示词", lines=3)
                img_negative = gr.Textbox(
                    label="负面提示词", value=DEFAULT_NEGATIVE_PROMPT, lines=2
                )
                with gr.Row():
                    img_init = gr.Image(label="初始图片", type="pil")
                    img_output = gr.Image(label="生成结果", type="pil")
                with gr.Row():
                    img_strength = gr.Slider(0.0, 1.0, value=0.75, label="重绘强度")
                    img_seed = gr.Number(
                        label="Seed", value=config.inference.default_seed, precision=0
                    )
                    img_seed_out = gr.Number(label="实际 Seed", interactive=False)
                img_button = gr.Button("生成", variant="primary")
                img_button.click(
                    fn=on_generate_img2img,
                    inputs=[
                        img_prompt,
                        img_negative,
                        img_init,
                        img_strength,
                        *common_generation_inputs,
                        img_seed,
                        clip_skip,
                    ],
                    outputs=[img_output, vram_display, img_seed_out],
                    **gpu_event,
                )

            with gr.TabItem("局部重绘"):
                paint_prompt = gr.Textbox(label="提示词", lines=3)
                paint_negative = gr.Textbox(
                    label="负面提示词", value=DEFAULT_NEGATIVE_PROMPT, lines=2
                )
                with gr.Row():
                    paint_init = gr.Image(label="原图", type="pil")
                    paint_mask = gr.Image(
                        label="蒙版（白色区域重绘）", type="pil", image_mode="L"
                    )
                    paint_output = gr.Image(label="生成结果", type="pil")
                with gr.Row():
                    paint_strength = gr.Slider(0.0, 1.0, value=0.75, label="重绘强度")
                    paint_seed = gr.Number(
                        label="Seed", value=config.inference.default_seed, precision=0
                    )
                    paint_seed_out = gr.Number(label="实际 Seed", interactive=False)
                paint_button = gr.Button("局部重绘", variant="primary")
                paint_button.click(
                    fn=on_generate_inpaint,
                    inputs=[
                        paint_prompt,
                        paint_negative,
                        paint_init,
                        paint_mask,
                        paint_strength,
                        *common_generation_inputs,
                        paint_seed,
                        clip_skip,
                    ],
                    outputs=[paint_output, vram_display, paint_seed_out],
                    **gpu_event,
                )

            with gr.TabItem("ControlNet"):
                control_prompt = gr.Textbox(label="提示词", lines=3)
                control_negative = gr.Textbox(
                    label="负面提示词", value=DEFAULT_NEGATIVE_PROMPT, lines=2
                )
                with gr.Row():
                    control_input = gr.Image(label="条件输入图", type="pil")
                    control_output = gr.Image(label="生成结果", type="pil")
                with gr.Row():
                    control_mode = gr.Dropdown(
                        choices=[mode.value for mode in ControlMode],
                        value=ControlMode.LINEART.value,
                        label="控制模式",
                    )
                    control_weight = gr.Slider(
                        0.0, 1.5, value=0.8, step=0.05, label="控制权重"
                    )
                    control_seed = gr.Number(
                        label="Seed", value=config.inference.default_seed, precision=0
                    )
                    control_seed_out = gr.Number(label="实际 Seed", interactive=False)
                control_button = gr.Button("生成", variant="primary")
                control_button.click(
                    fn=on_generate_control,
                    inputs=[
                        control_prompt,
                        control_negative,
                        control_input,
                        control_mode,
                        control_weight,
                        *common_generation_inputs,
                        control_seed,
                        clip_skip,
                    ],
                    outputs=[control_output, vram_display, control_seed_out],
                    **gpu_event,
                )

            with gr.TabItem("系统"):
                gr.Markdown(
                    "首次运行 `anime-gen-download` 下载基础模型；ControlNet 模型请按需下载。"
                )
                unload_button = gr.Button("卸载模型并释放内存")
                unload_status = gr.Textbox(label="状态", interactive=False)
                unload_button.click(
                    fn=on_unload,
                    outputs=[unload_status, vram_display],
                    **gpu_event,
                )

    return demo


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()
    if config.webui.share and not config.inference.enable_safety_checker:
        logger.warning("WebUI 开启公网分享，但 safety checker 已关闭")
    demo = create_ui(config)
    demo.queue(max_size=10, default_concurrency_limit=1)
    demo.launch(
        server_name=config.webui.host,
        server_port=config.webui.port,
        share=config.webui.share,
    )


if __name__ == "__main__":
    main()
