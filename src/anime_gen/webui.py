"""
Gradio Web UI — 提供可视化的动漫图像生成交互界面

运行:
    python -m anime_gen.webui
    或
    anime-gen-webui
"""

import logging

from .config import AppConfig, load_config
from .controlnet import ControlMode
from .pipeline import AnimePipeline, GenParams

logger = logging.getLogger(__name__)


def create_ui(config: AppConfig | None = None):
    """构建 Gradio 界面"""
    import gradio as gr

    config = config or load_config()

    # --- 自定义 CSS ---
    css = """
    .anime-title {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .anime-subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 20px;
    }
    .vram-info {
        color: #4ade80;
        font-weight: bold;
    }
    """

    pipe_holder = {"pipeline": None}

    def get_pipeline():
        if pipe_holder["pipeline"] is None:
            pipe_holder["pipeline"] = AnimePipeline(config)
        return pipe_holder["pipeline"]

    # ================================================================
    # 事件处理
    # ================================================================

    def on_generate_txt2img(
        prompt, negative_prompt, width, height, steps, cfg, seed, batch_size,
        progress=gr.Progress(),
    ):
        pipe = get_pipeline()
        params = GenParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            cfg_scale=float(cfg),
            seed=int(seed),
            batch_size=int(batch_size),
        )
        progress(0.3, desc="生成中...")
        result = pipe.txt2img(params)
        progress(1.0, desc=f"完成 ({result.elapsed:.1f}s, seed={result.seed_used})")
        vram = pipe.engine.get_vram_info()
        return result.images[0], vram, result.seed_used

    def on_generate_img2img(
        prompt, negative_prompt, init_image, strength,
        width, height, steps, cfg, seed,
        progress=gr.Progress(),
    ):
        if init_image is None:
            raise gr.Error("请上传初始图片")

        pipe = get_pipeline()
        params = GenParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            cfg_scale=float(cfg),
            seed=int(seed),
            init_image=init_image,
            strength=float(strength),
        )
        progress(0.3, desc="生成中...")
        result = pipe.img2img(params)
        progress(1.0, desc=f"完成 ({result.elapsed:.1f}s)")
        vram = pipe.engine.get_vram_info()
        return result.images[0], vram, result.seed_used

    def on_generate_control(
        prompt, negative_prompt, control_image, control_mode, control_weight,
        width, height, steps, cfg, seed,
        progress=gr.Progress(),
    ):
        if control_image is None:
            raise gr.Error("请上传 ControlNet 条件图")

        pipe = get_pipeline()
        params = GenParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            cfg_scale=float(cfg),
            seed=int(seed),
            control_weight=float(control_weight),
        )
        progress(0.2, desc="预处理条件图...")
        result = pipe.generate_with_control(
            params,
            control_image=control_image,
            control_mode=control_mode,
        )
        progress(1.0, desc=f"完成 ({result.elapsed:.1f}s)")
        vram = pipe.engine.get_vram_info()
        return result.images[0], vram, result.seed_used

    def on_unload():
        if pipe_holder["pipeline"] is not None:
            pipe_holder["pipeline"].cleanup()
            pipe_holder["pipeline"] = None
        return "模型已卸载", "VRAM 已释放"

    # ================================================================
    # 界面布局
    # ================================================================

    with gr.Blocks(
        css=css,
        title="Anime Gen Pipeline",
        theme=config.webui.theme,
    ) as demo:
        gr.HTML('<div class="anime-title">Anime Gen Pipeline</div>')
        gr.HTML('<div class="anime-subtitle">2D 动漫可控图像生成 · 低配 PC 优化版</div>')

        vram_display = gr.Textbox(
            label="VRAM 状态",
            value="模型未加载",
            interactive=False,
        )

        with gr.Tabs():
            # ==================== Tab 1: 文生图 ====================
            with gr.TabItem("文生图 (txt2img)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        txt2img_prompt = gr.Textbox(
                            label="提示词 (Prompt)",
                            placeholder="1girl, solo, long hair, blue eyes, "
                                        "school uniform, looking at viewer, "
                                        "masterpiece, best quality, 8k",
                            lines=3,
                        )
                        txt2img_neg = gr.Textbox(
                            label="负面提示词",
                            value="lowres, bad anatomy, bad hands, extra fingers, "
                                  "worst quality, low quality, ugly, deformed",
                            lines=2,
                        )
                        with gr.Row():
                            txt2img_seed = gr.Number(
                                label="Seed (-1=随机)",
                                value=config.inference.default_seed,
                                precision=0,
                            )
                            txt2img_batch = gr.Slider(
                                1,
                                4,
                                value=config.inference.default_batch_size,
                                step=1,
                                label="Batch Size",
                            )
                        with gr.Row():
                            txt2img_btn = gr.Button("生成", variant="primary", scale=3)
                            txt2img_seed_out = gr.Number(label="实际 Seed", interactive=False)

                    with gr.Column(scale=1):
                        txt2img_out = gr.Image(label="生成结果", type="pil", height=400)

                with gr.Accordion("高级设置", open=False):
                    with gr.Row():
                        txt2img_w = gr.Slider(
                            256,
                            1024,
                            value=config.inference.default_width,
                            step=64,
                            label="宽度",
                        )
                        txt2img_h = gr.Slider(
                            256,
                            1024,
                            value=config.inference.default_height,
                            step=64,
                            label="高度",
                        )
                    with gr.Row():
                        txt2img_steps = gr.Slider(
                            1,
                            50,
                            value=config.inference.default_steps,
                            step=1,
                            label="步数",
                        )
                        txt2img_cfg = gr.Slider(
                            0.0,
                            15.0,
                            value=config.inference.default_cfg,
                            step=0.5,
                            label="CFG Scale",
                        )

                txt2img_btn.click(
                    fn=on_generate_txt2img,
                    inputs=[
                        txt2img_prompt, txt2img_neg,
                        txt2img_w, txt2img_h, txt2img_steps, txt2img_cfg,
                        txt2img_seed, txt2img_batch,
                    ],
                    outputs=[txt2img_out, vram_display, txt2img_seed_out],
                )

            # ==================== Tab 2: 图生图 ====================
            with gr.TabItem("图生图 (img2img)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        img2img_prompt = gr.Textbox(label="提示词", lines=3)
                        img2img_neg = gr.Textbox(label="负面提示词", lines=2)
                        with gr.Row():
                            img2img_strength = gr.Slider(
                                0.1, 1.0, value=0.75, step=0.05, label="重绘强度"
                            )
                            img2img_seed = gr.Number(
                                label="Seed",
                                value=config.inference.default_seed,
                                precision=0,
                            )
                        with gr.Row():
                            img2img_btn = gr.Button("生成", variant="primary")
                            img2img_seed_out = gr.Number(label="实际 Seed", interactive=False)

                    with gr.Column(scale=1):
                        img2img_init = gr.Image(label="初始图片", type="pil")
                        img2img_out = gr.Image(label="生成结果", type="pil")

                img2img_btn.click(
                    fn=on_generate_img2img,
                    inputs=[
                        img2img_prompt, img2img_neg, img2img_init, img2img_strength,
                        txt2img_w, txt2img_h, txt2img_steps, txt2img_cfg, img2img_seed,
                    ],
                    outputs=[img2img_out, vram_display, img2img_seed_out],
                )

            # ==================== Tab 3: ControlNet ====================
            with gr.TabItem("ControlNet 可控生成"):
                with gr.Row():
                    with gr.Column(scale=2):
                        cn_prompt = gr.Textbox(label="提示词", lines=3)
                        cn_neg = gr.Textbox(label="负面提示词", lines=2)
                        cn_mode = gr.Dropdown(
                            choices=[m.value for m in ControlMode],
                            value="lineart",
                            label="控制模式",
                        )
                        cn_weight = gr.Slider(
                            0.1, 1.5, value=0.8, step=0.05, label="ControlNet 权重"
                        )
                        with gr.Row():
                            cn_seed = gr.Number(
                                label="Seed",
                                value=config.inference.default_seed,
                                precision=0,
                            )
                            cn_btn = gr.Button("生成", variant="primary")
                            cn_seed_out = gr.Number(label="实际 Seed", interactive=False)

                    with gr.Column(scale=1):
                        cn_image = gr.Image(label="条件输入图 (照片/草图)", type="pil")
                        cn_out = gr.Image(label="生成结果", type="pil")

                cn_btn.click(
                    fn=on_generate_control,
                    inputs=[
                        cn_prompt, cn_neg, cn_image, cn_mode, cn_weight,
                        txt2img_w, txt2img_h, txt2img_steps, txt2img_cfg, cn_seed,
                    ],
                    outputs=[cn_out, vram_display, cn_seed_out],
                )

            # ==================== Tab 4: 系统 ====================
            with gr.TabItem("系统"):
                gr.Markdown("""
                ### 使用说明

                1. **首次使用**：运行 `anime-gen-download` 下载基础模型
                2. **配置**：编辑 `configs/default.yaml` 调整模型路径和优化选项
                3. **ControlNet**：上传任意图片 + 选择控制模式 → 生成对应风格动漫图
                   - **Pose**: 提取骨骼姿态 → 人物动作控制
                   - **Lineart**: 提取线稿 → 线稿上色
                   - **Canny**: 边缘检测 → 轮廓保持
                   - **Depth**: 深度估计 → 空间结构保持
                   - **SoftEdge**: 软边缘 → 柔和的结构引导
                   - **Scribble**: 涂鸦 → 自由构图

                ### 低配优化建议
                - 4GB VRAM: 开启 `model_cpu_offload`
                - 6GB VRAM: 开启 `attention_slicing` + `vae_tiling`
                - 8GB+ VRAM: 可关闭部分优化以获得更好速度

                ### 模型推荐
                - 基础模型: Anything V5 / Counterfeit / AOM3
                - LoRA: 在 CivitAI 下载对应角色/风格 LoRA
                """)

                unload_btn = gr.Button("卸载模型 (释放 VRAM)", variant="secondary")
                unload_status = gr.Textbox(label="状态", interactive=False)
                unload_btn.click(fn=on_unload, outputs=[unload_status, vram_display])

    return demo


def main():
    """WebUI 入口"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()
    wc = config.webui

    demo = create_ui(config)
    demo.queue(max_size=10)
    demo.launch(
        server_name=wc.host,
        server_port=wc.port,
        share=wc.share,
    )


if __name__ == "__main__":
    main()
