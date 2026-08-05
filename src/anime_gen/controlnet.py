"""
ControlNet 预处理与集成模块

支持的控制模式:
  - pose:     OpenPose 骨骼 (适合人物姿态控制)
  - lineart:  线稿提取 (适合线稿上色)
  - scribble: 涂鸦引导 (适合草图生成)
  - canny:    Canny 边缘检测 (适合轮廓控制)
  - depth:    深度图 (适合空间结构控制)
  - softedge: 软边缘 (HED, 更柔和的边缘)
"""

import gc
import importlib
import logging
from enum import Enum
from typing import Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    POSE = "pose"
    LINEART = "lineart"
    SCRIBBLE = "scribble"
    CANNY = "canny"
    DEPTH = "depth"
    SOFTEDGE = "softedge"


class ControlNetProcessor:
    """
    ControlNet 预处理器集合。
    输入一张普通图片, 输出对应的条件图 (如骨骼图、线稿图等)。
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._preprocessors: dict = {}

    # ----------------------------------------------------------------
    # 公开 API
    # ----------------------------------------------------------------

    def process(
        self,
        image: Union[Image.Image, np.ndarray, str],
        mode: ControlMode | str,
    ) -> Image.Image:
        """
        对输入图片进行预处理, 返回条件图

        Args:
            image: PIL Image / numpy array / 文件路径
            mode:  控制模式

        Returns:
            PIL Image (条件图)
        """
        if isinstance(mode, str):
            mode = ControlMode(mode)

        img = self._load_image(image)
        method = getattr(self, f"_preprocess_{mode.value}", None)
        if method is None:
            raise ValueError(f"不支持的控制模式: {mode}")

        result = method(img)
        return Image.fromarray(result) if isinstance(result, np.ndarray) else result

    def process_pose(self, image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
        """提取 OpenPose 骨骼图"""
        return self.process(image, ControlMode.POSE)

    def process_lineart(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> Image.Image:
        """提取线稿"""
        return self.process(image, ControlMode.LINEART)

    def process_scribble(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> Image.Image:
        """生成涂鸦引导图"""
        return self.process(image, ControlMode.SCRIBBLE)

    def process_canny(
        self,
        image: Union[Image.Image, np.ndarray, str],
        low: int = 100,
        high: int = 200,
    ) -> Image.Image:
        """Canny 边缘检测"""
        img = self._load_image(image)
        return self._preprocess_canny(img, low, high)

    def process_depth(self, image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
        """深度估计"""
        return self.process(image, ControlMode.DEPTH)

    def process_softedge(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> Image.Image:
        """软边缘 (HED)"""
        return self.process(image, ControlMode.SOFTEDGE)

    # ----------------------------------------------------------------
    # 预处理器实现
    # ----------------------------------------------------------------

    def _preprocess_pose(self, img: Image.Image) -> Image.Image:
        """OpenPose 骨骼提取"""
        proc = self._get_preprocessor("openpose")
        if proc is None:
            OpenposeDetector = self._get_aux_class("OpenposeDetector")
            proc = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            self._cache_preprocessor("openpose", proc)
        return proc(img)

    def _preprocess_lineart(self, img: Image.Image) -> Image.Image:
        """线稿提取 (使用 LineartDetector)"""
        proc = self._get_preprocessor("lineart")
        if proc is None:
            LineartDetector = self._get_aux_class("LineartDetector")
            proc = LineartDetector.from_pretrained("lllyasviel/Annotators")
            self._cache_preprocessor("lineart", proc)
        return proc(img)

    def _preprocess_scribble(self, img: Image.Image) -> Image.Image:
        """涂鸦 (HED 软边缘 + 简化)"""
        proc = self._get_preprocessor("scribble")
        if proc is None:
            HEDdetector = self._get_aux_class("HEDdetector")
            proc = HEDdetector.from_pretrained("lllyasviel/Annotators")
            self._cache_preprocessor("scribble", proc)
        return proc(img, scribble=True)

    def _preprocess_canny(
        self, img: Image.Image, low: int = 100, high: int = 200
    ) -> Image.Image:
        """Canny 边缘检测"""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                'Canny 依赖未安装，请运行: pip install -e ".[controlnet]"'
            ) from exc

        arr = np.array(img)
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        edges = cv2.Canny(gray, low, high)
        return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2), mode="RGB")

    def _preprocess_depth(self, img: Image.Image) -> Image.Image:
        """深度图估计"""
        proc = self._get_preprocessor("depth")
        if proc is None:
            MidasDetector = self._get_aux_class("MidasDetector")
            proc = MidasDetector.from_pretrained("lllyasviel/Annotators")
            self._cache_preprocessor("depth", proc)
        return proc(img)

    def _preprocess_softedge(self, img: Image.Image) -> Image.Image:
        """软边缘 (HED)"""
        proc = self._get_preprocessor("softedge")
        if proc is None:
            HEDdetector = self._get_aux_class("HEDdetector")
            proc = HEDdetector.from_pretrained("lllyasviel/Annotators")
            self._cache_preprocessor("softedge", proc)
        return proc(img)

    # ----------------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------------

    def _get_preprocessor(self, name: str):
        return self._preprocessors.get(name)

    def _cache_preprocessor(self, name: str, processor):
        """低配设备只保留最近使用的一个神经网络预处理器。"""
        self._preprocessors.clear()
        self._preprocessors[name] = processor

    def cleanup(self):
        self._preprocessors.clear()
        gc.collect()

    @staticmethod
    def _get_aux_class(name: str):
        try:
            module = importlib.import_module("controlnet_aux")
        except ImportError as exc:
            raise RuntimeError(
                'ControlNet 预处理依赖未安装，请运行: pip install -e ".[controlnet]"'
            ) from exc
        return getattr(module, name)

    @staticmethod
    def _load_image(image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return image.convert("RGB")
