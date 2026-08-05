"""
Anime Gen Pipeline — 2D 动漫可控图像生成流水线
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .pipeline import AnimePipeline, GenParams, GenResult
from .engine import ModelEngine
from .controlnet import ControlNetProcessor, ControlMode

__all__ = [
    "AnimePipeline",
    "GenParams",
    "GenResult",
    "ModelEngine",
    "ControlNetProcessor",
    "ControlMode",
]
