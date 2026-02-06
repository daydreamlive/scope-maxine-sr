"""NVIDIA Maxine Super Resolution plugin for Daydream Scope.

This plugin provides AI-powered video upscaling using NVIDIA's Maxine Video Effects SDK.
Windows only - requires an NVIDIA RTX GPU with Tensor Cores (20xx/30xx/40xx/50xx series).
"""

from .plugin import MaxineSRPlugin

plugin = MaxineSRPlugin()

__all__ = ["plugin", "MaxineSRPlugin"]
