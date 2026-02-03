"""NVIDIA Maxine Super Resolution plugin for Daydream Scope.

This plugin provides AI-powered video upscaling using NVIDIA's Maxine Video Effects SDK.
Windows only - requires an NVIDIA RTX GPU with Tensor Cores (20xx/30xx/40xx/50xx series).
"""

import logging
import sys

from scope.core.plugins import hookimpl

logger = logging.getLogger(__name__)

__version__ = "0.1.0"


def is_supported() -> bool:
    """Check if this plugin is supported on the current platform."""
    return sys.platform == "win32"


@hookimpl
def register_pipelines(register):
    """Register the Maxine Super Resolution pipeline."""
    if not is_supported():
        logger.warning(
            "Maxine SR plugin is only supported on Windows. Skipping registration."
        )
        return

    try:
        from .pipeline import MaxineSRPipeline

        register(MaxineSRPipeline)
        logger.info("Registered Maxine Super Resolution pipeline")
    except Exception as e:
        logger.warning(f"Failed to register Maxine SR pipeline: {e}")


__all__ = ["register_pipelines", "is_supported", "__version__"]
