"""Plugin registration for Maxine Super Resolution."""

import logging
import sys

from scope.core.plugins import hookimpl

logger = logging.getLogger(__name__)


class MaxineSRPlugin:
    """NVIDIA Maxine Super Resolution plugin for Daydream Scope."""

    @hookimpl
    def register_pipelines(self, register):
        """Register the Maxine Super Resolution pipeline."""
        if sys.platform != "win32":
            logger.warning(
                "Maxine SR plugin is only supported on Windows. "
                "Skipping registration."
            )
            return

        try:
            from .pipeline import MaxineSRPipeline

            register(MaxineSRPipeline)
            logger.info("Registered Maxine Super Resolution pipeline")
        except Exception as e:
            logger.warning(f"Failed to register Maxine SR pipeline: {e}")
