"""NVIDIA Maxine Super Resolution pipeline for Daydream Scope.

This pipeline uses NVIDIA's Maxine Video Effects SDK to upscale video
frames using AI-powered super resolution.
"""

import logging
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope.core.pipelines.process import normalize_frame_sizes

from .schema import MaxineSRConfig

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class MaxineSRPipeline(Pipeline):
    """NVIDIA Maxine Super Resolution pipeline.

    This pipeline upscales video frames using NVIDIA's Maxine SDK.
    It acts as a postprocessor, taking input video and producing
    higher-resolution output.

    Requirements:
    - Windows only
    - NVIDIA RTX GPU with Tensor Cores
    - NVIDIA Video Effects SDK installed
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return MaxineSRConfig

    def __init__(
        self,
        config: MaxineSRConfig | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        """Initialize the Maxine Super Resolution pipeline.

        Args:
            config: Pipeline configuration (optional, can pass fields as kwargs)
            device: Target device (must be CUDA)
            dtype: Data type for processing (unused - Maxine uses F32 internally)
            **kwargs: Config fields passed directly (scale_factor, mode, etc.)
        """
        from .modules import MaxineSuperResolution

        # Support both config object and kwargs (pipeline manager passes kwargs)
        if config is not None:
            self.config = config
        else:
            # Filter kwargs to only include valid config fields
            config_fields = {
                k: v for k, v in kwargs.items()
                if k in MaxineSRConfig.model_fields
            }
            self.config = MaxineSRConfig(**config_fields)

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Maxine Super Resolution requires a CUDA-capable GPU. "
                "No CUDA device found."
            )

        # Initialize the Maxine upscaler (lazy initialization - will init on first frame)
        logger.info(
            f"Initializing Maxine Super Resolution: "
            f"scale={self.config.scale_factor}x, mode={self.config.mode}"
        )
        self.upscaler = MaxineSuperResolution(
            scale_factor=self.config.scale_factor,
            mode=self.config.mode,
            device=self.device,
        )
        logger.info("Maxine Super Resolution pipeline initialized")

    def prepare(self, **kwargs) -> Requirements:
        """Prepare the pipeline for processing.

        Returns:
            Requirements specifying input size needed.
        """
        # We process one frame at a time but accept any number
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Process video frames through Maxine Super Resolution.

        Args:
            **kwargs: Must contain "video" key with input frames.
                The video value is a list of tensors in THWC format,
                [0, 255] range (uint8), one tensor per frame.

        Returns:
            Dictionary with "video" key containing upscaled frames
            as a tensor in THWC format, [0, 1] range (float).
        """
        input_frames = kwargs.get("video")

        if input_frames is None:
            raise ValueError("Input 'video' cannot be None for MaxineSRPipeline")

        # Handle list of frames input
        if isinstance(input_frames, list):
            # Normalize frame sizes to handle resolution changes
            input_frames = normalize_frame_sizes(input_frames)

            # Process each frame
            output_frames = []
            for frame in input_frames:
                # frame is (1, H, W, C) uint8 [0, 255]
                # Convert to (H, W, C) float32 [0, 1]
                frame_hwc = frame.squeeze(0).float() / 255.0

                # Process through Maxine
                upscaled = self.upscaler.process_frame(frame_hwc)

                # Add back the T dimension
                output_frames.append(upscaled.unsqueeze(0))

            # Stack all frames: list of (1, H, W, C) -> (T, H, W, C)
            output_tensor = torch.cat(output_frames, dim=0)

        elif isinstance(input_frames, torch.Tensor):
            # Handle tensor input (THWC format)
            if input_frames.dim() == 4:
                # THWC format
                t, h, w, c = input_frames.shape
                output_frames = []

                for i in range(t):
                    frame = input_frames[i]  # (H, W, C)

                    # Convert to float [0, 1] if needed
                    if frame.dtype == torch.uint8:
                        frame = frame.float() / 255.0
                    elif frame.max() > 1.0:
                        frame = frame / 255.0

                    upscaled = self.upscaler.process_frame(frame)
                    output_frames.append(upscaled)

                output_tensor = torch.stack(output_frames, dim=0)

            elif input_frames.dim() == 3:
                # Single frame HWC
                frame = input_frames
                if frame.dtype == torch.uint8:
                    frame = frame.float() / 255.0
                elif frame.max() > 1.0:
                    frame = frame / 255.0

                upscaled = self.upscaler.process_frame(frame)
                output_tensor = upscaled.unsqueeze(0)  # Add T dimension

            else:
                raise ValueError(
                    f"Expected 3D (HWC) or 4D (THWC) tensor, got {input_frames.dim()}D"
                )
        else:
            raise ValueError(
                f"Expected list or tensor for 'video', got {type(input_frames)}"
            )

        # Ensure output is in [0, 1] range and float
        output_tensor = output_tensor.clamp(0, 1).float()

        return {"video": output_tensor}

    def cleanup(self):
        """Release resources."""
        if hasattr(self, "upscaler") and self.upscaler is not None:
            self.upscaler.cleanup()
            logger.info("Maxine Super Resolution pipeline cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
