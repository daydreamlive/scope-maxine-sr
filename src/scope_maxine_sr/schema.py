"""Configuration schema for Maxine Super Resolution pipeline."""

from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class MaxineSRConfig(BasePipelineConfig):
    """Configuration for NVIDIA Maxine Super Resolution pipeline.

    This pipeline uses NVIDIA's Maxine Video Effects SDK to upscale video
    using AI-powered super resolution. It enhances details, sharpens output,
    and preserves content while increasing resolution.

    Requirements:
    - Windows only
    - NVIDIA RTX GPU with Tensor Cores (20xx/30xx/40xx/50xx series)
    - NVIDIA Video Effects SDK installed

    Supported scale factors: 1.33x (4/3), 1.5x, 2x, 3x, 4x
    Note: 4x scaling is limited to inputs up to 540p.
    """

    pipeline_id = "maxine_sr"
    pipeline_name = "Maxine Super Resolution"
    pipeline_description = (
        "AI-powered video upscaling using NVIDIA Maxine. "
        "Enhances details and sharpens output while preserving content. "
        "Windows only, requires RTX GPU."
    )
    docs_url = "https://developer.nvidia.com/maxine"
    artifacts = []  # SDK must be installed separately
    supports_prompts = False
    modified = False

    usage = [UsageType.POSTPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    # Super Resolution specific settings
    scale_factor: Literal["1.33", "1.5", "2", "3", "4"] = Field(
        default="2",
        description=(
            "Upscaling factor. Supported: 1.33x (4/3), 1.5x, 2x, 3x, 4x. "
            "Note: 4x is limited to inputs up to 540p."
        ),
        json_schema_extra=ui_field_config(
            order=1, label="Scale Factor", is_load_param=True
        ),
    )

    mode: Literal[0, 1] = Field(
        default=0,
        description=(
            "Super Resolution mode. "
            "0 = optimized for lossy/encoded content (recommended for most video). "
            "1 = optimized for lossless content."
        ),
        json_schema_extra=ui_field_config(
            order=2, label="SR Mode", is_load_param=True
        ),
    )
