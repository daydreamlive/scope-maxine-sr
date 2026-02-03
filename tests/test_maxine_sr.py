"""Tests for Maxine Super Resolution plugin."""

import sys

import pytest
import torch


# Skip all tests if not on Windows
pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Maxine SR only works on Windows",
)


class TestMaxineSRSchema:
    """Tests for the configuration schema."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from scope_maxine_sr.schema import MaxineSRConfig

        config = MaxineSRConfig()
        assert config.scale_factor == "2"
        assert config.mode == 0

    def test_config_metadata(self):
        """Test pipeline metadata."""
        from scope_maxine_sr.schema import MaxineSRConfig

        assert MaxineSRConfig.pipeline_id == "maxine_sr"
        assert MaxineSRConfig.pipeline_name == "Maxine Super Resolution"
        assert MaxineSRConfig.supports_prompts is False

    def test_config_scale_factors(self):
        """Test valid scale factors."""
        from scope_maxine_sr.schema import MaxineSRConfig

        for scale in ["1.33", "1.5", "2", "3", "4"]:
            config = MaxineSRConfig(scale_factor=scale)
            assert config.scale_factor == scale

    def test_config_invalid_scale_factor(self):
        """Test invalid scale factor raises error."""
        from pydantic import ValidationError

        from scope_maxine_sr.schema import MaxineSRConfig

        with pytest.raises(ValidationError):
            MaxineSRConfig(scale_factor="5")


class TestMaxineSRPlugin:
    """Tests for plugin registration."""

    def test_is_supported(self):
        """Test platform support check."""
        from scope_maxine_sr import is_supported

        # On Windows this should be True, on other platforms False
        assert is_supported() == (sys.platform == "win32")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestMaxineSRPipeline:
    """Tests for the pipeline (requires CUDA and SDK)."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        from scope_maxine_sr.schema import MaxineSRConfig

        return MaxineSRConfig(scale_factor="2", mode=0)

    def test_pipeline_init(self, config):
        """Test pipeline initialization."""
        pytest.importorskip("scope_maxine_sr.modules.nvvfx")

        from scope_maxine_sr.pipeline import MaxineSRPipeline

        try:
            pipeline = MaxineSRPipeline(config)
            assert pipeline.upscaler is not None
            pipeline.cleanup()
        except RuntimeError as e:
            if "SDK not found" in str(e):
                pytest.skip("NVIDIA Video Effects SDK not installed")
            raise

    def test_pipeline_process_frame(self, config):
        """Test processing a single frame."""
        pytest.importorskip("scope_maxine_sr.modules.nvvfx")

        from scope_maxine_sr.pipeline import MaxineSRPipeline

        try:
            pipeline = MaxineSRPipeline(config)

            # Create a test frame (512x288 minimum for SuperRes)
            input_frame = torch.randint(0, 256, (1, 288, 512, 3), dtype=torch.uint8)

            result = pipeline(video=[input_frame])

            assert "video" in result
            output = result["video"]

            # Check output dimensions (2x upscale)
            assert output.shape[1] == 576  # 288 * 2
            assert output.shape[2] == 1024  # 512 * 2
            assert output.shape[3] == 3  # RGB

            # Check output range
            assert output.min() >= 0
            assert output.max() <= 1

            pipeline.cleanup()

        except RuntimeError as e:
            if "SDK not found" in str(e):
                pytest.skip("NVIDIA Video Effects SDK not installed")
            raise
