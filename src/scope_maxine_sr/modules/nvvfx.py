"""NVIDIA Video Effects SDK ctypes wrapper for Super Resolution.

This module provides Python bindings for the NVIDIA Maxine Video Effects SDK
using ctypes. It wraps the Super Resolution filter functionality.

Requires:
- Windows OS
- NVIDIA Video Effects SDK installed
- NVIDIA RTX GPU with Tensor Cores
"""

import ctypes
import logging
import os
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_float,
    c_int,
    c_uint,
    c_uint8,
    c_void_p,
)
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# NVIDIA VFX SDK constants
NVVFX_FX_SUPER_RES = b"SuperRes"

# NvCVImage pixel formats (from nvCVImage.h)
NVCV_FORMAT_UNKNOWN = 0
NVCV_Y = 1       # Luminance (grayscale)
NVCV_A = 2       # Alpha
NVCV_YA = 3      # Luminance + Alpha
NVCV_RGB = 4     # RGB
NVCV_BGR = 5     # BGR
NVCV_RGBA = 6    # RGBA
NVCV_BGRA = 7    # BGRA

# NvCVImage component types (from nvCVImage.h)
NVCV_TYPE_UNKNOWN = 0
NVCV_U8 = 1      # Unsigned 8-bit
NVCV_U16 = 2     # Unsigned 16-bit
NVCV_S16 = 3     # Signed 16-bit
NVCV_F16 = 4     # 16-bit float
NVCV_U32 = 5     # Unsigned 32-bit
NVCV_S32 = 6     # Signed 32-bit
NVCV_F32 = 7     # 32-bit float
NVCV_U64 = 8     # Unsigned 64-bit
NVCV_S64 = 9     # Signed 64-bit
NVCV_F64 = 10    # 64-bit float

# NvCVImage layout (from nvCVImage.h)
NVCV_INTERLEAVED = 0  # Chunky: RGBRGBRGB...
NVCV_PLANAR = 1       # Planar: RRR...GGG...BBB...
NVCV_YCBCR = 2        # YCbCr

# NvCVImage memory location (from nvCVImage.h)
NVCV_CPU = 0
NVCV_GPU = 1
NVCV_CUDA = 1         # Alias for GPU
NVCV_CPU_PINNED = 2

# Parameter selectors (must match nvVideoEffects.h exactly)
NVVFX_INPUT_IMAGE = b"SrcImage0"
NVVFX_OUTPUT_IMAGE = b"DstImage0"
NVVFX_MODEL_DIRECTORY = b"ModelDir"
NVVFX_CUDA_STREAM = b"CudaStream"
NVVFX_SCALE = b"Scale"
NVVFX_STRENGTH = b"Strength"
NVVFX_MODE = b"Mode"
NVVFX_GPU = b"GPU"

# Status codes
NVCV_SUCCESS = 0
NVCV_ERR_GENERAL = -1
NVCV_ERR_UNIMPLEMENTED = -2
NVCV_ERR_MEMORY = -3
NVCV_ERR_EFFECT = -4
NVCV_ERR_SELECTOR = -5
NVCV_ERR_BUFFER = -6
NVCV_ERR_PARAMETER = -7
NVCV_ERR_MISMATCH = -8
NVCV_ERR_PIXELFORMAT = -9
NVCV_ERR_MODEL = -10
NVCV_ERR_LIBRARY = -11
NVCV_ERR_INITIALIZATION = -12
NVCV_ERR_FILE = -13
NVCV_ERR_FEATURENOTFOUND = -14
NVCV_ERR_MISSINGINPUT = -15
NVCV_ERR_RESOLUTION = -16
NVCV_ERR_UNSUPPORTEDGPU = -17
NVCV_ERR_WRONGGPU = -18
NVCV_ERR_UNSUPPORTEDDRIVER = -19
NVCV_ERR_MODELPERMISSIONS = -20
NVCV_ERR_CUDA = -99

STATUS_MESSAGES = {
    NVCV_SUCCESS: "Success",
    NVCV_ERR_GENERAL: "General error",
    NVCV_ERR_UNIMPLEMENTED: "Unimplemented feature",
    NVCV_ERR_MEMORY: "Memory allocation error",
    NVCV_ERR_EFFECT: "Effect error",
    NVCV_ERR_SELECTOR: "Invalid selector",
    NVCV_ERR_BUFFER: "Buffer error",
    NVCV_ERR_PARAMETER: "Invalid parameter",
    NVCV_ERR_MISMATCH: "Mismatch error",
    NVCV_ERR_PIXELFORMAT: "Unsupported pixel format",
    NVCV_ERR_MODEL: "Model error",
    NVCV_ERR_LIBRARY: "Library error",
    NVCV_ERR_INITIALIZATION: "Initialization error",
    NVCV_ERR_FILE: "File error",
    NVCV_ERR_FEATURENOTFOUND: "Feature not found",
    NVCV_ERR_MISSINGINPUT: "Missing input",
    NVCV_ERR_RESOLUTION: "Resolution error (check min/max dimensions)",
    NVCV_ERR_UNSUPPORTEDGPU: "Unsupported GPU (requires RTX with Tensor Cores)",
    NVCV_ERR_WRONGGPU: "Wrong GPU selected",
    NVCV_ERR_UNSUPPORTEDDRIVER: "Unsupported driver version",
    NVCV_ERR_MODELPERMISSIONS: "Model permissions error",
    NVCV_ERR_CUDA: "CUDA error",
}


class NvCVImage(Structure):
    """NvCVImage structure for image data."""

    _fields_ = [
        ("width", c_uint),  # Width in pixels
        ("height", c_uint),  # Height in pixels
        ("pitch", c_int),  # Byte stride between rows
        ("pixelFormat", c_uint8),  # Pixel format (NVCV_BGR, etc.)
        ("componentType", c_uint8),  # Component type (NVCV_U8, NVCV_F32, etc.)
        ("pixelBytes", c_uint8),  # Bytes per pixel
        ("componentBytes", c_uint8),  # Bytes per component
        ("numComponents", c_uint8),  # Number of components per pixel
        ("planar", c_uint8),  # Layout (NVCV_INTERLEAVED, NVCV_PLANAR)
        ("gpuMem", c_uint8),  # Memory location (NVCV_CPU, NVCV_GPU)
        ("reserved", c_uint8 * 1),  # Padding
        ("pixels", c_void_p),  # Pointer to pixel data
        ("deletePtr", c_void_p),  # Pointer for deallocation
        ("deleteProc", c_void_p),  # Deallocation callback
        ("bufferBytes", c_uint),  # Total buffer size in bytes
    ]


# Type aliases
NvVFX_Handle = c_void_p
NvCV_Status = c_int
CUstream = c_void_p


def _get_sdk_path() -> Path | None:
    """Find the NVIDIA Video Effects SDK installation path.

    Searches in order:
    1. NVVFX_SDK_DIR environment variable
    2. Official NVIDIA Video Effects SDK installation
    3. TouchDesigner bundled SDK (if installed)
    """
    # Check environment variable first
    env_path = os.environ.get("NVVFX_SDK_DIR")
    if env_path:
        return Path(env_path)

    # Default installation paths (in order of preference)
    default_paths = [
        # Official NVIDIA SDK
        Path(r"C:\Program Files\NVIDIA Corporation\NVIDIA Video Effects"),
        Path(r"C:\Program Files (x86)\NVIDIA Corporation\NVIDIA Video Effects"),
        # TouchDesigner bundled SDK
        Path(r"C:\Program Files\Derivative\TouchDesigner\bin"),
    ]

    for path in default_paths:
        if path.exists() and (path / "NVVideoEffects.dll").exists():
            return path

    return None


def _get_models_path() -> Path | None:
    """Find the NVIDIA Video Effects models directory.

    Searches in order:
    1. NVVFX_MODEL_DIR environment variable
    2. Official NVIDIA SDK models directory
    3. TouchDesigner bundled models
    """
    # Check environment variable first
    env_path = os.environ.get("NVVFX_MODEL_DIR")
    if env_path:
        return Path(env_path)

    # Check official SDK location
    sdk_path = _get_sdk_path()
    if sdk_path:
        # Official SDK has models in a subdirectory
        official_models = sdk_path / "models"
        if official_models.exists():
            return official_models

        # TouchDesigner has models in Config/Models relative to bin
        if "TouchDesigner" in str(sdk_path):
            td_models = sdk_path.parent / "Config" / "Models"
            if td_models.exists():
                return td_models

    return None


def _check_status(status: int, operation: str = ""):
    """Check NvCV_Status and raise exception on error."""
    if status != NVCV_SUCCESS:
        msg = STATUS_MESSAGES.get(status, f"Unknown error code: {status}")
        raise RuntimeError(f"NVIDIA VFX SDK error{' during ' + operation if operation else ''}: {msg}")


class NvVFXLibrary:
    """Wrapper for NVIDIA Video Effects SDK library."""

    def __init__(self):
        self._nvvfx = None
        self._nvcvimage = None
        self._loaded = False
        self._sdk_path = None

    def load(self) -> bool:
        """Load the NVIDIA VFX SDK libraries."""
        if self._loaded:
            return True

        self._sdk_path = _get_sdk_path()
        if self._sdk_path is None:
            raise RuntimeError(
                "NVIDIA Video Effects SDK not found. "
                "Please install it from: https://developer.nvidia.com/maxine"
            )

        try:
            # Load NVVideoEffects.dll
            nvvfx_path = self._sdk_path / "NVVideoEffects.dll"
            if not nvvfx_path.exists():
                raise FileNotFoundError(f"NVVideoEffects.dll not found at {nvvfx_path}")

            self._nvvfx = ctypes.WinDLL(str(nvvfx_path))
            self._setup_nvvfx_functions()

            # Load NVCVImage.dll
            nvcvimage_path = self._sdk_path / "NVCVImage.dll"
            if not nvcvimage_path.exists():
                raise FileNotFoundError(f"NVCVImage.dll not found at {nvcvimage_path}")

            self._nvcvimage = ctypes.WinDLL(str(nvcvimage_path))
            self._setup_nvcvimage_functions()

            self._loaded = True
            logger.info(f"Loaded NVIDIA VFX SDK from {self._sdk_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load NVIDIA VFX SDK: {e}")
            raise

    def _setup_nvvfx_functions(self):
        """Set up function signatures for NVVideoEffects.dll."""
        # NvVFX_CreateEffect
        self._nvvfx.NvVFX_CreateEffect.argtypes = [c_char_p, POINTER(NvVFX_Handle)]
        self._nvvfx.NvVFX_CreateEffect.restype = NvCV_Status

        # NvVFX_DestroyEffect
        self._nvvfx.NvVFX_DestroyEffect.argtypes = [NvVFX_Handle]
        self._nvvfx.NvVFX_DestroyEffect.restype = NvCV_Status

        # NvVFX_SetString
        self._nvvfx.NvVFX_SetString.argtypes = [NvVFX_Handle, c_char_p, c_char_p]
        self._nvvfx.NvVFX_SetString.restype = NvCV_Status

        # NvVFX_SetU32
        self._nvvfx.NvVFX_SetU32.argtypes = [NvVFX_Handle, c_char_p, c_uint]
        self._nvvfx.NvVFX_SetU32.restype = NvCV_Status

        # NvVFX_SetF32
        self._nvvfx.NvVFX_SetF32.argtypes = [NvVFX_Handle, c_char_p, c_float]
        self._nvvfx.NvVFX_SetF32.restype = NvCV_Status

        # NvVFX_SetCudaStream
        self._nvvfx.NvVFX_SetCudaStream.argtypes = [NvVFX_Handle, c_char_p, CUstream]
        self._nvvfx.NvVFX_SetCudaStream.restype = NvCV_Status

        # NvVFX_SetImage
        self._nvvfx.NvVFX_SetImage.argtypes = [NvVFX_Handle, c_char_p, POINTER(NvCVImage)]
        self._nvvfx.NvVFX_SetImage.restype = NvCV_Status

        # NvVFX_Load
        self._nvvfx.NvVFX_Load.argtypes = [NvVFX_Handle]
        self._nvvfx.NvVFX_Load.restype = NvCV_Status

        # NvVFX_Run
        self._nvvfx.NvVFX_Run.argtypes = [NvVFX_Handle, c_int]
        self._nvvfx.NvVFX_Run.restype = NvCV_Status

    def _setup_nvcvimage_functions(self):
        """Set up function signatures for NVCVImage.dll."""
        # NvCVImage_Init
        self._nvcvimage.NvCVImage_Init.argtypes = [
            POINTER(NvCVImage),  # image
            c_uint,  # width
            c_uint,  # height
            c_int,  # pitch
            c_void_p,  # pixels
            c_uint8,  # pixelFormat
            c_uint8,  # componentType
            c_uint8,  # layout
            c_uint8,  # memLocation
        ]
        self._nvcvimage.NvCVImage_Init.restype = NvCV_Status

        # NvCVImage_Alloc
        self._nvcvimage.NvCVImage_Alloc.argtypes = [
            POINTER(NvCVImage),  # image
            c_uint,  # width
            c_uint,  # height
            c_uint8,  # pixelFormat
            c_uint8,  # componentType
            c_uint8,  # layout
            c_uint8,  # memLocation
            c_uint,  # alignment
        ]
        self._nvcvimage.NvCVImage_Alloc.restype = NvCV_Status

        # NvCVImage_Dealloc
        self._nvcvimage.NvCVImage_Dealloc.argtypes = [POINTER(NvCVImage)]
        self._nvcvimage.NvCVImage_Dealloc.restype = NvCV_Status

        # NvCVImage_Transfer
        self._nvcvimage.NvCVImage_Transfer.argtypes = [
            POINTER(NvCVImage),  # src
            POINTER(NvCVImage),  # dst
            c_float,  # scale
            CUstream,  # stream
            POINTER(NvCVImage),  # staging buffer (can be NULL)
        ]
        self._nvcvimage.NvCVImage_Transfer.restype = NvCV_Status

    @property
    def nvvfx(self):
        if not self._loaded:
            self.load()
        return self._nvvfx

    @property
    def nvcvimage(self):
        if not self._loaded:
            self.load()
        return self._nvcvimage

    @property
    def model_dir(self) -> str:
        models_path = _get_models_path()
        if models_path is None:
            raise RuntimeError(
                "NVIDIA Video Effects models not found. "
                "Set NVVFX_MODEL_DIR environment variable or install the SDK."
            )
        return str(models_path)


# Global library instance
_lib = NvVFXLibrary()


class MaxineSuperResolution:
    """NVIDIA Maxine Super Resolution effect wrapper.

    This class provides a high-level interface for the Maxine Super Resolution
    filter, handling GPU memory management and frame processing.
    """

    SCALE_FACTORS = {
        "1.33": 4.0 / 3.0,
        "1.5": 1.5,
        "2": 2.0,
        "3": 3.0,
        "4": 4.0,
    }

    def __init__(
        self,
        scale_factor: str = "2",
        mode: int = 0,
        device: torch.device | None = None,
    ):
        """Initialize the Maxine Super Resolution effect.

        Args:
            scale_factor: Upscaling factor ("1.33", "1.5", "2", "3", "4")
            mode: 0 = lossy content, 1 = lossless content
            device: CUDA device to use
        """
        self.scale_factor = scale_factor
        self.scale = self.SCALE_FACTORS[scale_factor]
        self.mode = mode
        self.device = device or torch.device("cuda")

        self._effect = NvVFX_Handle()
        self._input_image = NvCVImage()
        self._output_image = NvCVImage()
        self._staging_image = NvCVImage()  # For CPU<->GPU transfers
        self._initialized = False
        self._current_input_size = None
        self._current_output_size = None

        # Load library
        _lib.load()

    def _create_effect(self):
        """Create the Super Resolution effect.

        Note: Scale factor is NOT set via a parameter. Instead, it's determined
        by the ratio of output dimensions to input dimensions when allocating
        the input/output images.

        Super Resolution only supports:
        - Mode: 0 (lossy content) or 1 (lossless content)
        - Model directory path
        """
        status = _lib.nvvfx.NvVFX_CreateEffect(NVVFX_FX_SUPER_RES, byref(self._effect))
        _check_status(status, "CreateEffect")

        # Set model directory
        model_dir = _lib.model_dir.encode("utf-8")
        status = _lib.nvvfx.NvVFX_SetString(self._effect, NVVFX_MODEL_DIRECTORY, model_dir)
        _check_status(status, "SetString(MODEL_DIRECTORY)")

        # Set mode (0 = lossy/encoded content, 1 = lossless content)
        status = _lib.nvvfx.NvVFX_SetU32(self._effect, NVVFX_MODE, c_uint(self.mode))
        _check_status(status, "SetU32(MODE)")

        logger.info(
            f"Created Maxine Super Resolution effect: "
            f"scale={self.scale}x, mode={self.mode}"
        )

    def _allocate_images(self, input_h: int, input_w: int):
        """Allocate GPU images for input and output."""
        output_h = int(input_h * self.scale)
        output_w = int(input_w * self.scale)

        # Allocate input image on GPU (BGR, F32, planar as required by SuperRes)
        status = _lib.nvcvimage.NvCVImage_Alloc(
            byref(self._input_image),
            c_uint(input_w),
            c_uint(input_h),
            c_uint8(NVCV_BGR),
            c_uint8(NVCV_F32),
            c_uint8(NVCV_PLANAR),
            c_uint8(NVCV_GPU),
            c_uint(0),  # Default alignment
        )
        _check_status(status, "Alloc input image")

        # Allocate output image on GPU
        status = _lib.nvcvimage.NvCVImage_Alloc(
            byref(self._output_image),
            c_uint(output_w),
            c_uint(output_h),
            c_uint8(NVCV_BGR),
            c_uint8(NVCV_F32),
            c_uint8(NVCV_PLANAR),
            c_uint8(NVCV_GPU),
            c_uint(0),
        )
        _check_status(status, "Alloc output image")

        # Allocate staging image on CPU for transfers
        status = _lib.nvcvimage.NvCVImage_Alloc(
            byref(self._staging_image),
            c_uint(max(input_w, output_w)),
            c_uint(max(input_h, output_h)),
            c_uint8(NVCV_BGR),
            c_uint8(NVCV_F32),
            c_uint8(NVCV_PLANAR),
            c_uint8(NVCV_CPU),
            c_uint(0),
        )
        _check_status(status, "Alloc staging image")

        self._current_input_size = (input_h, input_w)
        self._current_output_size = (output_h, output_w)

        logger.info(f"Allocated images: input={input_w}x{input_h}, output={output_w}x{output_h}")

    def _load_effect(self):
        """Load the effect (must be called after setting images)."""
        # Set input/output images
        status = _lib.nvvfx.NvVFX_SetImage(self._effect, NVVFX_INPUT_IMAGE, byref(self._input_image))
        _check_status(status, "SetImage(INPUT)")

        status = _lib.nvvfx.NvVFX_SetImage(self._effect, NVVFX_OUTPUT_IMAGE, byref(self._output_image))
        _check_status(status, "SetImage(OUTPUT)")

        # Load the effect (initializes CUDA kernels)
        status = _lib.nvvfx.NvVFX_Load(self._effect)
        _check_status(status, "Load")

        self._initialized = True
        logger.info("Maxine Super Resolution effect loaded and ready")

    def initialize(self, input_h: int, input_w: int):
        """Initialize the effect for a given input resolution.

        Args:
            input_h: Input height in pixels
            input_w: Input width in pixels
        """
        # Check minimum resolution requirements
        if input_h < 288 or input_w < 512:
            raise ValueError(
                f"Input resolution {input_w}x{input_h} is too small. "
                f"Minimum is 512x288 for Super Resolution."
            )

        # Check 4x scale limitation
        if self.scale_factor == "4" and input_h > 540:
            raise ValueError(
                f"4x scaling is limited to inputs up to 540p. "
                f"Input height {input_h} exceeds this limit."
            )

        self._create_effect()
        self._allocate_images(input_h, input_w)
        self._load_effect()

    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Process a single frame through Super Resolution.

        Args:
            frame: Input frame tensor in HWC format, RGB, float32 [0, 1]

        Returns:
            Upscaled frame tensor in HWC format, RGB, float32 [0, 1]
        """
        if frame.dim() != 3:
            raise ValueError(f"Expected 3D tensor (HWC), got {frame.dim()}D")

        h, w, c = frame.shape
        if c != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {c}")

        # Initialize on first frame or if resolution changed
        if not self._initialized or self._current_input_size != (h, w):
            if self._initialized:
                self.cleanup()
            self.initialize(h, w)

        # Convert RGB to BGR and rearrange to planar format for Maxine
        frame_bgr = frame[:, :, [2, 1, 0]]  # RGB -> BGR
        frame_planar = frame_bgr.permute(2, 0, 1).contiguous()  # HWC -> CHW (planar)

        # Copy to GPU input buffer via staging
        frame_np = frame_planar.cpu().numpy()
        self._copy_to_input(frame_np)

        # Run the effect
        status = _lib.nvvfx.NvVFX_Run(self._effect, c_int(0))  # 0 = synchronous
        _check_status(status, "Run")

        # Copy from GPU output buffer
        output_np = self._copy_from_output()

        # Convert back to HWC RGB tensor
        output_h, output_w = self._current_output_size
        output_tensor = torch.from_numpy(output_np).reshape(3, output_h, output_w)
        output_tensor = output_tensor.permute(1, 2, 0)  # CHW -> HWC
        output_tensor = output_tensor[:, :, [2, 1, 0]]  # BGR -> RGB

        return output_tensor.to(self.device)

    def _copy_to_input(self, data: np.ndarray):
        """Copy numpy array to GPU input image via staging buffer."""
        # Create a temporary CPU NvCVImage wrapping the numpy data
        h, w = self._current_input_size
        temp_cpu = NvCVImage()

        # data is in CHW planar format (3, H, W)
        data_contiguous = np.ascontiguousarray(data, dtype=np.float32)
        pitch = w * 4  # bytes per row for F32

        status = _lib.nvcvimage.NvCVImage_Init(
            byref(temp_cpu),
            c_uint(w),
            c_uint(h),
            c_int(pitch),
            data_contiguous.ctypes.data_as(c_void_p),
            c_uint8(NVCV_BGR),
            c_uint8(NVCV_F32),
            c_uint8(NVCV_PLANAR),
            c_uint8(NVCV_CPU),
        )
        _check_status(status, "Init temp CPU image")

        # Transfer CPU -> GPU
        status = _lib.nvcvimage.NvCVImage_Transfer(
            byref(temp_cpu),
            byref(self._input_image),
            c_float(1.0),
            CUstream(0),  # Default stream
            byref(self._staging_image),
        )
        _check_status(status, "Transfer CPU->GPU")

    def _copy_from_output(self) -> np.ndarray:
        """Copy GPU output image to numpy array via staging buffer."""
        output_h, output_w = self._current_output_size

        # Allocate output numpy array
        output_data = np.zeros((3, output_h, output_w), dtype=np.float32)
        pitch = output_w * 4

        # Create temporary CPU NvCVImage
        temp_cpu = NvCVImage()
        status = _lib.nvcvimage.NvCVImage_Init(
            byref(temp_cpu),
            c_uint(output_w),
            c_uint(output_h),
            c_int(pitch),
            output_data.ctypes.data_as(c_void_p),
            c_uint8(NVCV_BGR),
            c_uint8(NVCV_F32),
            c_uint8(NVCV_PLANAR),
            c_uint8(NVCV_CPU),
        )
        _check_status(status, "Init output CPU image")

        # Transfer GPU -> CPU
        status = _lib.nvcvimage.NvCVImage_Transfer(
            byref(self._output_image),
            byref(temp_cpu),
            c_float(1.0),
            CUstream(0),
            byref(self._staging_image),
        )
        _check_status(status, "Transfer GPU->CPU")

        return output_data

    def cleanup(self):
        """Release all resources."""
        if self._effect:
            _lib.nvvfx.NvVFX_DestroyEffect(self._effect)
            self._effect = NvVFX_Handle()

        if self._input_image.pixels:
            _lib.nvcvimage.NvCVImage_Dealloc(byref(self._input_image))

        if self._output_image.pixels:
            _lib.nvcvimage.NvCVImage_Dealloc(byref(self._output_image))

        if self._staging_image.pixels:
            _lib.nvcvimage.NvCVImage_Dealloc(byref(self._staging_image))

        self._initialized = False
        self._current_input_size = None
        self._current_output_size = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during destruction
