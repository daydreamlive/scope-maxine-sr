# scope-maxine-sr

NVIDIA Maxine Super Resolution plugin for Daydream Scope.

AI-powered video upscaling using NVIDIA's Maxine Video Effects SDK.

## Important: SDK License

This plugin provides Python bindings to interface with NVIDIA's Video Effects SDK.
**The SDK itself is not included** - you must download and install it separately from NVIDIA.

By installing and using the NVIDIA Video Effects SDK, you agree to [NVIDIA's SDK License Terms](https://developer.nvidia.com/downloads/maxine-sdk-license).

## Requirements

- **Windows only** (Linux/Mac not supported by NVIDIA Maxine)
- **NVIDIA RTX GPU** with Tensor Cores (20xx/30xx/40xx/50xx series)
- **NVIDIA Video Effects SDK** (see installation below)

## Installation

### Step 1: Install NVIDIA Video Effects SDK

You have two options:

#### Option A: Install Official NVIDIA SDK (Recommended)

This is the recommended option, especially for RTX 50 series (Blackwell) GPUs.

1. Go to [NVIDIA Broadcast Download](https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/) or [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/maxine/resources/maxine_windows_video_effects_sdk_ga)
2. Download the **Video Effects SDK** installer for your GPU:
   - RTX 20xx series → Turing
   - RTX 30xx series → Ampere
   - RTX 40xx series → Ada
   - RTX 50xx series → Blackwell
3. Run the installer (installs to `C:\Program Files\NVIDIA Corporation\NVIDIA Video Effects\`)

#### Option B: Use TouchDesigner's Bundled SDK (Fallback)

If you have [TouchDesigner](https://derivative.ca/) installed, the plugin can use its bundled Maxine SDK as a fallback.

**Note:** TouchDesigner's bundled SDK may not support the latest GPUs (e.g., RTX 50 series). If you get "Unsupported GPU" errors, install the official SDK instead.

The plugin looks for:
- DLLs: `C:\Program Files\Derivative\TouchDesigner\bin\`
- Models: `C:\Program Files\Derivative\TouchDesigner\Config\Models\`

#### Option C: Manual Path Configuration

Set environment variables to point to custom locations:
```bash
# Path to directory containing NVVideoEffects.dll and NVCVImage.dll
set NVVFX_SDK_DIR=C:\path\to\sdk

# Path to directory containing model files (*.trtpkg)
set NVVFX_MODEL_DIR=C:\path\to\models
```

### Step 2: Install the Plugin

```bash
# Install from local path (editable/development mode)
daydream-scope install -e /path/to/scope-maxine-sr

# Or install from PyPI (when published)
daydream-scope install scope-maxine-sr
```

## Usage

Once installed, "Maxine Super Resolution" appears as a postprocessor pipeline in Daydream Scope.

### Configuration Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `scale_factor` | "1.33", "1.5", "2", "3", "4" | "2" | Upscaling multiplier |
| `mode` | 0, 1 | 0 | 0 = lossy/encoded content, 1 = lossless content |

### Input Requirements

- Minimum resolution: 512x288
- For 4x scaling: input must be 540p or lower
- Recommended aspect ratio: 16:9 (other ratios supported)

## Troubleshooting

### "NVIDIA Video Effects SDK not found"

The plugin searches these locations in order:
1. `NVVFX_SDK_DIR` environment variable
2. `C:\Program Files\NVIDIA Corporation\NVIDIA Video Effects\`
3. `C:\Program Files\Derivative\TouchDesigner\bin\`

Ensure one of these contains `NVVideoEffects.dll`.

### "NVIDIA Video Effects models not found"

The plugin searches for models in:
1. `NVVFX_MODEL_DIR` environment variable
2. `<SDK_DIR>\models\`
3. `C:\Program Files\Derivative\TouchDesigner\Config\Models\`

Models are files like `SR_2x_*.trtpkg`.

### "Unsupported GPU"

The SDK requires an NVIDIA RTX GPU with Tensor Cores:
- RTX 20xx (Turing)
- RTX 30xx (Ampere)
- RTX 40xx (Ada)
- RTX 50xx (Blackwell)

GTX cards and older GPUs are **not supported**.

### "Resolution error"

- Input must be at least 512x288 pixels
- For 4x upscaling, input must be 540p (960x540) or smaller

## License

This plugin is open source under the MIT License.

**Note:** This plugin provides bindings only. The NVIDIA Video Effects SDK is a separate product subject to [NVIDIA's SDK License Agreement](https://developer.nvidia.com/downloads/maxine-sdk-license). You must download and install the SDK directly from NVIDIA and agree to their terms.
