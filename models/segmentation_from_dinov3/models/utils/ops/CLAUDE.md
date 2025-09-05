# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Multi-Scale Deformable Attention (MSDA) operations module, part of the DINOv3 downstream segmentation models. It provides PyTorch bindings for CUDA-accelerated multi-scale deformable attention operations, originally derived from Deformable DETR.

## Core Architecture

### Module Structure
- `functions/`: PyTorch autograd functions for MSDA operations
  - `ms_deform_attn_func.py`: Main MSDA function with forward/backward passes
- `modules/`: High-level PyTorch modules
  - `ms_deform_attn.py`: MSDeformAttn neural network module
- `src/`: C++/CUDA implementation
  - `cpu/`: CPU fallback implementations  
  - `cuda/`: CUDA kernels for GPU acceleration
  - `vision.cpp`: Main C++ interface

### Key Components
- **MSDeformAttnFunction**: Low-level autograd function calling CUDA kernels
- **MSDeformAttn**: High-level nn.Module providing the attention mechanism
- **CUDA Extensions**: Custom C++/CUDA code compiled at install time

## Development Commands

### Building the Extension
```bash
# Build and install the CUDA extension
python setup.py build_ext --inplace

# Full installation (recommended for development)
pip install -e .
```

### Testing
```bash
# Run comprehensive tests (forward pass, gradients, numerical checks)
python test.py
```

The test suite validates:
- Forward pass equivalence between CUDA and PyTorch implementations
- Gradient correctness through numerical differentiation
- Multiple precision levels (float32/float64)
- Various channel dimensions (30, 32, 64, 71, 1025, 2048, 3096)

## Requirements

### System Dependencies
- CUDA toolkit with development headers
- PyTorch with CUDA support
- C++ compiler compatible with CUDA

### Critical Notes
- **CUDA Required**: This module will not build without CUDA availability
- **Memory Usage**: im2col_step parameter (default: 64) controls memory vs speed tradeoff
- **Power of 2 Dimensions**: For optimal CUDA performance, ensure d_model/n_heads is a power of 2

## Usage Pattern

```python
from modules import MSDeformAttn

# Initialize attention module
attn = MSDeformAttn(d_model=256, n_levels=4, n_heads=8, n_points=4)

# Forward pass requires:
# - query: (N, Length_query, C)
# - reference_points: (N, Length_query, n_levels, 2 or 4)  
# - input_flatten: (N, sum(H_l * W_l), C)
# - input_spatial_shapes: (n_levels, 2)
# - input_level_start_index: (n_levels,)
output = attn(query, reference_points, input_flatten, 
              input_spatial_shapes, input_level_start_index)
```

## License and Attribution

This code combines:
- Meta Platforms DINOv3 License Agreement
- SenseTime Deformable DETR (Apache License 2.0)  
- Modified from Deformable Convolution V2 PyTorch implementation

Always respect both licensing requirements when modifying or distributing.