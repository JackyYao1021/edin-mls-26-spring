"""
Template Baseline - Triton Student Assignment
Performance: TBD (Torch baseline with Triton kernels available)

Key Characteristics:
- Pure Torch tensor operations
- Triton kernels for core ops (student TODOs)
"""

import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from . import layers

# Runtime defaults (cublas Linear, fused MLP off) are set in layers.py for ~10GB GPUs.
# For Triton matmul + fused MLP/Encoder: GLM_ASR_LINEAR_BACKEND=triton GLM_ASR_MLP_FUSED=1 GLM_ASR_ENCODER_MLP_FUSED=1

from . import model
from . import rope
from . import conv
from . import weight_loader
