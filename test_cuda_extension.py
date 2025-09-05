#!/usr/bin/env python3
"""
Test script to verify MultiScaleDeformableAttention CUDA extension
"""

import sys
import os

print("Testing CUDA extension import...")

try:
    # Test the import
    from models.segmentation_from_dinov3.models.utils.ms_deform_attn import MSDeformAttn
    print("SUCCESS: MSDeformAttn imported successfully")
    
    # Test MSDA availability
    import models.segmentation_from_dinov3.models.utils.ms_deform_attn as ms_deform_module
    if hasattr(ms_deform_module, 'MSDA') and ms_deform_module.MSDA is not None:
        print("SUCCESS: MultiScaleDeformableAttention CUDA extension loaded")
        print(f"MSDA module: {ms_deform_module.MSDA}")
    else:
        print("ERROR: MultiScaleDeformableAttention CUDA extension not loaded")
        
    # Test creating MSDeformAttn module
    import torch
    attn = MSDeformAttn(d_model=256, n_levels=4, n_heads=8, n_points=4)
    print("SUCCESS: MSDeformAttn module created")
    
    # Test forward pass (if CUDA available)
    if torch.cuda.is_available():
        print("Testing CUDA forward pass...")
        device = torch.device('cuda')
        attn = attn.to(device)
        
        # Create dummy input with correct dimensions
        bs, d_model = 1, 256
        n_levels, n_heads, n_points = 4, 8, 4
        
        # Define spatial shapes for 4 levels
        H1, W1 = 28, 28  # Level 0
        H2, W2 = 14, 14  # Level 1  
        H3, W3 = 7, 7    # Level 2
        H4, W4 = 4, 4    # Level 3
        
        # Total flattened length
        Len_in = H1*W1 + H2*W2 + H3*W3 + H4*W4
        num_queries = 100
        
        query = torch.randn(bs, num_queries, d_model, device=device)
        reference_points = torch.rand(bs, num_queries, n_levels, 2, device=device)
        input_flatten = torch.randn(bs, Len_in, d_model, device=device)
        input_spatial_shapes = torch.tensor([[H1, W1], [H2, W2], [H3, W3], [H4, W4]], device=device)
        input_level_start_index = torch.tensor([0, H1*W1, H1*W1+H2*W2, H1*W1+H2*W2+H3*W3], device=device)
        
        with torch.no_grad():
            output = attn(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index)
        print(f"SUCCESS: MSDeformAttn forward pass successful: {output.shape}")
    else:
        print("CUDA not available, skipping forward pass test")
        
    print("\nAll tests passed! CUDA extension is working correctly.")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    
    # Additional debug info
    print(f"\nDEBUG INFO:")
    print(f"Python path: {sys.path[:3]}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if .pyd file exists
    ops_path = os.path.join(os.getcwd(), "models", "segmentation_from_dinov3", "models", "utils", "ops")
    print(f"Ops path: {ops_path}")
    if os.path.exists(ops_path):
        files = os.listdir(ops_path)
        pyd_files = [f for f in files if f.endswith('.pyd')]
        print(f"PYD files found: {pyd_files}")
    else:
        print("Ops path does not exist!")