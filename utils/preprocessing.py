"""
Image Preprocessing Module (Legacy/Disabled)
============================================

This module previously handled EfficientNet preprocessing with PyTorch.
It is now disabled as we use the Hugging Face Inference API.
"""

# import torch
# from torchvision import transforms

def load_and_preprocess_image(*args, **kwargs):
    raise NotImplementedError("Local preprocessing is disabled. Use Hugging Face API.")

def get_image_transforms(*args, **kwargs):
    raise NotImplementedError("Local preprocessing is disabled. Use Hugging Face API.")
