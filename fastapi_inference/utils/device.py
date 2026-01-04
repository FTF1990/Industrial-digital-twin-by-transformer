"""
GPU/CPU Device Management Utilities
"""
import torch
import gc


def setup_device():
    """
    Setup computing device with GPU detection and configuration

    Returns:
        torch.device: PyTorch device (cuda or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f" GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    else:
        device = torch.device('cpu')
        print("   GPU not available, using CPU")
        return device


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(">ù GPU memory cleared")


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"=Ê GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
