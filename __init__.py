from .color_correct_gpu import ColorCorrectGPU


NODE_CLASS_MAPPINGS = {
    "ColorCorrectGPU": ColorCorrectGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorCorrectGPU": "Color Correct GPU",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
