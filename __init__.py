"""
即梦AI图像生成节点包
支持文生图和图生图
"""

from .jimeng_image_node import JimengImageNode

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "Jimeng_Image": JimengImageNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng_Image": "即梦AI生图（Token版）",
}

__version__ = "1.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]



