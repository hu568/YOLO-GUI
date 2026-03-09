#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ultralytics 库是否可以正确导入
"""

import sys
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")

# 尝试导入 ultralytics
try:
    from ultralytics import YOLO
    print("✅ 成功导入 ultralytics 库")
    print(f"ultralytics 版本: {YOLO.__version__}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("尝试查看 sys.path:")
    for path in sys.path:
        print(f"  - {path}")
