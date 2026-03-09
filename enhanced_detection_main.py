#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Universal Object Detection System v2.0
优化的通用目标检测系统 - 主程序

新功能特性:
✨ 渐变UI样式效果
📱 优化的响应式布局
📊 增强的日志显示（类别识别信息）
📁 支持自定义模型目录加载
📹 多摄像头支持和选择
🖥️ 实时监控页面
🎨 优化的图标设计
⚡ 性能优化和错误处理
"""

import sys
import os
import cv2
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 请安装ultralytics库: pip install ultralytics")
    sys.exit(1)


class StyleManager:
    """样式管理器 - 提供渐变和现代化UI样式"""

    @staticmethod
    def get_main_stylesheet():
        return """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }

            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid rgba(52, 152, 219, 0.3);
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.9), stop:1 rgba(245, 245, 245, 0.9));
                color: #2c3e50;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #2c3e50;
                font-size: 13px;
                font-weight: bold;
            }

            QPushButton {
                padding: 2px 8px;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                color: white;
                min-width: 65px;
                min-height: 25px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
                transform: translateY(-1px);
            }

            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980b9, stop:1 #1f618d);
            }

            QPushButton:disabled {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #bdc3c7, stop:1 #95a5a6);
                color: #7f8c8d;
            }

            QComboBox {
                padding: 2px 8px;
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 white, stop:1 #f8f9fa);
                font-size: 12px;
                min-width: 150px;
                min-height: 25px;
                color: #2c3e50;
            }

            QComboBox:focus {
                border-color: #3498db;
                background: white;
            }

            QProgressBar {
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 11px;
                max-height: 20px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ecf0f1, stop:1 #d5dbdb);
                color: #2c3e50;
            }

            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2ecc71, stop:1 #27ae60);
                border-radius: 6px;
                margin: 1px;
            }

            QTextEdit {
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.95);
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 8px;
                selection-background-color: #3498db;
                color: #2c3e50;
            }

            QSlider::groove:horizontal {
                border: 1px solid rgba(189, 195, 199, 0.5);
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ecf0f1, stop:1 #bdc3c7);
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                border: 2px solid #2980b9;
                width: 20px;
                height: 20px;
                margin: -8px 0;
                border-radius: 12px;
            }

            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
            }

            QSpinBox, QDoubleSpinBox {
                padding: 6px 10px;
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 6px;
                background: white;
                min-width: 80px;
                font-size: 12px;
                color: #2c3e50;
            }

            QTabWidget::pane {
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.95), stop:1 rgba(245, 245, 245, 0.95));
                margin-top: 5px;
            }

            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ecf0f1, stop:1 #bdc3c7);
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-bottom: none;
                border-radius: 8px 8px 0 0;
                padding: 12px 25px;
                margin-right: 3px;
                font-weight: bold;
                font-size: 12px;
                color: #2c3e50;
            }

            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border-color: rgba(52, 152, 219, 0.7);
            }

            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #d5dbdb, stop:1 #bdc3c7);
            }

            QTableWidget {
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 8px;
                background: white;
                gridline-color: rgba(189, 195, 199, 0.3);
                selection-background-color: rgba(52, 152, 219, 0.2);
                alternate-background-color: rgba(248, 249, 250, 0.5);
                color: #2c3e50;
            }

            QTableWidget::item {
                padding: 8px;
                border: none;
                color: #2c3e50;
            }

            QTableWidget::item:selected {
                background: rgba(52, 152, 219, 0.3);
                color: #2c3e50;
            }

            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #34495e, stop:1 #2c3e50);
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }

            QListWidget {
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 8px;
                background: white;
                selection-background-color: rgba(52, 152, 219, 0.2);
                color: #2c3e50;
            }

            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid rgba(189, 195, 199, 0.2);
                color: #2c3e50;
            }

            QListWidget::item:selected {
                background: rgba(52, 152, 219, 0.3);
                color: #2c3e50;
            }

            QScrollBar:vertical {
                background: rgba(236, 240, 241, 0.5);
                width: 12px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #bdc3c7, stop:1 #95a5a6);
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #95a5a6, stop:1 #7f8c8d);
            }

            QLabel {
                color: #2c3e50;
            }
        """

    @staticmethod
    def get_image_label_style():
        return """
            border: 3px solid rgba(52, 152, 219, 0.3);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(248, 249, 250, 0.9), stop:1 rgba(233, 236, 239, 0.9));
            color: #2c3e50;
            font-weight: bold;
            font-size: 14px;
            border-radius: 10px;
            padding: 15px;
        """


class CameraManager:
    """摄像头管理器 - 处理多摄像头检测和管理"""

    def __init__(self):
        self.cameras = []
        self.scan_cameras()

    def scan_cameras(self):
        """扫描可用摄像头"""
        self.cameras = []

        # 检测摄像头（检测前8个索引）
        for i in range(8):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # 获取摄像头信息
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    camera_info = {
                        'id': i,
                        'name': f"摄像头 {i}",
                        'resolution': f"{width}x{height}",
                        'fps': fps if fps > 0 else 30,
                        'available': True
                    }
                    self.cameras.append(camera_info)
                cap.release()

        # 如果没有摄像头，添加虚拟摄像头用于测试
        if not self.cameras:
            self.cameras.append({
                'id': -1,
                'name': "未检测到摄像头",
                'resolution': "N/A",
                'fps': 0,
                'available': False
            })

    def get_available_cameras(self):
        """获取可用摄像头列表"""
        return [cam for cam in self.cameras if cam['available']]

    def get_camera_info(self, camera_id):
        """获取摄像头信息"""
        for cam in self.cameras:
            if cam['id'] == camera_id:
                return cam
        return None


class ModelManager:
    """模型管理器 - 处理模型扫描和加载"""

    def __init__(self):
        self.models_paths = [
            Path("pt_models"),
            Path("models"),
            Path("weights"),
            Path("yolo_models"),
            Path.home() / "yolo_models",
            Path("/usr/local/share/yolo_models"),  # Linux系统路径
            Path("C:/yolo_models"),  # Windows系统路径
        ]
        self.current_model = None
        self.class_names = []

    def scan_models(self, custom_path=None):
        """扫描模型文件"""
        models = []
        search_paths = self.models_paths.copy()

        if custom_path and Path(custom_path).exists():
            search_paths.insert(0, Path(custom_path))

        for model_dir in search_paths:
            if model_dir.exists():
                try:
                    pt_files = sorted(model_dir.glob("*.pt"))
                    for pt_file in pt_files:
                        models.append({
                            'name': pt_file.name,
                            'path': str(pt_file),
                            'size': self._get_file_size(pt_file),
                            'modified': self._get_modification_time(pt_file)
                        })
                except Exception as e:
                    print(f"扫描目录 {model_dir} 时出错: {e}")

        return models

    def load_model(self, model_path):
        """加载模型"""
        try:
            self.current_model = YOLO(model_path)
            self.class_names = list(self.current_model.names.values())
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def get_class_names(self):
        """获取类别名称"""
        return self.class_names

    def _get_file_size(self, file_path):
        """获取文件大小"""
        try:
            size = file_path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "Unknown"

    def _get_modification_time(self, file_path):
        """获取修改时间"""
        try:
            timestamp = file_path.stat().st_mtime
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
        except:
            return "Unknown"


class DetectionThread(QThread):
    """增强的检测线程"""
    result_ready = Signal(object, object, float, object, list)  # 原图, 结果图, 耗时, 检测结果, 类别名称
    progress_updated = Signal(int)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    fps_updated = Signal(float)
    finished = Signal()

    def __init__(self, model, source_type, source_path=None, camera_id=0, confidence_threshold=0.25):
        super().__init__()
        self.model = model
        self.source_type = source_type
        self.source_path = source_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()

    def run(self):
        self.is_running = True
        try:
            if self.source_type == 'image':
                self._process_image()
            elif self.source_type == 'video':
                self._process_video()
            elif self.source_type == 'camera':
                self._process_camera()
        except Exception as e:
            self.error_occurred.emit(f"检测过程发生错误: {str(e)}")
        finally:
            self.is_running = False
            self.finished.emit()

    def _process_image(self):
        """处理单张图片"""
        if not self.source_path or not Path(self.source_path).exists():
            self.error_occurred.emit("图片文件不存在")
            return

        self.status_changed.emit("正在处理图片...")

        start_time = time.time()
        results = self.model(self.source_path, conf=self.confidence_threshold, verbose=False)
        end_time = time.time()

        original_img = cv2.imread(self.source_path)
        if original_img is None:
            self.error_occurred.emit("无法读取图片文件")
            return

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        class_names = list(self.model.names.values())

        self.result_ready.emit(original_img, result_img, end_time - start_time, results, class_names)
        self.progress_updated.emit(100)

    def _process_video(self):
        """处理视频文件"""
        if not self.source_path or not Path(self.source_path).exists():
            self.error_occurred.emit("视频文件不存在")
            return

        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            self.error_occurred.emit("无法打开视频文件")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        class_names = list(self.model.names.values())

        self.status_changed.emit(f"开始处理视频 (共{total_frames}帧)...")

        while cap.isOpened() and self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            end_time = time.time()

            original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            self.result_ready.emit(original_img, result_img, end_time - start_time, results, class_names)

            frame_count += 1
            if total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)

            # 更新FPS
            self._update_fps()

            # 状态更新（每30帧更新一次）
            if frame_count % 30 == 0:
                current_fps = self._get_current_fps()
                self.status_changed.emit(f"处理中... {frame_count}/{total_frames} 帧 (FPS: {current_fps:.1f})")

            time.sleep(0.033)  # 约30fps

        cap.release()

    def _process_camera(self):
        """处理摄像头"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.error_occurred.emit(f"无法打开摄像头 {self.camera_id}")
            return

        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        class_names = list(self.model.names.values())
        self.status_changed.emit(f"摄像头 {self.camera_id} 已启动...")

        while cap.isOpened() and self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            end_time = time.time()

            original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            self.result_ready.emit(original_img, result_img, end_time - start_time, results, class_names)

            # 更新FPS
            self._update_fps()

            # 状态更新（每60帧更新一次）
            if self.frame_count % 60 == 0:
                current_fps = self._get_current_fps()
                self.status_changed.emit(f"摄像头运行中 (FPS: {current_fps:.1f})")

            time.sleep(0.033)  # 约30fps

        cap.release()

    def _update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        self.fps_counter += 1

        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_updated.emit(fps)
            self.fps_counter = 0
            self.last_fps_time = current_time

    def _get_current_fps(self):
        """获取当前FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time > 0:
            return self.fps_counter / (current_time - self.last_fps_time)
        return 0

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_running = False

# 继续下一部分...