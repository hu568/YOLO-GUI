#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Components - 增强组件模块
包含批量检测、结果显示、监控等组件
"""

import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *


class BatchDetectionThread(QThread):
    """批量检测线程"""
    result_ready = Signal(str, object, object, float, object, list)  # 文件路径, 原图, 结果图, 耗时, 检测结果, 类别名称
    progress_updated = Signal(int)
    current_file_changed = Signal(str)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, model, folder_path, confidence_threshold=0.25, supported_formats=None):
        super().__init__()
        self.model = model
        self.folder_path = folder_path
        self.confidence_threshold = confidence_threshold
        self.supported_formats = supported_formats or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.tif']
        self.is_running = False
        self.processed_count = 0
        self.error_count = 0

    def run(self):
        self.is_running = True

        try:
            # 收集所有支持的图片文件
            image_files = []
            for fmt in self.supported_formats:
                image_files.extend(Path(self.folder_path).rglob(f'*{fmt}'))
                image_files.extend(Path(self.folder_path).rglob(f'*{fmt.upper()}'))

            total_files = len(image_files)
            if total_files == 0:
                self.status_changed.emit("文件夹中没有找到支持的图片格式")
                self.finished.emit()
                return

            self.status_changed.emit(f"开始批量处理 {total_files} 个文件...")

            # 获取类别名称
            class_names = list(self.model.names.values())

            for i, img_path in enumerate(image_files):
                if not self.is_running:
                    break

                self.current_file_changed.emit(str(img_path))

                try:
                    # 处理单个图片
                    start_time = time.time()
                    results = self.model(str(img_path), conf=self.confidence_threshold, verbose=False)
                    end_time = time.time()

                    # 获取原图
                    original_img = cv2.imread(str(img_path))
                    if original_img is not None:
                        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                        # 获取结果图
                        result_img = results[0].plot()
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                        self.result_ready.emit(str(img_path), original_img, result_img,
                                               end_time - start_time, results, class_names)
                        self.processed_count += 1

                except Exception as e:
                    self.error_occurred.emit(f"处理文件 {img_path.name} 时发生错误: {str(e)}")
                    self.error_count += 1

                # 更新进度
                progress = int(((i + 1) / total_files) * 100)
                self.progress_updated.emit(progress)

                # 状态更新
                if (i + 1) % 10 == 0 or i == total_files - 1:
                    self.status_changed.emit(
                        f"处理进度: {i + 1}/{total_files} (成功: {self.processed_count}, 错误: {self.error_count})")

        except Exception as e:
            self.error_occurred.emit(f"批量处理发生错误: {str(e)}")
        finally:
            self.is_running = False
            self.finished.emit()

    def stop(self):
        """停止批量检测"""
        self.is_running = False


class MultiCameraMonitorThread(QThread):
    """多摄像头监控线程"""
    camera_result_ready = Signal(int, object, object, float, object, list)  # 摄像头ID, 原图, 结果图, 耗时, 检测结果, 类别名称
    camera_error = Signal(int, str)  # 摄像头ID, 错误信息
    camera_status_changed = Signal(int, str)  # 摄像头ID, 状态信息
    finished = Signal()

    def __init__(self, model, camera_ids, confidence_threshold=0.25):
        super().__init__()
        self.model = model
        self.camera_ids = camera_ids
        self.confidence_threshold = confidence_threshold
        self.is_running = False
        self.cameras = {}
        self.last_frame_times = {}

    def run(self):
        self.is_running = True

        # 初始化所有摄像头
        for camera_id in self.camera_ids:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                self.cameras[camera_id] = cap
                self.last_frame_times[camera_id] = time.time()
                self.camera_status_changed.emit(camera_id, "已连接")
            else:
                self.camera_error.emit(camera_id, "无法打开摄像头")

        if not self.cameras:
            self.finished.emit()
            return

        # 获取类别名称
        class_names = list(self.model.names.values())

        try:
            while self.is_running:
                for camera_id, cap in self.cameras.items():
                    if not self.is_running:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        self.camera_error.emit(camera_id, "读取帧失败")
                        continue

                    # 控制检测频率（每个摄像头约10fps）
                    current_time = time.time()
                    if current_time - self.last_frame_times[camera_id] < 0.1:
                        continue

                    self.last_frame_times[camera_id] = current_time

                    try:
                        start_time = time.time()
                        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                        end_time = time.time()

                        # 获取原图和结果图
                        original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result_img = results[0].plot()
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                        self.camera_result_ready.emit(camera_id, original_img, result_img,
                                                      end_time - start_time, results, class_names)

                    except Exception as e:
                        self.camera_error.emit(camera_id, f"检测错误: {str(e)}")

                time.sleep(0.01)  # 小延迟避免CPU占用过高

        except Exception as e:
            for camera_id in self.cameras:
                self.camera_error.emit(camera_id, f"线程错误: {str(e)}")
        finally:
            # 释放所有摄像头
            for cap in self.cameras.values():
                cap.release()
            self.cameras.clear()
            self.is_running = False
            self.finished.emit()

    def stop(self):
        """停止多摄像头检测"""
        self.is_running = False


class ModelSelectionDialog(QDialog):
    """模型选择对话框"""

    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.selected_model = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🔧 高级模型选择")
        self.setModal(True)
        self.resize(700, 450)

        layout = QVBoxLayout(self)

        # 自定义路径
        path_group = QGroupBox("📁 自定义模型路径")
        path_layout = QHBoxLayout(path_group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("输入自定义模型目录路径...")
        path_layout.addWidget(self.path_edit)

        browse_btn = QPushButton("📂 浏览")
        browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(browse_btn)

        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self.refresh_models)
        path_layout.addWidget(refresh_btn)

        layout.addWidget(path_group)

        # 模型列表
        models_group = QGroupBox("📋 可用模型")
        models_layout = QVBoxLayout(models_group)

        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(["模型名称", "大小", "修改时间", "路径"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.model_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.doubleClicked.connect(self.accept)

        models_layout.addWidget(self.model_table)
        layout.addWidget(models_group)

        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # 设置样式
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
        """)

        self.refresh_models()

    def browse_path(self):
        """浏览自定义路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self.path_edit.setText(path)
            self.refresh_models()

    def refresh_models(self):
        """刷新模型列表"""
        custom_path = self.path_edit.text() if self.path_edit.text() else None
        models = self.model_manager.scan_models(custom_path)

        self.model_table.setRowCount(len(models))

        for i, model in enumerate(models):
            self.model_table.setItem(i, 0, QTableWidgetItem(model['name']))
            self.model_table.setItem(i, 1, QTableWidgetItem(model['size']))
            self.model_table.setItem(i, 2, QTableWidgetItem(model['modified']))
            self.model_table.setItem(i, 3, QTableWidgetItem(model['path']))

    def accept(self):
        """确认选择"""
        current_row = self.model_table.currentRow()
        if current_row >= 0:
            self.selected_model = self.model_table.item(current_row, 3).text()
        super().accept()


class DetectionResultWidget(QWidget):
    """检测结果显示组件"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("🎯 检测结果详情")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)

        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(["序号", "类别", "置信度", "坐标 (x,y)", "尺寸 (w×h)"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setMaximumHeight(200)
        self.result_table.setAlternatingRowColors(True)

        layout.addWidget(self.result_table)

        # 统计信息
        self.stats_label = QLabel("等待检测结果...")
        self.stats_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(236, 240, 241, 0.9), stop:1 rgba(189, 195, 199, 0.9));
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            color: #2c3e50;
            font-weight: bold;
        """)
        layout.addWidget(self.stats_label)

    def update_results(self, results, class_names, inference_time):
        """更新检测结果"""
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            self.result_table.setRowCount(0)
            self.stats_label.setText("❌ 未检测到目标")
            return

        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        # 更新表格
        self.result_table.setRowCount(len(confidences))

        class_counts = {}
        for i, (conf, cls, box) in enumerate(zip(confidences, classes, xyxy)):
            class_name = class_names[cls] if cls < len(class_names) else f"类别{cls}"

            # 统计类别数量
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            self.result_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.result_table.setItem(i, 1, QTableWidgetItem(class_name))

            # 置信度带颜色
            conf_item = QTableWidgetItem(f"{conf:.3f}")
            if conf > 0.8:
                conf_item.setBackground(QColor(46, 204, 113, 100))  # 绿色
            elif conf > 0.5:
                conf_item.setBackground(QColor(241, 196, 15, 100))  # 黄色
            else:
                conf_item.setBackground(QColor(231, 76, 60, 100))  # 红色
            self.result_table.setItem(i, 2, conf_item)

            self.result_table.setItem(i, 3, QTableWidgetItem(f"({box[0]:.0f},{box[1]:.0f})"))
            self.result_table.setItem(i, 4, QTableWidgetItem(f"{box[2] - box[0]:.0f}×{box[3] - box[1]:.0f}"))

        # 更新统计信息
        total_objects = len(confidences)
        avg_confidence = np.mean(confidences)

        stats_text = f"✅ 检测到 {total_objects} 个目标 | "
        stats_text += f"🎯 平均置信度: {avg_confidence:.3f} | "
        stats_text += f"⏱️ 耗时: {inference_time:.3f}秒\n"
        stats_text += "📊 类别统计: " + " | ".join([f"{name}: {count}" for name, count in class_counts.items()])

        self.stats_label.setText(stats_text)


class MonitoringWidget(QWidget):
    """监控页面组件"""

    def __init__(self, model_manager, camera_manager):
        super().__init__()
        self.model_manager = model_manager
        self.camera_manager = camera_manager
        self.monitoring_thread = None
        self.camera_labels = {}
        self.current_model = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 控制面板
        control_group = QGroupBox("🖥️ 监控控制")
        control_layout = QVBoxLayout(control_group)

        # 模型选择
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("模型:"))

        self.model_combo = QComboBox()
        self.model_combo.addItem("请先加载模型")
        self.model_combo.setMinimumWidth(80)
        model_layout.addWidget(self.model_combo)

        select_model_btn = QPushButton("🔧 选择模型")
        select_model_btn.clicked.connect(self.select_model)
        model_layout.addWidget(select_model_btn)

        control_layout.addLayout(model_layout)

        # 摄像头选择
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(QLabel("摄像头:"))

        self.camera_list = QListWidget()
        # self.camera_list.setMaximumHeight(50)
        self.camera_list.setSelectionMode(QListWidget.MultiSelection)
        self.refresh_cameras()
        camera_layout.addWidget(self.camera_list)

        refresh_camera_btn = QPushButton("🔄 刷新")
        refresh_camera_btn.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(refresh_camera_btn)

        control_layout.addLayout(camera_layout)

        # 控制按钮
        btn_layout = QHBoxLayout()

        self.start_monitor_btn = QPushButton("🚀 开始监控")
        self.start_monitor_btn.clicked.connect(self.start_monitoring)
        self.start_monitor_btn.setEnabled(False)
        btn_layout.addWidget(self.start_monitor_btn)

        self.stop_monitor_btn = QPushButton("⏹️ 停止监控")
        self.stop_monitor_btn.clicked.connect(self.stop_monitoring)
        self.stop_monitor_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_monitor_btn)

        control_layout.addLayout(btn_layout)

        layout.addWidget(control_group)


        # 监控显示区域
        self.monitor_scroll = QScrollArea()
        self.monitor_widget = QWidget()
        self.monitor_layout = QGridLayout(self.monitor_widget)
        self.monitor_scroll.setWidget(self.monitor_widget)
        self.monitor_scroll.setWidgetResizable(True)

        layout.addWidget(self.monitor_scroll)

    def select_model(self):
        """选择模型"""
        from enhanced_detection_main import YOLO

        dialog = ModelSelectionDialog(self.model_manager, self)
        if dialog.exec() == QDialog.Accepted and dialog.selected_model:
            try:
                self.current_model = YOLO(dialog.selected_model)
                model_name = Path(dialog.selected_model).name
                self.model_combo.clear()
                self.model_combo.addItem(model_name)
                self.start_monitor_btn.setEnabled(True)
                QMessageBox.information(self, "成功", f"模型加载成功: {model_name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")

    def refresh_cameras(self):
        """刷新摄像头列表"""
        self.camera_manager.scan_cameras()
        self.camera_list.clear()

        for camera in self.camera_manager.get_available_cameras():
            item = QListWidgetItem(f"📹 {camera['name']} ({camera['resolution']})")
            item.setData(Qt.UserRole, camera['id'])
            self.camera_list.addItem(item)

    def start_monitoring(self):
        """开始监控"""
        if not self.current_model:
            QMessageBox.warning(self, "警告", "请先选择模型")
            return

        selected_items = self.camera_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请选择至少一个摄像头")
            return

        camera_ids = [item.data(Qt.UserRole) for item in selected_items]

        # 清空之前的显示
        self.clear_monitor_display()

        # 创建显示标签
        self.create_camera_labels(camera_ids)

        # 启动监控线程
        self.monitoring_thread = MultiCameraMonitorThread(self.current_model, camera_ids)
        self.monitoring_thread.camera_result_ready.connect(self.update_camera_display)
        self.monitoring_thread.camera_error.connect(self.handle_camera_error)
        self.monitoring_thread.finished.connect(self.on_monitoring_finished)

        self.monitoring_thread.start()

        self.start_monitor_btn.setEnabled(False)
        self.stop_monitor_btn.setEnabled(True)

    def stop_monitoring(self):
        """停止监控"""
        if self.monitoring_thread and self.monitoring_thread.is_running:
            self.monitoring_thread.stop()
            self.monitoring_thread.wait()

    def create_camera_labels(self, camera_ids):
        """创建摄像头显示标签"""
        self.camera_labels = {}

        cols = 2  # 每行2个摄像头
        for i, camera_id in enumerate(camera_ids):
            row = i // cols
            col = i % cols

            # 创建摄像头组
            camera_group = QGroupBox(f"📹 摄像头 {camera_id}")
            camera_layout = QVBoxLayout(camera_group)

            # 图像显示标签
            image_label = QLabel("等待连接...")
            image_label.setMinimumSize(320, 240)
            image_label.setStyleSheet("""
                border: 3px solid rgba(52, 152, 219, 0.3);
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(248, 249, 250, 0.9), stop:1 rgba(233, 236, 239, 0.9));
                color: #7f8c8d;
                font-weight: bold;
                font-size: 14px;
                border-radius: 10px;
                padding: 15px;
            """)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setScaledContents(True)

            camera_layout.addWidget(image_label)

            # 状态标签
            status_label = QLabel("状态: 初始化中...")
            status_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
            camera_layout.addWidget(status_label)

            self.camera_labels[camera_id] = {
                'image': image_label,
                'status': status_label,
                'group': camera_group
            }

            self.monitor_layout.addWidget(camera_group, row, col)

    def clear_monitor_display(self):
        """清空监控显示"""
        for camera_id in list(self.camera_labels.keys()):
            self.camera_labels[camera_id]['group'].deleteLater()
        self.camera_labels.clear()

    def update_camera_display(self, camera_id, original_img, result_img, inference_time, results, class_names):
        """更新摄像头显示"""
        if camera_id not in self.camera_labels:
            return

        # 显示结果图
        self.display_image(result_img, self.camera_labels[camera_id]['image'])

        # 更新状态
        if results and results[0].boxes and len(results[0].boxes) > 0:
            object_count = len(results[0].boxes)
            self.camera_labels[camera_id]['status'].setText(
                f"状态: 检测到 {object_count} 个目标 | 耗时: {inference_time:.3f}s"
            )
        else:
            self.camera_labels[camera_id]['status'].setText(
                f"状态: 无目标 | 耗时: {inference_time:.3f}s"
            )

    def handle_camera_error(self, camera_id, error_msg):
        """处理摄像头错误"""
        if camera_id in self.camera_labels:
            self.camera_labels[camera_id]['status'].setText(f"错误: {error_msg}")
            self.camera_labels[camera_id]['status'].setStyleSheet("color: red; font-size: 10px;")

    def on_monitoring_finished(self):
        """监控结束"""
        self.start_monitor_btn.setEnabled(True)
        self.stop_monitor_btn.setEnabled(False)

        for camera_id in self.camera_labels:
            self.camera_labels[camera_id]['status'].setText("状态: 已停止")

    def display_image(self, img_array, label):
        """显示图像"""
        if img_array is None:
            return

        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)