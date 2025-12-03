import sys
import os
import numpy as np
import pydicom
import cv2
from PyQt5.QtCore import Qt, QTime, QRectF, QPointF, QThread, pyqtSignal
from PIL import Image
from ultralytics import SAM, YOLO
import highdicom.seg
from pydicom.uid import generate_uid
from highdicom.seg import SegmentDescription
from PyQt5.QtGui import QCursor,QBrush, QPen, QPixmap, QKeySequence, QColor, QImage, QPainter
from PyQt5.QtWidgets import (QMessageBox, QDialog, QSlider, QScrollBar,QProgressBar, QMenuBar, QMenu, QTextEdit, QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsScene, QGraphicsView, QShortcut, QFileDialog, QSizePolicy, QLabel, QGraphicsRectItem, QGraphicsEllipseItem, QGridLayout, QProgressDialog, QSlider, QComboBox, QGroupBox, QCheckBox, QDoubleSpinBox, QAction, QToolBar, QLineEdit)
from highdicom import AlgorithmIdentificationSequence
from highdicom.seg import Segmentation
from highdicom.content import Code
from pydicom.valuerep import PersonName
from pydicom.sr.codedict import codes
import highdicom
import vtk
import SimpleITK as sitk
import traceback
import math
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
vtk.vtkObject.GlobalWarningDisplayOff() 
from utils import intensity_to_color, intensity_to_label, np2pixmap, dicom_to_png, get_yolo_box, get_data_path, create_dicom_segmentation

class CustomDrawView(QGraphicsView):
    def __init__(self, scene, parent, mask_rgb, current_slice_img):
        super().__init__(scene, parent)
        self.parent_widget = parent
        self.mask_rgb = mask_rgb
        self.current_slice_img = current_slice_img
        self.is_drawing = False
        self.is_erasing = False
        self.last_point = None
        self.setRenderHint(QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setStyleSheet("""
            QGraphicsView {
                background-color: #F0F8FF;
                border: 2px solid #4682B4;
                border-radius: 5px;
            }
        """)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            if 0 <= x < self.mask_rgb.shape[1] and 0 <= y < self.mask_rgb.shape[0]:
                self.is_drawing = True
                self.last_point = (x, y)
                if self.is_erasing:
                    self.mask_rgb[y, x] = [0, 0, 0]  # Erase to black
                else:
                    self.mask_rgb[y, x] = self.parent_widget.current_color
                self.parent_widget.update_display()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            if 0 <= x < self.mask_rgb.shape[1] and 0 <= y < self.mask_rgb.shape[0] and self.last_point:
                if self.is_erasing:
                    cv2.line(self.mask_rgb, self.last_point, (x, y), (0, 0, 0), 5)
                else:
                    cv2.line(self.mask_rgb, self.last_point, (x, y), self.parent_widget.current_color, 1)
                self.last_point = (x, y)
                self.parent_widget.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.last_point = None

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def set_eraser_mode(self, enabled):
        self.is_erasing = enabled
        if enabled:
            # Set a square cursor (e.g., 20x20 pixels)
            square_cursor = QPixmap(20, 20)
            square_cursor.fill(Qt.transparent)  # Make the background transparent

            # Draw a solid square to represent the eraser
            painter = QPainter(square_cursor)
            painter.setBrush(Qt.white)
            painter.drawRect(0, 0, 10, 10)  # Draw a 20x20 pixel square
            painter.end()

            # Set the custom cursor
            self.setCursor(QCursor(square_cursor))
        else:
            self.setCursor(Qt.ArrowCursor)  # Reset to default cursor

class ManualDrawWindow(QWidget):
    def __init__(self, parent_viewer=None, current_slice_img=None, dicom_path=None, slice_index=0):
        super().__init__()
        self.setWindowTitle("Manual Drawing Tool")
        self.setGeometry(300, 300, 800, 600)
        self.setStyleSheet("background-color: #F0F8FF;")
        self.parent_viewer = parent_viewer
        if parent_viewer is None:
            self.dicom_images = []
            self.dicom_files = []
            self.original_slice_img = np.zeros((512, 512, 3), dtype=np.uint8) if current_slice_img is None else current_slice_img.copy()
            self.dicom_path = dicom_path if dicom_path else ""
            self.slice_index = slice_index
        else:
            self.dicom_images = parent_viewer.dicom_images
            self.dicom_files = parent_viewer.dicom_files
            self.original_slice_img = current_slice_img.copy()
            self.dicom_path = dicom_path
            self.slice_index = slice_index
        self.current_slice_img = self.original_slice_img.copy()
        self.mask_rgb = np.zeros_like(self.current_slice_img, dtype=np.uint8)
        self.current_color = [255, 0, 0]
        self.contrast_factor = 1.0
        self.brightness_offset = 0
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        controls_layout = QHBoxLayout()
        SLIDER_STYLESHEET = """
        QSlider:horizontal {
            border: 1px solid #2F4F4F; /* Darker border for depth */
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #D3D9DF, stop:0.5 #B0BEC5, stop:1 #A0A8B0); /* Gradient for groove effect */
            height: 15px;
            margin: 0px;
            border-radius: 3px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #2F4F4F;
            height: 8px; /* Groove height */
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #D3D9DF, stop:0.5 #B0BEC5, stop:1 #A0A8B0);
            margin: 2px 0;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #FF7F50, stop:0.5 #FF4500, stop:1 #CD5C5C); /* Gradient for raised handle */
            border-top: 1px solid #FFFFFF; /* Light top border for highlight */
            border-left: 1px solid #FFFFFF; /* Light left border for highlight */
            border-right: 1px solid #8B0000; /* Dark right border for shadow */
            border-bottom: 1px solid #8B0000; /* Dark bottom border for shadow */
            border-radius: 5px;
            width: 20px; /* Handle width */
            margin: -4px 0; /* Center handle vertically */
        }
        QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #FF8C00, stop:0.5 #FF6347, stop:1 #DC143C); /* Brighter gradient on hover */
            border-top: 1px solid #F0F8FF; /* Slightly brighter highlight */
            border-left: 1px solid #F0F8FF;
            border-right: 1px solid #A52A2A;
            border-bottom: 1px solid #A52A2A;
        }
        QSlider::handle:horizontal:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #CD5C5C, stop:0.5 #DC143C, stop:1 #FF4500); /* Sunken gradient on press */
            border-top: 1px solid #8B0000; /* Dark top border for sunken effect */
            border-left: 1px solid #8B0000;
            border-right: 1px solid #FFFFFF; /* Light right border for contrast */
            border-bottom: 1px solid #FFFFFF;
            padding-top: 1px; /* Slight offset to enhance sunken effect */
        }
        QSlider::sub-page:horizontal {
            background: none; /* Transparent to show groove gradient */
        }
        QSlider::add-page:horizontal {
            background: none; /* Transparent to show groove gradient */
        }
        """
        # Contrast slider
        contrast_label = QLabel("Contrast: 100%")
        self.contrast_slider = QSlider(Qt.Horizontal, self)
        self.contrast_slider.setRange(0, 200)  # 0% to 200%
        self.contrast_slider.setValue(100)     # Default: 100%
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.valueChanged.connect(lambda value: self.adjust_contrast(value, contrast_label))
        controls_layout.addWidget(contrast_label)
        controls_layout.addWidget(self.contrast_slider)

        # Brightness slider
        brightness_label = QLabel("Brightness: 0")
        self.brightness_slider = QSlider(Qt.Horizontal, self)
        self.brightness_slider.setRange(-100, 100)  # -100 to +100
        self.brightness_slider.setValue(0)          # Default: 0
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.setTickInterval(10)
        self.brightness_slider.valueChanged.connect(lambda value: self.adjust_brightness(value, brightness_label))
        controls_layout.addWidget(brightness_label)
        controls_layout.addWidget(self.brightness_slider)

        layout.addLayout(controls_layout)

        # Slice navigation slider
        slice_layout = QHBoxLayout()
        slice_count = len(self.parent_viewer.dicom_images) if self.parent_viewer is not None else len(self.dicom_images)
        slice_label = QLabel(f"Slice: {self.slice_index + 1}/{slice_count}")
        self.slice_slider = QSlider(Qt.Horizontal, self)
        self.slice_slider.setRange(0, slice_count - 1 if slice_count > 0 else 0)
        self.slice_slider.setValue(self.slice_index)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.setTickInterval(1)
        self.slice_slider.valueChanged.connect(lambda value: self.update_slice(value, slice_label))
        slice_layout.addWidget(slice_label)
        slice_layout.addWidget(self.slice_slider)
        layout.addLayout(slice_layout)

        self.contrast_slider.setStyleSheet(SLIDER_STYLESHEET)
        self.brightness_slider.setStyleSheet(SLIDER_STYLESHEET)
        self.slice_slider.setStyleSheet(SLIDER_STYLESHEET)

        # Graphics view
        self.scene = QGraphicsScene()
        self.view = CustomDrawView(self.scene, self, self.mask_rgb, self.current_slice_img)
        self.view.setStyleSheet("""
        QGraphicsView {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #E6ECEF, stop:0.5 #F0F8FF, stop:1 #D3D9DF);
            border-top: 1px solid #4682B4; /* Dark top border for sunken effect */
            border-left: 1px solid #4682B4; /* Dark left border for sunken effect */
            border-right: 1px solid #FFFFFF; /* Light right border for highlight */
            border-bottom: 1px solid #FFFFFF; /* Light bottom border for highlight */
            border-radius: 5px;
            padding: 3px; /* Padding to create inset effect */
        }
        """)
        layout.addWidget(self.view)
        self.pixmap = np2pixmap(self.current_slice_img)
        self.bg_item = self.scene.addPixmap(self.pixmap)
        self.scene.setSceneRect(0, 0, self.current_slice_img.shape[1], self.current_slice_img.shape[0])
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Button layouts
        button_layout1 = QHBoxLayout()
        self.button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                           stop:0 #E6ECEF, stop:1 #B0BEC5);
                border: 1px solid #78909C;
                border-top-color: #FFFFFF;
                border-left-color: #FFFFFF;
                border-right-color: #546E7A;
                border-bottom-color: #546E7A;
                border-radius: 5px;
                padding: 5px;
                color: #263238;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                           stop:0 #CFD8DC, stop:1 #90A4AE);
                border: 1px solid #546E7A;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                           stop:0 #90A4AE, stop:1 #CFD8DC);
                border: 1px solid #546E7A;
                border-top-color: #546E7A;
                border-left-color: #546E7A;
                border-right-color: #FFFFFF;
                border-bottom-color: #FFFFFF;
                padding-top: 6px;
                padding-bottom: 4px;
            }
        """

        self.load_dicom_btn = QPushButton("Load DICOM Folder")
        self.load_dicom_btn.setStyleSheet(self.button_style)
        self.load_dicom_btn.clicked.connect(self.load_dicom_folder)
        button_layout1.addWidget(self.load_dicom_btn)

        self.save_btn = QPushButton("Save Mask")
        self.save_btn.setStyleSheet(self.button_style)
        self.save_btn.clicked.connect(self.save_mask)
        button_layout1.addWidget(self.save_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet(self.button_style)
        self.refresh_btn.clicked.connect(self.refresh)
        button_layout1.addWidget(self.refresh_btn)

        self.eraser_btn = QPushButton("Eraser")
        self.eraser_btn.setStyleSheet(self.button_style)
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(self.toggle_eraser)
        button_layout1.addWidget(self.eraser_btn)

        self.load_auto_btn = QPushButton("Load Annotation")
        self.load_auto_btn.setStyleSheet(self.button_style)
        self.load_auto_btn.clicked.connect(self.load_auto_annotation)
        button_layout1.addWidget(self.load_auto_btn)

        button_layout2 = QHBoxLayout()
        self.color_buttons = []
        self.color_map = {
            "🎨 Red (255, 0, 0):Tumor | Brain": (255, 0, 0),
            "🎨 Green (0, 255, 0):Tumor | Dura": (0, 255, 0),
            "🎨 Blue (0, 0, 255):Tumor | Necrosis": (0, 0, 255),
            "🎨 Yellow (255, 255, 0):Edema | Brain": (255, 255, 0),
            "🎨 Cyan (0, 255, 255):Tumor | Edema": (0, 255, 255),
            "🎨 Magenta (255, 0, 255)": (255, 0, 255)
        }
        for name, rgb in self.color_map.items():
            btn = QPushButton(name)
            btn.setStyleSheet("background-color: white; border: 1px solid #4682B4; border-radius: 5px; padding: 2px;")
            btn.clicked.connect(lambda checked, c=rgb: self.set_color(c))
            self.color_buttons.append(btn)
            button_layout2.addWidget(btn)
        self.color_buttons[0].setStyleSheet("background-color: rgb(255, 0, 0); color: white; border: 1px solid #4682B4; border-radius: 5px; padding: 2px;")

        layout.addLayout(button_layout1)
        layout.addLayout(button_layout2)
        self.setLayout(layout)

    def load_dicom_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder_path:
            return
        
        try:
            # Collect all DICOM files from the folder
            files = [f for f in os.listdir(folder_path) if f.endswith('.dcm') and '_seg' not in f]
            files.sort(key=lambda f: int(pydicom.dcmread(os.path.join(folder_path, f)).InstanceNumber))
            dicom_files = [os.path.join(folder_path, f) for f in files]
            print(folder_path)
            print(dicom_files)
            if not dicom_files:
                QMessageBox.warning(self, "Warning", "No DICOM files found in the selected folder.")
                return

            # Read DICOM files and extract pixel arrays
            dicom_images = []
            for file in sorted(dicom_files):  # Sort for consistent order
                ds = pydicom.dcmread(file)
                img = ds.pixel_array
                dicom_images.append(img.astype(np.uint8))

            # Initialize attributes for standalone mode
            self.dicom_images = dicom_images
            self.dicom_files = dicom_files
            self.slice_index = 0
            self.dicom_path = dicom_files[0]
            self.original_slice_img = dicom_images[0].copy()
            self.current_slice_img = self.original_slice_img.copy()
            self.mask_rgb = np.zeros_like(self.current_slice_img, dtype=np.uint8)

            # Update UI elements
            self.slice_slider.setRange(0, len(self.dicom_images) - 1)
            self.slice_slider.setValue(0)
            self.scene.setSceneRect(0, 0, self.current_slice_img.shape[1], self.current_slice_img.shape[0])
            self.pixmap = np2pixmap(self.current_slice_img)
            self.scene.removeItem(self.bg_item)
            self.bg_item = self.scene.addPixmap(self.pixmap)
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.view.current_slice_img = self.current_slice_img
            self.view.mask_rgb = self.mask_rgb

            # Update slice label
            slice_label = self.slice_slider.parent().findChild(QLabel)
            slice_label.setText(f"Slice: {self.slice_index + 1}/{len(self.dicom_images)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load DICOM folder: {str(e)}")

    def update_slice(self, value, slice_label):
        self.slice_index = value
        try:
            # Update slice data, ensuring 3D RGB
            if self.parent_viewer is not None:
                new_slice = self.parent_viewer.dicom_images[self.slice_index]
                self.dicom_path = self.parent_viewer.dicom_files[self.slice_index]
            else:
                new_slice = self.dicom_images[self.slice_index]
                self.dicom_path = self.dicom_files[self.slice_index]
            self.original_slice_img = np.repeat(new_slice[:, :, None], 3, axis=-1) if new_slice.ndim == 2 else new_slice.copy()
            self.current_slice_img = self.original_slice_img.copy()
            self.mask_rgb = np.zeros_like(self.current_slice_img, dtype=np.uint8)
            self.contrast_factor = 1.0
            self.brightness_offset = 0
            self.contrast_slider.setValue(100)
            self.brightness_slider.setValue(0)

            # Update view and scene
            self.view.current_slice_img = self.current_slice_img
            self.view.mask_rgb = self.mask_rgb
            self.scene.setSceneRect(0, 0, self.current_slice_img.shape[1], self.current_slice_img.shape[0])
            self.pixmap = np2pixmap(self.current_slice_img)
            self.scene.removeItem(self.bg_item)
            self.bg_item = self.scene.addPixmap(self.pixmap)
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

            # Reset drawing tools
            self.view.set_eraser_mode(False)
            self.eraser_btn.setChecked(False)
            self.eraser_btn.setStyleSheet(self.button_style)

            # Sync parent viewer if it exists
            if self.parent_viewer is not None:
                self.parent_viewer.current_slice = self.slice_index
                self.parent_viewer.scrollbar.setValue(self.slice_index)
            slice_label.setText(f"Slice: {self.slice_index + 1}/{len(self.dicom_images)}")

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load slice {self.slice_index + 1}: {str(e)}")
        
    def load_auto_annotation(self):
        # Check if corresponding seg file exists
        base_name = os.path.splitext(self.dicom_path)[0]
        seg_path = f"{base_name}_seg.dcm"

        if os.path.exists(seg_path):
            # Load the segmentation DICOM file
            try:
                seg_dicom = pydicom.dcmread(seg_path)
                segment_labels = [segment.get('SegmentLabel', 'No Label') for segment in seg_dicom.SegmentSequence]
                seg_data = seg_dicom.pixel_array  # Assuming seg_data is a grayscale mask
                if seg_data.ndim > 2:  # Multi-frame case
                    combined_mask = np.zeros_like(seg_data, dtype=int)
                    for i in range(seg_data.shape[0]):
                        intensity_value = intensity_to_label(segment_labels[i])
                        seg_data[i] = (seg_data[i] > 0) * (0 + intensity_value)
                    combined_mask = np.sum(seg_data, axis=0)  # Combine frames
                else:
                    combined_mask = np.zeros_like(seg_data, dtype=int)
                    for segment_num, label in enumerate(segment_labels, start=1):
                        intensity_value = intensity_to_label(label)
                        seg_data = (seg_data > 0) * (0 + intensity_value)
                    combined_mask = seg_data  # Single frame case

                # Convert seg_data to RGB mask using the same color map as manual drawing
                seg_data = combined_mask
                self.mask_rgb = np.zeros_like(self.current_slice_img, dtype=np.uint8)
                intensity_to_rgb = {
                    1: (255, 0, 0),    # Red
                    2: (0, 255, 0),    # Green
                    3: (0, 0, 255),    # Blue
                    4: (255, 255, 0),  # Yellow
                    5: (0, 255, 255),  # Cyan
                    6: (255, 0, 255)   # Magenta
                }

                for intensity, rgb in intensity_to_rgb.items():
                    self.mask_rgb[seg_data == intensity] = rgb

                # Ensure the view is updated to use the new mask
                self.view.mask_rgb = self.mask_rgb  # Update the view's mask reference
                self.update_display()  # Refresh the display
                self.view.set_eraser_mode(False)  # Reset to drawing mode
                self.eraser_btn.setChecked(False)
                self.eraser_btn.setStyleSheet(self.button_style)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load segmentation: {str(e)}")
                return
        else:
            # No seg file found, show popup
            QMessageBox.information(self, "No Auto Annotation", "No auto-detected marking available for this slice.")

    def adjust_contrast(self, value, label):
        self.contrast_factor = value / 100.0  # Convert percentage to factor
        label.setText(f"Contrast: {value}%")
        self.update_display()

    def adjust_brightness(self, value, label):
        self.brightness_offset = value
        label.setText(f"Brightness: {value}")
        self.update_display()

    def toggle_eraser(self):
        is_checked = self.eraser_btn.isChecked()
        self.view.set_eraser_mode(is_checked)
        if is_checked:
            self.eraser_btn.setStyleSheet(self.button_style)
        else:
            self.eraser_btn.setStyleSheet(self.button_style)

    def set_color(self, rgb):
        self.current_color = list(rgb)
        self.view.set_eraser_mode(False)
        self.eraser_btn.setChecked(False)
        self.eraser_btn.setStyleSheet(self.button_style)
        for btn in self.color_buttons:
            btn.setStyleSheet("background-color: white; border: 1px solid #4682B4; border-radius: 5px; padding: 2px;")
        clicked_btn = self.sender()
        r, g, b = rgb
        text_color = "white" if (r + g + b) < 300 else "black"
        clicked_btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); color: {text_color}; border: 1px solid #4682B4; border-radius: 5px; padding: 2px;")

    def update_display(self):
        # Apply contrast and brightness to the original image
        adjusted_img = self.original_slice_img.astype(np.float32)
        adjusted_img = adjusted_img * self.contrast_factor + self.brightness_offset
        adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)
        self.current_slice_img = adjusted_img

        # Overlay the mask on the adjusted image
        overlay = np.repeat(self.current_slice_img[:, :, None], 3, axis=-1) if self.current_slice_img.ndim == 2 else self.current_slice_img.copy()
        mask_nonzero = np.any(self.mask_rgb > 0, axis=2)
        overlay[mask_nonzero] = self.mask_rgb[mask_nonzero]
        
        pixmap = np2pixmap(overlay)
        self.scene.removeItem(self.bg_item)
        self.bg_item = self.scene.addPixmap(pixmap)

    def refresh(self):
        self.mask_rgb.fill(0)
        self.contrast_factor = 1.0
        self.brightness_offset = 0
        self.contrast_slider.setValue(100)
        self.brightness_slider.setValue(0)
        self.current_slice_img = self.original_slice_img.copy()
        self.scene.removeItem(self.bg_item)
        self.pixmap = np2pixmap(self.current_slice_img)
        self.bg_item = self.scene.addPixmap(self.pixmap)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.view.set_eraser_mode(False)
        self.eraser_btn.setChecked(False)
        self.eraser_btn.setStyleSheet(self.button_style)

    def save_mask(self):
        intensity_map = {
            (255, 0, 0): 1,
            (0, 255, 0): 2,
            (0, 0, 255): 3,
            (255, 255, 0): 4,
            (0, 255, 255): 5,
            (255, 0, 255): 6
        }
        mask_gray = np.zeros((self.mask_rgb.shape[0], self.mask_rgb.shape[1]), dtype=np.uint8)
        for rgb, intensity in intensity_map.items():
            rgb_tuple = np.array(rgb, dtype=np.uint8)
            mask_gray[np.all(self.mask_rgb == rgb_tuple, axis=2)] = intensity

        gt_rgb_slice = mask_gray.astype(int)
        base_name = os.path.splitext(self.dicom_path)[0]
        output_path = f"{base_name}_seg.dcm"
        create_dicom_segmentation(gt_rgb_slice, self.dicom_path, output_path, self.slice_index)
        if self.parent_viewer is not None:
            self.parent_viewer.saved_masks.add(output_path)
            self.parent_viewer.message_label.setVisible(True)
        else:
            QMessageBox.information(self, "Success", f"Segmentation saved to {output_path}")
        #self.close()

    def closeEvent(self, event):
        if self.parent_viewer is not None:
            self.parent_viewer.display_slice()
        event.accept()

# Optional: test/debug when run directly
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ManualDrawWindow(parent_viewer=None)
    window.show()
    sys.exit(app.exec_())
