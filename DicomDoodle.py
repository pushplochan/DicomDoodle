#!/usr/bin/env python3
# Author: Pushp Lochan Kumar
import sys
import os
os.environ["ULTRALYTICS_CACHE"] = "0"
import numpy as np
import pydicom
import cv2
from PyQt5.QtCore import Qt, QTime, QRectF, QPointF, QThread, pyqtSignal
from PIL import Image
from ultralytics import SAM, YOLO, YOLOE
from PyQt5.QtGui import QIcon, QCursor,QBrush, QPen, QPixmap, QKeySequence, QColor, QImage, QPainter
from PyQt5.QtWidgets import (QStyle, QMessageBox, QDialog, QSlider, QScrollBar,QProgressBar, QMenuBar, QMenu, QTextEdit, QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsScene, QGraphicsView, QShortcut, QFileDialog, QSizePolicy, QLabel, QGraphicsRectItem, QGraphicsEllipseItem, QGridLayout, QProgressDialog, QSlider, QComboBox, QGroupBox, QCheckBox, QDoubleSpinBox, QAction, QToolBar, QLineEdit)
###################--local import--######################################## 
from create_manual import CustomDrawView, ManualDrawWindow
#from manual import CustomDrawView, ManualDrawWindow
from create_about import AboutViewer
from create_htu import HowToUseViewer
from create_cit import CitationsViewer
from create_oly import DICOMOverlayViewer
from create_3d import DicomViewer3D
from calculate_3dv import volume_calculator
from utils import create_dicom_segmentation, ResultsViewer, intensity_to_color, intensity_to_label, np2pixmap, dicom_to_png, get_data_path, get_yolo_box, get_yoloe_box
from convert_MultiFrameDicom2SingleDicom import Multi_frame_DicomViewer
#Model Path
yolo_path = get_data_path('best_flair.pt')
sam2_1_l_path = get_data_path('sam2.1_l.pt')
# Load SAM model
sam_model = SAM(sam2_1_l_path)

class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        
        # Attributes for box manipulation
        self.selected_box = None  # Currently selected box
        self.resize_handle = None  # Handle being resized (e.g., 'top-left', 'bottom-right')
        self.drag_start_pos = None  # Starting position for dragging
        self.box_start_pos = None  # Original box position for dragging
        self.is_dragging = False  # Flag for dragging the entire box
        self.handle_size = 8  # Size of resize handles

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.parent_widget.dicom_images:
            pos = self.mapToScene(event.pos())
            x, y = pos.x(), pos.y()
            
            # Check if clicking on an existing box or its handles
            item = self.itemAt(event.pos())
            if isinstance(item, QGraphicsRectItem):
                self.selected_box = item
                self.drag_start_pos = pos
                self.box_start_pos = QPointF(self.selected_box.rect().x(), self.selected_box.rect().y())
                self.is_dragging = True
                self.resize_handle = None
            elif isinstance(item, QGraphicsEllipseItem):  # Handle detection
                self.selected_box = item.data(0)  # Rectangle associated with handle
                self.resize_handle = item.data(1)  # Handle type (e.g., 'top-left')
                self.drag_start_pos = pos
                self.is_dragging = False
            else:
                # Start drawing a new box
                self.parent_widget.is_mouse_down = True
                self.parent_widget.start_pos = (x, y)
                self.parent_widget.start_point = self.scene().addEllipse(
                    x - 5, y - 5, 10, 10,
                    pen=QPen(QColor("red")),
                    brush=QBrush(QColor("red"))
                )
                self.selected_box = None
                self.resize_handle = None

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        x, y = pos.x(), pos.y()
        
        # Update cursor based on position
        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsRectItem) and not self.is_dragging and not self.resize_handle:
            rect = item.rect()
            if abs(rect.left() - x) < self.handle_size and abs(rect.top() - y) < self.handle_size:
                self.setCursor(QCursor(Qt.SizeFDiagCursor))  # Top-left
            elif abs(rect.right() - x) < self.handle_size and abs(rect.top() - y) < self.handle_size:
                self.setCursor(QCursor(Qt.SizeBDiagCursor))  # Top-right
            elif abs(rect.left() - x) < self.handle_size and abs(rect.bottom() - y) < self.handle_size:
                self.setCursor(QCursor(Qt.SizeBDiagCursor))  # Bottom-left
            elif abs(rect.right() - x) < self.handle_size and abs(rect.bottom() - y) < self.handle_size:
                self.setCursor(QCursor(Qt.SizeFDiagCursor))  # Bottom-right
            else:
                self.setCursor(QCursor(Qt.CrossCursor))  # Center (move)
        elif not self.parent_widget.is_mouse_down and not self.is_dragging and not self.resize_handle:
            self.setCursor(QCursor(Qt.ArrowCursor))

        # Handle resizing (only for corners)
        if self.resize_handle and self.selected_box:
            rect = self.selected_box.rect()
            dx = x - self.drag_start_pos.x()
            dy = y - self.drag_start_pos.y()
            new_rect = rect

            if 'left' in self.resize_handle and 'top' in self.resize_handle:
                new_rect.setLeft(rect.left() + dx)
                new_rect.setTop(rect.top() + dy)
            elif 'right' in self.resize_handle and 'top' in self.resize_handle:
                new_rect.setRight(rect.right() + dx)
                new_rect.setTop(rect.top() + dy)
            elif 'left' in self.resize_handle and 'bottom' in self.resize_handle:
                new_rect.setLeft(rect.left() + dx)
                new_rect.setBottom(rect.bottom() + dy)
            elif 'right' in self.resize_handle and 'bottom' in self.resize_handle:
                new_rect.setRight(rect.right() + dx)
                new_rect.setBottom(rect.bottom() + dy)

            self.selected_box.setRect(new_rect.normalized())
            self.update_handles()
            self.drag_start_pos = pos

        # Handle dragging the entire box
        elif self.is_dragging and self.selected_box:
            dx = x - self.drag_start_pos.x()
            dy = y - self.drag_start_pos.y()
            new_x = self.box_start_pos.x() + dx
            new_y = self.box_start_pos.y() + dy
            rect = self.selected_box.rect()
            rect.moveTo(new_x, new_y)
            self.selected_box.setRect(rect)
            self.update_handles()

        # Handle drawing a new box
        elif self.parent_widget.is_mouse_down and self.parent_widget.dicom_images:
            if self.parent_widget.end_point:
                self.scene().removeItem(self.parent_widget.end_point)
            self.parent_widget.end_point = self.scene().addEllipse(
                x - 5, y - 5, 10, 10,
                pen=QPen(QColor("red")),
                brush=QBrush(QColor("red"))
            )

            if self.parent_widget.rect:
                self.scene().removeItem(self.parent_widget.rect)
            sx, sy = self.parent_widget.start_pos
            xmin, xmax = sorted([x, sx])
            ymin, ymax = sorted([y, sy])
            self.parent_widget.rect = self.scene().addRect(
                xmin, ymin, xmax - xmin, ymax - ymin,
                pen=QPen(QColor("red"))
            )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.parent_widget.is_mouse_down and self.parent_widget.dicom_images:
            pos = self.mapToScene(event.pos())
            x, y = pos.x(), pos.y()
            sx, sy = self.parent_widget.start_pos
            xmin, xmax = sorted([x, sx])
            ymin, ymax = sorted([y, sy])

            self.parent_widget.is_mouse_down = False
            self.selected_box = self.parent_widget.rect
            self.parent_widget.rect = None
            self.add_resize_handles()

            # Clean up start and end points
            if self.parent_widget.start_point:
                self.scene().removeItem(self.parent_widget.start_point)
                self.parent_widget.start_point = None
            if self.parent_widget.end_point:
                self.scene().removeItem(self.parent_widget.end_point)
                self.parent_widget.end_point = None

        elif event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.resize_handle = None
            self.drag_start_pos = None
            self.box_start_pos = None

    def contextMenuEvent(self, event):
        if self.selected_box:
            menu = QMenu(self)
            detect_action = menu.addAction("Detect")
            action = menu.exec_(self.mapToGlobal(event.pos()))
            if action == detect_action:
                # Ensure the progress label is visible and updated
                self.parent_widget.progress_label.setText("Generating annotation")
                self.parent_widget.progress_label.setVisible(True)
                QApplication.processEvents()  # Force UI update
                rect = self.selected_box.rect()
                x1, y1 = rect.x(), rect.y()
                x2, y2 = rect.right(), rect.bottom()
                self.parent_widget.run_sam_model(x1, y1, x2, y2)
                self.scene().removeItem(self.selected_box)
                for item in self.scene().items():
                    if isinstance(item, QGraphicsEllipseItem) and item.data(0) == self.selected_box:
                        self.scene().removeItem(item)
                self.selected_box = None
                self.parent_widget.progress_label.setVisible(False)
                QApplication.processEvents()  # Ensure the label disappears

    def add_resize_handles(self):
        if not self.selected_box:
            return
        rect = self.selected_box.rect()
        # Only add handles at the vertices (corners)
        handles = [
            ('top-left', rect.left(), rect.top()),
            ('top-right', rect.right(), rect.top()),
            ('bottom-left', rect.left(), rect.bottom()),
            ('bottom-right', rect.right(), rect.bottom())
        ]
        for handle_type, hx, hy in handles:
            handle = self.scene().addEllipse(
                hx - self.handle_size / 2, hy - self.handle_size / 2,
                self.handle_size, self.handle_size,
                pen=QPen(QColor("blue")),
                brush=QBrush(QColor("blue"))
            )
            handle.setData(0, self.selected_box)  # Associate handle with box
            handle.setData(1, handle_type)  # Store handle type

    def update_handles(self):
        if not self.selected_box:
            return
        # Remove existing handles
        for item in self.scene().items():
            if isinstance(item, QGraphicsEllipseItem) and item.data(0) == self.selected_box:
                self.scene().removeItem(item)
        # Add new handles
        self.add_resize_handles()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent_widget.bg_img:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

class DICOMSegmentationViewer(QMainWindow):  # Changed from QWidget to QMainWindow
    def __init__(self):
        super().__init__()
        self.dicom_images = []
        self.dicom_files = []
        self.original_dicoms = []
        self.current_slice = 0
        self.is_mouse_down = False
        self.rect = None
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.prev_mask = None
        self.color_idx = 0
        self.colors = [(255, 0, 0)]
        self.mask_c = None
        self.img_3c = None
        self.bg_img = None
        self.saved_masks = set()
        self.blank_range1_start = None
        self.blank_range1_end = None
        self.blank_range2_start = None
        self.blank_range2_end = None
        self.yolo_range_start = None
        self.yolo_range_end = None
        self.intensity_map = {
            (255, 0, 0): 1.0,
            (0, 255, 0): 2.0,
            (0, 0, 255): 3.0,
            (255, 255, 0): 4.0,
            (0, 255, 255): 5.0,
            (255, 0, 255): 6.0
        }
        # Progress-related attributes
        self.splash = None
        self.progress_bar = None
        self.total_slices = 0
        self.processed_slices = 0
        self.initUI()

    def initUI(self):
        # Set up the central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create a horizontal layout for menu bar and scrollbar
        top_layout = QHBoxLayout()
        
        # Add Menu Bar
        menu_bar = self.menuBar()  # Use QMainWindow's built-in menu bar
        menu_bar.setStyleSheet("""
            QMenuBar {
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
        menu = QMenu("☰", self)
        menu.addAction("About", self.show_about)
        menu.addAction("How to Use", self.show_how_to_use)
        menu.addAction("Citations", self.show_citations)
        menu.addAction("Annotated Dicom Viewer", self.show_overlay_viewer)
        menu.addAction("View 3D Model", self.open_3d_viewer)
        menu.addAction("Convert Multi Frame Dicom to Single Dicom", self.open_converter)
        menu.addAction("Lesion Volume Calculator", self.calculate_volume)
        menu_bar.addMenu(menu)
        
        # Add horizontal scrollbar
        self.scrollbar = QScrollBar(Qt.Horizontal, self)
        self.scrollbar.setStyleSheet("""
            QScrollBar:horizontal {
                border: 1px solid #2F4F4F;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                           stop:0 #D3D9DF, stop:0.5 #B0BEC5, stop:1 #A0A8B0);
                height: 15px;
                margin: 0px;
                border-radius: 3px;
            }
            QScrollBar::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                           stop:0 #FF7F50, stop:0.5 #FF4500, stop:1 #CD5C5C);
                border-top: 1px solid #FFFFFF;
                border-left: 1px solid #FFFFFF;
                border-right: 1px solid #8B0000;
                border-bottom: 1px solid #8B0000;
                border-radius: 5px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                           stop:0 #FF8C00, stop:0.5 #FF6347, stop:1 #DC143C);
                border-top: 1px solid #F0F8FF;
                border-left: 1px solid #F0F8FF;
                border-right: 1px solid #A52A2A;
                border-bottom: 1px solid #A52A2A;
            }
            QScrollBar::handle:horizontal:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                           stop:0 #CD5C5C, stop:0.5 #DC143C, stop:1 #FF4500);
                border-top: 1px solid #8B0000;
                border-left: 1px solid #8B0000;
                border-right: 1px solid #FFFFFF;
                border-bottom: 1px solid #FFFFFF;
                padding-top: 1px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)
        self.scrollbar.valueChanged.connect(self.scrollbar_changed)
        self.update_scrollbar_range()
        
        # Add menu bar and scrollbar to the top layout
        top_layout.addWidget(menu_bar)
        top_layout.addWidget(self.scrollbar, stretch=1)  # Stretch scrollbar to take available space
        
        # Add the top layout to the main layout
        main_layout.addLayout(top_layout)
        
        # Rest of the initUI code remains unchanged
        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self)
        self.view.setScene(self.scene)
        
        # Set colorful background for the central widget
        central_widget.setStyleSheet("""
        QWidget {
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
        
        # Style the graphics view
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
        
        main_layout.addWidget(self.view)
        
        # Existing message label for "Mask already saved" with enhanced color
        self.message_label = QLabel("Mask already saved for this slice", self.view)
        self.message_label.setStyleSheet("background-color: rgba(255, 215, 0, 180); color: black; padding: 5px; border-radius: 3px;")
        self.message_label.setVisible(False)
        self.message_label.move(10, 10)
        
        # New message label for "Annotation in progress" and "Finished" with enhanced color
        self.progress_label = QLabel("", self.view)
        self.progress_label.setStyleSheet("background-color: rgba(50, 205, 50, 180); color: black; padding: 5px; border-radius: 3px;")
        self.progress_label.setVisible(False)
        self.progress_label.move(10, 10)
        
        # First row of buttons with colorful styles
        button_layout1 = QHBoxLayout()
        self.load_btn = QPushButton("Load DICOM Folder")
        self.yolo_btn = QPushButton("Auto Detect")
        self.plot_btn = QPushButton("Plot Results")
        self.save_btn = QPushButton("Save Results")
        self.refresh_btn = QPushButton("Refresh")
        
        # Style each button
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
        self.load_btn.setStyleSheet(self.button_style)
        self.load_btn.setIcon(app.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.yolo_btn.setStyleSheet(self.button_style)
        self.plot_btn.setStyleSheet(self.button_style)
        self.save_btn.setStyleSheet(self.button_style)
        self.save_btn.setIcon(app.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.refresh_btn.setStyleSheet(self.button_style)
        self.refresh_btn.setIcon(app.style().standardIcon(QStyle.SP_BrowserReload))
        
        button_layout1.addWidget(self.load_btn)
        button_layout1.addWidget(self.yolo_btn)
        button_layout1.addWidget(self.plot_btn)
        button_layout1.addWidget(self.save_btn)
        button_layout1.addWidget(self.refresh_btn)
        
        # Second row of buttons with colorful styles
        button_layout2 = QHBoxLayout()
        self.blank1_start_btn = QPushButton("Set Blank Mask Range 1 Start")
        self.blank1_end_btn = QPushButton("Set Blank Mask Range 1 End")
        self.blank2_start_btn = QPushButton("Set Blank Mask Range 2 Start")
        self.blank2_end_btn = QPushButton("Set Blank Mask Range 2 End")
        self.save_blank_btn = QPushButton("Save Blank Masks")
        self.yolo_start_btn = QPushButton("Set Auto Detect Range Start")
        self.yolo_end_btn = QPushButton("Set Auto Detect Range End")
        self.save_yolo_btn = QPushButton("Save Auto Detected Masks")
        self.manual_draw_btn = QPushButton("Manual Draw")
        
        # Style each button in the second row
        self.blank1_start_btn.setStyleSheet(self.button_style)
        self.blank1_end_btn.setStyleSheet(self.button_style)
        self.blank2_start_btn.setStyleSheet(self.button_style)
        self.blank2_end_btn.setStyleSheet(self.button_style)
        self.save_blank_btn.setStyleSheet(self.button_style)
        self.save_blank_btn.setIcon(app.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.yolo_start_btn.setStyleSheet(self.button_style)
        self.yolo_end_btn.setStyleSheet(self.button_style)
        self.save_yolo_btn.setStyleSheet(self.button_style)
        self.save_yolo_btn.setIcon(app.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.manual_draw_btn.setStyleSheet(self.button_style)
        
        button_layout2.addWidget(self.blank1_start_btn)
        button_layout2.addWidget(self.blank1_end_btn)
        button_layout2.addWidget(self.blank2_start_btn)
        button_layout2.addWidget(self.blank2_end_btn)
        button_layout2.addWidget(self.save_blank_btn)
        button_layout2.addWidget(self.yolo_start_btn)
        button_layout2.addWidget(self.yolo_end_btn)
        button_layout2.addWidget(self.save_yolo_btn)
        button_layout2.addWidget(self.manual_draw_btn)
        # Third Button Layout (Color Buttons)
        button_layout3 = QHBoxLayout()
        self.color_buttons = []
        self.color_map = {
            "🎨 Red (255, 0, 0):Tumor | Brain": (255, 0, 0),
            "🎨 Green (0, 255, 0):Tumor | Dura": (0, 255, 0),
            "🎨 Blue (0, 0, 255):Tumor | Necrosis": (0, 0, 255),
            "🎨 Yellow (255, 255, 0):Edema | Brain": (255, 255, 0),
            "🎨 Cyan (0, 255, 255):Tumor | Edema": (0, 255, 255),
            "🎨 Magenta (255, 0, 255):": (255, 0, 255)
        }
        for name, rgb in self.color_map.items():
            btn = QPushButton(name)
            btn.setStyleSheet(self.button_style)
            btn.clicked.connect(lambda checked, c=rgb: self.set_color(c))
            self.color_buttons.append(btn)
            button_layout3.addWidget(btn)
        # Set the first button (red) as selected by default
        self.color_buttons[0].setStyleSheet("background-color: rgb(255, 0, 0); color: white; border: 1px solid #4682B4; border-radius: 5px; padding: 2px;")
        
        main_layout.addLayout(button_layout1)
        main_layout.addLayout(button_layout2)
        main_layout.addLayout(button_layout3)
        
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo)
        self.load_btn.clicked.connect(self.load_series)
        self.yolo_btn.clicked.connect(self.run_yolo_segmentation)
        self.plot_btn.clicked.connect(self.plot_results)
        self.save_btn.clicked.connect(self.save_segmentation)
        self.refresh_btn.clicked.connect(self.refresh_slice)
        self.blank1_start_btn.clicked.connect(self.set_blank_range1_start)
        self.blank1_end_btn.clicked.connect(self.set_blank_range1_end)
        self.blank2_start_btn.clicked.connect(self.set_blank_range2_start)
        self.blank2_end_btn.clicked.connect(self.set_blank_range2_end)
        self.save_blank_btn.clicked.connect(self.save_blank_masks)
        self.yolo_start_btn.clicked.connect(self.set_yolo_range_start)
        self.yolo_end_btn.clicked.connect(self.set_yolo_range_end)
        self.save_yolo_btn.clicked.connect(self.save_yolo_masks)
        self.manual_draw_btn.clicked.connect(self.open_manual_draw_window)

    def open_3d_viewer(self):
        self.viewer_window = DicomViewer3D()
        self.viewer_window.show()

    def open_converter(self):
        self.viewer_window = Multi_frame_DicomViewer()
        self.viewer_window.show()

    def calculate_volume(self):
        self.viewer_window = volume_calculator()
        self.viewer_window.show()

    def update_scrollbar_range(self):
        if self.dicom_images:
            self.scrollbar.setRange(0, len(self.dicom_images) - 1)
            self.scrollbar.setValue(self.current_slice)
        else:
            self.scrollbar.setRange(0, 0)

    def scrollbar_changed(self, value):
        if not self.dicom_images:
            return
        self.current_slice = value
        self.display_slice()

    def set_color(self, rgb):
        """Set the drawing color and update button appearances."""
        self.colors = list(rgb)
        for btn in self.color_buttons:
            btn.setStyleSheet(self.button_style)
        clicked_btn = self.sender()
        r, g, b = rgb
        # Ensure text is readable (white if dark color, black if light)
        text_color = "white" if (r + g + b) < 300 else "black"
        clicked_btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); color: {text_color}; border: 1px solid #4682B4; border-radius: 5px; padding: 2px;")
        print(f"Drawing color set to: {self.colors}")

    def create_splash_screen(self):
        # Create a custom QWidget as a splash screen with progress bar
        self.splash = QWidget(self)
        self.splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.splash.setStyleSheet("background-color: #F0F8FF; border: 2px solid #4682B4; border-radius: 5px;")
        
        layout = QVBoxLayout()
        label = QLabel(f"Processing 0/{self.total_slices} slices")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; color: #4682B4;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.total_slices)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #4682B4;
                border-radius: 5px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #FF6347;
            }
        """)
        
        layout.addWidget(label)
        layout.addWidget(self.progress_bar)
        
        self.splash.setLayout(layout)
        self.splash.resize(300, 100)
        self.splash.move(self.geometry().center() - self.splash.rect().center())  # Center on main window
        self.splash.show()

    def open_manual_draw_window(self):
        if not self.dicom_images:
            QMessageBox.warning(self, "Warning", "Please load a DICOM series first!")
            return
        # Open the manual drawing window with the current slice image and DICOM path
        self.manual_draw_window = ManualDrawWindow(self, self.img_3c, self.dicom_files[self.current_slice], self.current_slice)
        self.manual_draw_window.show()
        # Update the main viewer after saving to reflect the new mask
        #self.manual_draw_window.destroyed.connect(self.display_slice)

    def update_progress(self, processed, total):
        if self.splash and self.progress_bar:
            self.splash.findChild(QLabel).setText(f"Processing {processed}/{total} slices")
            self.progress_bar.setValue(processed)
            QApplication.processEvents()

    def show_finished_message(self, elapsed_time):
        if self.splash:
            self.splash.close()
        
        # Create finished dialog
        finished_dialog = QWidget()
        finished_dialog.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        finished_dialog.setStyleSheet("background-color: #F0F8FF; border: 2px solid #4682B4; border-radius: 5px;")
        
        layout = QVBoxLayout()
        label = QLabel(f"Finished - Total time: {elapsed_time}s")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; color: #4682B4;")
        ok_btn = QPushButton("OK")
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6347;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover { background-color: #FF4500; }
            QPushButton:pressed { background-color: #CD5C5C; }
        """)
        ok_btn.clicked.connect(finished_dialog.close)
        ok_btn.clicked.connect(self.display_slice)  # Refresh the main screen
        
        layout.addWidget(label)
        layout.addWidget(ok_btn)
        
        finished_dialog.setLayout(layout)
        finished_dialog.resize(200, 100)
        finished_dialog.move(self.geometry().center() - finished_dialog.rect().center())
        finished_dialog.show()
        self.splash = finished_dialog
        
    def show_about(self):
        self.about_viewer = AboutViewer()
        self.about_viewer.show()

    def show_how_to_use(self):
        self.how_to_use_viewer = HowToUseViewer()
        self.how_to_use_viewer.show()

    def show_citations(self):
        self.citations_viewer = CitationsViewer()
        self.citations_viewer.show()

    def show_overlay_viewer(self):
        self.overlay_viewer = DICOMOverlayViewer()
        self.overlay_viewer.show()
        
    def load_series(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.dicom_images, self.dicom_files, self.original_dicoms = self.read_dicom_series(folder)
            self.current_slice = 0
            self.saved_masks.clear()
            self.blank_range1_start = None
            self.blank_range1_end = None
            self.blank_range2_start = None
            self.blank_range2_end = None
            self.yolo_range_start = None
            self.yolo_range_end = None
            self.update_scrollbar_range()
            self.display_slice()

    def read_dicom_series(self, folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.dcm') and "seg" not in f] 
        files.sort(key=lambda f: int(pydicom.dcmread(f).InstanceNumber))
        images = [self.normalize_image(pydicom.dcmread(d).pixel_array) for d in files]
        file_paths = [os.path.join(folder, os.path.basename(f)) for f in files]
        return images, file_paths, files

    def normalize_image(self, image_array):
        img_np = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
        return img_np.astype(np.uint8)

    def display_slice(self):
        if not self.dicom_images:
            return
        self.scene.clear()
        self.start_point = None
        self.end_point = None
        self.rect = None
        img_np = self.dicom_images[self.current_slice]
        self.img_3c = np.repeat(img_np[:, :, None], 3, axis=-1) if img_np.ndim == 2 else img_np
        self.mask_c = np.zeros_like(self.img_3c)
        self.bg_img = self.scene.addPixmap(np2pixmap(self.img_3c))
        self.scene.setSceneRect(0, 0, self.img_3c.shape[1], self.img_3c.shape[0])
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.setWindowTitle(f"DICOM Viewer - Slice {self.current_slice+1}/{len(self.dicom_images)}")
        
        # Update "Mask already saved" message
        seg_file = f"{os.path.splitext(self.dicom_files[self.current_slice])[0]}_seg.dcm"
        self.message_label.setVisible(seg_file in self.saved_masks)

    def wheelEvent(self, event):
        if not self.dicom_images:
            return
        angle = event.angleDelta().y()
        if angle > 0 and self.current_slice > 0:
            self.current_slice -= 1
            self.display_slice()
        elif angle < 0 and self.current_slice < len(self.dicom_images) - 1:
            self.current_slice += 1
            self.display_slice()
        self.scrollbar.setValue(self.current_slice)  # Sync scrollbar with wheel
        self.display_slice()

    def run_sam_model(self, x1, y1, x2, y2):
        if sam_model is None:
            print("SAM model not loaded. Cannot perform segmentation.")
            return
        self.results = sam_model.predict(self.img_3c, bboxes=[x1, y1, x2, y2])
        mask = self.results[0].masks.data[0].cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255
        self.prev_mask = self.mask_c.copy()
        self.mask_c[mask != 0] = self.colors#[self.color_idx % len(self.colors)]
        #self.color_idx += 1

        edges = cv2.Canny(mask, 100, 200)
        edge_overlay = self.img_3c.copy()
        edge_color = self.colors
        edge_overlay[edges != 0] = edge_color

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(edge_overlay))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.dicom_images[self.current_slice] = self.normalize_image(edge_overlay[:, :, 0])
        self.img_3c = edge_overlay
        #self.mask_c = np.zeros_like(self.img_3c)

    def run_yolo_segmentation(self):
        if not self.dicom_images:
            return
        self.progress_label.setText("Generating annotation")
        self.progress_label.setVisible(True)
        QApplication.processEvents()
        
        dicom_file_path = self.dicom_files[self.current_slice]
        yolo_box = get_yolo_box(dicom_file_path)
        if not yolo_box:
            yolo_box = get_yoloe_box(dicom_file_path)

        if yolo_box is None:
            print("No bounding box detected by YOLO.")
            self.progress_label.setVisible(False)  # Hide the generating message
            QMessageBox.information(self, "No Detection", "No detection found for this slice.")
            return
        
        y1, x1, y2, x2 = yolo_box
        
        self.scene.addRect(x1, y1, x2 - x1, y2 - y1, pen=QPen(QColor("green")))
        self.run_sam_model(x1, y1, x2, y2)
        self.progress_label.setVisible(False)

    def undo(self):
        if self.prev_mask is None or not self.dicom_images:
            return
        self.color_idx -= 1
        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.prev_mask.astype("uint8"), "RGB")
        blended = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(blended)))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.mask_c = self.prev_mask
        self.dicom_images[self.current_slice] = self.normalize_image(np.array(blended)[:, :, 0])
        self.prev_mask = None

    def plot_results(self):
        if not self.dicom_images or self.mask_c is None:
            return
            
        #mask = self.results[0].masks.data[0].cpu().numpy()
        #mask = (mask > 0).astype(np.uint8) * 255
        #self.prev_mask = self.mask_c.copy()
        #self.mask_c[mask != 0] = colors[self.color_idx % len(colors)]
        
        # Create blended image
        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask_img = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
        blended = Image.blend(bg, mask_img, 0.2)
        blended_edge = Image.blend(bg, mask_img, 0)
        
        # Create edge overlay
        mask_gray = cv2.cvtColor(self.mask_c, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(mask_gray, 100, 200)
        edge_overlay = self.img_3c.copy()
        edge_color = self.colors
        edge_overlay[edges != 0] = edge_color
        
        # Show results in new viewer
        self.results_viewer = ResultsViewer(blended, blended_edge)
        self.results_viewer.show()

    def save_segmentation(self):
        if not self.dicom_images or self.mask_c is None or not hasattr(self, 'results'):
            return
        
        dicom_path = self.dicom_files[self.current_slice]
        base_name = os.path.splitext(dicom_path)[0]
        output_path = f"{base_name}_seg.dcm"
        slice_index = self.current_slice
        #mask = self.results[0].masks.data[0].cpu().numpy()
        #mask = (mask > 0).astype(np.uint8) * 255
        #self.prev_mask = self.mask_c.copy()
        #self.mask_c[mask != 0] = colors[self.color_idx % len(colors)]

        gt_rgb_slice = np.zeros((self.mask_c.shape[0], self.mask_c.shape[1]), dtype=np.uint8)
        #mask_gray = cv2.cvtColor(self.mask_c, cv2.COLOR_RGB2GRAY)
        #edges = cv2.Canny(mask_gray, 100, 200)
        for rgb, intensity in self.intensity_map.items():
            rgb_tuple = np.array(rgb, dtype=np.uint8)
            color_mask = np.all(self.mask_c == rgb_tuple, axis=2)
            color_mask_uint8 = np.uint8(color_mask) * 255
            edges = cv2.Canny(color_mask_uint8, 100, 200)
            gt_rgb_slice[edges != 0] = intensity

        create_dicom_segmentation(gt_rgb_slice, dicom_path, output_path, slice_index)
        self.saved_masks.add(output_path)
        self.message_label.setVisible(True)

    def refresh_slice(self):
        if not self.dicom_images:
            return
        
        original_img = self.normalize_image(pydicom.dcmread(self.original_dicoms[self.current_slice]).pixel_array)
        self.dicom_images[self.current_slice] = original_img
        self.mask_c = np.zeros_like(self.img_3c)
        self.prev_mask = None
        self.color_idx = 0
        self.display_slice()

    def set_blank_range1_start(self):
        if not self.dicom_images:
            return
        self.blank_range1_start = self.current_slice
        print(f"Blank Range 1 Start set to slice {self.blank_range1_start}")

    def set_blank_range1_end(self):
        if not self.dicom_images:
            return
        self.blank_range1_end = self.current_slice
        print(f"Blank Range 1 End set to slice {self.blank_range1_end}")

    def set_blank_range2_start(self):
        if not self.dicom_images:
            return
        self.blank_range2_start = self.current_slice
        print(f"Blank Range 2 Start set to slice {self.blank_range2_start}")

    def set_blank_range2_end(self):
        if not self.dicom_images:
            return
        self.blank_range2_end = self.current_slice
        print(f"Blank Range 2 End set to slice {self.blank_range2_end}")

    def set_yolo_range_start(self):
        if not self.dicom_images:
            return
        self.yolo_range_start = self.current_slice
        print(f"YOLO Range Start set to slice {self.yolo_range_start}")

    def set_yolo_range_end(self):
        if not self.dicom_images:
            return
        self.yolo_range_end = self.current_slice
        print(f"YOLO Range End set to slice {self.yolo_range_end}")

    def save_blank_masks(self):
        if not self.dicom_images:
            return
        
        # Calculate total slices to process
        ranges = []
        if self.blank_range1_start is not None and self.blank_range1_end is not None:
            start1 = min(self.blank_range1_start, self.blank_range1_end)
            end1 = max(self.blank_range1_start, self.blank_range1_end)
            ranges.extend(range(start1, end1 + 1))
        if self.blank_range2_start is not None and self.blank_range2_end is not None:
            start2 = min(self.blank_range2_start, self.blank_range2_end)
            end2 = max(self.blank_range2_start, self.blank_range2_end)
            ranges.extend(range(start2, end2 + 1))
        self.total_slices = len(set(ranges))  # Unique slices to process
        self.processed_slices = 0

        # Start timing and show splash screen
        start_time = QTime.currentTime()
        self.create_splash_screen()
        
        # Save blank masks for Range 1
        if self.blank_range1_start is not None and self.blank_range1_end is not None:
            start = min(self.blank_range1_start, self.blank_range1_end)
            end = max(self.blank_range1_start, self.blank_range1_end)
            for i in range(start, end + 1):
                if i < len(self.dicom_files):
                    dicom_path = self.dicom_files[i]
                    base_name = os.path.splitext(dicom_path)[0]
                    output_path = f"{base_name}_seg.dcm"
                    if output_path not in self.saved_masks:
                        gt_rgb_slice = np.zeros_like(self.dicom_images[i], dtype=np.uint8)
                        create_dicom_segmentation(gt_rgb_slice, dicom_path, output_path, i)
                        self.saved_masks.add(output_path)
                    self.processed_slices += 1
                    self.update_progress(self.processed_slices, self.total_slices)
            print(f"Saved blank masks for Range 1: slices {start} to {end}")

        # Save blank masks for Range 2
        if self.blank_range2_start is not None and self.blank_range2_end is not None:
            start = min(self.blank_range2_start, self.blank_range2_end)
            end = max(self.blank_range2_start, self.blank_range2_end)
            for i in range(start, end + 1):
                if i < len(self.dicom_files):
                    dicom_path = self.dicom_files[i]
                    base_name = os.path.splitext(dicom_path)[0]
                    output_path = f"{base_name}_seg.dcm"
                    if output_path not in self.saved_masks:
                        gt_rgb_slice = np.zeros_like(self.dicom_images[i], dtype=np.uint8)
                        create_dicom_segmentation(gt_rgb_slice, dicom_path, output_path, i)
                        self.saved_masks.add(output_path)
                    self.processed_slices += 1
                    self.update_progress(self.processed_slices, self.total_slices)
            print(f"Saved blank masks for Range 2: slices {start} to {end}")

        # Show finished message
        elapsed = start_time.elapsed() // 1000
        self.show_finished_message(elapsed)
        self.total_slices = 0
        self.processed_slices = 0

    def save_yolo_masks(self):
        if not self.dicom_images or self.yolo_range_start is None or self.yolo_range_end is None:
            return
        
        # Calculate total slices to process
        start = min(self.yolo_range_start, self.yolo_range_end)
        end = max(self.yolo_range_start, self.yolo_range_end)
        self.total_slices = end - start + 1
        self.processed_slices = 0

        # Start timing and show splash screen
        start_time = QTime.currentTime()
        self.create_splash_screen()
        
        for i in range(start, end + 1):
            if i < len(self.dicom_files):
                dicom_path = self.dicom_files[i]
                base_name = os.path.splitext(dicom_path)[0]
                output_path = f"{base_name}_seg.dcm"
                if output_path not in self.saved_masks:
                    yolo_box = get_yolo_box(dicom_path)
                    if not yolo_box:
                       yolo_box = get_yoloe_box(dicom_file_path)
                    if yolo_box:
                        y1, x1, y2, x2 = yolo_box
                        img_3c = np.repeat(self.dicom_images[i][:, :, None], 3, axis=-1)
                        mask_c = np.zeros_like(img_3c)
                        if sam_model:
                            results = sam_model.predict(img_3c, bboxes=[x1, y1, x2, y2])
                            mask = results[0].masks.data[0].cpu().numpy()
                            mask = (mask > 0).astype(np.uint8) * 255
                            mask_c[mask != 0] = self.colors
                            edges = cv2.Canny(mask, 100, 200)
                            edge_overlay = img_3c.copy()
                            edge_overlay[edges != 0] = self.colors
                            self.dicom_images[i] = self.normalize_image(edge_overlay[:, :, 0])
                            #mask_gray = mask
                            gt_rgb_slice = np.zeros((mask_c.shape[0], mask_c.shape[1]), dtype=np.uint8)
                            #gt_rgb_slice = cv2.Canny(mask_gray, 100, 200)
                            #gt_rgb_slice = gt_rgb_slice.astype(int)
                            for rgb, intensity in self.intensity_map.items():
                              rgb_tuple = np.array(rgb, dtype=np.uint8)
                              color_mask = np.all(mask_c == rgb_tuple, axis=2)
                              color_mask_uint8 = np.uint8(color_mask) * 255
                              edges = cv2.Canny(color_mask_uint8, 100, 200)
                              gt_rgb_slice[edges != 0] = intensity
                            create_dicom_segmentation(gt_rgb_slice, dicom_path, output_path, i)
                            self.saved_masks.add(output_path)
                        else:
                            print(f"SAM model not loaded, saving blank mask for slice {i}")
                            gt_rgb_slice = np.zeros_like(self.dicom_images[i], dtype=np.uint8)
                            create_dicom_segmentation(gt_rgb_slice, dicom_path, output_path, i)
                            self.saved_masks.add(output_path)
                    else:
                        print(f"No YOLO box detected for slice {i}, saving blank mask.")
                        gt_rgb_slice = np.zeros_like(self.dicom_images[i], dtype=np.uint8)
                        create_dicom_segmentation(gt_rgb_slice, dicom_path, output_path, i)
                        self.saved_masks.add(output_path)
                self.processed_slices += 1
                self.update_progress(self.processed_slices, self.total_slices)
        
        print(f"Saved YOLO masks for slices {start} to {end}")
        # Show finished message
        elapsed = start_time.elapsed() // 1000
        self.show_finished_message(elapsed)
        self.total_slices = 0
        self.processed_slices = 0

if __name__ == '__main__':
    app = QApplication(sys.argv) 
    viewer = DICOMSegmentationViewer()
    viewer.setWindowTitle("DICOM Viewer and Annotator")
    viewer.setGeometry(100, 100, 800, 800)
    viewer.show()
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass
    sys.exit(app.exec_())
