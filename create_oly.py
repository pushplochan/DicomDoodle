import sys
import os
import numpy as np
import pydicom
import cv2
from PyQt5.QtCore import Qt, QTime, QRectF, QPointF, QThread, pyqtSignal
from PIL import Image
from PyQt5.QtGui import QIcon, QCursor,QBrush, QPen, QPixmap, QKeySequence, QColor, QImage, QPainter
from PyQt5.QtWidgets import (QStyle, QMessageBox, QDialog, QSlider, QScrollBar,QProgressBar, QMenuBar, QMenu, QTextEdit, QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsScene, QGraphicsView, QShortcut, QFileDialog, QSizePolicy, QLabel, QGraphicsRectItem, QGraphicsEllipseItem, QGridLayout, QProgressDialog, QSlider, QComboBox, QGroupBox, QCheckBox, QDoubleSpinBox, QAction, QToolBar, QLineEdit)
from utils import intensity_to_color, intensity_to_label, np2pixmap, dicom_to_png, get_yolo_box, get_data_path
from create_ontology import BioPortalSearchGUI
from pydicom.sequence import Sequence
from highdicom.sr.coding import CodedConcept
#from uploadpacs import UploadWindow
class DICOMOverlayViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DICOM Annotated Image Viewer")
        self.setGeometry(300, 300, 800, 600)
        self.dicom_images = []
        self.dicom_files = []
        self.seg_files = {}
        self.folder_path = "" 
        self.current_slice = 0
        self.zoom_factor = 1.0  # Default zoom level (100%)
        self.zoom_mode = False  # Zoom mode off by default
        self.selected_category = None
        self.selected_type = None
        self.show_mask = False
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        app = QApplication.instance()

        # Horizontal scrollbar for slice navigation
        self.scrollbar = QScrollBar(Qt.Horizontal, self)
        self.scrollbar.setStyleSheet("""
        QScrollBar:horizontal {
            border: 1px solid #2F4F4F; /* Darker border for depth */
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #D3D9DF, stop:0.5 #B0BEC5, stop:1 #A0A8B0); /* Gradient for groove effect */
            height: 15px;
            margin: 0px;
            border-radius: 3px;
        }
        QScrollBar::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #FF7F50, stop:0.5 #FF4500, stop:1 #CD5C5C); /* Gradient for raised handle */
            border-top: 1px solid #FFFFFF; /* Light top border for highlight */
            border-left: 1px solid #FFFFFF; /* Light left border for highlight */
            border-right: 1px solid #8B0000; /* Dark right border for shadow */
            border-bottom: 1px solid #8B0000; /* Dark bottom border for shadow */
            border-radius: 5px;
            min-width: 20px; /* Ensure handle is visible */
        }
        QScrollBar::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #FF8C00, stop:0.5 #FF6347, stop:1 #DC143C); /* Brighter gradient on hover */
            border-top: 1px solid #F0F8FF; /* Slightly brighter highlight */
            border-left: 1px solid #F0F8FF;
            border-right: 1px solid #A52A2A;
            border-bottom: 1px solid #A52A2A;
        }
        QScrollBar::handle:horizontal:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                       stop:0 #CD5C5C, stop:0.5 #DC143C, stop:1 #FF4500); /* Sunken gradient on press */
            border-top: 1px solid #8B0000; /* Dark top border for sunken effect */
            border-left: 1px solid #8B0000;
            border-right: 1px solid #FFFFFF; /* Light right border for contrast */
            border-bottom: 1px solid #FFFFFF;
            padding-top: 1px; /* Slight offset to enhance sunken effect */
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px; /* Hide arrows */
        }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: none; /* Transparent to show track gradient */
        }
        """)
        self.scrollbar.valueChanged.connect(self.scrollbar_changed)
        self.update_scrollbar_range()
        layout.addWidget(self.scrollbar)

        # Graphics View for displaying images
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        # Enable scrollbars for adjusting zoomed image
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
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

        # Button Layout
        button_layout = QHBoxLayout()

        # Load Button
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
        # Labels for Segment Category and Type
        self.category_label = QLabel("Segment Category: Not loaded", self.view)
        self.category_label.setStyleSheet("background-color: rgba(255, 215, 0, 180); color: black; padding: 5px; border-radius: 3px;")
        self.category_label.move(10, 10)
        self.category_label.setVisible(False)

        self.type_label = QLabel("Segment Type: Not loaded", self.view)
        self.type_label.setStyleSheet("background-color: rgba(255, 215, 0, 180); color: black; padding: 5px; border-radius: 3px;")
        self.type_label.move(10, 35)
        self.type_label.setVisible(False)
        
        self.load_btn = QPushButton("Load DICOM Folder")
        self.load_btn.setStyleSheet(self.button_style)
        self.load_btn.setIcon(app.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.load_btn.clicked.connect(self.load_series)
        button_layout.addWidget(self.load_btn)

        # Zoom Button
        self.zoom_btn = QPushButton("Zoom")
        self.zoom_btn.setStyleSheet(self.button_style)
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.clicked.connect(self.toggle_zoom_mode)
        button_layout.addWidget(self.zoom_btn)

        # Show Mask Button
        self.show_mask_btn = QPushButton("Show Mask")
        self.show_mask_btn.setStyleSheet(self.button_style)
        self.show_mask_btn.setCheckable(True)
        self.show_mask_btn.clicked.connect(self.toggle_mask_display)
        button_layout.addWidget(self.show_mask_btn)

        # Add Ontology Button
        self.add_ontology_btn = QPushButton("Add Ontology")
        self.add_ontology_btn.setStyleSheet(self.button_style)
        self.add_ontology_btn.clicked.connect(self.open_ontology_search)
        button_layout.addWidget(self.add_ontology_btn)

        # Add Ontology Button
        #self.add_ex_pacs_btn = QPushButton("Export To PACS")
        #self.add_ex_pacs_btn.setStyleSheet(self.button_style)
        #self.add_ex_pacs_btn.clicked.connect(self.export_data_to_pacs)
        #button_layout.addWidget(self.add_ex_pacs_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_series(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.dicom_images, self.dicom_files, self.seg_files = self.read_dicom_series(folder)
            self.current_slice = 0
            self.zoom_factor = 1.0  # Reset zoom on new series
            self.zoom_mode = False  # Reset zoom mode
            self.zoom_btn.setChecked(False)
            self.zoom_btn.setStyleSheet(self.button_style)
            self.show_mask = False
            self.show_mask_btn.setChecked(False)
            self.show_mask_btn.setStyleSheet(self.button_style)
            self.update_scrollbar_range()
            self.display_slice()

    def read_dicom_series(self, folder):
        # Load DICOM files excluding segmentation files
        self.folder_path = folder
        files = [f for f in os.listdir(folder) if f.endswith('.dcm') and '_seg' not in f]
        files.sort(key=lambda f: int(pydicom.dcmread(os.path.join(folder, f)).InstanceNumber))
        dicom_files = [os.path.join(folder, f) for f in files]
        images = [self.normalize_image(pydicom.dcmread(f).pixel_array) for f in dicom_files]
        
        # Load corresponding segmentation files
        seg_files = {}
        for dicom_file in dicom_files:
            base_name = os.path.splitext(dicom_file)[0]
            seg_file = f"{base_name}_seg.dcm"
            if os.path.exists(seg_file):
                seg_ds = pydicom.dcmread(seg_file)
                seg_mask = seg_ds.pixel_array
                segmented_property_category = [segment.get('SegmentedPropertyCategoryCodeSequence', [{}])[0].get('CodeMeaning', 'No Meaning') for segment in seg_ds.SegmentSequence]
                segmented_property_type = [segment.get('SegmentedPropertyTypeCodeSequence', [{}])[0].get('CodeMeaning', 'No Meaning') for segment in seg_ds.SegmentSequence]
                segment_labels = [segment.get('SegmentLabel', 'No Label') for segment in seg_ds.SegmentSequence]
                if seg_mask.ndim > 2:  # Multi-frame case
                    combined_mask = np.zeros_like(seg_mask, dtype=int)
                    for i in range(seg_mask.shape[0]):
                        intensity_value = intensity_to_label(segment_labels[i])
                        seg_mask[i] = (seg_mask[i] > 0) * (0 + intensity_value)
                    combined_mask = np.sum(seg_mask, axis=0)  # Combine frames
                else:
                    combined_mask = np.zeros_like(seg_mask, dtype=int)
                    for segment_num, label in enumerate(segment_labels, start=1):
                        intensity_value = intensity_to_label(label)
                        seg_mask = (seg_mask > 0) * (0 + intensity_value)
                    combined_mask = seg_mask  # Single frame case
                seg_files[dicom_file] = (combined_mask.astype(np.uint8),segmented_property_category,segmented_property_type)
            else:
                seg_files[dicom_file] = (None, ["None"], ["None"])
        
        return images, dicom_files, seg_files

    def normalize_image(self, image_array):
        img_np = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
        return img_np.astype(np.uint8)

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

    def toggle_zoom_mode(self):
        self.zoom_mode = self.zoom_btn.isChecked()
        if self.zoom_mode:
            self.zoom_btn.setStyleSheet("""
                QPushButton {
                    background-color: #32CD32;
                    color: white;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover { background-color: #228B22; }
                QPushButton:pressed { background-color: #2E8B57; }
            """)
        else:
            self.zoom_btn.setStyleSheet(self.button_style)
        self.display_slice()  # Refresh display to ensure zoom state is applied

    def toggle_mask_display(self):
        self.show_mask = self.show_mask_btn.isChecked()
        if self.show_mask:
            self.show_mask_btn.setStyleSheet("""
                QPushButton {
                    background-color: #32CD32;
                    color: white;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover { background-color: #228B22; }
                QPushButton:pressed { background-color: #2E8B57; }
            """)
        else:
            self.show_mask_btn.setStyleSheet(self.button_style)
        self.display_slice()
        
    def open_ontology_search(self):
        """Open BioPortal search window and connect signals"""
        if not self.folder_path:
            QMessageBox.warning(self, "Warning", "Please load a DICOM folder first.")
            return
        self.search_window = BioPortalSearchGUI()
        self.search_window.categorySelected.connect(self.set_category)
        self.search_window.typeSelected.connect(self.set_type)
        self.search_window.show()

    def export_data_to_pacs(self):
        """Open BioPortal search window and connect signals"""
        if not self.folder_path:
            QMessageBox.warning(self, "Warning", "Please load a DICOM folder first.")
            return
        self.export_window = UploadWindow()
        self.export_window.show()

    def set_category(self, concept):
        """Set selected segment category and process files"""
        self.selected_category = concept
        self.process_dicom_files()

    def set_type(self, concept):
        """Set selected segment type and process files"""
        self.selected_type = concept
        self.process_dicom_files()

    def process_dicom_files(self):
        """Process DICOM files with selected ontology concepts"""
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "No folder loaded.")
            return
        if not self.selected_category or not self.selected_type:
            QMessageBox.warning(self, "Error", "Please select both segment category and type.")
            return

        dicom_files = [f for f in os.listdir(self.folder_path) if f.endswith('_seg.dcm')]
        if not dicom_files:
            QMessageBox.warning(self, "Error", "No _seg.dcm files found in the selected folder.")
            return

        for dicom_file in dicom_files:
            file_path = os.path.join(self.folder_path, dicom_file)
            try:
                ds = pydicom.dcmread(file_path)
                pixel_array = ds.pixel_array
                has_non_zero = pixel_array.any()

                if has_non_zero:
                    category_concept = self.selected_category
                    type_concept = self.selected_type
                else:
                    category_concept = CodedConcept(value='91772007', scheme_designator='SCT', meaning='Organ')
                    type_concept = CodedConcept(value='12738006', scheme_designator='SCT', meaning='Brain')

                if hasattr(ds, 'SegmentSequence') and ds.SegmentSequence:
                    for segment in ds.SegmentSequence:
                        segment.SegmentedPropertyCategoryCodeSequence = Sequence([category_concept])
                        segment.SegmentedPropertyTypeCodeSequence = Sequence([type_concept])
                    ds.save_as(file_path)
                else:
                    QMessageBox.warning(self, "Warning", f"No SegmentSequence found in {dicom_file}")
                    continue

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error processing {dicom_file}: {str(e)}")
                return

        QMessageBox.information(self, "Success", f"Successfully processed {len(dicom_files)} DICOM files.")
        # Reload the series to update displayed labels
        self.dicom_images, self.dicom_files, self.seg_files = self.read_dicom_series(self.folder_path)
        self.display_slice()

    def display_slice(self):
        if not self.dicom_images:
            return
        
        self.scene.clear()
        # Load current grayscale DICOM image
        img_gray = self.dicom_images[self.current_slice]
        # Convert to RGB for overlay purposes
        img_3c = np.repeat(img_gray[:, :, None], 3, axis=-1)
        
        # Overlay segmentation mask if available
        seg_mask,segmented_property_category,segmented_property_type = self.seg_files[self.dicom_files[self.current_slice]]
        if seg_mask is not None:
            gt_slice_rgb = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
            for y in range(seg_mask.shape[0]):
                for x in range(seg_mask.shape[1]):
                    gt_value = seg_mask[y, x]
                    if gt_value != 0:
                        color = intensity_to_color(gt_value)
                        gt_slice_rgb[y, x] = color
            if seg_mask.shape != img_gray.shape:
                print(f"Warning: Mask shape {seg_mask.shape} does not match image shape {img_gray.shape} for slice {self.current_slice+1}")
                gt_slice_rgb = cv2.resize(gt_slice_rgb, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Blend the grayscale image with the colored mask
            if self.show_mask:
                blended = img_3c.copy()
                contours, hierarchy = cv2.findContours(seg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                colors = [
                    (255, 255, 0),   # Yellow
                    (0, 255, 255),   # Cyan
                    (255, 0, 255),   # Magenta
                    (0, 255, 0),     # Green
                    (255, 0, 0),     # Red
                    (0, 165, 255),   # Orange
                    (128, 0, 128),   # Purple
                    (255, 255, 255), # White
                    (0, 0, 255),     # Blue
                    (255, 192, 203), # Pink
                    (139, 69, 19),   # Brown
                    (128, 128, 128), # Gray
                    (0, 128, 128),   # Teal
                    (255, 215, 0),   # Gold
                    (75, 0, 130),    # Indigo
                    (173, 216, 230), # Light Blue
                    (255, 165, 0),   # Dark Orange
                    (127, 255, 212), # Aquamarine
                    (220, 20, 60),   # Crimson
                    (124, 252, 0)    # Lawn Green
                ]
                
                gt_slice_rgb = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Iterate through contours and assign colors
                for i, contour in enumerate(contours):
                    # Use modulo to cycle through colors if there are more contours than colors
                    color = colors[i % len(colors)]
                    
                    # For concentric contours, use hierarchy to detect if it's an inner contour
                    # hierarchy[0][i][3] = -1 means it's an outer contour, otherwise it's inner
                    thickness = cv2.FILLED
                    cv2.drawContours(gt_slice_rgb, [contour], -1, color, thickness=thickness)
                
                # Blend the colored contours with the original image
                non_black_mask = np.any(gt_slice_rgb != [0, 0, 0], axis=-1)
                blended[non_black_mask] = gt_slice_rgb[non_black_mask]
            else:
                blended = img_3c.copy()
                non_black_mask = np.any(gt_slice_rgb != [0, 0, 0], axis=-1)
                blended[non_black_mask] = gt_slice_rgb[non_black_mask]
            self.category_label.setText(f"Segment Category: {segmented_property_category[0]}")
            self.type_label.setText(f"Segment Type: {segmented_property_type[0]}")
            self.category_label.setVisible(True)
            self.category_label.adjustSize()
            self.type_label.setVisible(True)
            self.type_label.adjustSize()
        else:
            blended = img_3c  # Show grayscale image as RGB without overlay
            self.category_label.setText("Segment Category: None")
            self.type_label.setText("Segment Type: None")
            self.category_label.setVisible(False)
            self.type_label.setVisible(False)
        
        # Display the blended image
        pixmap = np2pixmap(blended)
        self.bg_img = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, blended.shape[1], blended.shape[0])
        
        # Apply zoom
        self.view.resetTransform()
        self.view.scale(self.zoom_factor, self.zoom_factor)

        self.setWindowTitle(f"DICOM Mask Overlay Viewer - Slice {self.current_slice+1}/{len(self.dicom_images)}")

    def wheelEvent(self, event):
        if not self.dicom_images:
            return
        angle = event.angleDelta().y()
        if self.zoom_mode:
            # Zoom in or out
            if angle > 0:
                self.zoom_factor *= 1.1  # Zoom in by 10%
            elif angle < 0:
                self.zoom_factor /= 1.1  # Zoom out by 10%
            # Limit zoom factor to a reasonable range (e.g., 0.5 to 5.0)
            self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))
            self.display_slice()
        else:
            # Change slice
            if angle > 0 and self.current_slice > 0:
                self.current_slice -= 1
            elif angle < 0 and self.current_slice < len(self.dicom_images) - 1:
                self.current_slice += 1
            self.scrollbar.setValue(self.current_slice)  # Sync scrollbar with wheel
            self.display_slice()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.dicom_images:
            self.display_slice()  # Reapply zoom on resize

# Optional: test/debug when run directly
if __name__ == "__main__":

    app = QApplication([])
    window = DICOMOverlayViewer()
    window.show()
    app.exec_()
