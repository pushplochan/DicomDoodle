import os
import pydicom
import numpy as np
import cv2
from PyQt5.QtWidgets import (QStyle, QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
class volume_calculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Tumor Volume Calculator")

        # Window size
        app = QApplication.instance()
        self.setGeometry(100, 100, 600, 300)
        self.setStyleSheet("""
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

        # Central widget and main vertical layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Add spacing
        main_layout.addStretch(1)

        # Title
        self.label = QLabel("Select DICOM Folders for Axial, Coronal, and Sagittal", central_widget)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(self.label)

        # Add spacing
        main_layout.addStretch(1)

        # Folder selection buttons
        folder_layout = QVBoxLayout()
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
        # Axial folder
        axial_layout = QHBoxLayout()
        self.axial_label = QLabel("Axial: Not selected", central_widget)
        self.axial_label.setStyleSheet("font-size: 12px;")
        self.select_axial_button = QPushButton("Browse Axial Folder", central_widget)
        self.select_axial_button.setStyleSheet(self.button_style)
        self.select_axial_button.setIcon(app.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.select_axial_button.clicked.connect(lambda: self.select_folder("axial"))
        self.select_axial_button.setMinimumWidth(150)
        axial_layout.addWidget(self.axial_label)
        axial_layout.addWidget(self.select_axial_button)
        folder_layout.addLayout(axial_layout)

        # Coronal folder
        coronal_layout = QHBoxLayout()
        self.coronal_label = QLabel("Coronal: Not selected", central_widget)
        self.coronal_label.setStyleSheet("font-size: 12px;")
        self.select_coronal_button = QPushButton("Browse Coronal Folder", central_widget)
        self.select_coronal_button.setStyleSheet(self.button_style)
        self.select_coronal_button.setIcon(app.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.select_coronal_button.clicked.connect(lambda: self.select_folder("coronal"))
        self.select_coronal_button.setMinimumWidth(150)
        coronal_layout.addWidget(self.coronal_label)
        coronal_layout.addWidget(self.select_coronal_button)
        folder_layout.addLayout(coronal_layout)

        # Sagittal folder
        sagittal_layout = QHBoxLayout()
        self.sagittal_label = QLabel("Sagittal: Not selected", central_widget)
        self.sagittal_label.setStyleSheet("font-size: 12px;")
        self.select_sagittal_button = QPushButton("Browse Sagittal Folder", central_widget)
        self.select_sagittal_button.setStyleSheet(self.button_style)
        self.select_sagittal_button.setIcon(app.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.select_sagittal_button.clicked.connect(lambda: self.select_folder("sagittal"))
        self.select_sagittal_button.setMinimumWidth(150)
        sagittal_layout.addWidget(self.sagittal_label)
        sagittal_layout.addWidget(self.select_sagittal_button)
        folder_layout.addLayout(sagittal_layout)

        main_layout.addLayout(folder_layout)

        # Add spacing
        main_layout.addStretch(1)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        self.process_button = QPushButton("Calculate Tumor Volumes", central_widget)
        self.process_button.setStyleSheet(self.button_style)
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_dicom)
        self.process_button.setMinimumWidth(150)
        button_layout.addWidget(self.process_button)

        self.quit_button = QPushButton("Quit", central_widget)
        self.quit_button.setStyleSheet(self.button_style)
        self.quit_button.clicked.connect(self.close)
        self.quit_button.setMinimumWidth(150)
        button_layout.addWidget(self.quit_button)

        button_layout.addSpacing(10)
        button_layout.addStretch(1)

        main_layout.addLayout(button_layout)
        main_layout.addStretch(1)

        self.folder_paths = {"axial": None, "coronal": None, "sagittal": None}

    def select_folder(self, orientation):
        folder_path = QFileDialog.getExistingDirectory(self, f"Select {orientation.capitalize()} DICOM Folder")
        if folder_path:
            self.folder_paths[orientation] = folder_path
            getattr(self, f"{orientation}_label").setText(f"{orientation.capitalize()}: {os.path.basename(folder_path)}")
            # Enable process button only if at least one folder is selected
            if any(self.folder_paths.values()):
                self.process_button.setEnabled(True)

    def read_dicom_series(self, folder_path):
        if not folder_path:
            return None, None, None, None

        # Get all DICOM files
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                     if f.endswith('.dcm')]
        dicom_files = [f for f in all_files if '_seg' not in f]
        seg_files = [f for f in all_files if '_seg' in f]

        if not dicom_files:
            return None, None, None, None

        # Read regular DICOM files with InstanceNumber
        slices = []
        slice_thickness = None
        pixel_spacing = None
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f)
                slices.append((ds.InstanceNumber, f))
                if slice_thickness is None and hasattr(ds, 'SliceThickness'):
                    slice_thickness = float(ds.SliceThickness)
                if pixel_spacing is None and hasattr(ds, 'PixelSpacing'):
                    pixel_spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to read {f}: {str(e)}")
                continue

        if not slices:
            return None, None, None, None

        if slice_thickness is None:
            slice_thickness = 1.0
            QMessageBox.warning(self, "Warning", f"SliceThickness not found in {os.path.basename(folder_path)}, using default 1.0 mm")
        if pixel_spacing is None:
            pixel_spacing = [1.0, 1.0]
            QMessageBox.warning(self, "Warning", f"PixelSpacing not found in {os.path.basename(folder_path)}, using default 1.0 mm x 1.0 mm")

        # Sort by InstanceNumber
        slices.sort(key=lambda x: x[0])
        min_instance = slices[0][0]

        # Get dimensions from first file
        ref_ds = pydicom.dcmread(slices[0][1])
        ConstPixelDims = (int(ref_ds.Rows), int(ref_ds.Columns), len(slices))

        # Create 3D arrays
        ArrayDicom = np.zeros(ConstPixelDims, dtype=ref_ds.pixel_array.dtype)
        tumor_mask = np.zeros_like(ArrayDicom, dtype=np.uint8)

        # Read segmentation files and match by InstanceNumber
        seg_dict = {}
        for f in seg_files:
            try:
                seg_ds = pydicom.dcmread(f)
                seg_dict[seg_ds.InstanceNumber] = seg_ds.pixel_array
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to read seg file {f}: {str(e)}")
                continue

        # Process each slice and overlay segmentation with filled contours
        max_intensity = np.iinfo(ArrayDicom.dtype).max
        for instance_num, filename in slices:
            slice_idx = instance_num - min_instance
            ds = pydicom.dcmread(filename)
            ArrayDicom[:, :, slice_idx] = ds.pixel_array

            if instance_num in seg_dict:
                seg_data = seg_dict[instance_num]
                if seg_data.shape[0] == ArrayDicom[:, :, 0].shape[0] and seg_data.shape[1] == ArrayDicom[:, :, 0].shape[1]:
                    if seg_data.ndim > 2:  # Multi-frame case
                        combined_mask = np.sum(seg_data, axis=0)
                    else:
                        combined_mask = seg_data

                    if np.any(combined_mask > 0):
                        binary_mask = (combined_mask > 0).astype(np.uint8)
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        filled_mask = np.zeros_like(binary_mask)
                        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
                        tumor_mask[:, :, slice_idx] = filled_mask
                        ArrayDicom[:, :, slice_idx][filled_mask > 0] = max_intensity

        return ArrayDicom, tumor_mask, slice_thickness, pixel_spacing

    def calculate_tumor_volume(self, tumor_mask, slice_thickness, pixel_spacing):
        # Calculate voxel volume in mm^3
        voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
        # Count non-zero voxels in tumor mask
        tumor_voxel_count = np.sum(tumor_mask > 0)
        # Total volume in mm^3
        tumor_volume_mm3 = tumor_voxel_count * voxel_volume
        # Convert to cm^3 (1 cm^3 = 1000 mm^3)
        tumor_volume_cm3 = tumor_volume_mm3 / 1000.0
        return tumor_volume_cm3

    def process_dicom(self):
        if not any(self.folder_paths.values()):
            QMessageBox.critical(self, "Error", "Please select at least one folder!")
            return

        try:
            self.label.setText("Processing... Please wait")
            self.update()

            volumes = {}
            valid_orientations = []
            for orientation in ["axial", "coronal", "sagittal"]:
                folder_path = self.folder_paths[orientation]
                if folder_path:
                    volume, tumor_mask, slice_thickness, pixel_spacing = self.read_dicom_series(folder_path)
                    if volume is None:
                        QMessageBox.critical(self, "Error", f"No valid DICOM files found in {orientation} folder!")
                        self.label.setText("Select DICOM Folders for Axial, Coronal, and Sagittal")
                        return
                    tumor_volume_cm3 = self.calculate_tumor_volume(tumor_mask, slice_thickness, pixel_spacing)
                    volumes[orientation] = tumor_volume_cm3
                    valid_orientations.append(orientation)

            if not volumes:
                QMessageBox.critical(self, "Error", "No valid volumes calculated!")
                self.label.setText("Select DICOM Folders for Axial, Coronal, and Sagittal")
                return

            # Calculate average volume
            average_volume = np.mean(list(volumes.values())) if volumes else 0.0

            # Prepare result message
            result_message = ""
            for orientation in ["axial", "coronal", "sagittal"]:
                if orientation in volumes:
                    result_message += f"{orientation.capitalize()} Volume: {volumes[orientation]:.2f} cm³\n"
                else:
                    result_message += f"{orientation.capitalize()}: Not processed\n"
            result_message += f"\nAverage Volume: {average_volume:.2f} cm³"

            # Display the result
            QMessageBox.information(self, "Tumor Volumes", result_message)
            self.label.setText("Select DICOM Folders for Axial, Coronal, and Sagittal")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            self.label.setText("Select DICOM Folders for Axial, Coronal, and Sagittal")

if __name__ == "__main__":
    # Required dependencies
    required_packages = ['pydicom', 'numpy', 'opencv-python']

    app = QApplication([])
    window = volume_calculator()
    window.show()
    app.exec_()
