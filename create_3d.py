import sys
import os
import numpy as np
import SimpleITK as sitk
import pydicom
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QLineEdit, QGridLayout, QMessageBox, QProgressDialog,
    QSlider, QComboBox, QGroupBox, QCheckBox, QDoubleSpinBox, QDialog, QAction,
    QToolBar, QMenu, QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal,QSize
import vtk
import math
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import traceback
# Optional import for texture analysis
try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
vtk.vtkObject.GlobalWarningDisplayOff()

# DicomProcessingWorkerTumor (modified to increase tumor region intensity)
class DicomProcessingWorkerTumor(QThread):
    finished = pyqtSignal(object, object, dict, str)  # (numpy_array, tumor_mask, image_properties, message)
    progress = pyqtSignal(int, str)

    def __init__(self, axial_dir, coronal_dir, sagittal_dir):
        super().__init__()
        self.axial_dir = axial_dir
        self.coronal_dir = coronal_dir
        self.sagittal_dir = sagittal_dir
        self.error_message_accumulator = []
        self.image_properties = {}

    def _log_error(self, msg):
        print(msg)
        self.error_message_accumulator.append(msg)

    def load_dicom_series(self, directory_path, series_name, is_segmentation=False):
        current_progress_base = self.progress_dialog_current_value if hasattr(self, 'progress_dialog_current_value') else 0
        self.progress.emit(current_progress_base, f"Reading {series_name} {'segmentation' if is_segmentation else 'image'} DICOM series from: {directory_path}")
        if not directory_path or not os.path.isdir(directory_path):
            if directory_path:
                self._log_error(f"Warning: {series_name} directory not found: {directory_path}. Skipping.")
            return None, []
        dicom_files = []
        try:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.lower().endswith('.dcm') and (('_seg' in file.lower() or '_segmentation' in file.lower()) == is_segmentation):
                        dicom_files.append(os.path.join(root, file))
            if not dicom_files:
                self._log_error(f"Warning: No {'segmentation' if is_segmentation else 'image'} .dcm files found in {series_name} directory: {directory_path}. Skipping.")
                return None, []
        except Exception as e:
            self._log_error(f"Error scanning {series_name} directory {directory_path}: {e}")
            return None, []
        if is_segmentation:
            return None, dicom_files
        dicom_files_with_instance = []
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                instance_number = int(ds.get((0x0020, 0x0013), 0).value or 0)
                dicom_files_with_instance.append((file_path, instance_number))
            except Exception as e:
                self._log_error(f"Warning: Could not read Instance Number from {file_path}: {e}. Skipping file.")
                continue
        if not dicom_files_with_instance:
            self._log_error(f"Warning: No valid image DICOM files in {series_name} directory: {directory_path}. Skipping.")
            return None, []
        dicom_files_with_instance.sort(key=lambda x: x[1])
        sorted_dicom_files = [file_path for file_path, _ in dicom_files_with_instance]
        reader = sitk.ImageSeriesReader()
        try:
            reader.SetFileNames(sorted_dicom_files)
            image_sitk = reader.Execute()
            self.progress.emit(current_progress_base + 5, f"Successfully read {series_name} image series.")
            return image_sitk, sorted_dicom_files
        except RuntimeError as e:
            self._log_error(f"Error reading {series_name} image series from {directory_path}: {e}")
            return None, []
        except Exception as e:
            self._log_error(f"Unexpected error reading {series_name} from {directory_path}: {e}")
            return None, []

    def process_segmentation(self, seg_files, series_name, ref_image_sitk, ref_instance_numbers):
        if not seg_files or not ref_image_sitk:
            return None
        ref_array = sitk.GetArrayFromImage(ref_image_sitk)
        nz, ny, nx = ref_array.shape
        tumor_mask = np.zeros((nz, ny, nx), dtype=np.uint8)
        seg_dict = {}
        for file_path in seg_files:
            try:
                ds = pydicom.dcmread(file_path)
                instance_number = int(ds.get((0x0020, 0x0013), 0).value or 0)
                pixel_array = ds.pixel_array
                if pixel_array.ndim == 3:
                    for i in range(pixel_array.shape[0]):
                        seg_dict[instance_number + i] = pixel_array[i]
                else:
                    seg_dict[instance_number] = pixel_array
            except Exception as e:
                self._log_error(f"Warning: Error reading segmentation file {file_path}: {e}")
                continue
        for z, instance_num in enumerate(ref_instance_numbers):
            if instance_num in seg_dict:
                seg_data = seg_dict[instance_num]
                if seg_data.shape == (ny, nx):
                    if seg_data.ndim > 2:
                        combined_mask = np.sum(seg_data, axis=0)
                    else:
                        combined_mask = seg_data
                    if np.any(combined_mask > 0):
                        binary_mask = (combined_mask > 0).astype(np.uint8)
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        filled_mask = np.zeros_like(binary_mask)
                        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
                        tumor_mask[z] = filled_mask
                else:
                    self._log_error(f"Warning: Segmentation shape mismatch for {series_name}, instance {instance_num}: expected ({ny}, {nx}), got {seg_data.shape}")
        tumor_sitk = sitk.GetImageFromArray(tumor_mask)
        tumor_sitk.CopyInformation(ref_image_sitk)
        return tumor_sitk

    def run(self):
        self.error_message_accumulator = []
        images_to_combine_sitk = []
        loaded_series_info = []
        seg_files_dict = {}
        instance_numbers_dict = {}
        self.progress_dialog_current_value = 0
        for dir_path, name in [(self.axial_dir, "Axial"), (self.coronal_dir, "Coronal"), (self.sagittal_dir, "Sagittal")]:
            if dir_path:
                img_sitk, img_files = self.load_dicom_series(dir_path, name, is_segmentation=False)
                if img_sitk:
                    images_to_combine_sitk.append(img_sitk)
                    loaded_series_info.append({"image": img_sitk, "name": name})
                    instance_numbers = []
                    for f in img_files:
                        try:
                            ds = pydicom.dcmread(f, stop_before_pixels=True)
                            instance_numbers.append(int(ds.get((0x0020, 0x0013), 0).value or 0))
                        except:
                            continue
                    instance_numbers.sort()
                    instance_numbers_dict[name] = instance_numbers
                _, seg_files = self.load_dicom_series(dir_path, f"{name} Segmentation", is_segmentation=True)
                if seg_files:
                    seg_files_dict[name] = seg_files
            self.progress_dialog_current_value += 10
        if not images_to_combine_sitk:
            final_msg = "No DICOM series loaded. " + " ".join(self.error_message_accumulator)
            self.finished.emit(None, None, {}, final_msg.strip())
            return
        preferred_order = ["Axial", "Coronal", "Sagittal"]
        reference_image_info = None
        for name_pref in preferred_order:
            for info_item in loaded_series_info:
                if info_item["name"] == name_pref:
                    reference_image_info = info_item
                    break
            if reference_image_info:
                break
        if not reference_image_info:
            reference_image_info = loaded_series_info[0]
        reference_image = reference_image_info["image"]
        reference_name = reference_image_info["name"]
        self.progress.emit(self.progress_dialog_current_value, f"Using {reference_name} as reference. Resampling...")
        self.image_properties = {
            "spacing": reference_image.GetSpacing(),
            "origin": reference_image.GetOrigin(),
            "dimensions": reference_image.GetSize(),
            "direction": reference_image.GetDirection()
        }
        resampled_images_np = []
        resampled_tumor_masks_sitk = []
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        try:
            min_val_ref = np.min(sitk.GetArrayViewFromImage(reference_image))
            resampler.SetDefaultPixelValue(float(min_val_ref))
        except Exception:
            resampler.SetDefaultPixelValue(0.0)
        num_images = len(images_to_combine_sitk)
        for i, info_item in enumerate(loaded_series_info):
            img_sitk = info_item["image"]
            img_name = info_item["name"]
            self.progress_dialog_current_value = 30 + int(i * (50.0 / num_images))
            self.progress.emit(self.progress_dialog_current_value, f"Processing {img_name} image...")
            img_sitk_float = sitk.Cast(img_sitk, sitk.sitkFloat32)
            if img_sitk is reference_image:
                resampled_img_float = img_sitk_float
            else:
                try:
                    resampled_img_float = resampler.Execute(img_sitk_float)
                except RuntimeError as e:
                    self._log_error(f"Error resampling {img_name} image: {e}. Skipping.")
                    continue
            resampled_images_np.append(sitk.GetArrayFromImage(resampled_img_float))
            self.progress.emit(self.progress_dialog_current_value, f"Processing {img_name} segmentation...")
            if img_name in seg_files_dict and img_name in instance_numbers_dict:
                tumor_sitk = self.process_segmentation(
                    seg_files_dict[img_name], img_name, img_sitk, instance_numbers_dict[img_name]
                )
                if tumor_sitk is not None:
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler.SetDefaultPixelValue(0)
                    try:
                        resampled_tumor_sitk = resampler.Execute(tumor_sitk)
                        resampled_tumor_masks_sitk.append(resampled_tumor_sitk)
                    except RuntimeError as e:
                        self._log_error(f"Error resampling {img_name} segmentation: {e}. Skipping.")
        if not resampled_images_np:
            final_msg = "No images resampled. " + " ".join(self.error_message_accumulator)
            self.finished.emit(None, None, {}, final_msg.strip())
            return
        self.progress.emit(80, "Averaging processed images...")
        try:
            first_shape = resampled_images_np[0].shape
            compatible_arrays = [arr for arr in resampled_images_np if arr.shape == first_shape]
            if len(compatible_arrays) != len(resampled_images_np):
                self._log_error("Warning: Shape mismatch during image averaging.")
            if not compatible_arrays:
                final_msg = "Critical error: No compatible arrays for averaging. " + " ".join(self.error_message_accumulator)
                self.finished.emit(None, None, {}, final_msg.strip())
                return
            combined_np = np.mean(np.stack(compatible_arrays, axis=0), axis=0)
        except Exception as e:
            final_msg = f"Error during image averaging: {e}. " + " ".join(self.error_message_accumulator)
            self.finished.emit(None, None, {}, final_msg.strip())
            return
        combined_tumor_mask_np = None
        if resampled_tumor_masks_sitk:
            self.progress.emit(90, "Combining tumor masks...")
            try:
                tumor_arrays = [sitk.GetArrayFromImage(t) for t in resampled_tumor_masks_sitk]
                compatible_masks = [m for m in tumor_arrays if m.shape == first_shape]
                if compatible_masks:
                    combined_tumor_mask_np = np.max(np.stack(compatible_masks, axis=0), axis=0).astype(np.uint8)
                    # Enhance intensity in tumor regions
                    tumor_mask = combined_tumor_mask_np > 0
                    # Check if 'flair' is in any directory path (case-insensitive)
                    is_flair = any(
                        'flair' in dir_path.lower()
                        for dir_path in [self.axial_dir, self.coronal_dir, self.sagittal_dir]
                        if dir_path
                    )
                    if is_flair:
                        # Intensity factors for FLAIR sequences
                        tumor_intensity_factor = 1.4  # Increase tumor region intensity by 50%
                        non_tumor_intensity_factor = 0.2  # Decrease non-tumor region intensity by 30%
                        # Enhance tumor region intensity
                        combined_np[tumor_mask] *= tumor_intensity_factor
                        # Reduce non-tumor region intensity
                        combined_np[~tumor_mask] *= non_tumor_intensity_factor
                        # Clip intensities to stay within original data range
                        combined_np = np.clip(combined_np, combined_np.min(), combined_np.max())
                    else:
                        # Default intensity adjustment for non-FLAIR sequences
                        tumor_intensity_factor = 1.99  # Original intensity increase
                        combined_np[tumor_mask] *= tumor_intensity_factor
                        combined_np[tumor_mask] = np.clip(combined_np[tumor_mask], combined_np.min(), combined_np.max())
                else:
                    self._log_error("Warning: No compatible tumor masks for combining.")
            except Exception as e:
                self._log_error(f"Error during tumor mask combining: {e}.")
        combined_np = np.ascontiguousarray(combined_np, dtype=np.float32)
        if combined_tumor_mask_np is not None:
            combined_tumor_mask_np = np.ascontiguousarray(combined_tumor_mask_np, dtype=np.uint8)
        self.progress.emit(100, "Processing complete.")
        final_message = "Success"
        if self.error_message_accumulator:
            final_message += " with warnings: " + " ".join(self.error_message_accumulator)
        self.finished.emit(combined_np, combined_tumor_mask_np, self.image_properties, final_message.strip())

# DicomProcessingWorkerVolume (unchanged)
class DicomProcessingWorkerVolume(QThread):
    finished = pyqtSignal(object, dict, str)
    progress = pyqtSignal(int, str)

    def __init__(self, axial_dir, coronal_dir, sagittal_dir):
        super().__init__()
        self.axial_dir = axial_dir
        self.coronal_dir = coronal_dir
        self.sagittal_dir = sagittal_dir
        self.error_message_accumulator = []
        self.image_properties = {}

    def _log_error(self, msg):
        print(msg)
        self.error_message_accumulator.append(msg)

    def load_dicom_series(self, directory_path, series_name):
        current_progress_base = self.progress_dialog_current_value if hasattr(self, 'progress_dialog_current_value') else 0
        self.progress.emit(current_progress_base, f"Reading {series_name} DICOM series from: {directory_path}")
        if not directory_path or not os.path.isdir(directory_path):
            if directory_path:
                self._log_error(f"Warning: {series_name} directory not found: {directory_path}. Skipping.")
            return None
        dicom_files = []
        try:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        dicom_files.append(os.path.join(root, file))
            if not dicom_files:
                self._log_error(f"Warning: No .dcm files found in {series_name} directory: {directory_path}. Skipping.")
                return None
        except Exception as e:
            self._log_error(f"Error scanning {series_name} directory {directory_path}: {e}")
            return None
        dicom_files = [f for f in dicom_files if '_seg' not in f]
        dicom_files_with_instance = []
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                instance_number = int(ds.get((0x0020, 0x0013), 0).value or 0)
                dicom_files_with_instance.append((file_path, instance_number))
            except Exception as e:
                self._log_error(f"Warning: Could not read Instance Number from {file_path}: {e}. Skipping file.")
                continue
        if not dicom_files_with_instance:
            self._log_error(f"Warning: No valid DICOM files with readable metadata in {series_name} directory: {directory_path}. Skipping.")
            return None
        dicom_files_with_instance.sort(key=lambda x: x[1])
        sorted_dicom_files = [file_path for file_path, _ in dicom_files_with_instance]
        reader = sitk.ImageSeriesReader()
        try:
            reader.SetFileNames(sorted_dicom_files)
            image_sitk = reader.Execute()
            self.progress.emit(current_progress_base + 5, f"Successfully read {series_name} series.")
            return image_sitk
        except RuntimeError as e:
            self._log_error(f"Error reading {series_name} DICOM series from {directory_path}: {e}")
            return None
        except Exception as e:
            self._log_error(f"Unexpected error reading {series_name} from {directory_path}: {e}")
            return None

    def run(self):
        self.error_message_accumulator = []
        images_to_combine_sitk = []
        loaded_series_info = []
        self.progress_dialog_current_value = 0
        if self.axial_dir:
            axial_sitk = self.load_dicom_series(self.axial_dir, "Axial")
            if axial_sitk: images_to_combine_sitk.append(axial_sitk); loaded_series_info.append({"image": axial_sitk, "name": "Axial"})
        self.progress_dialog_current_value = 10
        if self.coronal_dir:
            coronal_sitk = self.load_dicom_series(self.coronal_dir, "Coronal")
            if coronal_sitk: images_to_combine_sitk.append(coronal_sitk); loaded_series_info.append({"image": coronal_sitk, "name": "Coronal"})
        self.progress_dialog_current_value = 20
        if self.sagittal_dir:
            sagittal_sitk = self.load_dicom_series(self.sagittal_dir, "Sagittal")
            if sagittal_sitk: images_to_combine_sitk.append(sagittal_sitk); loaded_series_info.append({"image": sagittal_sitk, "name": "Sagittal"})
        self.progress_dialog_current_value = 30
        if not images_to_combine_sitk:
            final_msg = "No DICOM series loaded. " + " ".join(self.error_message_accumulator)
            self.finished.emit(None, {}, final_msg.strip())
            return
        preferred_order = ["Axial", "Coronal", "Sagittal"]
        reference_image_info = None
        for name_pref in preferred_order:
            for info_item in loaded_series_info:
                if info_item["name"] == name_pref:
                    reference_image_info = info_item
                    break
            if reference_image_info:
                break
        if not reference_image_info: reference_image_info = loaded_series_info[0]
        reference_image = reference_image_info["image"]
        reference_name = reference_image_info["name"]
        self.progress.emit(self.progress_dialog_current_value, f"Using {reference_name} as reference. Resampling...")
        self.image_properties = {
            "spacing": reference_image.GetSpacing(),
            "origin": reference_image.GetOrigin(),
            "dimensions": reference_image.GetSize(),
            "direction": reference_image.GetDirection()
        }
        resampled_images_np = []
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        try:
            min_val_ref = np.min(sitk.GetArrayViewFromImage(reference_image))
            resampler.SetDefaultPixelValue(float(min_val_ref))
        except Exception: resampler.SetDefaultPixelValue(0.0)
        num_images = len(images_to_combine_sitk)
        for i, current_info_item in enumerate(loaded_series_info):
            img_sitk = current_info_item["image"]
            img_name = current_info_item["name"]
            self.progress_dialog_current_value = 30 + int(i * (50.0 / num_images))
            self.progress.emit(self.progress_dialog_current_value, f"Processing {img_name}...")
            img_sitk_float = sitk.Cast(img_sitk, sitk.sitkFloat32)
            if img_sitk is reference_image:
                resampled_img_float = img_sitk_float
            else:
                try: resampled_img_float = resampler.Execute(img_sitk_float)
                except RuntimeError as e: self._log_error(f"Error resampling {img_name}: {e}. Skip."); continue
            resampled_images_np.append(sitk.GetArrayFromImage(resampled_img_float))
        if not resampled_images_np:
            final_msg = "No images resampled. " + " ".join(self.error_message_accumulator)
            self.finished.emit(None, {}, final_msg.strip())
            return
        self.progress.emit(80, "Averaging processed images...")
        try:
            first_shape = resampled_images_np[0].shape
            compatible_arrays = [arr for arr in resampled_images_np if arr.shape == first_shape]
            if len(compatible_arrays) != len(resampled_images_np): self._log_error("Warn: Shape mismatch during avg.")
            if not compatible_arrays:
                final_msg = "Crit err: No compatible arrays for avg. " + " ".join(self.error_message_accumulator)
                self.finished.emit(None, {}, final_msg.strip()); return
            combined_np = np.mean(np.stack(compatible_arrays, axis=0), axis=0)
        except Exception as e:
            final_msg = f"Error during averaging: {e}. " + " ".join(self.error_message_accumulator)
            self.finished.emit(None, {}, final_msg.strip()); return
        combined_np = np.ascontiguousarray(combined_np, dtype=np.float32)
        self.progress.emit(100, "Processing complete.")
        final_message = "Success"
        if self.error_message_accumulator:
            final_message += " with warnings: " + " ".join(self.error_message_accumulator)
        self.finished.emit(combined_np, self.image_properties, final_message.strip())

# Load DICOM Dialog (unchanged)
class LoadDicomDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load DICOM Series")
        self.setGeometry(200, 200, 600, 200)
        self.layout = QVBoxLayout(self)
        self.input_layout = QGridLayout()
        self.path_edits = {}
        for i, name in enumerate(["Axial", "Coronal", "Sagittal"]):
            label = QLabel(f"{name} DICOM Series Folder:")
            self.path_edits[name] = QLineEdit()
            button = QPushButton("Browse...")
            button.clicked.connect(lambda _, n=name: self.browse_folder(n))
            self.input_layout.addWidget(label, i, 0)
            self.input_layout.addWidget(self.path_edits[name], i, 1)
            self.input_layout.addWidget(button, i, 2)
        self.layout.addLayout(self.input_layout)
        self.process_button = QPushButton("Process 3D Plot")
        self.process_button.clicked.connect(self.accept)
        self.layout.addWidget(self.process_button)

    def browse_folder(self, name):
        current_path = self.path_edits[name].text()
        if not current_path or not os.path.isdir(current_path):
            current_path = os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(self, f"Select {name} DICOM Folder", current_path)
        if directory:
            self.path_edits[name].setText(directory)

    def get_paths(self):
        return {
            "Axial": self.path_edits["Axial"].text(),
            "Coronal": self.path_edits["Coronal"].text(),
            "Sagittal": self.path_edits["Sagittal"].text()
        }

# Anatomy Visualization Dialog (new with advanced features)
class AnatomyVisualizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anatomy Visualization Customization")
        self.setGeometry(300, 300, 600, 700)
        self.layout = QVBoxLayout(self)
        vis_control_group = QGroupBox("Anatomy Visualization Controls")
        vis_control_layout = QFormLayout()
        # Basic Controls
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue(50)
        vis_control_layout.addRow("Contrast:", self.contrast_slider)
        self.ambience_slider = QSlider(Qt.Horizontal)
        self.ambience_slider.setRange(0, 100)
        self.ambience_slider.setValue(20)
        vis_control_layout.addRow("Ambience:", self.ambience_slider)
        self.opacity_threshold_slider = QSlider(Qt.Horizontal)
        self.opacity_threshold_slider.setRange(0, 100)
        self.opacity_threshold_slider.setValue(0)
        vis_control_layout.addRow("Opacity Threshold:", self.opacity_threshold_slider)
        self.gradient_opacity_slider = QSlider(Qt.Horizontal)
        self.gradient_opacity_slider.setRange(0, 100)
        self.gradient_opacity_slider.setValue(50)
        vis_control_layout.addRow("Gradient Opacity:", self.gradient_opacity_slider)
        self.specular_power_slider = QSlider(Qt.Horizontal)
        self.specular_power_slider.setRange(1, 100)
        self.specular_power_slider.setValue(10)
        vis_control_layout.addRow("Specular Power:", self.specular_power_slider)
        self.scalar_min_spinbox = QDoubleSpinBox()
        self.scalar_min_spinbox.setRange(-10000, 10000)
        self.scalar_min_spinbox.setValue(0)
        vis_control_layout.addRow("Scalar Min:", self.scalar_min_spinbox)
        self.scalar_max_spinbox = QDoubleSpinBox()
        self.scalar_max_spinbox.setRange(-10000, 10000)
        self.scalar_max_spinbox.setValue(1000)
        vis_control_layout.addRow("Scalar Max:", self.scalar_max_spinbox)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "Grayscale", "Bone", "Hot", "Jet", "CoolWarm", "Viridis",
            "Magma", "Rainbow", "Spectral", "BrainTumorSpecific", "Parula",
            "Ocean", "Autumn"
        ])
        vis_control_layout.addRow("Colormap:", self.colormap_combo)
        # Advanced Controls for Surgery Planning
        self.shading_toggle = QCheckBox("Enable Advanced Shading")
        self.shading_toggle.setChecked(True)
        vis_control_layout.addRow(self.shading_toggle)
        self.edge_enhancement_slider = QSlider(Qt.Horizontal)
        self.edge_enhancement_slider.setRange(0, 100)
        self.edge_enhancement_slider.setValue(20)
        vis_control_layout.addRow("Edge Enhancement:", self.edge_enhancement_slider)
        self.tissue_density_slider = QSlider(Qt.Horizontal)
        self.tissue_density_slider.setRange(0, 100)
        self.tissue_density_slider.setValue(50)
        vis_control_layout.addRow("Tissue Density:", self.tissue_density_slider)
        self.annotation_toggle = QCheckBox("Show Tumor Annotations")
        self.annotation_toggle.setChecked(False)
        vis_control_layout.addRow(self.annotation_toggle)
        self.measurement_tool_toggle = QCheckBox("Enable Measurement Tool")
        self.measurement_tool_toggle.setChecked(False)
        vis_control_layout.addRow(self.measurement_tool_toggle)
        # Clipping Planes in All Orientations
        self.clipping_x_toggle = QCheckBox("Enable X-Axis Clipping")
        vis_control_layout.addRow(self.clipping_x_toggle)
        self.clipping_x_position = QSlider(Qt.Horizontal)
        self.clipping_x_position.setRange(0, 100)
        self.clipping_x_position.setValue(50)
        vis_control_layout.addRow("X-Axis Clip Position:", self.clipping_x_position)
        self.clipping_y_toggle = QCheckBox("Enable Y-Axis Clipping")
        vis_control_layout.addRow(self.clipping_y_toggle)
        self.clipping_y_position = QSlider(Qt.Horizontal)
        self.clipping_y_position.setRange(0, 100)
        self.clipping_y_position.setValue(50)
        vis_control_layout.addRow("Y-Axis Clip Position:", self.clipping_y_position)
        self.clipping_z_toggle = QCheckBox("Enable Z-Axis Clipping")
        vis_control_layout.addRow(self.clipping_z_toggle)
        self.clipping_z_position = QSlider(Qt.Horizontal)
        self.clipping_z_position.setRange(0, 100)
        self.clipping_z_position.setValue(50)
        vis_control_layout.addRow("Z-Axis Clip Position:", self.clipping_z_position)
        vis_control_group.setLayout(vis_control_layout)
        self.layout.addWidget(vis_control_group)
        # Lighting Controls
        lighting_group = QGroupBox("Lighting Controls")
        lighting_layout = QFormLayout()
        self.light_azimuth_slider = QSlider(Qt.Horizontal)
        self.light_azimuth_slider.setRange(-180, 180)
        self.light_azimuth_slider.setValue(30)
        lighting_layout.addRow("Light Azimuth:", self.light_azimuth_slider)
        self.light_elevation_slider = QSlider(Qt.Horizontal)
        self.light_elevation_slider.setRange(-90, 90)
        self.light_elevation_slider.setValue(30)
        lighting_layout.addRow("Light Elevation:", self.light_elevation_slider)
        self.light_intensity_slider = QSlider(Qt.Horizontal)
        self.light_intensity_slider.setRange(0, 100)
        self.light_intensity_slider.setValue(60)
        lighting_layout.addRow("Light Intensity:", self.light_intensity_slider)
        lighting_group.setLayout(lighting_layout)
        self.layout.addWidget(lighting_group)

# Volume Visualization Dialog (new with advanced features)
class VolumeVisualizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Volume Visualization Customization")
        self.setGeometry(300, 300, 600, 700)
        self.layout = QVBoxLayout(self)
        vis_control_group = QGroupBox("Volume Visualization Controls")
        vis_control_layout = QFormLayout()
        # Basic Controls
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue(50)
        vis_control_layout.addRow("Contrast:", self.contrast_slider)
        self.ambience_slider = QSlider(Qt.Horizontal)
        self.ambience_slider.setRange(0, 100)
        self.ambience_slider.setValue(20)
        vis_control_layout.addRow("Ambience:", self.ambience_slider)
        self.opacity_threshold_slider = QSlider(Qt.Horizontal)
        self.opacity_threshold_slider.setRange(0, 100)
        self.opacity_threshold_slider.setValue(0)
        vis_control_layout.addRow("Opacity Threshold:", self.opacity_threshold_slider)
        self.gradient_opacity_slider = QSlider(Qt.Horizontal)
        self.gradient_opacity_slider.setRange(0, 100)
        self.gradient_opacity_slider.setValue(50)
        vis_control_layout.addRow("Gradient Opacity:", self.gradient_opacity_slider)
        self.specular_power_slider = QSlider(Qt.Horizontal)
        self.specular_power_slider.setRange(1, 100)
        self.specular_power_slider.setValue(10)
        vis_control_layout.addRow("Specular Power:", self.specular_power_slider)
        self.scalar_min_spinbox = QDoubleSpinBox()
        self.scalar_min_spinbox.setRange(-10000, 10000)
        self.scalar_min_spinbox.setValue(0)
        vis_control_layout.addRow("Scalar Min:", self.scalar_min_spinbox)
        self.scalar_max_spinbox = QDoubleSpinBox()
        self.scalar_max_spinbox.setRange(-10000, 10000)
        self.scalar_max_spinbox.setValue(1000)
        vis_control_layout.addRow("Scalar Max:", self.scalar_max_spinbox)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "Grayscale", "Bone", "Hot", "Jet", "CoolWarm", "Viridis",
            "Magma", "Rainbow", "Spectral", "BrainTumorSpecific", "Parula",
            "Ocean", "Autumn"
        ])
        vis_control_layout.addRow("Colormap:", self.colormap_combo)
        # Advanced Controls for Surgery Planning
        self.shading_toggle = QCheckBox("Enable Advanced Shading")
        self.shading_toggle.setChecked(True)
        vis_control_layout.addRow(self.shading_toggle)
        self.edge_enhancement_slider = QSlider(Qt.Horizontal)
        self.edge_enhancement_slider.setRange(0, 100)
        self.edge_enhancement_slider.setValue(20)
        vis_control_layout.addRow("Edge Enhancement:", self.edge_enhancement_slider)
        self.tissue_density_slider = QSlider(Qt.Horizontal)
        self.tissue_density_slider.setRange(0, 100)
        self.tissue_density_slider.setValue(50)
        vis_control_layout.addRow("Tissue Density:", self.tissue_density_slider)
        self.annotation_toggle = QCheckBox("Show Volume Annotations")
        self.annotation_toggle.setChecked(False)
        vis_control_layout.addRow(self.annotation_toggle)
        self.measurement_tool_toggle = QCheckBox("Enable Measurement Tool")
        self.measurement_tool_toggle.setChecked(False)
        vis_control_layout.addRow(self.measurement_tool_toggle)
        # Clipping Planes in All Orientations
        self.clipping_x_toggle = QCheckBox("Enable X-Axis Clipping")
        vis_control_layout.addRow(self.clipping_x_toggle)
        self.clipping_x_position = QSlider(Qt.Horizontal)
        self.clipping_x_position.setRange(0, 100)
        self.clipping_x_position.setValue(50)
        vis_control_layout.addRow("X-Axis Clip Position:", self.clipping_x_position)
        self.clipping_y_toggle = QCheckBox("Enable Y-Axis Clipping")
        vis_control_layout.addRow(self.clipping_y_toggle)
        self.clipping_y_position = QSlider(Qt.Horizontal)
        self.clipping_y_position.setRange(0, 100)
        self.clipping_y_position.setValue(50)
        vis_control_layout.addRow("Y-Axis Clip Position:", self.clipping_y_position)
        self.clipping_z_toggle = QCheckBox("Enable Z-Axis Clipping")
        vis_control_layout.addRow(self.clipping_z_toggle)
        self.clipping_z_position = QSlider(Qt.Horizontal)
        self.clipping_z_position.setRange(0, 100)
        self.clipping_z_position.setValue(50)
        vis_control_layout.addRow("Z-Axis Clip Position:", self.clipping_z_position)
        vis_control_group.setLayout(vis_control_layout)
        self.layout.addWidget(vis_control_group)
        # Lighting Controls
        lighting_group = QGroupBox("Lighting Controls")
        lighting_layout = QFormLayout()
        self.light_azimuth_slider = QSlider(Qt.Horizontal)
        self.light_azimuth_slider.setRange(-180, 180)
        self.light_azimuth_slider.setValue(30)
        lighting_layout.addRow("Light Azimuth:", self.light_azimuth_slider)
        self.light_elevation_slider = QSlider(Qt.Horizontal)
        self.light_elevation_slider.setRange(-90, 90)
        self.light_elevation_slider.setValue(30)
        lighting_layout.addRow("Light Elevation:", self.light_elevation_slider)
        self.light_intensity_slider = QSlider(Qt.Horizontal)
        self.light_intensity_slider.setRange(0, 100)
        self.light_intensity_slider.setValue(60)
        lighting_layout.addRow("Light Intensity:", self.light_intensity_slider)
        lighting_group.setLayout(lighting_layout)
        self.layout.addWidget(lighting_group)

# Enhanced DicomViewer3D (modified for separate dialogs and advanced features)
class DicomViewer3D(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM 3D Volume Viewer")
        self.setGeometry(100, 100, 1200, 600)
        self.toolbar = QToolBar("Main Menu")
        self.addToolBar(self.toolbar)
        self.menu_action = QAction("☰", self)
        self.menu_action.setMenu(self.create_menu())
        self.toolbar.addAction(self.menu_action)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        vtk_layout = QHBoxLayout()
        self.tumor_vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        vtk_layout.addWidget(self.tumor_vtk_widget, 1)
        self.tumor_renderer = vtk.vtkRenderer()
        self.tumor_vtk_widget.GetRenderWindow().AddRenderer(self.tumor_renderer)
        self.tumor_interactor = self.tumor_vtk_widget.GetRenderWindow().GetInteractor()
        self.anatomy_volume_actor = vtk.vtkVolume()
        self.tumor_renderer.SetBackground(0.1, 0.1, 0.1)
        self.anatomy_clipping_planes = {
            'x': vtk.vtkPlane(),
            'y': vtk.vtkPlane(),
            'z': vtk.vtkPlane()
        }
        self.volume_vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        vtk_layout.addWidget(self.volume_vtk_widget, 1)
        self.volume_renderer = vtk.vtkRenderer()
        self.volume_vtk_widget.GetRenderWindow().AddRenderer(self.volume_renderer)
        self.volume_interactor = self.volume_vtk_widget.GetRenderWindow().GetInteractor()
        self.volume_actor = vtk.vtkVolume()
        self.volume_renderer.SetBackground(0.1, 0.1, 0.1)
        self.volume_clipping_planes = {
            'x': vtk.vtkPlane(),
            'y': vtk.vtkPlane(),
            'z': vtk.vtkPlane()
        }
        self.main_layout.addLayout(vtk_layout)
        self.status_label = QLabel("Status: Ready")
        self.main_layout.addWidget(self.status_label)
        self.progress_dialog = None
        self.tumor_worker = None
        self.volume_worker = None
        self.anatomy_data_range = None
        self.volume_data_range = None
        self.anatomy_lights = []
        self.volume_lights = []
        self.anatomy_vis_dialog = None
        self.volume_vis_dialog = None
        self.anatomy_image_data = None
        self.tumor_mask_np = None
        self.image_properties = None

        self.tumor_metrics = {}
        self.tumor_surface_actor = None
        self.annotation_actor = None

    def create_menu(self):
        menu = self.menu_action.menu()
        if menu is None:
            menu = self.menu_action.setMenu(QMenu())
            menu = self.menu_action.menu()
        menu.clear()
        load_action = QAction("Load", self)
        load_action.triggered.connect(self.open_load_dialog)
        menu.addAction(load_action)
        anatomy_vis_action = QAction("Anatomy Visualization", self)
        anatomy_vis_action.triggered.connect(self.open_anatomy_visualization_dialog)
        menu.addAction(anatomy_vis_action)
        volume_vis_action = QAction("Volume Visualization", self)
        volume_vis_action.triggered.connect(self.open_volume_visualization_dialog)
        menu.addAction(volume_vis_action)
        return menu

    def open_load_dialog(self):
        dialog = LoadDicomDialog(self)
        if dialog.exec_():
            paths = dialog.get_paths()
            self.start_processing(paths["Axial"], paths["Coronal"], paths["Sagittal"])

    def open_anatomy_visualization_dialog(self):
        if self.anatomy_vis_dialog is None:
            self.anatomy_vis_dialog = AnatomyVisualizationDialog(self)
            self.anatomy_vis_dialog.contrast_slider.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.ambience_slider.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.opacity_threshold_slider.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.gradient_opacity_slider.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.specular_power_slider.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.scalar_min_spinbox.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.scalar_max_spinbox.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.colormap_combo.currentIndexChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.shading_toggle.stateChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.edge_enhancement_slider.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.tissue_density_slider.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.annotation_toggle.stateChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.measurement_tool_toggle.stateChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.clipping_x_toggle.stateChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.clipping_x_position.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.clipping_y_toggle.stateChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.clipping_y_position.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.clipping_z_toggle.stateChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.clipping_z_position.valueChanged.connect(self.update_anatomy_visualization)
            self.anatomy_vis_dialog.light_azimuth_slider.valueChanged.connect(self.update_anatomy_lighting)
            self.anatomy_vis_dialog.light_elevation_slider.valueChanged.connect(self.update_anatomy_lighting)
            self.anatomy_vis_dialog.light_intensity_slider.valueChanged.connect(self.update_anatomy_lighting)
            self.update_anatomy_lighting()
        self.anatomy_vis_dialog.show()

    def open_volume_visualization_dialog(self):
        if self.volume_vis_dialog is None:
            self.volume_vis_dialog = VolumeVisualizationDialog(self)
            self.volume_vis_dialog.contrast_slider.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.ambience_slider.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.opacity_threshold_slider.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.gradient_opacity_slider.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.specular_power_slider.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.scalar_min_spinbox.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.scalar_max_spinbox.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.colormap_combo.currentIndexChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.shading_toggle.stateChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.edge_enhancement_slider.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.tissue_density_slider.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.annotation_toggle.stateChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.measurement_tool_toggle.stateChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.clipping_x_toggle.stateChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.clipping_x_position.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.clipping_y_toggle.stateChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.clipping_y_position.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.clipping_z_toggle.stateChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.clipping_z_position.valueChanged.connect(self.update_volume_visualization)
            self.volume_vis_dialog.light_azimuth_slider.valueChanged.connect(self.update_volume_lighting)
            self.volume_vis_dialog.light_elevation_slider.valueChanged.connect(self.update_volume_lighting)
            self.volume_vis_dialog.light_intensity_slider.valueChanged.connect(self.update_volume_lighting)
            self.update_volume_lighting()
        self.volume_vis_dialog.show()

    def get_colormap(self, colormap_name, smin, smax):
        color_tf = vtk.vtkColorTransferFunction()
        delta = smax - smin
        if delta == 0:
            delta = 1.0
        if colormap_name == "Grayscale":
            color_tf.AddRGBPoint(smin, 0.0, 0.0, 0.0)
            color_tf.AddRGBPoint(smin + 0.1 * delta, 0.2, 0.2, 0.2)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 0.5, 0.5, 0.5)
            color_tf.AddRGBPoint(smax, 1.0, 1.0, 1.0)
        elif colormap_name == "Bone":
            color_tf.AddRGBPoint(smin, 0.0, 0.0, 0.0)
            color_tf.AddRGBPoint(smin + 0.1 * delta, 0.8, 0.8, 0.8)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 0.9, 0.9, 0.8)
            color_tf.AddRGBPoint(smax, 1.0, 1.0, 0.9)
        elif colormap_name == "Hot":
            color_tf.AddRGBPoint(smin, 0.0, 0.0, 0.0)
            color_tf.AddRGBPoint(smin + 0.1 * delta, 1.0, 0.0, 0.0)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 1.0, 1.0, 0.0)
            color_tf.AddRGBPoint(smax, 1.0, 1.0, 1.0)
        elif colormap_name == "Jet":
            color_tf.AddRGBPoint(smin, 0.0, 0.0, 1.0)
            color_tf.AddRGBPoint(smin + 0.33 * delta, 0.0, 1.0, 1.0)
            color_tf.AddRGBPoint(smin + 0.66 * delta, 1.0, 1.0, 0.0)
            color_tf.AddRGBPoint(smax, 1.0, 0.0, 0.0)
        elif colormap_name == "CoolWarm":
            color_tf.AddRGBPoint(smin, 0.0, 0.0, 1.0)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 1.0, 1.0, 1.0)
            color_tf.AddRGBPoint(smax, 1.0, 0.0, 0.0)
        elif colormap_name == "Viridis":
            color_tf.AddRGBPoint(smin, 0.267, 0.0, 0.329)
            color_tf.AddRGBPoint(smin + 0.25 * delta, 0.0, 0.247, 0.541)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 0.0, 0.506, 0.529)
            color_tf.AddRGBPoint(smin + 0.75 * delta, 0.471, 0.886, 0.298)
            color_tf.AddRGBPoint(smax, 0.992, 0.906, 0.144)
        elif colormap_name == "Magma":
            color_tf.AddRGBPoint(smin, 0.0, 0.0, 0.0)
            color_tf.AddRGBPoint(smin + 0.25 * delta, 0.188, 0.071, 0.271)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 0.557, 0.173, 0.408)
            color_tf.AddRGBPoint(smin + 0.75 * delta, 0.969, 0.471, 0.349)
            color_tf.AddRGBPoint(smax, 1.0, 0.988, 0.706)
        elif colormap_name == "Rainbow":
            color_tf.AddRGBPoint(smin, 1.0, 0.0, 0.0)
            color_tf.AddRGBPoint(smin + 0.2 * delta, 1.0, 0.647, 0.0)
            color_tf.AddRGBPoint(smin + 0.4 * delta, 1.0, 1.0, 0.0)
            color_tf.AddRGBPoint(smin + 0.6 * delta, 0.0, 1.0, 0.0)
            color_tf.AddRGBPoint(smin + 0.8 * delta, 0.0, 0.0, 1.0)
            color_tf.AddRGBPoint(smax, 0.294, 0.0, 0.51)
        elif colormap_name == "Spectral":
            color_tf.AddRGBPoint(smin, 0.619, 0.0, 0.278)
            color_tf.AddRGBPoint(smin + 0.25 * delta, 0.957, 0.427, 0.263)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 1.0, 1.0, 0.749)
            color_tf.AddRGBPoint(smin + 0.75 * delta, 0.0, 0.557, 0.58)
            color_tf.AddRGBPoint(smax, 0.0, 0.0, 0.502)
        elif colormap_name == "BrainTumorSpecific":
            color_tf.AddRGBPoint(smin, 0.2, 0.2, 0.2)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 0.5, 0.5, 0.5)
            color_tf.AddRGBPoint(smax - 0.1 * delta, 0.8, 0.8, 0.8)
            color_tf.AddRGBPoint(smax, 1.0, 0.0, 0.0)
        elif colormap_name == "Parula":
            color_tf.AddRGBPoint(smin, 0.208, 0.169, 0.647)
            color_tf.AddRGBPoint(smin + 0.25 * delta, 0.0, 0.408, 0.784)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 0.0, 0.659, 0.663)
            color_tf.AddRGBPoint(smin + 0.75 * delta, 0.631, 0.851, 0.0)
            color_tf.AddRGBPoint(smax, 0.996, 0.878, 0.0)
        elif colormap_name == "Ocean":
            color_tf.AddRGBPoint(smin, 0.0, 0.0, 0.2)
            color_tf.AddRGBPoint(smin + 0.25 * delta, 0.0, 0.2, 0.4)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 0.0, 0.4, 0.6)
            color_tf.AddRGBPoint(smin + 0.75 * delta, 0.0, 0.6, 0.4)
            color_tf.AddRGBPoint(smax, 0.0, 0.8, 0.2)
        elif colormap_name == "Autumn":
            color_tf.AddRGBPoint(smin, 1.0, 0.0, 0.0)
            color_tf.AddRGBPoint(smin + 0.5 * delta, 1.0, 0.5, 0.0)
            color_tf.AddRGBPoint(smax, 1.0, 1.0, 0.0)
        return color_tf

    def update_anatomy_lighting(self):
        if not self.anatomy_vis_dialog:
            azimuth = 30
            elevation = 30
            intensity = 0.6
        else:
            azimuth = self.anatomy_vis_dialog.light_azimuth_slider.value()
            elevation = self.anatomy_vis_dialog.light_elevation_slider.value()
            intensity = self.anatomy_vis_dialog.light_intensity_slider.value() / 100.0
        for light in self.anatomy_lights:
            self.tumor_renderer.RemoveLight(light)
        self.anatomy_lights.clear()
        rad_azimuth = math.radians(azimuth)
        rad_elevation = math.radians(elevation)
        light1_pos = [
            math.cos(rad_elevation) * math.cos(rad_azimuth),
            math.cos(rad_elevation) * math.sin(rad_azimuth),
            math.sin(rad_elevation)
        ]
        light1 = vtk.vtkLight()
        light1.SetPosition(light1_pos)
        light1.SetFocalPoint(0, 0, 0)
        light1.SetIntensity(intensity)
        self.tumor_renderer.AddLight(light1)
        self.anatomy_lights.append(light1)
        light2 = vtk.vtkLight()
        light2.SetPosition(0, 1, 0)
        light2.SetFocalPoint(0, 0, 0)
        light2.SetIntensity(intensity * 0.4)
        self.tumor_renderer.AddLight(light2)
        self.anatomy_lights.append(light2)
        self.tumor_vtk_widget.GetRenderWindow().Render()

    def update_volume_lighting(self):
        if not self.volume_vis_dialog:
            azimuth = 30
            elevation = 30
            intensity = 0.6
        else:
            azimuth = self.volume_vis_dialog.light_azimuth_slider.value()
            elevation = self.volume_vis_dialog.light_elevation_slider.value()
            intensity = self.volume_vis_dialog.light_intensity_slider.value() / 100.0
        for light in self.volume_lights:
            self.volume_renderer.RemoveLight(light)
        self.volume_lights.clear()
        rad_azimuth = math.radians(azimuth)
        rad_elevation = math.radians(elevation)
        light1_pos = [
            math.cos(rad_elevation) * math.cos(rad_azimuth),
            math.cos(rad_elevation) * math.sin(rad_azimuth),
            math.sin(rad_elevation)
        ]
        light1 = vtk.vtkLight()
        light1.SetPosition(light1_pos)
        light1.SetFocalPoint(0, 0, 0)
        light1.SetIntensity(intensity)
        self.volume_renderer.AddLight(light1)
        self.volume_lights.append(light1)
        light2 = vtk.vtkLight()
        light2.SetPosition(0, 1, 0)
        light2.SetFocalPoint(0, 0, 0)
        light2.SetIntensity(intensity * 0.4)
        self.volume_renderer.AddLight(light2)
        self.volume_lights.append(light2)
        self.volume_vtk_widget.GetRenderWindow().Render()

    def _calculate_tumor_features(self):
        """Calculates tumor metrics and creates display actors for irregular 3D tumor shapes."""
        if self.tumor_mask_np is None or self.image_properties is None or self.anatomy_image_data is None:
            return
        
        self.tumor_metrics.clear()
        mask = self.tumor_mask_np

        if np.sum(mask) == 0:
            print("No tumor region found in the mask.")
            return

        spacing = self.image_properties['spacing']
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        voxel_volume_cc = voxel_volume_mm3 / 1000.0  # Convert mm³ to cc
        
        # --- Volume ---
        num_voxels = np.count_nonzero(mask)
        tumor_volume_cc = num_voxels * voxel_volume_cc
        self.tumor_metrics['Volume (cc)'] = tumor_volume_cc

        # --- Surface Area & Sphericity using VTK Marching Cubes ---
        mask_importer = vtk.vtkImageImport()
        mask_string = mask.tobytes('C')
        mask_importer.CopyImportVoidPointer(mask_string, len(mask_string))
        mask_importer.SetDataScalarTypeToUnsignedChar()
        mask_importer.SetNumberOfScalarComponents(1)
        dims = mask.shape
        mask_importer.SetDataExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
        mask_importer.SetWholeExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
        mask_importer.SetDataSpacing(spacing)
        mask_importer.SetDataOrigin(self.image_properties['origin'])
        mask_importer.Update()
        
        # Use standard Marching Cubes for smoother surfaces
        mc = vtk.vtkMarchingCubes()
        mc.SetInputConnection(mask_importer.GetOutputPort())
        mc.SetValue(0, 1.0)  # Extract surface for voxel value 1
        mc.ComputeNormalsOn()
        mc.Update()

        # Handle multiple disconnected regions
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputConnection(mc.GetOutputPort())
        connectivity.SetExtractionModeToAllRegions()
        connectivity.Update()
        
        mass_props = vtk.vtkMassProperties()
        mass_props.SetInputConnection(connectivity.GetOutputPort())
        surface_area = mass_props.GetSurfaceArea()
        self.tumor_metrics['Surface Area (mm²)'] = surface_area
        
        if surface_area > 0:
            tumor_volume_mm3 = tumor_volume_cc * 1000.0  # Convert cc to mm³ for sphericity
            sphericity = (math.pi**(1/3) * (6 * tumor_volume_mm3)**(2/3)) / surface_area
            self.tumor_metrics['Sphericity'] = sphericity
        else:
            self.tumor_metrics['Sphericity'] = 0.0

        # --- Create Tumor Surface Actor ---
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(connectivity.GetOutputPort())
        self.tumor_surface_actor = vtk.vtkActor()
        self.tumor_surface_actor.SetMapper(mapper)
        self.tumor_surface_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
        self.tumor_surface_actor.GetProperty().SetOpacity(0.4)  # Semi-transparent

        # --- Texture Analysis (GLCM) ---
        if SKIMAGE_AVAILABLE:
            try:
                z_slice_areas = np.sum(mask, axis=(1, 2))
                if np.max(z_slice_areas) > 0:
                    best_slice_idx = np.argmax(z_slice_areas)
                    image_slice = self.anatomy_image_data[best_slice_idx, :, :]
                    mask_slice = mask[best_slice_idx, :, :]

                    min_val, max_val = np.min(image_slice), np.max(image_slice)
                    if max_val > min_val:
                        image_slice_8bit = ((image_slice - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
                    else:
                        image_slice_8bit = np.zeros_like(image_slice, dtype=np.uint8)

                    glcm = graycomatrix(image_slice_8bit, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                    glcm_masked = glcm[:, :, 0, 0] * np.sum(mask_slice)  # Use mask to weight
                    
                    self.tumor_metrics['Texture Contrast'] = graycoprops(glcm, 'contrast')[0, 0]
                    self.tumor_metrics['Texture Homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
            except Exception as e:
                print(f"Texture analysis failed: {e}")
                self.tumor_metrics['Texture Contrast'] = 'N/A'
                self.tumor_metrics['Texture Homogeneity'] = 'N/A'

        # --- Create Annotation Actor ---
        annotation_text = "Tumor Metrics:\n\n"
        for key, value in self.tumor_metrics.items():
            if isinstance(value, float):
                if key == 'Volume (cc)':
                    annotation_text += f"{key}: {value:.3f} cc\n"
                elif key == 'Surface Area (mm²)':
                    annotation_text += f"{key}: {value:.2f} mm²\n"
                else:
                    annotation_text += f"{key}: {value:.3f}\n"
            else:
                annotation_text += f"{key}: {value}\n"

        self.annotation_actor = vtk.vtkCornerAnnotation()
        self.annotation_actor.SetLinearFontScaleFactor(2)
        self.annotation_actor.SetNonlinearFontScaleFactor(1)
        self.annotation_actor.SetMaximumFontSize(20)
        self.annotation_actor.SetText(vtk.vtkCornerAnnotation.UpperRight, annotation_text)
        self.annotation_actor.GetTextProperty().SetColor(1.0, 1.0, 0.0)  # Yellow text
        self.annotation_actor.GetTextProperty().SetFontFamilyToArial()
        self.annotation_actor.GetTextProperty().BoldOn()

    def update_anatomy_visualization(self):
        if not self.anatomy_volume_actor.GetMapper():
            return
        if not self.anatomy_vis_dialog:
            contrast_value = 0.5
            ambience_value = 0.2
            opacity_threshold = 0.0
            gradient_opacity_value = 0.5
            specular_power = 10
            scalar_min = -10000
            scalar_max = 10000
            colormap_name = "Grayscale"
            enable_shading = True
            edge_enhancement = 0.2
            tissue_density = 0.5
            enable_annotations = False
            enable_measurement = False
            clip_x = False
            clip_x_pos = 0.5
            clip_y = False
            clip_y_pos = 0.5
            clip_z = False
            clip_z_pos = 0.5
        else:
            contrast_value = self.anatomy_vis_dialog.contrast_slider.value() / 100.0
            ambience_value = self.anatomy_vis_dialog.ambience_slider.value() / 100.0
            opacity_threshold = self.anatomy_vis_dialog.opacity_threshold_slider.value() / 100.0
            gradient_opacity_value = self.anatomy_vis_dialog.gradient_opacity_slider.value() / 100.0
            specular_power = self.anatomy_vis_dialog.specular_power_slider.value()
            scalar_min = self.anatomy_vis_dialog.scalar_min_spinbox.value()
            scalar_max = self.anatomy_vis_dialog.scalar_max_spinbox.value()
            colormap_name = self.anatomy_vis_dialog.colormap_combo.currentText()
            enable_shading = self.anatomy_vis_dialog.shading_toggle.isChecked()
            edge_enhancement = self.anatomy_vis_dialog.edge_enhancement_slider.value() / 100.0
            tissue_density = self.anatomy_vis_dialog.tissue_density_slider.value() / 100.0
            enable_annotations = self.anatomy_vis_dialog.annotation_toggle.isChecked()
            enable_measurement = self.anatomy_vis_dialog.measurement_tool_toggle.isChecked()
            clip_x = self.anatomy_vis_dialog.clipping_x_toggle.isChecked()
            clip_x_pos = self.anatomy_vis_dialog.clipping_x_position.value() / 100.0
            clip_y = self.anatomy_vis_dialog.clipping_y_toggle.isChecked()
            clip_y_pos = self.anatomy_vis_dialog.clipping_y_position.value() / 100.0
            clip_z = self.anatomy_vis_dialog.clipping_z_toggle.isChecked()
            clip_z_pos = self.anatomy_vis_dialog.clipping_z_position.value() / 100.0
        smin, smax = self.anatomy_data_range if self.anatomy_data_range else (0, 1)
        if smin == smax:
            smax = smin + 1
        smin = max(smin, scalar_min)
        smax = min(smax, scalar_max)
        if smin >= smax:
            smin = scalar_min
            smax = scalar_max if scalar_max > scalar_min else scalar_min + 1
        opacity_tf = vtk.vtkPiecewiseFunction()
        threshold_value = smin + opacity_threshold * (smax - smin)
        contrast_range = (smax - smin) * (1.0 - contrast_value * 0.8)
        opacity_tf.AddPoint(smin, 0.0)
        opacity_tf.AddPoint(threshold_value, 0.0)
        opacity_tf.AddPoint(threshold_value + 0.1 * contrast_range, 0.05)
        opacity_tf.AddPoint(threshold_value + 0.3 * contrast_range, 0.3 * tissue_density)
        opacity_tf.AddPoint(smin + contrast_range, 0.8 * tissue_density)
        gradient_opacity_tf = vtk.vtkPiecewiseFunction()
        gradient_opacity_tf.AddPoint(0, 0.0)
        gradient_opacity_tf.AddPoint(10, gradient_opacity_value * (1.0 + edge_enhancement))
        gradient_opacity_tf.AddPoint(255, gradient_opacity_value * (1.0 + edge_enhancement))
        color_tf = self.get_colormap(colormap_name, smin, smax)
        anatomy_volume_property = self.anatomy_volume_actor.GetProperty()
        anatomy_volume_property.SetColor(color_tf)
        anatomy_volume_property.SetScalarOpacity(opacity_tf)
        anatomy_volume_property.SetGradientOpacity(gradient_opacity_tf)
        anatomy_volume_property.SetAmbient(ambience_value)
        anatomy_volume_property.SetDiffuse(0.8)
        anatomy_volume_property.SetSpecular(0.1 + edge_enhancement * 0.2)
        anatomy_volume_property.SetSpecularPower(specular_power)
        if enable_shading:
            anatomy_volume_property.ShadeOn()
        else:
            anatomy_volume_property.ShadeOff()
        if enable_annotations:
            # Placeholder for annotations (e.g., tumor boundary labels)
            pass
        if enable_measurement:
            if self.tumor_surface_actor and not self.tumor_renderer.HasViewProp(self.tumor_surface_actor):
                self.tumor_renderer.AddActor(self.tumor_surface_actor)
            if self.annotation_actor and not self.tumor_renderer.HasViewProp(self.annotation_actor):
                self.tumor_renderer.AddViewProp(self.annotation_actor)
        else:
            if self.tumor_surface_actor and self.tumor_renderer.HasViewProp(self.tumor_surface_actor):
                self.tumor_renderer.RemoveActor(self.tumor_surface_actor)
            if self.annotation_actor and self.tumor_renderer.HasViewProp(self.annotation_actor):
                self.tumor_renderer.RemoveViewProp(self.annotation_actor)
        mapper = self.anatomy_volume_actor.GetMapper()
        mapper.RemoveAllClippingPlanes()
        if clip_x:
            bounds = self.anatomy_volume_actor.GetBounds()
            x_min, x_max = bounds[0], bounds[1]
            clip_x_val = x_min + clip_x_pos * (x_max - x_min)
            self.anatomy_clipping_planes['x'].SetOrigin(clip_x_val, 0, 0)
            self.anatomy_clipping_planes['x'].SetNormal(1, 0, 0)
            mapper.AddClippingPlane(self.anatomy_clipping_planes['x'])
        if clip_y:
            bounds = self.anatomy_volume_actor.GetBounds()
            y_min, y_max = bounds[2], bounds[3]
            clip_y_val = y_min + clip_y_pos * (y_max - y_min)
            self.anatomy_clipping_planes['y'].SetOrigin(0, clip_y_val, 0)
            self.anatomy_clipping_planes['y'].SetNormal(0, 1, 0)
            mapper.AddClippingPlane(self.anatomy_clipping_planes['y'])
        if clip_z:
            bounds = self.anatomy_volume_actor.GetBounds()
            z_min, z_max = bounds[4], bounds[5]
            clip_z_val = z_min + clip_z_pos * (z_max - z_min)
            self.anatomy_clipping_planes['z'].SetOrigin(0, 0, clip_z_val)
            self.anatomy_clipping_planes['z'].SetNormal(0, 0, 1)
            mapper.AddClippingPlane(self.anatomy_clipping_planes['z'])
        self.tumor_vtk_widget.GetRenderWindow().Render()

    def update_volume_visualization(self):
        if not self.volume_actor.GetMapper():
            return
        if not self.volume_vis_dialog:
            contrast_value = 0.5
            ambience_value = 0.2
            opacity_threshold = 0.0
            gradient_opacity_value = 0.5
            specular_power = 10
            scalar_min = -10000
            scalar_max = 10000
            colormap_name = "Grayscale"
            enable_shading = True
            edge_enhancement = 0.2
            tissue_density = 0.5
            enable_annotations = False
            enable_measurement = False
            clip_x = False
            clip_x_pos = 0.5
            clip_y = False
            clip_y_pos = 0.5
            clip_z = False
            clip_z_pos = 0.5
        else:
            contrast_value = self.volume_vis_dialog.contrast_slider.value() / 100.0
            ambience_value = self.volume_vis_dialog.ambience_slider.value() / 100.0
            opacity_threshold = self.volume_vis_dialog.opacity_threshold_slider.value() / 100.0
            gradient_opacity_value = self.volume_vis_dialog.gradient_opacity_slider.value() / 100.0
            specular_power = self.volume_vis_dialog.specular_power_slider.value()
            scalar_min = self.volume_vis_dialog.scalar_min_spinbox.value()
            scalar_max = self.volume_vis_dialog.scalar_max_spinbox.value()
            colormap_name = self.volume_vis_dialog.colormap_combo.currentText()
            enable_shading = self.volume_vis_dialog.shading_toggle.isChecked()
            edge_enhancement = self.volume_vis_dialog.edge_enhancement_slider.value() / 100.0
            tissue_density = self.volume_vis_dialog.tissue_density_slider.value() / 100.0
            enable_annotations = self.volume_vis_dialog.annotation_toggle.isChecked()
            enable_measurement = self.volume_vis_dialog.measurement_tool_toggle.isChecked()
            clip_x = self.volume_vis_dialog.clipping_x_toggle.isChecked()
            clip_x_pos = self.volume_vis_dialog.clipping_x_position.value() / 100.0
            clip_y = self.volume_vis_dialog.clipping_y_toggle.isChecked()
            clip_y_pos = self.volume_vis_dialog.clipping_y_position.value() / 100.0
            clip_z = self.volume_vis_dialog.clipping_z_toggle.isChecked()
            clip_z_pos = self.volume_vis_dialog.clipping_z_position.value() / 100.0
        smin, smax = self.volume_data_range if self.volume_data_range else (0, 1)
        if smin == smax:
            smax = smin + 1
        smin = max(smin, scalar_min)
        smax = min(smax, scalar_max)
        if smin >= smax:
            smin = scalar_min
            smax = scalar_max if scalar_max > scalar_min else scalar_min + 1
        opacity_tf = vtk.vtkPiecewiseFunction()
        threshold_value = smin + opacity_threshold * (smax - smin)
        contrast_range = (smax - smin) * (1.0 - contrast_value * 0.8)
        opacity_tf.AddPoint(smin, 0.0)
        opacity_tf.AddPoint(threshold_value, 0.0)
        opacity_tf.AddPoint(threshold_value + 0.1 * contrast_range, 0.05)
        opacity_tf.AddPoint(threshold_value + 0.3 * contrast_range, 0.3 * tissue_density)
        opacity_tf.AddPoint(smin + contrast_range, 0.8 * tissue_density)
        gradient_opacity_tf = vtk.vtkPiecewiseFunction()
        gradient_opacity_tf.AddPoint(0, 0.0)
        gradient_opacity_tf.AddPoint(10, gradient_opacity_value * (1.0 + edge_enhancement))
        gradient_opacity_tf.AddPoint(255, gradient_opacity_value * (1.0 + edge_enhancement))
        color_tf = self.get_colormap(colormap_name, smin, smax)
        volume_property = self.volume_actor.GetProperty()
        volume_property.SetColor(color_tf)
        volume_property.SetScalarOpacity(opacity_tf)
        volume_property.SetGradientOpacity(gradient_opacity_tf)
        volume_property.SetAmbient(ambience_value)
        volume_property.SetDiffuse(0.8)
        volume_property.SetSpecular(0.1 + edge_enhancement * 0.2)
        volume_property.SetSpecularPower(specular_power)
        if enable_shading:
            volume_property.ShadeOn()
        else:
            volume_property.ShadeOff()
        if enable_annotations:
            # Placeholder for annotations
            pass
        if enable_measurement:
            # Placeholder for measurement tool
            pass
        mapper = self.volume_actor.GetMapper()
        mapper.RemoveAllClippingPlanes()
        if clip_x:
            bounds = self.volume_actor.GetBounds()
            x_min, x_max = bounds[0], bounds[1]
            clip_x_val = x_min + clip_x_pos * (x_max - x_min)
            self.volume_clipping_planes['x'].SetOrigin(clip_x_val, 0, 0)
            self.volume_clipping_planes['x'].SetNormal(1, 0, 0)
            mapper.AddClippingPlane(self.volume_clipping_planes['x'])
        if clip_y:
            bounds = self.volume_actor.GetBounds()
            y_min, y_max = bounds[2], bounds[3]
            clip_y_val = y_min + clip_y_pos * (y_max - y_min)
            self.volume_clipping_planes['y'].SetOrigin(0, clip_y_val, 0)
            self.volume_clipping_planes['y'].SetNormal(0, 1, 0)
            mapper.AddClippingPlane(self.volume_clipping_planes['y'])
        if clip_z:
            bounds = self.volume_actor.GetBounds()
            z_min, z_max = bounds[4], bounds[5]
            clip_z_val = z_min + clip_z_pos * (z_max - z_min)
            self.volume_clipping_planes['z'].SetOrigin(0, 0, clip_z_val)
            self.volume_clipping_planes['z'].SetNormal(0, 0, 1)
            mapper.AddClippingPlane(self.volume_clipping_planes['z'])
        self.volume_vtk_widget.GetRenderWindow().Render()

    def start_processing(self, axial_dir, coronal_dir, sagittal_dir):
        if not axial_dir and not coronal_dir and not sagittal_dir:
            QMessageBox.warning(self, "Input Error", "Please select at least one DICOM series folder.")
            return
        self.status_label.setText("Status: Processing...")
        self.tumor_renderer.RemoveAllViewProps()
        self.volume_renderer.RemoveAllViewProps()
        self.progress_dialog = QProgressDialog("Processing DICOM data...", "Cancel", 0, 200, self)
        self.progress_dialog.setWindowTitle("Working")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setValue(0)
        # Set fixed size for the progress dialog
        self.progress_dialog.setFixedSize(400, 100)  # Set desired width and height
        
        # Center the progress dialog relative to the main window
        main_window_rect = self.geometry()
        dialog_rect = self.progress_dialog.geometry()
        center_x = main_window_rect.x() + (main_window_rect.width() - dialog_rect.width()) // 2
        center_y = main_window_rect.y() + (main_window_rect.height() - dialog_rect.height()) // 2
        self.progress_dialog.move(center_x, center_y)
        
        # Optionally, remove window decorations for a cleaner look
        self.progress_dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        
        # Ensure the dialog stays on top of the main window
        self.progress_dialog.setAttribute(Qt.WA_AlwaysStackOnTop)
        self.tumor_worker = DicomProcessingWorkerTumor(axial_dir, coronal_dir, sagittal_dir)
        self.tumor_worker.progress.connect(self.update_tumor_progress)
        self.tumor_worker.finished.connect(self.on_tumor_processing_finished)
        self.volume_worker = DicomProcessingWorkerVolume(axial_dir, coronal_dir, sagittal_dir)
        self.volume_worker.progress.connect(self.update_volume_progress)
        self.volume_worker.finished.connect(self.on_volume_processing_finished)
        self.progress_dialog.canceled.connect(self.cancel_processing)
        self.tumor_worker.start()
        self.volume_worker.start()
        self.progress_dialog.show()

    def cancel_processing(self):
        for worker in [self.tumor_worker, self.volume_worker]:
            if worker and worker.isRunning():
                worker.requestInterruption()
                worker.quit()
                worker.wait()
        self.status_label.setText("Status: Processing Canceled.")
        if self.progress_dialog:
            self.progress_dialog.close()
        self.tumor_worker = None
        self.volume_worker = None

    def update_tumor_progress(self, value, message):
        if self.progress_dialog:
            current_value = self.progress_dialog.value()
            self.progress_dialog.setValue(max(current_value, value))
            self.progress_dialog.setLabelText(f"Tumor Processing: {message}")

    def update_volume_progress(self, value, message):
        if self.progress_dialog:
            current_value = self.progress_dialog.value()
            self.progress_dialog.setValue(max(current_value, value + 100))
            self.progress_dialog.setLabelText(f"Volume Processing: {message}")

    def on_tumor_processing_finished(self, combined_np_array, tumor_mask_np, image_props, message):
        if combined_np_array is None or not image_props:
            self.status_label.setText(f"Status: Tumor Error - {message}")
            QMessageBox.critical(self, "Tumor Processing Error", message)
            self.tumor_worker = None
            self.check_processing_completion()
            return
        try:
            self.anatomy_image_data = combined_np_array
            self.tumor_mask_np = tumor_mask_np
            self.image_properties = image_props
            nz, ny, nx = combined_np_array.shape
            data_importer = vtk.vtkImageImport()
            data_string = combined_np_array.tobytes('C')
            data_importer.CopyImportVoidPointer(data_string, len(data_string))
            data_importer.SetDataScalarTypeToFloat()
            data_importer.SetNumberOfScalarComponents(1)
            data_importer.SetDataExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
            data_importer.SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
            data_importer.SetDataSpacing(image_props["spacing"])
            data_importer.SetDataOrigin(image_props["origin"])
            data_importer.Update()
            anatomy_vtk_image = data_importer.GetOutput()
            self.anatomy_data_range = (combined_np_array.min(), combined_np_array.max())
            if self.anatomy_vis_dialog:
                self.anatomy_vis_dialog.scalar_min_spinbox.setRange(self.anatomy_data_range[0], self.anatomy_data_range[1])
                self.anatomy_vis_dialog.scalar_max_spinbox.setRange(self.anatomy_data_range[0], self.anatomy_data_range[1])
                self.anatomy_vis_dialog.scalar_min_spinbox.setValue(self.anatomy_data_range[0])
                self.anatomy_vis_dialog.scalar_max_spinbox.setValue(self.anatomy_data_range[1])
            self.update_anatomy_visualization()
            anatomy_mapper = vtk.vtkSmartVolumeMapper()
            anatomy_mapper.SetInputData(anatomy_vtk_image)
            self.anatomy_volume_actor.SetMapper(anatomy_mapper)
            self.tumor_renderer.AddVolume(self.anatomy_volume_actor)
            # --- Calculate advanced features now that data is ready ---
            if self.tumor_mask_np is not None:
                self._calculate_tumor_features()
            self.tumor_renderer.ResetCamera()
            self.tumor_interactor.Initialize()
            self.tumor_vtk_widget.GetRenderWindow().Render()
            self.status_label.setText(f"Status: Tumor - {message}. VTK Volume rendered.")
        except Exception as e:
            self.status_label.setText(f"Status: Tumor Rendering Error - {e}")
            QMessageBox.critical(self, "Tumor Rendering Error", f"An error occurred: {e}\n\nTraceback:\n{traceback.format_exc()}")
            traceback.print_exc()
        self.tumor_worker = None
        self.check_processing_completion()

    def on_volume_processing_finished(self, combined_np_array, image_props, message):
        if combined_np_array is None or not image_props:
            self.status_label.setText(f"Status: Volume Error - {message}")
            QMessageBox.critical(self, "Volume Processing Error", message)
            self.volume_worker = None
            self.check_processing_completion()
            return
        try:
            nz, ny, nx = combined_np_array.shape
            data_importer = vtk.vtkImageImport()
            data_string = combined_np_array.tobytes('C')
            data_importer.CopyImportVoidPointer(data_string, len(data_string))
            data_importer.SetDataScalarTypeToFloat()
            data_importer.SetNumberOfScalarComponents(1)
            data_importer.SetDataExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
            data_importer.SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
            data_importer.SetDataSpacing(image_props["spacing"])
            data_importer.SetDataOrigin(image_props["origin"])
            data_importer.Update()
            vtk_image_data = data_importer.GetOutput()
            self.volume_data_range = (combined_np_array.min(), combined_np_array.max())
            if self.volume_vis_dialog:
                self.volume_vis_dialog.scalar_min_spinbox.setRange(self.volume_data_range[0], self.volume_data_range[1])
                self.volume_vis_dialog.scalar_max_spinbox.setRange(self.volume_data_range[0], self.volume_data_range[1])
                self.volume_vis_dialog.scalar_min_spinbox.setValue(self.volume_data_range[0])
                self.volume_vis_dialog.scalar_max_spinbox.setValue(self.volume_data_range[1])
            self.update_volume_visualization()
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(vtk_image_data)
            self.volume_actor.SetMapper(mapper)
            self.volume_renderer.AddVolume(self.volume_actor)
            self.volume_renderer.ResetCamera()
            self.volume_interactor.Initialize()
            self.volume_vtk_widget.GetRenderWindow().Render()
            self.status_label.setText(f"Status: Volume - {message}. VTK Volume rendered.")
        except Exception as e:
            self.status_label.setText(f"Status: Volume Rendering Error - {e}")
            QMessageBox.critical(self, "Volume Rendering Error", f"An error occurred: {e}\n\nTraceback:\n{traceback.format_exc()}")
            traceback.print_exc()
        self.volume_worker = None
        self.check_processing_completion()

    def check_processing_completion(self):
        if self.tumor_worker is None and self.volume_worker is None:
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None

if __name__ == "__main__":
    try:
        print(f"VTK version: {vtk.vtkVersion().GetVTKVersion()}")
    except Exception as e:
        QMessageBox.critical(None, "VTK Error", f"VTK library not found or could not be imported: {e}. The application cannot run.")
        sys.exit(1)
    app = QApplication(sys.argv)
    viewer = DicomViewer3D()
    viewer.show()
    sys.exit(app.exec_())
