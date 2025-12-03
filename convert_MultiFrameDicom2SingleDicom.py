import sys
import pydicom
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QSlider, QMessageBox, QProgressDialog)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QCoreApplication
import os
import copy
from pydicom import Dataset, Sequence
# We need to subclass QLabel to override the mouse wheel event
class DicomImageLabel(QLabel):
    """A QLabel subclass to handle mouse wheel events for slice scrolling."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent

    def wheelEvent(self, event):
        """Overrides the wheelEvent to scroll through DICOM slices."""
        if self.parent_viewer and self.parent_viewer.dcm_data:
            delta = event.angleDelta().y()
            if delta > 0:
                self.parent_viewer.scroll_slices(1)
            elif delta < 0:
                self.parent_viewer.scroll_slices(-1)
        event.accept()

class Multi_frame_DicomViewer(QMainWindow):
    """A simple DICOM viewer for multi-frame DICOM files."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D DICOM Viewer")
        self.setGeometry(100, 100, 288, 288)

        # --- Data Attributes ---
        self.dcm_data = None
        self.pixel_array = None
        self.current_slice_index = 0
        self.total_slices = 0
        self.original_file_path = None

        # --- Main Widget and Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- Widgets ---
        self.btn_open = QPushButton("Open DICOM File")
        self.btn_open.setFont(QFont("Arial", 12))
        
        self.btn_save_slices = QPushButton("Save Slices")
        self.btn_save_slices.setFont(QFont("Arial", 12))
        self.btn_save_slices.setEnabled(False)
        
        self.image_label = DicomImageLabel(self)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setEnabled(False)
        
        self.info_label = QLabel("Please open a DICOM file.")
        self.info_label.setFont(QFont("Arial", 11))
        self.info_label.setAlignment(Qt.AlignCenter)
        
        # --- Add Widgets to Layout ---
        self.layout.addWidget(self.btn_open)
        self.layout.addWidget(self.btn_save_slices)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.slice_slider)
        self.layout.addWidget(self.info_label)

        # --- Connect Signals to Slots ---
        self.btn_open.clicked.connect(self.open_dicom_file)
        self.btn_save_slices.clicked.connect(self.save_slices_as_dicom)
        self.slice_slider.valueChanged.connect(self.slider_changed)

    def open_dicom_file(self):
        """Opens a file dialog to select a DICOM file and loads it."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM File", "", "DICOM Files (*.dcm);;All Files (*)", options=options
        )
        
        if file_path:
            try:
                self.dcm_data = pydicom.dcmread(file_path)
                
                if not hasattr(self.dcm_data, 'pixel_array') or self.dcm_data.pixel_array.ndim < 3:
                    self.show_error_message("This is not a multi-frame 3D DICOM file.")
                    self.dcm_data = None
                    return

                self.pixel_array = self.dcm_data.pixel_array
                self.total_slices = self.pixel_array.shape[0]
                self.original_file_path = file_path
                
                self.slice_slider.setMinimum(0)
                self.slice_slider.setMaximum(self.total_slices - 1)
                self.slice_slider.setEnabled(True)
                
                self.btn_save_slices.setEnabled(True)
                self.slice_slider.setValue(self.total_slices // 2)
                
            except Exception as e:
                self.show_error_message(f"Could not read or process the DICOM file.\nError: {e}")
                self.dcm_data = None
                self.btn_save_slices.setEnabled(False)

    def save_slices_as_dicom(self):
        """Saves each slice as an individual DICOM file with metadata similar to Metadata 1."""
        if not self.dcm_data or not self.original_file_path:
            self.show_error_message("No DICOM file loaded.")
            return

        try:
            # Get the directory and base name of the original file
            directory = os.path.dirname(self.original_file_path)
            base_name = os.path.splitext(os.path.basename(self.original_file_path))[0]
            # Create a new folder with the base name
            output_folder = os.path.join(directory, base_name)
            os.makedirs(output_folder, exist_ok=True)

            # Create a progress dialog with percentage display
            progress = QProgressDialog("Saving slices ...", "Cancel", 0, self.total_slices, self)
            progress.setWindowTitle("Saving Slices")
            progress.setWindowModality(Qt.WindowModal)  # Modal to main window but allows interaction
            progress.setMinimumDuration(0)  # Show immediately
            progress.setAutoClose(True)  # Automatically close when complete
            progress.setAutoReset(True)  # Reset when complete
            progress.show()
            QApplication.processEvents()

            # Get consistent metadata from the source dataset
            series_instance_uid = self.dcm_data.get('SeriesInstanceUID', pydicom.uid.generate_uid())
            frame_of_reference_uid = self.dcm_data.get('FrameOfReferenceUID', pydicom.uid.generate_uid())
            pixel_spacing = self.dcm_data.get('PixelSpacing', [1.0, 1.0])
            slice_thickness = self.dcm_data.get('SliceThickness', 1.0)
            image_orientation = self.dcm_data.get('ImageOrientationPatient', [1, 0, 0, 0, 1, 0])
            first_position = self.dcm_data.get('ImagePositionPatient', [0.0, 0.0, 0.0])

            # If PerFrameFunctionalGroupsSequence exists, use it for initial metadata
            if (hasattr(self.dcm_data, 'PerFrameFunctionalGroupsSequence') and 
                self.dcm_data.PerFrameFunctionalGroupsSequence and
                len(self.dcm_data.PerFrameFunctionalGroupsSequence) > 0):
                frame_0 = self.dcm_data.PerFrameFunctionalGroupsSequence[0]
                if hasattr(frame_0, 'PlaneOrientationSequence') and frame_0.PlaneOrientationSequence:
                    image_orientation = frame_0.PlaneOrientationSequence[0].ImageOrientationPatient
                if hasattr(frame_0, 'PlanePositionSequence') and frame_0.PlanePositionSequence:
                    first_position = frame_0.PlanePositionSequence[0].ImagePositionPatient
                if hasattr(frame_0, 'PixelMeasuresSequence') and frame_0.PixelMeasuresSequence:
                    pixel_spacing = frame_0.PixelMeasuresSequence[0].get('PixelSpacing', pixel_spacing)
                    slice_thickness = frame_0.PixelMeasuresSequence[0].get('SliceThickness', slice_thickness)

            # Calculate the normal vector for slice progression
            orientation = np.array(image_orientation).reshape(2, 3)
            normal = np.cross(orientation[0], orientation[1])  # Cross product of row and column vectors
            normal = normal / np.linalg.norm(normal)  # Normalize

            for slice_idx in range(self.total_slices):
                # Update progress bar
                progress.setValue(slice_idx)
                if progress.wasCanceled():
                    progress.close()
                    QMessageBox.information(self, "Cancelled", "Slice saving was cancelled.")
                    return
                new_dcm = copy.deepcopy(self.dcm_data)

                # Update pixel data
                new_dcm.PixelData = self.pixel_array[slice_idx].tobytes()
                new_dcm.Rows, new_dcm.Columns = self.pixel_array[slice_idx].shape

                # Remove NumberOfFrames if present to match classic single-frame
                if 'NumberOfFrames' in new_dcm:
                    del new_dcm.NumberOfFrames

                # Remove multi-frame sequences to match Metadata 1
                for seq in [
                    'PerFrameFunctionalGroupsSequence',
                    'SharedFunctionalGroupsSequence',
                    'FrameContentSequence',
                    'PlanePositionSequence',
                    'PlaneOrientationSequence',
                    'PixelMeasuresSequence',
                    'FrameVOILUTSequence',
                    'MREchoSequence',
                    'MRAveragesSequence',
                    'MRImageFrameTypeSequence',
                    'PixelValueTransformationSequence'
                ]:
                    if seq in new_dcm:
                        del new_dcm[seq]

                # Extract slice-specific metadata from PerFrameFunctionalGroupsSequence
                position_number = slice_idx + 1
                image_position = first_position
                window_center = self.dcm_data.get('WindowCenter', '344')
                window_width = self.dcm_data.get('WindowWidth', '565')
                slice_location = str(float(first_position[2]) + slice_idx * float(slice_thickness))

                if (hasattr(self.dcm_data, 'PerFrameFunctionalGroupsSequence') and 
                    len(self.dcm_data.PerFrameFunctionalGroupsSequence) > slice_idx):
                    frame_data = self.dcm_data.PerFrameFunctionalGroupsSequence[slice_idx]
                    if hasattr(frame_data, 'PlanePositionSequence') and frame_data.PlanePositionSequence:
                        image_position = frame_data.PlanePositionSequence[0].ImagePositionPatient
                    else:
                        # Calculate position
                        z_offset = slice_idx * float(slice_thickness)
                        image_position = [
                            float(first_position[0]) + normal[0] * z_offset,
                            float(first_position[1]) + normal[1] * z_offset,
                            float(first_position[2]) + normal[2] * z_offset
                        ]
                    if hasattr(frame_data, 'PixelMeasuresSequence') and frame_data.PixelMeasuresSequence:
                        pixel_spacing = frame_data.PixelMeasuresSequence[0].get('PixelSpacing', pixel_spacing)
                        slice_thickness = frame_data.PixelMeasuresSequence[0].get('SliceThickness', slice_thickness)
                    if hasattr(frame_data, 'FrameVOILUTSequence') and frame_data.FrameVOILUTSequence:
                        window_center = frame_data.FrameVOILUTSequence[0].WindowCenter
                        window_width = frame_data.FrameVOILUTSequence[0].WindowWidth
                    if hasattr(frame_data, 'FrameContentSequence') and frame_data.FrameContentSequence:
                        position_number = frame_data.FrameContentSequence[0].InStackPositionNumber
                        slice_location = str(image_position[2])

                # Set top-level metadata to match Metadata 1
                new_dcm.ImagePositionPatient = image_position
                new_dcm.ImageOrientationPatient = image_orientation
                new_dcm.PixelSpacing = pixel_spacing
                new_dcm.SliceThickness = slice_thickness
                new_dcm.WindowCenter = window_center
                new_dcm.WindowWidth = window_width
                new_dcm.InstanceNumber = str(position_number)
                new_dcm.SliceLocation = slice_location

                # Set ImageType to match Metadata 1
                new_dcm.ImageType = ['DERIVED', 'SECONDARY', 'REFORMATTED', 'AVERAGE']

                # Ensure consistent UIDs
                new_dcm.SeriesInstanceUID = series_instance_uid
                new_dcm.FrameOfReferenceUID = frame_of_reference_uid
                new_dcm.SOPInstanceUID = pydicom.uid.generate_uid()
                if 'DeviceSerialNumber' in self.dcm_data:
                    new_dcm.DeviceSerialNumber = self.dcm_data.DeviceSerialNumber

                # Update File Meta Information to match Metadata 1
                new_dcm.file_meta = pydicom.dataset.FileMetaDataset()
                new_dcm.file_meta.FileMetaInformationGroupLength = self.dcm_data.file_meta.get('FileMetaInformationGroupLength', 194)
                new_dcm.file_meta.FileMetaInformationVersion = b'\x00\x01'
                new_dcm.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
                new_dcm.file_meta.MediaStorageSOPInstanceUID = new_dcm.SOPInstanceUID
                new_dcm.file_meta.TransferSyntaxUID = self.dcm_data.file_meta.get('TransferSyntaxUID', '1.2.840.10008.1.2.1')
                new_dcm.file_meta.ImplementationClassUID = self.dcm_data.file_meta.get('ImplementationClassUID', '1.2.276.0.7230010.3.0.3.5.3')
                new_dcm.file_meta.ImplementationVersionName = self.dcm_data.file_meta.get('ImplementationVersionName', 'OFFIS_DCMTK_353')

                # Ensure critical image tags
                new_dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
                new_dcm.SamplesPerPixel = self.dcm_data.get('SamplesPerPixel', 1)
                new_dcm.PhotometricInterpretation = self.dcm_data.get('PhotometricInterpretation', 'MONOCHROME2')
                new_dcm.BitsAllocated = self.dcm_data.get('BitsAllocated', 16)
                new_dcm.BitsStored = self.dcm_data.get('BitsStored', 16)
                new_dcm.HighBit = self.dcm_data.get('HighBit', 15)
                new_dcm.PixelRepresentation = self.dcm_data.get('PixelRepresentation', 1)

                # Copy sequence items from source
                for tag in [
                    'ProcedureCodeSequence', 'ReferencedStudySequence', 'ContributingEquipmentSequence',
                    'RequestAttributesSequence'
                ]:
                    if tag in self.dcm_data:
                        setattr(new_dcm, tag, copy.deepcopy(getattr(self.dcm_data, tag)))

                # Copy private tags from source (including Siemens-specific tags)
                for group in [0x0009, 0x0019, 0x0021, 0x0025, 0x0027, 0x0029, 0x0043, 0x0903, 0x0905, 0x7FD1]:
                    for elem in self.dcm_data.group_dataset(group):
                        if elem.tag not in new_dcm:
                            new_dcm.add(elem)

                # Set output file name
                output_filename = f"{base_name}_{position_number}.dcm"
                output_path = os.path.join(output_folder, output_filename)

                # Save the new DICOM file
                new_dcm.save_as(output_path)
            progress.setValue(self.total_slices)
            progress.close()
            QMessageBox.information(self, "Success", f"Saved {self.total_slices} slices as individual DICOM files in {directory}.")

        except Exception as e:
            progress.close()
            self.show_error_message(f"Could not save slices.\nError: {e}")

    def update_view(self):
        """Updates the image label and info label for the current slice."""
        if self.pixel_array is not None:
            slice_2d = self.pixel_array[self.current_slice_index, :, :]
            slice_normalized = self.normalize_pixel_data(slice_2d)
            height, width = slice_normalized.shape
            q_image = QImage(slice_normalized.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            )
            self.update_info_label()

    def normalize_pixel_data(self, pixel_data):
        """Normalize pixel data to a 0-255 range for 8-bit display."""
        wc = None
        ww = None
        if 'WindowCenter' in self.dcm_data and 'WindowWidth' in self.dcm_data:
            wc = self.dcm_data.WindowCenter
            ww = self.dcm_data.WindowWidth
        elif (hasattr(self.dcm_data, 'PerFrameFunctionalGroupsSequence') and
              len(self.dcm_data.PerFrameFunctionalGroupsSequence) > self.current_slice_index):
            frame_info = self.dcm_data.PerFrameFunctionalGroupsSequence[self.current_slice_index]
            if hasattr(frame_info, 'FrameVOILUTSequence') and frame_info.FrameVOILUTSequence:
                voi_lut_info = frame_info.FrameVOILUTSequence[0]
                if 'WindowCenter' in voi_lut_info and 'WindowWidth' in voi_lut_info:
                    wc = voi_lut_info.WindowCenter
                    ww = voi_lut_info.WindowWidth
        if wc is not None and ww is not None:
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = wc[0]
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = ww[0]
            min_val = wc - ww / 2
            max_val = wc + ww / 2
        else:
            min_val = np.min(pixel_data)
            max_val = np.max(pixel_data)
        if max_val == min_val:
            return np.zeros(pixel_data.shape, dtype=np.uint8)
        windowed_data = np.clip(pixel_data, min_val, max_val)
        normalized_data = ((windowed_data - min_val) / (max_val - min_val)) * 255.0
        return normalized_data.astype(np.uint8)

    def update_info_label(self):
        """Updates the text label with current slice and DICOM metadata."""
        if self.dcm_data:
            patient_name = self.dcm_data.get("PatientName", "N/A")
            study_date = self.dcm_data.get("StudyDate", "N/A")
            rows = self.dcm_data.get("Rows", "N/A")
            cols = self.dcm_data.get("Columns", "N/A")
            position_number = self.current_slice_index + 1
            if (hasattr(self.dcm_data, 'PerFrameFunctionalGroupsSequence') and
                len(self.dcm_data.PerFrameFunctionalGroupsSequence) > self.current_slice_index and
                hasattr(self.dcm_data.PerFrameFunctionalGroupsSequence[self.current_slice_index], 'FrameContentSequence') and
                self.dcm_data.PerFrameFunctionalGroupsSequence[self.current_slice_index].FrameContentSequence):
                position_number = self.dcm_data.PerFrameFunctionalGroupsSequence[self.current_slice_index].FrameContentSequence[0].InStackPositionNumber
            info_text = (
                f"Patient: {patient_name} | Study Date: {study_date}\n"
                f"Dimensions: {cols} x {rows} | "
                f"Slice_position: {position_number}\n"
                f"Slice: {self.current_slice_index + 1} / {self.total_slices}"
            )
            self.info_label.setText(info_text)

    def slider_changed(self, value):
        """Handles the slider's valueChanged signal."""
        if self.dcm_data is not None:
            self.current_slice_index = value
            self.update_view()

    def scroll_slices(self, direction):
        """Scrolls through slices by changing the slider's value."""
        if self.dcm_data:
            current_value = self.slice_slider.value()
            self.slice_slider.setValue(current_value + direction)

    def show_error_message(self, message):
        """Displays an error message in a dialog box."""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_() 

if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    viewer = Multi_frame_DicomViewer()
    viewer.show()
    sys.exit(app.exec_())
