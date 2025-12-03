import sys
import os
import numpy as np
import pydicom
import cv2
from PyQt5.QtCore import Qt, QTime, QRectF, QPointF, QThread, pyqtSignal
from PIL import Image
import highdicom.seg
from pydicom.uid import generate_uid
from highdicom.seg import SegmentDescription
from PyQt5.QtGui import QCursor,QBrush, QPen, QPixmap, QKeySequence, QColor, QImage, QPainter
from PyQt5.QtWidgets import (QMessageBox, QDialog, QSlider, QScrollBar,QProgressBar, QMenuBar, QMenu, QTextEdit, QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsScene, QGraphicsView, QShortcut, QFileDialog, QSizePolicy, QLabel, QGraphicsRectItem, QGraphicsEllipseItem, QGridLayout, QProgressDialog, QSlider, QComboBox, QGroupBox, QCheckBox, QDoubleSpinBox, QAction, QToolBar, QLineEdit)
from highdicom import AlgorithmIdentificationSequence
from highdicom.seg import Segmentation
from highdicom.sr.coding import CodedConcept
from pydicom.valuerep import PersonName
from pydicom.sr.codedict import codes
import highdicom
from ultralytics import YOLO, YOLOE
from scipy.ndimage import label
from scipy.spatial.distance import cdist

def get_neighbors(y, x, image):
    """Get the 8 neighbors of image[y, x]."""
    return np.array([
        image[y-1, x], image[y-1, x+1], image[y, x+1], image[y+1, x+1],
        image[y+1, x], image[y+1, x-1], image[y, x-1], image[y-1, x-1]
    ])

def get_transitions(p):
    """Count the number of 0-to-1 transitions around the neighbors."""
    return np.sum((p[:-1] == 0) & (p[1:] == 1)) + int((p[-1] == 0) & (p[0] == 1))

def thin(binary_image):
    """Apply Zhang-Suen thinning to make the binary image one-pixel thick."""
    image = binary_image.copy().astype(int)
    while True:
        prev = image.copy()
        # Step 1
        to_delete = []
        rows, cols = image.shape
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if image[y, x] == 0:
                    continue
                p = get_neighbors(y, x, image)
                n = np.sum(p)
                t = get_transitions(p)
                if 2 <= n <= 6 and t == 1 and (p[0] * p[2] * p[4] == 0) and (p[2] * p[4] * p[6] == 0):
                    to_delete.append((y, x))
        for y, x in to_delete:
            image[y, x] = 0
        # Step 2
        to_delete = []
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if image[y, x] == 0:
                    continue
                p = get_neighbors(y, x, image)
                n = np.sum(p)
                t = get_transitions(p)
                if 2 <= n <= 6 and t == 1 and (p[0] * p[2] * p[6] == 0) and (p[0] * p[4] * p[6] == 0):
                    to_delete.append((y, x))
        for y, x in to_delete:
            image[y, x] = 0
        if np.all(prev == image):
            break
    return image

def draw_line(img, p1, p2):
    """Draw a straight line between points p1 and p2 on a binary image."""
    y1, x1 = p1
    y2, x2 = p2
    points = []
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return points
    x_inc = dx / steps
    y_inc = dy / steps
    x, y = x1, y1
    for _ in range(steps + 1):
        points.append((int(round(y)), int(round(x))))
        x += x_inc
        y += y_inc
    return points

def fill_discontinuities(img):
    
    binary = (img > 0).astype(int)
    
    # Find endpoints (pixels with exactly one neighbor)
    endpoints = []
    rows, cols = binary.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if binary[y, x] == 1 and np.sum(get_neighbors(y, x, binary)) == 1:
                endpoints.append((y, x))
    
    if len(endpoints) < 2:
        return img  # No gaps to fill
    
    # Convert endpoints to array for distance computation
    endpoints = np.array(endpoints)
    
    # Compute pairwise distances
    distances = cdist(endpoints, endpoints)
    
    # Create a mask for new connections
    connection_mask = np.zeros_like(binary)
    
    # Track used endpoints to avoid reusing
    used = set()
    endpoint_intensities = {}
    
    for i, (y1, x1) in enumerate(endpoints):
        if i in used:
            continue
        # Find nearest unused endpoint within 5 pixels
        valid_distances = distances[i].copy()
        valid_distances[list(used) + [i]] = np.inf
        j = np.argmin(valid_distances)
        if valid_distances[j] > 20:  # Skip if gap is too large
            continue
        y2, x2 = endpoints[j]
        
        # Draw line between endpoints
        line_points = draw_line(binary, (y1, x1), (y2, x2))
        for y, x in line_points:
            if 0 <= y < rows and 0 <= x < cols:
                connection_mask[y, x] = 1
        used.add(i)
        used.add(j)
        
        # Store intensity of the first endpoint
        endpoint_intensities[(y1, x1)] = img[y1, x1]
        endpoint_intensities[(y2, x2)] = img[y2, x2]
    
    # Combine original and new connections
    combined = np.maximum(binary, connection_mask)
    
    # Thin to one-pixel thickness
    thinned = thin(combined)
    
    # Assign intensities: original pixels keep their intensity, new pixels get nearest endpoint intensity
    output = np.zeros_like(img)
    output[thinned > 0] = np.mean(img[img > 0])  # Fallback intensity
    output[binary > 0] = img[binary > 0]  # Restore original intensities
    
    # For new pixels, assign intensity from the nearest endpoint
    new_pixels = (thinned > 0) & (binary == 0)
    if np.any(new_pixels):
        new_coords = np.where(new_pixels)
        new_coords = np.array(list(zip(new_coords[0], new_coords[1])))
        endpoint_coords = np.array(list(endpoint_intensities.keys()))
        distances = cdist(new_coords, endpoint_coords)
        nearest_indices = np.argmin(distances, axis=1)
        for idx, (y, x) in enumerate(new_coords):
            nearest_endpoint = endpoint_coords[nearest_indices[idx]]
            output[y, x] = endpoint_intensities[tuple(nearest_endpoint)]
    
    return output

def intensity_to_color(gt_value):
        """ Map grayscale value to a specific color (Red, Green, Blue, Yellow) """
        if gt_value == 0:
         return [0, 0, 0]  # Black for background
        elif gt_value == 1:
         return [255, 0, 0]  # Red
        elif gt_value == 2:
         return [0, 255, 0]  # Green
        elif gt_value == 3:
         return [0, 0, 255]  # Blue
        elif gt_value == 4:
         return [255, 255, 0]  
        elif gt_value == 5:
         return [0, 255, 255]  
        elif gt_value == 6:
         return [255, 0, 255]  
        else:
         return [0, 0, 0]  # Black for any unexpected values
def intensity_to_label(gt_value):
        """ Map grayscale value to a specific color (Red, Green, Blue, Yellow) """
        if gt_value == "No Boundary":
         return 0  # Black for background
        elif gt_value == "Tumor & Brain Boundary":
         return 1  # Red
        elif gt_value == "Tumor & Dura Boundary":
         return 2  # Green
        elif gt_value == "Tumor & Necrosis Boundary":
         return 3  # Blue
        elif gt_value == "Edema & Brain Boundary":
         return 4  
        elif gt_value == "Tumor & Edema Boundary":
         return 5  
        elif gt_value == "No Boundary":
         return 6  
        else:
         return 0  # Black for any unexpected values
        
def np2pixmap(np_img):
    height, width = np_img.shape[:2]
    bytesPerLine = 3 * width if np_img.ndim == 3 else width
    format_ = QImage.Format_RGB888 if np_img.ndim == 3 else QImage.Format_Grayscale8
    qImg = QImage(np_img.data, width, height, bytesPerLine, format_)
    return QPixmap.fromImage(qImg)

def dicom_to_png(dicom_file_path):
    ds = pydicom.dcmread(dicom_file_path)
    img_array = ds.pixel_array
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255
    img_array = img_array.astype(np.uint8)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    return img_rgb

def get_yolo_box(dicom_file_path):
    model = YOLO(yolo_path)
    image = dicom_to_png(dicom_file_path)
    results = model.predict(image)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
        return (y1, x1, y2, x2)
    else:
        return None

def get_yoloe_box(dicom_file_path):
    model = YOLOE(yoloe_path)
    image = dicom_to_png(dicom_file_path)
    results = model.predict(image)
    boxes = results[0].boxes
    
    max_avg_intensity = -1
    tumor_box = None

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[0].xyxy[0])
        
        # Extract the region of interest (ROI) within the bounding box
        roi = image[y1:y2, x1:x2]
        if roi.size > 0:  # Ensure ROI is valid
            # Calculate the average pixel intensity in the ROI
            avg_intensity = np.mean(roi)
            
            # Update tumor box if this box has higher intensity
            if avg_intensity > max_avg_intensity:
                max_avg_intensity = avg_intensity
                tumor_box = (y1, x1, y2, x2)
    
    if tumor_box is None:
        print("No valid tumor box found based on intensity.")
        return None
    
    return tumor_box

def get_data_path(filename):
    if getattr(sys, 'frozen', False):
        # Running as a bundled executable
        application_path = sys._MEIPASS
    else:
        # Running as a script
        application_path = os.path.dirname(__file__)

    return os.path.join(application_path, filename)

yolo_path = get_data_path('best_flair.pt')
yoloe_path = get_data_path('yoloe-11l-seg-pf.pt')
def create_dicom_segmentation(gt_rgb_slice, dicom_path, output_path, slice_index):
    """
    Create DICOM Segmentation (SEG) file from ground truth segmentation colors, excluding black.
    Creates a blank segmentation if no matching colors are found in the slice.
    """
    # Load the source DICOM file
    try:
        source_image = pydicom.dcmread(dicom_path)
    except Exception as e:
        raise ValueError(f"Could not read source DICOM file: {e}")

    # Define the intensity mappings (ground truth intensity to label mapping)
    intensity_mapping = {
        1: ("Tumor & Brain Boundary", "TUMOR"),
        2: ("Tumor & Dura Boundary", "DURA"),
        3: ("Tumor & Necrosis Boundary", "NECROSIS"),
        4: ("Edema & Brain Boundary", "EDEMA"),
        5: ("Tumor & Edema Boundary", "EDEMA"),
        6: ("Magenta Boundary", "MAGENTA"),
    }

    # Prepare segmentation masks and segment descriptions
    # Check if the slice contains any non-zero values
    gt_rgb_slice = fill_discontinuities(gt_rgb_slice)
    is_non_zero_present = np.any(gt_rgb_slice != 0)
    segmentation_masks = []
    segment_descriptions = []
    #print(gt_rgb_slice)

   

    if is_non_zero_present:
        # Process each intensity level (1-6) in the mapping
        segment_number = 1
        for intensity, (label, category) in intensity_mapping.items():
         # Create a binary mask where non-zero values represent segmentation regions
         mask = (gt_rgb_slice == intensity).astype(np.uint8)
         #mask = gt_rgb_slice #np.zeros_like(gt_rgb_slice, dtype=np.uint8)
         #mask[gt_rgb_slice != 0]  # Set non-zero pixels as part of the segmentation
         #print(mask)
         if np.any(mask):  # Only add the segment if there are any non-zero pixels
            segmentation_masks.append(mask)

            algorithm_identification = AlgorithmIdentificationSequence(
                name='Segmentation Algorithm',
                version='v1.0',
                family=codes.cid7162.ArtificialIntelligence
            )
            segment_descriptions.append(SegmentDescription(
                segment_number=segment_number,
                segment_label=label,
                segmented_property_category=codes.cid7150.Tissue,
                segmented_property_type=codes.cid7166.ConnectiveTissue,
                algorithm_type=highdicom.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm_identification,
                tracking_uid=highdicom.UID(),
                tracking_id=f"segmentation-{segment_number}"
            ))
            segment_number += 1
        

    # Handle case where no segmentation colors are found
    else:
        print(f"No segmentation colors found for slice {slice_index}. Creating a blank segmentation.")
        mask = gt_rgb_slice #np.zeros_like(gt_rgb_slice, dtype=np.uint8)
        segmentation_masks.append(mask)
        algorithm_identification = AlgorithmIdentificationSequence(
                name='Segmentation Algorithm',
                version='v1.0',
                family=codes.cid7162.ArtificialIntelligence
            )
        segment_descriptions.append(SegmentDescription(
            segment_number=1,
            segment_label="No Segmentation",
            segmented_property_category=CodedConcept(value='91772007', scheme_designator='SCT', meaning='Organ'),
            segmented_property_type=CodedConcept(value='12738006', scheme_designator='SCT', meaning='Brain'),
            algorithm_type=highdicom.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification,
            tracking_uid=highdicom.UID(),
            tracking_id="segmentation"
        ))

    # Combine masks into a single 3D array
    pixel_array = np.zeros_like(gt_rgb_slice, dtype=np.uint8)
    for idx, mask in enumerate(segmentation_masks):
        pixel_array[mask > 0] = idx + 1  # Label each segment with a unique integer

    # Set default values for missing attributes before passing to Segmentation
    if "PatientName" not in source_image:
        source_image.PatientName = "Unknown^Patient"
    else:
        source_image.PatientName = str(source_image.PatientName) + "^" if "^" not in source_image.PatientName else source_image.PatientName
    if "PatientID" not in source_image:
        source_image.PatientID = "Unknown"
    if "PatientBirthDate" not in source_image:
        source_image.PatientBirthDate = "20121219"  # Default birth date
    if "PatientSex" not in source_image:
        source_image.PatientSex = "O"  # Other
    if "StudyInstanceUID" not in source_image:
        source_image.StudyInstanceUID = "1.2.3.4"  # Default UID
    if "StudyID" not in source_image:
        source_image.StudyID = "Unknown"
    if "AccessionNumber" not in source_image:
        source_image.AccessionNumber = "Unknown"  # Default Accession Number
    if "StudyDate" not in source_image:
        source_image.StudyDate = "19000101"  # Default date (January 1, 1900)
    if "StudyTime" not in source_image:
        source_image.StudyTime = "000000"  # Default time (midnight)
    if "FrameOfReferenceUID" not in source_image:
        source_image.FrameOfReferenceUID = str(uuid.uuid4())
    if "PixelSpacing" not in source_image:
        raise ValueError("Missing PixelSpacing in the source image.")
    if "ImageOrientationPatient" not in source_image:
        raise ValueError("Missing ImageOrientationPatient in the source image.")
    if "ImagePositionPatient" not in source_image:
        raise ValueError("Missing ImagePositionPatient in the source image.")
    if "DeviceSerialNumber" not in source_image:
        source_image.DeviceSerialNumber = "12345"
    if "SoftwareVersions" not in source_image:
        source_image.DeviceSerialNumber = "1.0"

    # Create DICOM SEG object
    seg = highdicom.seg.Segmentation(
        source_images=[source_image],  # Pass the modified source DICOM
        pixel_array=pixel_array,
        segmentation_type=highdicom.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=generate_uid(),
        series_number=int(source_image.InstanceNumber), #int(slice_index)+1,
        sop_instance_uid=generate_uid(),
        instance_number=int(source_image.InstanceNumber), #slice_index + 1,
        manufacturer=source_image.Manufacturer, #"YourInstitution",
        manufacturer_model_name=source_image.ManufacturerModelName, #"ModelName",  # Replace with your model name
        device_serial_number=source_image.DeviceSerialNumber, #"12345",  # Replace with your device serial number
        software_versions=source_image.SoftwareVersions, #"1.0",  # Replace with your software version
    )
    # Explicitly set the Transfer Syntax UID for compatibility
    seg.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Save the DICOM SEG file
    seg.save_as(output_path)
    print(f"Saved DICOM Segmentation to: {output_path}")

class ResultsViewer(QWidget):
    def __init__(self, blended_img, edge_overlay, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation Results")
        self.setGeometry(200, 200, 1000, 500)
        
        # Convert images to QPixmap
        self.blended_pixmap = np2pixmap(np.array(blended_img))
        self.edge_pixmap = np2pixmap(np.array(edge_overlay))
        
        # Create layout
        layout = QHBoxLayout()
        
        # Create scenes and views
        self.scene1 = QGraphicsScene()
        self.view1 = QGraphicsView(self.scene1)
        self.scene2 = QGraphicsScene()
        self.view2 = QGraphicsView(self.scene2)
        
        # Add images to scenes
        self.scene1.addPixmap(self.blended_pixmap)
        self.scene2.addPixmap(self.edge_pixmap)
        
        # Set view properties
        for view in (self.view1, self.view2):
            view.setRenderHint(QPainter.Antialiasing)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.fitInView(view.scene().sceneRect(), Qt.KeepAspectRatio)
        
        # Add labels
        label1 = QLabel("Original Image with Segmented Tumor")
        label1.setAlignment(Qt.AlignCenter)
        label2 = QLabel("Original Image with Tumor Boundary")
        label2.setAlignment(Qt.AlignCenter)
        
        # Create vertical layouts for each view
        vlayout1 = QVBoxLayout()
        vlayout1.addWidget(label1)
        vlayout1.addWidget(self.view1)
        
        vlayout2 = QVBoxLayout()
        vlayout2.addWidget(label2)
        vlayout2.addWidget(self.view2)
        
        # Add to main layout
        layout.addLayout(vlayout1)
        layout.addLayout(vlayout2)
        
        self.setLayout(layout)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.view1.fitInView(self.scene1.sceneRect(), Qt.KeepAspectRatio)
        self.view2.fitInView(self.scene2.sceneRect(), Qt.KeepAspectRatio)

# Optional: test/debug when run directly
if __name__ == "__main__":
    print(f"Not Allowed")
