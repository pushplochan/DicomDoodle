<img width="819" height="819" alt="1000142045" src="https://github.com/user-attachments/assets/2879c9b9-9585-46c9-9fb4-6efffd88f607" />


# DicomDoodle - MRI 3D DICOM Image Annotation and Visualization Tool

DicomDoodle is a comprehensive Python-based graphical user interface (GUI) tool designed to streamline the annotation, segmentation, and visualization of DICOM medical images, with a particular focus on MRI brain tumor analysis.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-GPLv3-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)

---

## ✨ Features

### Core Functionality

- **DICOM Image Viewer**: Load and visualize DICOM series from MRI scans with smooth scrolling
- **Automated Segmentation**: Leverage YOLO and SAM (Segment Anything Model) for tumor detection with high accuracy
- **Manual Annotation**: Draw freehand annotations or define custom bounding boxes for precise control
- **Batch Processing**: Annotate multiple slices at once with range selection
- **Color-Coded Annotations**: Use 6 different colors to differentiate between boundary types
- **2D and 3D Visualization**: View segmentation results in 2D with overlaid boundaries
- **3D Model Visualization**: Explore 3D models of tumors using VTK-based rendering
- **DICOM Compatibility**: Save annotations as standard DICOM segmentation files (`_seg.dcm`)
- **Volume Calculation**: Calculate lesion/tumor volumes from annotated DICOM series in three orientations
- **Multi-Frame Conversion**: Convert multi-frame DICOM files to single-frame DICOM files

### Advanced Features

- **Overlay Viewer**: Visualize annotated DICOM images with color-coded overlays
- **3D Model Generation**: Create interactive 3D tumor models with customizable rendering
- **Advanced Shading**: Customizable visualization with lighting and material controls
- **Clipping Planes**: Explore internal structures with multi-axis clipping planes
- **Multiple Colormaps**: Choose from 13 visualization colormaps for optimal viewing

---

## 🖥️ System Requirements

### Minimum Requirements

- **Operating System**: 
  - Windows 8 or higher
  - macOS 
  - Linux 
  
- **Python**: 3.10 or higher
- **RAM**: (8 GB recommended for smooth operation)

### Recommended Requirements

- **GPU**: NVIDIA GPU with CUDA support (for faster inference - optional)

---

## 📦 Installation

### Step 1: Install Python

Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version
git clone https://github.com/pushplochan/DicomDoodle.git
cd DicomDoodle
pip install -r requirements.txt
python DicomDoodle.py
