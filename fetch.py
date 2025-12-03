import sys
import requests
from dicomweb_client.api import DICOMwebClient
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, 
                            QListWidget, QHBoxLayout, QDialog, QLineEdit, QPushButton, 
                            QFormLayout, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import io
import pydicom
import os
import tempfile
import shutil
import re

# === Step 1: Authenticate and fetch DICOM data ===
def authenticate(username, password):
    token_url = "https://staging.meningioma.midaspacs.in/auth/realms/midas/protocol/openid-connect/token"
    token_data = {
        "grant_type": "password",
        "client_id": "pacs-rs",
        "client_secret": "dU22uVdEKvR87qeswXvpeRlnsIIBllzW",
        "username": username,
        "password": password
    }
    try:
        token_resp = requests.post(token_url, data=token_data)
        token_resp.raise_for_status()
        return token_resp.json()["access_token"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Authentication failed: {str(e)}")

def fetch_dicom_volume(study_instance_uid, client, temp_dir):
    series_list = client.search_for_series(study_instance_uid)
    if not series_list:
        raise RuntimeError("No series found")
    series_instance_uid = series_list[0]['0020000E']['Value'][0]

    instances = client.search_for_instances(study_instance_uid, series_instance_uid)
    if not instances:
        raise RuntimeError("No instances found")

    slices = []
    instance_numbers = []
    dicom_buffers = []

    for instance in instances:
        sop_instance_uid = instance['00080018']['Value'][0]
        ds = client.retrieve_instance(study_instance_uid, series_instance_uid, sop_instance_uid)
        slices.append(ds.pixel_array)
        instance_number = instance.get('00200013', {}).get('Value', [len(instance_numbers)])[0]
        instance_numbers.append(instance_number)
        
        # Save .dcm content to memory
        buffer = io.BytesIO()
        pydicom.dcmwrite(buffer, ds)
        buffer.seek(0)
        dicom_buffers.append(buffer)
        
        # Save to temporary file using SOP Instance UID as filename
        filename = os.path.join(temp_dir, f"{sop_instance_uid}.dcm")
        pydicom.dcmwrite(filename, ds)

    print(dicom_buffers)
    sorted_slices = [s for _, s in sorted(zip(instance_numbers, slices), key=lambda x: int(x[0]))]
    return np.stack(sorted_slices, axis=0)

# === Step 2: DICOM Viewer Widget ===
class DicomViewer(QWidget):
    def __init__(self, volume):
        super().__init__()
        self.volume = volume
        self.num_slices = volume.shape[0]
        self.current_slice = 0

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.update_image()

    def update_image(self):
        slice_img = self.volume[self.current_slice]
        img_min = slice_img.min()
        img_max = slice_img.max()
        if img_max > img_min:
            img_norm = ((slice_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(slice_img, dtype=np.uint8)

        height, width = img_norm.shape
        qimg = QImage(img_norm.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setWindowTitle(f"DICOM Viewer - Slice {self.current_slice + 1} / {self.num_slices}")

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.current_slice = (self.current_slice + 1) % self.num_slices
        else:
            self.current_slice = (self.current_slice - 1) % self.num_slices
        self.update_image()

    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)

# === Step 3: Login Dialog ===
class LoginDialog(QDialog):
    access_token = None  # Class variable to store the access token

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login")
        self.setFixedSize(300, 150)

        layout = QFormLayout()
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        layout.addRow("Username:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addRow("Password:", self.password_input)

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.handle_login)
        layout.addRow(self.login_button)

        self.setLayout(layout)

    def handle_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        if not username or not password:
            QMessageBox.warning(self, "Input Error", "Please enter both username and password.")
            return
        try:
            LoginDialog.access_token = authenticate(username, password)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Login Failed", str(e))

# === Step 4: Main Window with Patient List ===
class PacsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Patient Viewer")
        self.client = None
        self.studies = []
        self.temp_dir = None

        if LoginDialog.access_token is None:
            login_dialog = LoginDialog()
            if login_dialog.exec_() != QDialog.Accepted:
                QMessageBox.critical(self, "Login Required", "Login is required to proceed.")
                self.close()
                return

        self.access_token = LoginDialog.access_token

        # Setup layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Patient list
        self.patient_list = QListWidget()
        self.patient_list.setMaximumWidth(300)
        self.patient_list.itemClicked.connect(self.load_study)
        main_layout.addWidget(self.patient_list)

        # Viewer container with persistent layout
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        main_layout.addWidget(self.viewer_container)

        # Initialize DICOM client and load studies
        self.init_dicom_client()
        self.load_patient_list()

    def init_dicom_client(self):
        try:
            self.client = DICOMwebClient(
                url="https://staging.meningioma.midaspacs.in/dcm4chee-arc/aets/DCM4CHEE/rs",
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            error_label = QLabel(f"Authentication failed: {str(e)}")
            error_label.setAlignment(Qt.AlignCenter)
            self.viewer_layout.addWidget(error_label)

    def load_patient_list(self):
        try:
            self.studies = self.client.search_for_studies()
            self.patient_list.clear()
            for study in self.studies:
                # Check if study modality is MR or SEG
                modalities = study.get('00080061', {}).get('Value', [])  # 00080061 = Modalities in Study
                if not any(mod in ['MR', 'SEG'] for mod in modalities):
                    continue  # Skip if neither MR nor SEG
                study_uid = study['0020000D']['Value'][0]
                print(study['0020000D'])
                patient_name = study.get('00100010', {}).get('Value', [{'Alphabetic': 'Unknown'}])[0]['Alphabetic']
                study_date = study.get('00080020', {}).get('Value', ['Unknown'])[0]
                patient_id = study.get('00100020', {}).get('Value', ['Unknown'])[0]
                study_description = study.get('00081030', {}).get('Value', ['No description'])[0]
                item_text = f"Name: {patient_name}\nID: {patient_id}\nDate: {study_date}\nDesc: {study_description}"
                self.patient_list.addItem(item_text)
                self.patient_list.item(self.patient_list.count() - 1).setData(Qt.UserRole, study_uid)
        except Exception as e:
            print(f"Error loading studies: {str(e)}")
            error_label = QLabel(f"Error loading studies: {str(e)}")
            error_label.setAlignment(Qt.AlignCenter)
            self.viewer_layout.addWidget(error_label)

    def load_study(self, item):
        study_instance_uid = item.data(Qt.UserRole)

        # Clear previous viewer widgets from layout
        while self.viewer_layout.count():
            child = self.viewer_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clean up previous temp dir if exists
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # Extract patient name from item text
        item_text = item.text()
        lines = item_text.split('\n')
        name_line = lines[0]
        patient_name = name_line.split(': ', 1)[1].strip()

        # Sanitize patient name for directory (replace spaces with underscores, remove invalid chars)
        sanitized_name = re.sub(r'[^\w\-_\.]', '_', patient_name.replace(' ', '_'))
        dir_name = f"PACS_{sanitized_name}"
        self.temp_dir = os.path.join(os.getcwd(), dir_name)

        # Create the directory
        os.makedirs(self.temp_dir, exist_ok=True)

        # Fetch and display volume
        try:
            print(f"Loading study {study_instance_uid}...")
            volume = fetch_dicom_volume(study_instance_uid, self.client, self.temp_dir)
            viewer = DicomViewer(volume)
            self.viewer_layout.addWidget(viewer)
            print(f"Loaded volume with shape {volume.shape}")
        except Exception as e:
            print(f"Error loading study: {str(e)}")
            error_label = QLabel(f"Error loading study: {str(e)}")
            error_label.setAlignment(Qt.AlignCenter)
            self.viewer_layout.addWidget(error_label)

    def closeEvent(self, event):
        # Clean up temp dir on app close
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    
    # Show login dialog
    login_dialog = LoginDialog()
    if login_dialog.exec_() == QDialog.Accepted:
        window = PacsWindow()
        window.resize(1000, 600)
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
