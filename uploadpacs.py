import sys
import os
import requests
import pydicom
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QLineEdit, QFormLayout, QDialog, QMessageBox
)
from dicomweb_client.api import DICOMwebClient


# === Authentication ===
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


# === Upload DICOMs ===
def upload_dicom_folder(client, folder_path):
    if not os.path.isdir(folder_path):
        raise Exception("Invalid folder path")

    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]

    if not dicom_files:
        raise Exception("No DICOM files found in the folder")

    datasets = []
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(path)
            sop_class_uid = ds.SOPClassUID.name if "SOPClassUID" in ds else "Unknown"
            print(f"Read file: {os.path.basename(path)} | SOP Class: {sop_class_uid}")
            datasets.append(ds)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not datasets:
        raise Exception("No valid DICOM files could be read")

    try:
        client.store_instances(datasets)
        print(f"Successfully uploaded {len(datasets)} DICOM files.")
    except Exception as e:
        raise Exception(f"Upload failed: {e}")


# === Login Dialog ===
class LoginDialog(QDialog):
    access_token = None  # Class variable to store the access token

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login to PACS")
        self.setFixedSize(300, 150)
        layout = QFormLayout()

        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        layout.addRow("Username:", self.username_input)
        layout.addRow("Password:", self.password_input)

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.handle_login)
        layout.addRow(self.login_button)

        self.setLayout(layout)

    def handle_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Username and password required")
            return

        try:
            LoginDialog.access_token = authenticate(username, password)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Login Failed", str(e))


# === Main GUI ===
class UploadWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Upload DICOM Folder to PACS")
        self.resize(400, 200)

        if LoginDialog.access_token is None:
            login_dialog = LoginDialog()
            if login_dialog.exec_() != QDialog.Accepted:
                QMessageBox.critical(self, "Login Required", "Login is required to proceed.")
                self.close()
                return

        self.access_token = LoginDialog.access_token

        # Setup DICOMweb client
        self.client = DICOMwebClient(
            url="https://staging.meningioma.midaspacs.in/dcm4chee-arc/aets/DCM4CHEE/rs",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )

        # GUI layout
        layout = QVBoxLayout()

        self.folder_label = QLabel("No folder selected")
        layout.addWidget(self.folder_label)

        self.select_button = QPushButton("Select DICOM Folder")
        self.select_button.clicked.connect(self.select_folder)
        layout.addWidget(self.select_button)

        self.upload_button = QPushButton("Upload to PACS")
        self.upload_button.clicked.connect(self.upload_folder)
        layout.addWidget(self.upload_button)

        self.setLayout(layout)
        self.folder_path = None

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path = folder
            self.folder_label.setText(f"Selected: {folder}")
        else:
            self.folder_label.setText("No folder selected")

    def upload_folder(self):
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "Please select a folder first")
            return
        try:
            upload_dicom_folder(self.client, self.folder_path)
            QMessageBox.information(self, "Success", "DICOM files uploaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Upload Failed", str(e))


# === Main App ===
def main():
    app = QApplication(sys.argv)

    login = LoginDialog()
    if login.exec_() == QDialog.Accepted:
        window = UploadWindow()
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
