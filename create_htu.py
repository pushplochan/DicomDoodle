import sys
import fitz
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea, QStyle)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from utils import get_data_path
pdf_path = get_data_path("DicomDoodle.pdf")
class HowToUseViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pdf_document = None
        self.current_page = 0
        self.scale = 1.0

        # Main layout
        layout = QVBoxLayout()

        # PDF display label
        self.pdf_label = QLabel()
        self.pdf_label.setAlignment(Qt.AlignCenter)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.pdf_label)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # Buttons with icons and text
        self.prev_button = QPushButton("<")
        self.next_button = QPushButton(">")
        self.page_button = QPushButton("Page 0/0")
        self.zoom_in_button = QPushButton("+")  
        self.zoom_out_button = QPushButton("-") 
        

        # Style buttons
        button_style = """
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
        self.prev_button.setStyleSheet(button_style)
        self.next_button.setStyleSheet(button_style)
        self.page_button.setStyleSheet(button_style)
        self.zoom_in_button.setStyleSheet(button_style)
        self.zoom_out_button.setStyleSheet(button_style)
        

        # Button layout (horizontal)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.page_button)
        button_layout.addWidget(self.zoom_in_button)
        button_layout.addWidget(self.zoom_out_button)
        button_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(button_layout)

        # Connect buttons to functions
        self.prev_button.clicked.connect(self.prev_page)
        self.next_button.clicked.connect(self.next_page)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.setLayout(layout)
        self.setWindowTitle("How to Use the DicomDoodle Tool")
        self.setGeometry(300, 300, 800, 600)
        self.setStyleSheet("background-color: #F0F8FF;")  # Light blue background

        # Load the PDF file automatically
        self.open_pdf(pdf_path)

    def open_pdf(self, file_name):
        try:
            self.pdf_document = fitz.open(file_name)
            self.current_page = 0
            self.scale = 1.0
            self.update_page_button()
            self.show_page()
        except Exception as e:
            print(f"Error loading PDF: {e}")
            self.page_button.setText("Page 0/0")

    def show_page(self):
        if self.pdf_document:
            page = self.pdf_document[self.current_page]
            pix = page.get_pixmap(matrix=fitz.Matrix(self.scale, self.scale))
            image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.pdf_label.setPixmap(pixmap)
            self.update_page_button()

    def update_page_button(self):
        if self.pdf_document:
            self.page_button.setText(f"Page {self.current_page + 1}/{len(self.pdf_document)}")
        else:
            self.page_button.setText("Page 0/0")

    def prev_page(self):
        if self.pdf_document and self.current_page > 0:
            self.current_page -= 1
            self.show_page()

    def next_page(self):
        if self.pdf_document and self.current_page < len(self.pdf_document) - 1:
            self.current_page += 1
            self.show_page()

    def zoom_in(self):
        self.scale *= 1.2
        self.show_page()

    def zoom_out(self):
        self.scale /= 1.2
        self.show_page()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HowToUseViewer()
    window.show()
    sys.exit(app.exec_())
