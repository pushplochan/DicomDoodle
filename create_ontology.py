import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QListWidget, QListWidgetItem, QLabel, QMenu)
from PyQt5.QtCore import Qt, pyqtSignal
from ontoportal_client import BioPortalClient
from highdicom.sr.coding import CodedConcept

class BioPortalSearchGUI(QMainWindow):
    categorySelected = pyqtSignal(object)  
    typeSelected = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioPortal Ontology Search")
        self.setGeometry(100, 100, 800, 400)
        
        # Initialize BioPortal client
        self.client = BioPortalClient(api_key="f46b5f95-b7f6-423a-a31e-111385f140ba")
        
        # Set up main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Search bar and button layout
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Enter search term")
        self.get_labels_button = QPushButton("Get Labels")
        self.get_labels_button.clicked.connect(self.search_ontologies)
        search_layout.addWidget(self.search_bar)
        search_layout.addWidget(self.get_labels_button)
        layout.addLayout(search_layout)
        
        # Results list widget
        self.results_list = QListWidget()
        self.results_list.setSelectionMode(QListWidget.MultiSelection)
        self.results_list.itemClicked.connect(self.handle_item_clicked)
        self.results_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_list.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.results_list)
        
        # Show all button
        self.show_all_button = QPushButton("Show All Results")
        self.show_all_button.clicked.connect(self.show_all_results)
        self.show_all_button.setVisible(False)  # Hidden until search
        layout.addWidget(self.show_all_button)
        
        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Store all results data (not QListWidgetItem objects)
        self.all_results_data = []
        
    def search_ontologies(self):
        """Perform search and populate results list"""
        search_term = self.search_bar.text().strip()
        if not search_term:
            self.status_label.setText("Please enter a search term.")
            return
        
        self.results_list.clear()
        self.all_results_data = []  # Clear previous results
        self.show_all_button.setVisible(False)
        self.status_label.setText(f"Searching for '{search_term}'...")
        
        try:
            results = self.client.search(search_term)
            matches = results.get("collection", [])
            
            if not matches:
                self.status_label.setText("No results found.")
                return
                
            for r in matches:
                label = r.get('prefLabel', '[no label]')
                if len(label) > 64:
                   sublabels = [s.strip() for s in label.split(',')]
                   label = next((s for s in sublabels if len(s) <= 64), None)
                   if not label:
                      print(f"[Skipping] Label too long (>64 chars): {label}")
                      continue  # Skip incomplete entries
                code_value = r.get('notation') or r.get('@id', '').split('/')[-1]
                if len(code_value) > 16:
                   print(f"[Skipping] code value too long (>16 chars): {code_value}")
                   continue  # Skip incomplete entries
                term_id = r.get('@id', '[no id]')
                self_link = r.get('links', {}).get('self', '[no self link]')
                ontology_link = r.get('links', {}).get('ontology', '[no ontology]')
                acronym = ontology_link.split('/')[-1] if ontology_link else '[unknown]'
                self.details_url = r.get('links', {}).get('self', '[no self link]')
                # Store data instead of QListWidgetItem
                if not (label and code_value and acronym):
                   continue  # Skip incomplete entries
                result_data = {
                    'label': label,
                    'id': term_id,
                    'self_link': self_link,
                    'ontology': ontology_link,
                    'Scheme': acronym,
                    'value': code_value
                }
                self.all_results_data.append(result_data)
                
                # Create list item for first 10 results
                if len(self.results_list) < 10:
                    item = QListWidgetItem(f"Label: {label}")
                    item.setData(Qt.UserRole, result_data)
                    self.results_list.addItem(item)
            
            # Show "Show All" button if there are more than 10 results
            if len(self.all_results_data) > 10:
                self.show_all_button.setVisible(True)
                
            self.status_label.setText(f"Showing {min(10, len(self.all_results_data))} of {len(self.all_results_data)} results for '{search_term}'")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
    
    def show_all_results(self):
        """Show all search results"""
        self.results_list.clear()
        for result_data in self.all_results_data:
            item = QListWidgetItem(f"Label: {result_data['label']}")
            item.setData(Qt.UserRole, result_data)
            self.results_list.addItem(item)
        self.show_all_button.setVisible(False)
        self.status_label.setText(f"Showing all {len(self.all_results_data)} results")
    
    def handle_item_clicked(self, item):
        """Handle click on list item"""
        data = item.data(Qt.UserRole)
        # Create CodedConcept from this
        concept = CodedConcept(
            value=data['value'],
            scheme_designator=data['Scheme'],
            meaning=data['label']
        )
        concept.LongCodeValue = f"{data['ontology'].replace('data', 'bioportal')}"
        concept.MappingResource = "BIOPORTAL"
        print(f"code_concept: \n{concept}\n")
    
    def show_context_menu(self, position):
        """Show context menu on right-click"""
        item = self.results_list.itemAt(position)
        if item:
            data = item.data(Qt.UserRole)
            menu = QMenu(self)
            add_segment_category = menu.addAction("Add Segment Category")
            segment_type = menu.addAction("Add Segment Type")
            details = menu.addAction("Definition And Synonyms")
            
            action = menu.exec_(self.results_list.mapToGlobal(position))
            if action == add_segment_category:
                concept = CodedConcept(
                    value=data['value'],
                    scheme_designator=data['Scheme'],
                    meaning=data['label']
                )
                concept.LongCodeValue = f"{data['ontology'].replace('data', 'bioportal')}"
                concept.MappingResource = "BIOPORTAL"
                self.categorySelected.emit(concept)
                print(f"Added Segment Category: \n{concept}\n")
            elif action == segment_type:
                concept = CodedConcept(
                    value=data['value'],
                    scheme_designator=data['Scheme'],
                    meaning=data['label']
                )
                concept.LongCodeValue = f"{data['ontology'].replace('data', 'bioportal')}"
                concept.MappingResource = "BIOPORTAL"
                self.typeSelected.emit(concept)
                print(f"Added Segment Type: \n{concept}\n")
            elif action == details:
                details = self.client.get_json(self.details_url)
                definition = details.get('definition', ['No definition found.'])
                synonyms = details.get('synonym', ['No synonyms found.'])
                print(f"definition for label: \n{definition}\n")
                print(f"synonyms for label: \n{synonyms}\n")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BioPortalSearchGUI()
    window.show()
    sys.exit(app.exec_())
