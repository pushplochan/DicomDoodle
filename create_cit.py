import sys
import math
import vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class CitationsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DicomDoodle Citation")
        self.setGeometry(100, 100, 800, 600)

        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.vtk_widget)

        # Initialize VTK renderer and interactor
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # Set up the scene
        self.setup_scene()

        # Animation setup
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # Update every 50ms

        # Initialize interactor
        self.interactor.Initialize()

    def setup_scene(self):
        # Define nodes with radius
        self.nodes = {
            "DicomDoodle": {"type": "tool", "color": (1, 1, 1), "size": 1.0},  # Yellow, larger
            "NumPy": {"type": "library", "color": (0, 1, 0), "size": 0.5},  # Green
            "OpenCV": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "Pillow": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "Ultralytics": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "SAM": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "pydicom": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "highdicom": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "SimpleITK": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "VTK": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "scikit-image": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "PyQt5": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "BioPortal": {"type": "library", "color": (0, 1, 0), "size": 0.5},
            "SciPy": {"type": "library", "color": (0, 1, 0), "size": 0.5},  # Added SciPy
            "Harris et al. (2020)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},  # Blue
            "Bradski (2000)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "Clark (2015)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "Jocher (2023)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "Kirillov et al. (2023)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "Mason (2011)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "Herrmann et al. (2023)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "Lowekamp et al. (2013)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "Schroeder et al. (2006)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "van der Walt et al. (2014)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "Thompson (2023)": {"type": "paper", "color": (1, 1, 0), "size": 0.5},
            "Noy et al. (2025)": {"type": "paper", "color": (1, 1, 0), "size": 0.5}
        }

        # Create VTK actors and label actors for nodes
        self.node_actors = {}
        self.node_positions = {}
        self.label_actors = {}
        num_libraries = sum(1 for n in self.nodes if self.nodes[n]["type"] == "library")
        num_papers = sum(1 for n in self.nodes if self.nodes[n]["type"] == "paper")
        
        for i, node in enumerate(self.nodes):
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(self.nodes[node]["size"])
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.nodes[node]["color"])
            
            # Position nodes
            if node == "DicomDoodle":
                pos = (0, 0, 0)
            elif self.nodes[node]["type"] == "library":
                angle = 2 * math.pi * (i - 1) / num_libraries  # Distribute libraries evenly
                pos = (5 * math.cos(angle), 5 * math.sin(angle), 0)  # Inner orbit, radius 5
            else:  # Papers
                angle = 2 * math.pi * (i - num_libraries - 1) / num_papers
                pos = (10 * math.cos(angle), 10 * math.sin(angle), 0)  # Outer orbit, radius 10
            
            actor.SetPosition(pos)
            self.node_actors[node] = actor
            self.node_positions[node] = [pos[0], pos[1], pos[2]]
            self.renderer.AddActor(actor)
            
            # Add label
            label = vtk.vtkVectorText()
            label.SetText(node)
            label_mapper = vtk.vtkPolyDataMapper()
            label_mapper.SetInputConnection(label.GetOutputPort())
            label_actor = vtk.vtkFollower()
            label_actor.SetMapper(label_mapper)
            label_actor.SetPosition(pos[0], pos[1] + self.nodes[node]["size"] + 0.2, pos[2])
            label_actor.SetScale(0.2)
            label_actor.GetProperty().SetColor(1, 1, 1)  # White labels
            label_actor.SetCamera(self.renderer.GetActiveCamera())
            self.label_actors[node] = label_actor
            self.renderer.AddActor(label_actor)

        # Set background and camera
        self.renderer.SetBackground(0, 0, 0.1)  # Dark blue background
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Dolly(0.8)  # Zoom out slightly

    def update_animation(self):
        self.angle += 0.01  # Slow rotation
        num_libraries = sum(1 for _, node in self.nodes.items() if node["type"] == "library")
        num_papers = sum(1 for _, node in self.nodes.items() if node["type"] == "paper")
        
        for i, (node, actor) in enumerate(self.node_actors.items()):
            if node == "DicomDoodle":
                continue  # Keep DicomDoodle stationary
            elif self.nodes[node]["type"] == "library":
                angle = 2 * math.pi * (i - 1) / num_libraries + self.angle
                pos = (5 * math.cos(angle), 5 * math.sin(angle), 0)
            else:  # Papers
                angle = 2 * math.pi * (i - num_libraries - 1) / num_papers + self.angle * 0.5  # Slower orbit
                pos = (10 * math.cos(angle), 10 * math.sin(angle), 0)
            
            actor.SetPosition(pos)
            self.node_positions[node] = [pos[0], pos[1], pos[2]]
            # Update label position using stored radius
            self.label_actors[node].SetPosition(pos[0], pos[1] + self.nodes[node]["size"] + 0.2, pos[2])

        self.vtk_widget.Render()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CitationsViewer()
    window.show()
    sys.exit(app.exec_())
