"""
DicomDoodle — MRI DICOM annotation, segmentation and visualization studio.

Package layout (the repository's README.md has the module map):

    dicomdoodle/
        app.py        builds the QApplication, splash screen and main window
        resources.py  finds the logo images and the guide's screenshots
        settings.py   persistent user preferences (JSON file)
        core/         pure Python logic: DICOM I/O, labels, masks, AI models ...
        workers/      background-thread helpers so the GUI never freezes
        help/         User Guide chapters, shortcut list and references (text only)
        ui/           everything Qt: theme, widgets, dialogs, tool windows, User Guide
"""

APP_NAME = "DicomDoodle"
APP_VERSION = "2.1.0"
APP_TAGLINE = "MRI DICOM Annotation & Visualization Studio"
AUTHOR = "Pushp Lochan Kumar"
AUTHOR_EMAIL = "pushpl@iisc.ac.in"
