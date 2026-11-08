"""
The main annotator window, split by responsibility:

    window.py      MainWindow — layout, slice display, shared state
    actions.py     every menu / toolbar action and keyboard shortcut
    annotation.py  single-slice work: auto detect, box detect, undo, save, plot
    drawing.py     drawing by hand: box / pen / eraser / pan tools, edit saved outline
    batch.py       auto-detect range batch job and the automatic blank masks
    tools.py       opens the tool windows and dialogs
    welcome.py     start page shown before a series is loaded
"""
from .window import MainWindow  # noqa: F401
