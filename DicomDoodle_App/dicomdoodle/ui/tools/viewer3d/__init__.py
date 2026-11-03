"""
The 3D parts of the Multiplanar & 3D Viewer (formerly the separate 3D Model
Viewer, the original "View 3D Model"; the window is mpr/window.py):

    controls.py  RenderSettings + the "3D view" tab's settings (replaces the
                 original's two identical "Anatomy / Volume Visualization" dialogs)
    panels.py    the right-hand panel: series folders, lesion volume (formerly the
                 Lesion Volume Calculator) and 3D lesion metrics
    view.py      the VTK volume view: transfer functions, colormaps, clipping, lights
"""
