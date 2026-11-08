"""Keyboard and mouse shortcuts (shown in the guide and in Help ▸ Keyboard Shortcuts)."""

SHORTCUTS = [
    ("Files", [
        ("Ctrl+O", "Open a DICOM folder"),
        ("Ctrl+S", "Save the current slice's annotation (_seg.dcm)"),
        ("Ctrl+Shift+S", "Save every modified slice"),
    ]),
    ("Moving around", [
        ("Mouse wheel", "Previous / next slice"),
        ("← → ↑ ↓", "Previous / next slice"),
        ("Page Up / Page Down", "Jump 10 slices"),
        ("Home / End", "First / last slice"),
        ("Ctrl + mouse wheel", "Zoom around the mouse pointer"),
        ("Ctrl+=  /  Ctrl+-  /  Ctrl+0", "Zoom in / zoom out / fit the image to the window"),
        ("Middle-drag  or  Space + drag", "Pan (move) the image"),
    ]),
    ("Annotating", [
        ("Left-drag on the image", "Box tool: draw a box for SAM (drag the box to move it, a corner to resize it)"),
        ("Enter  or  right-click ▸ Detect", "Segment the box with SAM"),
        ("Delete / Esc", "Remove the box"),
        ("Ctrl+D", "Auto Detect (YOLO → SAM) on the current slice"),
        ("1 … 9", "Choose the active label"),
        ("Ctrl+Z  /  Ctrl+Y", "Undo / redo on the current slice"),
        ("Ctrl+R", "Reset (clear) the current slice"),
        ("Ctrl+P", "Plot the results of the current slice"),
    ]),
    ("Drawing by hand", [
        ("B / P / E / H", "Box / pen / eraser / pan tool"),
        ("Ctrl+M", "Pen tool"),
        ("[  /  ]", "Smaller / bigger pen or eraser"),
        ("Ctrl+L", "Edit the saved outline of the slice"),
        ("Right-click on the image", "Choose a tool"),
    ]),
    ("Application", [
        ("Ctrl+T", "Switch between light and dark theme"),
        ("Ctrl+,", "Preferences"),
        ("F1", "Open the User Guide"),
    ]),
]
