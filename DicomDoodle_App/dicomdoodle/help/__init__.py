"""
Help content — plain text and data only, no Qt.

    chapter.py         Chapter: id, title, icon, summary, body (one guide page)
    contents.py        the User Guide's table of contents (GROUPS, find, neighbours)
    part_start.py      chapters: welcome, quick start, the window, opening scans, viewing
    part_annotate.py   chapters: labels, auto detect, box + SAM, drawing by hand, correcting
    part_save.py       chapters: saving, batch processing
    part_review.py     chapters: annotated viewer, ontology, lesion volume (3D viewer side panel)
    part_tools.py      chapters: 3D viewer, multiplanar view, converter, preferences
    part_reference.py  chapters: shortcuts, troubleshooting, glossary
    shortcuts.py       keyboard shortcut list (also used by the shortcuts dialog)
    references.py      how to cite DicomDoodle and the works it builds on

Chapters are written in a small markup (see ui/guide/markup.py) so the text
stays readable and easy to edit; ui/guide/ renders them.
"""
