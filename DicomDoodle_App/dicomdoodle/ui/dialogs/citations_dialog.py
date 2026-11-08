"""
References & Citations: how to cite DicomDoodle, and the papers and
software it builds on, grouped by topic.

Each reference shows what DicomDoodle uses it for, an APA-style citation
with a clickable DOI, and copy buttons; everything can be copied at once or
exported as a BibTeX file. The entries live in ``help/references.py``.
"""
from __future__ import annotations

import html
from typing import List, Tuple

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ... import APP_NAME, APP_VERSION
from ...help.references import (REPOSITORY_URL, SECTIONS, Reference, all_references, app_reference, bibtex_file,
                                plain_text)
from ..icons import icon, pixmap
from ..qt_utils import label, make_button, show_error
from ..theme import current_theme


def citation_html(ref: Reference) -> str:
    """APA-style citation with the venue in italics and a clickable DOI / URL."""
    title = ref.title if ref.title.endswith((".", "?", "!")) else ref.title + "."
    venue = f"<i>{html.escape(ref.venue)}</i>" + (f", {html.escape(ref.details)}" if ref.details else "") + "."
    if ref.status:
        venue += f" {html.escape(ref.status)}."
    link = f' <a href="{ref.link}">{html.escape(ref.link)}</a>' if ref.link else ""
    return f"{html.escape(ref.author_text())} ({ref.year}). {html.escape(title)} {venue}{link}"


def _copy(button: QPushButton, text: str) -> None:
    """Put ``text`` on the clipboard; the button says “Copied” for a moment."""
    QApplication.clipboard().setText(text)
    caption = button.text()
    if caption.startswith("Copied"):
        return
    button.setText("Copied ✓")
    QTimer.singleShot(1400, lambda: button.setText(caption) if not _deleted(button) else None)


def _deleted(widget: QWidget) -> bool:
    try:
        widget.objectName()
        return False
    except RuntimeError:
        return True


class _ReferenceCard(QFrame):
    def __init__(self, ref: Reference, parent=None):
        super().__init__(parent)
        self.setObjectName("RefCard")
        self.ref = ref
        self.search_text = " ".join([ref.component, ref.role, ref.text(), ref.key]).lower()

        name = label(ref.component, "Heading")
        role = label(ref.role, "Muted", wrap=True)
        citation = QLabel(citation_html(ref))
        citation.setWordWrap(True)
        citation.setOpenExternalLinks(True)
        citation.setTextInteractionFlags(Qt.TextBrowserInteraction)
        copy = make_button("Copy", "copy", role="ghost", tooltip="Copy this citation")
        copy.clicked.connect(lambda: _copy(copy, ref.text()))
        bibtex = make_button("BibTeX", role="ghost", tooltip="Copy this reference as a BibTeX entry")
        bibtex.clicked.connect(lambda: _copy(bibtex, ref.bibtex()))

        head = QHBoxLayout()
        head.setSpacing(4)
        head.addWidget(name)
        head.addStretch(1)
        head.addWidget(copy)
        head.addWidget(bibtex)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 10, 12)
        layout.setSpacing(4)
        layout.addLayout(head)
        layout.addWidget(role)
        layout.addSpacing(2)
        layout.addWidget(citation)


class CitationsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("References & Citations")
        self.resize(900, 780)

        # ---- header ---------------------------------------------------------
        mark = QLabel()
        mark.setPixmap(pixmap("bookmark", current_theme().accent, 30))
        title = label("References & Citations", "Title")
        intro = label(f"{APP_NAME} builds on the research and open-source work below. If {APP_NAME} helps your "
                      "work, please cite it together with the methods you used — for example SAM and YOLO "
                      "for AI segmentation.", "Muted", wrap=True)
        header = QHBoxLayout()
        header.setSpacing(14)
        header.addWidget(mark, 0, Qt.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(4)
        text.addWidget(title)
        text.addWidget(intro)
        header.addLayout(text, 1)

        # ---- how to cite DicomDoodle -------------------------------------------
        own = app_reference()
        cite = QFrame()
        cite.setObjectName("CiteCard")
        cite_text = QLabel(citation_html(own))
        cite_text.setWordWrap(True)
        cite_text.setOpenExternalLinks(True)
        cite_text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        copy_own = make_button("Copy citation", "copy", role="primary")
        copy_own.clicked.connect(lambda: _copy(copy_own, own.text()))
        copy_own_bib = make_button("Copy BibTeX", "file-text")
        copy_own_bib.clicked.connect(lambda: _copy(copy_own_bib, own.bibtex()))
        cite_buttons = QHBoxLayout()
        cite_buttons.addWidget(copy_own)
        cite_buttons.addWidget(copy_own_bib)
        cite_buttons.addStretch(1)
        cite_layout = QVBoxLayout(cite)
        cite_layout.setContentsMargins(16, 12, 16, 14)
        cite_layout.setSpacing(8)
        cite_layout.addWidget(label(f"HOW TO CITE {APP_NAME.upper()}", "SectionTitle"))
        software = QLabel(f"Software used: {APP_NAME} version {APP_VERSION} · "
                          f'<a href="{REPOSITORY_URL}">{html.escape(REPOSITORY_URL)}</a>')
        software.setObjectName("Muted")
        software.setWordWrap(True)
        software.setOpenExternalLinks(True)
        software.setTextInteractionFlags(Qt.TextBrowserInteraction)
        cite_layout.addWidget(cite_text)
        cite_layout.addWidget(software)
        cite_layout.addLayout(cite_buttons)

        # ---- filter -----------------------------------------------------------
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name, author, title or year …")
        self.filter.setClearButtonEnabled(True)
        self.filter.addAction(icon("search"), QLineEdit.LeadingPosition)
        self.filter.textChanged.connect(self._apply_filter)
        self.count = label("", "Faint")
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.filter, 1)
        filter_row.addWidget(self.count)

        # ---- the references ------------------------------------------------------
        body = QWidget()
        body.setObjectName("DockBody")
        listing = QVBoxLayout(body)
        listing.setContentsMargins(2, 2, 10, 2)
        listing.setSpacing(8)
        self.sections: List[Tuple[QLabel, List[_ReferenceCard]]] = []
        for section, refs in SECTIONS:
            heading = label(section.upper(), "SectionTitle")
            listing.addSpacing(6)
            listing.addWidget(heading)
            cards = [_ReferenceCard(ref) for ref in refs]
            for card in cards:
                listing.addWidget(card)
            self.sections.append((heading, cards))
        self.empty = label("No reference matches the filter.", "Muted")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.hide()
        listing.addWidget(self.empty)
        listing.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(body)

        # ---- footer --------------------------------------------------------------
        copy_all = make_button("Copy all", "copy", tooltip="Copy every citation as text")
        copy_all.clicked.connect(lambda: _copy(copy_all, plain_text()))
        export = make_button("Export BibTeX…", "download", tooltip="Save all references as a .bib file")
        export.clicked.connect(self.export_bibtex)
        close = make_button("Close", role="primary")
        close.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addWidget(copy_all)
        footer.addWidget(export)
        footer.addStretch(1)
        footer.addWidget(close)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)
        root.addLayout(header)
        root.addWidget(cite)
        root.addLayout(filter_row)
        root.addWidget(area, 1)
        root.addLayout(footer)
        self._apply_filter("")

    def _apply_filter(self, text: str) -> None:
        words = text.lower().split()
        shown = 0
        for heading, cards in self.sections:
            visible = 0
            for card in cards:
                match = all(w in card.search_text for w in words)
                card.setVisible(match)
                visible += match
            heading.setVisible(visible > 0)
            shown += visible
        total = len(all_references())
        self.count.setText(f"{total} references" if not words else f"{shown} of {total}")
        self.empty.setVisible(shown == 0)

    def export_bibtex(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export BibTeX", "dicomdoodle_references.bib",
                                              "BibTeX (*.bib);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(bibtex_file())
        except OSError as exc:
            show_error(self, "Export BibTeX", f"Could not write the file:\n{exc}")
