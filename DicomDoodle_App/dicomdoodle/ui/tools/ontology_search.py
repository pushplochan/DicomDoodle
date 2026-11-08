"""
Ontology Search — choose a *segment category* and a *segment type* and
write them into the annotation files of the series open in the Annotated
DICOM Viewer.

A search looks in DICOM's own code dictionary first (core/coding.py, no
internet needed); when that has no term, BioPortal is asked automatically,
and "Also search BioPortal" adds its results to the dictionary's. The
category is picked from the CID 7150 list the standard asks for, or from
a search result. The terms can go to every segment or to one label's
segments, and nothing is written until you press Apply.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
                             QMainWindow, QMenu, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from ...core.coding import CodedTerm, categories, search_dicom, unique_terms
from ...core.ontology import BioPortalClient
from ...core.seg_io import BLANK_CATEGORY, BLANK_TYPE, apply_concepts
from ...workers import run_task
from ..guide import help_button
from ..icons import icon
from ..qt_utils import ask_yes_no, label, make_button, show_error, show_info
from ..widgets.cards import Card

PAGE = 10


# Background jobs (module functions; results come back to bound methods, which
# Qt disconnects automatically if the window is closed while a job runs).
def _search_job(_ctx, client: BioPortalClient, text: str) -> List[CodedTerm]:
    return client.search(text)


def _details_job(_ctx, client: BioPortalClient, term: CodedTerm):
    return client.details(term)


def _apply_job(ctx, files: List[str], category: CodedTerm, type_: CodedTerm, segment_label: Optional[str]):
    return apply_concepts(files, category, type_, segment_label, ctx)


def term_text(term: CodedTerm) -> str:
    return f"{term.meaning}  ({term.scheme} {term.value})"


def source_text(term: CodedTerm) -> str:
    if term.source == "BioPortal":
        local = "" if term.is_standard_scheme else " — local coding scheme, described in the files"
        return f"BioPortal · {term.ontology}{local}"
    groups = term.context_group_names()
    return "DICOM code dictionary" + (" · " + "; ".join(groups[:4]) if groups else "")


class OntologySearchWindow(QMainWindow):
    applied = pyqtSignal(int)          # number of files updated

    def __init__(self, api_key: str, folder: str, seg_files: List[str],
                 segment_labels: Sequence[str] = (), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ontology Search — DICOM codes and BioPortal")
        self.resize(1080, 680)
        self.client = BioPortalClient(api_key)
        self.folder, self.seg_files = folder, seg_files
        self.results: List[CodedTerm] = []
        self._query = ""
        self._bioportal_done = False
        self.type: Optional[CodedTerm] = None

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search a term or code, e.g. neoplasm, edema, glioma, brain …")
        self.search_edit.setClearButtonEnabled(True)
        self.search_btn = make_button("Search", "search", role="primary")
        self.list = QListWidget()
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.show_all_btn = make_button("Show all results", role="ghost")
        self.show_all_btn.hide()
        self.bioportal_btn = make_button("Also search BioPortal", "external",
                                         tooltip="Add BioPortal's results (needs internet)")
        self.bioportal_btn.hide()
        self.status = label("Terms come from DICOM's own code dictionary first; BioPortal is asked when it "
                            "has none.", "Muted", wrap=True)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(self.search_edit, 1)
        row.addWidget(self.search_btn)
        left_layout.addLayout(row)
        left_layout.addWidget(self.list, 1)
        more = QHBoxLayout()
        more.addWidget(self.show_all_btn)
        more.addWidget(self.bioportal_btn)
        more.addStretch(1)
        left_layout.addLayout(more)
        left_layout.addWidget(self.status)

        # Details of the selected result
        self.detail_values = {}
        details = Card("Selected term")
        form = QFormLayout()
        for key in ("Meaning", "Code", "Source", "Identifier"):
            value = label("—", "Value", wrap=True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.detail_values[key] = value
            form.addRow(label(key, "Muted"), value)
        details.add_layout(form)
        self.definition = QTextBrowser()
        self.definition.setMaximumHeight(120)
        self.definition.setPlaceholderText("Definitions and synonyms appear here (from BioPortal).")
        self.definition_btn = make_button("Definition && synonyms", "book")
        buttons = QHBoxLayout()
        self.as_category_btn = make_button("Use as category", "tag")
        self.as_type_btn = make_button("Use as type", "tag")
        buttons.addWidget(self.as_category_btn)
        buttons.addWidget(self.as_type_btn)
        buttons.addWidget(self.definition_btn)
        details.add_layout(buttons)
        details.add(self.definition)

        # Choices + apply
        choice = Card("Apply to annotation files")
        choice.add_header_widget(help_button("ontology"))
        self.segment_combo = QComboBox()
        self.segment_combo.addItem("All labels", None)
        for name in segment_labels:
            self.segment_combo.addItem(name, name)
        self.category_combo = QComboBox()
        self.category_combo.addItem("Choose a category…", None)
        for term in categories():
            self.category_combo.addItem(term_text(term), term)
        self.category_note = label("", "Faint", wrap=True)
        self.type_label = label("— search a term and press “Use as type”", "Value", wrap=True)
        choice_form = QFormLayout()
        choice_form.addRow(label("Segments", "Muted"), self.segment_combo)
        choice_form.addRow(label("Segment category", "Muted"), self.category_combo)
        choice_form.addRow(label("", "Muted"), self.category_note)
        choice_form.addRow(label("Segment type", "Muted"), self.type_label)
        choice.add_layout(choice_form)
        choice.add(label(f"{len(seg_files)} annotation file(s) in “{os.path.basename(folder)}”. Blank masks get "
                         f"{BLANK_CATEGORY[2]} / {BLANK_TYPE[2]}.", "Faint", wrap=True))
        self.apply_btn = make_button("Apply to files", "check", role="primary")
        choice.add(self.apply_btn)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(details)
        right_layout.addWidget(choice)
        right_layout.addStretch(1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 520])
        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(14, 14, 14, 14)
        root.addWidget(splitter)
        self.setCentralWidget(container)

        self.search_btn.clicked.connect(self.search)
        self.search_edit.returnPressed.connect(self.search)
        self.show_all_btn.clicked.connect(lambda: self._fill(len(self.results)))
        self.bioportal_btn.clicked.connect(self.search_bioportal)
        self.list.currentItemChanged.connect(lambda item, _prev: self._show_term(self._term(item)))
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.as_category_btn.clicked.connect(lambda: self._assign("category", self._term(self.list.currentItem())))
        self.as_type_btn.clicked.connect(lambda: self._assign("type", self._term(self.list.currentItem())))
        self.definition_btn.clicked.connect(lambda: self._load_definition(self._term(self.list.currentItem())))
        self.category_combo.currentIndexChanged.connect(self._category_changed)
        self.apply_btn.clicked.connect(self.apply)
        self._update_buttons()

    @property
    def category(self) -> Optional[CodedTerm]:
        return self.category_combo.currentData()

    # ---------------------------------------------------------------- search
    def search(self) -> None:
        text = self.search_edit.text().strip()
        if not text:
            self.status.setText("Please enter a search term.")
            return
        self._query, self._bioportal_done = text, False
        local = search_dicom(text)
        if local:
            self._show_results(local)
            self.bioportal_btn.show()
            return
        self.list.clear()
        self.show_all_btn.hide()
        self.bioportal_btn.hide()
        self.results = []
        self.status.setText(f"No DICOM term for “{text}” — searching BioPortal…")
        self.search_bioportal()

    def search_bioportal(self) -> None:
        if not self._query or self._bioportal_done:
            return
        self.search_btn.setEnabled(False)
        self.bioportal_btn.setEnabled(False)
        if self.results:
            self.status.setText(f"Searching BioPortal for “{self._query}”…")
        run_task(_search_job, self.client, self._query, on_result=self._bioportal_results,
                 on_error=self._search_failed, on_finished=self._search_finished)

    def _bioportal_results(self, terms: List[CodedTerm]) -> None:
        self._bioportal_done = True
        self.bioportal_btn.hide()
        shown = max(PAGE, self.list.count())
        self._show_results(unique_terms(self.results + terms), shown)

    def _search_failed(self, message: str, _details: str) -> None:
        prefix = f"{len(self.results)} DICOM term(s) shown. " if self.results else ""
        self.status.setText(f"{prefix}BioPortal: {message}")

    def _search_finished(self) -> None:
        self.search_btn.setEnabled(True)
        self.bioportal_btn.setEnabled(True)

    def _show_results(self, terms: List[CodedTerm], count: int = PAGE) -> None:
        self.results = terms
        if not terms:
            self.list.clear()
            self.show_all_btn.hide()
            self.status.setText(f"No results for “{self._query}”.")
            return
        self._fill(count)
        local = sum(1 for t in terms if t.source == "DICOM")
        sources = f"{local} from the DICOM dictionary" + (f", {len(terms) - local} from BioPortal"
                                                           if self._bioportal_done else "")
        self.status.setText(f"{len(terms)} result(s) for “{self._query}” ({sources}). "
                            "Right-click a result for more options.")

    def _fill(self, count: int) -> None:
        self.list.clear()
        for term in self.results[:count]:
            where = "DICOM" if term.source == "DICOM" else f"BioPortal {term.ontology}"
            item = QListWidgetItem(f"{term.meaning}    ·  {term.scheme} {term.value}    [{where}]")
            item.setData(Qt.UserRole, term)
            item.setToolTip(source_text(term))
            self.list.addItem(item)
        self.show_all_btn.setVisible(len(self.results) > count)
        self.show_all_btn.setText(f"Show all {len(self.results)} results")

    @staticmethod
    def _term(item: Optional[QListWidgetItem]) -> Optional[CodedTerm]:
        return item.data(Qt.UserRole) if item is not None else None

    def _show_term(self, term: Optional[CodedTerm]) -> None:
        self.detail_values["Meaning"].setText(term.meaning if term else "—")
        self.detail_values["Code"].setText(f"{term.value}  (coding scheme {term.scheme})" if term else "—")
        self.detail_values["Source"].setText(source_text(term) if term else "—")
        self.detail_values["Identifier"].setText((term.term_id or "—") if term else "—")
        self.definition.clear()
        if term is not None and term.synonyms:
            self.definition.setHtml("<b>Also written as</b><br>" + ", ".join(term.synonyms))
        self._update_buttons()

    def _load_definition(self, term: Optional[CodedTerm]) -> None:
        if term is None:
            return
        self.definition.setPlainText("Loading from BioPortal…")
        run_task(_details_job, self.client, term, on_result=self._show_definition,
                 on_error=self._definition_failed)

    def _show_definition(self, result) -> None:
        definitions, synonyms = result
        html = "<b>Definition</b><br>" + ("<br>".join(definitions) or "No definition found.")
        html += "<br><br><b>Synonyms</b><br>" + (", ".join(synonyms) or "No synonyms found.")
        self.definition.setHtml(html)

    def _definition_failed(self, message: str, _details: str) -> None:
        self.definition.setPlainText(f"Could not load details: {message}")

    def _context_menu(self, pos) -> None:
        term = self._term(self.list.itemAt(pos))
        if term is None:
            return
        menu = QMenu(self)
        menu.addAction(icon("tag"), "Add Segment Category", lambda: self._assign("category", term))
        menu.addAction(icon("tag"), "Add Segment Type", lambda: self._assign("type", term))
        menu.addAction(icon("book"), "Definition And Synonyms", lambda: self._load_definition(term))
        menu.exec_(self.list.viewport().mapToGlobal(pos))

    # ----------------------------------------------------------------- apply
    def _assign(self, kind: str, term: Optional[CodedTerm]) -> None:
        if term is None:
            return
        if kind == "type":
            self.type = term
            self.type_label.setText(term_text(term))
        else:
            index = next((i for i in range(1, self.category_combo.count())
                          if self._same(self.category_combo.itemData(i), term)), -1)
            if index < 0:
                self.category_combo.addItem(term_text(term), term)
                index = self.category_combo.count() - 1
            self.category_combo.setCurrentIndex(index)
        self._update_buttons()

    @staticmethod
    def _same(a: Optional[CodedTerm], b: CodedTerm) -> bool:
        return a is not None and (a.scheme, a.value) == (b.scheme, b.value)

    def _category_changed(self, _index: int) -> None:
        term = self.category
        if term is not None and not term.is_category:
            self.category_note.setText("Not one of the standard categories (CID 7150). DICOM viewers expect "
                                       "one of the categories above; the type can be any term.")
        else:
            self.category_note.setText("")
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_term = self.list.currentItem() is not None
        for button in (self.as_category_btn, self.as_type_btn, self.definition_btn):
            button.setEnabled(has_term)
        self.apply_btn.setEnabled(self.category is not None and self.type is not None and bool(self.seg_files))

    def apply(self) -> None:
        category, type_ = self.category, self.type
        if category is None or type_ is None:
            show_info(self, "Ontology", "Please choose a segment category and a segment type.")
            return
        target = self.segment_combo.currentData()
        segments = f"the “{target}” segments of" if target else "every segment of"
        if not ask_yes_no(self, "Apply ontology terms",
                          f"Write “{category.meaning}” / “{type_.meaning}” into {segments} "
                          f"{len(self.seg_files)} annotation file(s)?", yes_text="Apply"):
            return
        self.apply_btn.setEnabled(False)
        run_task(_apply_job, list(self.seg_files), category, type_, target,
                 on_result=self._applied, on_error=self._apply_failed, on_finished=self._update_buttons)

    def _applied(self, result) -> None:
        updated, errors = result
        if errors:
            show_error(self, "Some files were not updated", f"{len(errors)} problem(s):\n" + "\n".join(errors[:10]))
        elif updated:
            show_info(self, "Success", f"Updated {updated} annotation file(s).")
        else:
            show_info(self, "Nothing to change", "The annotation files already hold these terms.")
        self.applied.emit(updated)

    def _apply_failed(self, message: str, details: str) -> None:
        show_error(self, "Ontology", message, details)
