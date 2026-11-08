"""
The bibliography shown under Help ▸ References & Citations.

Each entry was checked against Crossref / arXiv / the project's own citation
file. ``Reference.text()`` gives an APA-style citation and ``bibtex()`` a
BibTeX entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .. import APP_NAME, APP_VERSION

REPOSITORY_URL = "https://github.com/pushplochan/DicomDoodle"
ARTICLE_DOI = "10.1007/s10278-026-02485-y"   # the peer-reviewed article describing DicomDoodle


@dataclass(frozen=True)
class Reference:
    key: str                      # BibTeX key
    component: str                # what DicomDoodle uses
    role: str                     # why it matters in DicomDoodle
    authors: Tuple[str, ...]      # "Family, I. N." — already in citation order
    year: str
    title: str
    venue: str                    # journal, proceedings, publisher or "[Computer software]"
    details: str = ""             # "585(7825), 357–362"
    doi: str = ""
    url: str = ""
    kind: str = "article"         # article | inproceedings | software | book | misc
    et_al: bool = False           # author list shortened
    status: str = ""              # "Advance online publication" — its own sentence after the venue

    @property
    def link(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else self.url

    def author_text(self) -> str:
        names = list(self.authors)
        if self.et_al:
            return ", ".join(names) + ", et al."
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + ", & " + names[-1]

    def text(self) -> str:
        """APA-style citation as plain text."""
        title = self.title if self.title.endswith((".", "?", "!")) else self.title + "."
        venue = self.venue + (f", {self.details}" if self.details else "") + "."
        if self.status:
            venue += f" {self.status}."
        return f"{self.author_text()} ({self.year}). {title} {venue} {self.link}".strip()

    def bibtex(self) -> str:
        authors = " and ".join(self.authors) + (" and others" if self.et_al else "")
        fields = [("author", authors), ("title", "{" + self.title + "}"), ("year", self.year)]
        if self.kind == "article":
            fields.append(("journal", self.venue))
        elif self.kind == "inproceedings":
            fields.append(("booktitle", self.venue))
        elif self.kind == "book":
            fields.append(("publisher", self.venue))
        else:
            fields.append(("howpublished", self.venue))
        note = ". ".join(part for part in (self.details, self.status) if part)
        if note:
            fields.append(("note", note))
        if self.doi:
            fields.append(("doi", self.doi))
        if self.url:
            fields.append(("url", self.url))
        entry_type = {"software": "software", "misc": "misc"}.get(self.kind, self.kind)
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
        return f"@{entry_type}{{{self.key},\n{body}\n}}"


def app_reference() -> Reference:
    """How to cite DicomDoodle: the article in the Journal of Imaging Informatics in Medicine."""
    return Reference(
        "kumar2026dicomdoodle", APP_NAME, "",
        ("Kumar, P. L.", "Shahin, M.", "Kumar, R.", "Suri, V.", "Gaikwad, S. B.", "Deb, B.", "Salunke, P.",
         "Bajaj, A.", "Srinivas, D.", "Sahu, R. N.", "Jaiswal, A. K.", "Vilanilam, G. C.", "Kedia, S.",
         "Vyas, T.", "Pal, D."),
        "2026",
        "DicomDoodle: Human-in-the-loop annotation of 3D MRI DICOM images with AI support and DICOM-SEG "
        "compatibility",
        "Journal of Imaging Informatics in Medicine", doi=ARTICLE_DOI, status="Advance online publication")


def software_note() -> str:
    """Which program was used — for a methods section, next to the article citation."""
    return f"Software used: {APP_NAME} version {APP_VERSION}, {REPOSITORY_URL}"


# (section title, [references]) in display order
SECTIONS: List[Tuple[str, List[Reference]]] = [
    ("AI models", [
        Reference("kirillov2023sam", "Segment Anything (SAM)", "Turns a box into a precise lesion outline.",
                  ("Kirillov, A.", "Mintun, E.", "Ravi, N."), "2023", "Segment Anything",
                  "Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)", "3992–4003",
                  doi="10.1109/ICCV51070.2023.00371", kind="inproceedings", et_al=True),
        Reference("ravi2024sam2", "SAM 2", "The SAM 2.1 weights used for segmentation.",
                  ("Ravi, N.", "Gabeur, V.", "Hu, Y.-T."), "2024", "SAM 2: Segment anything in images and videos",
                  "arXiv preprint", "arXiv:2408.00714", url="https://arxiv.org/abs/2408.00714", kind="misc",
                  et_al=True),
        Reference("jocher2023ultralytics", "Ultralytics YOLO", "Detection model framework (lesion bounding boxes).",
                  ("Jocher, G.", "Qiu, J.", "Chaurasia, A."), "2023", "Ultralytics YOLO (Version 8.0.0)",
                  "[Computer software]", url="https://github.com/ultralytics/ultralytics", kind="software"),
        Reference("wang2025yoloe", "YOLOE", "Prompt-free fallback detector when YOLO finds nothing.",
                  ("Wang, A.", "Liu, L.", "Chen, H.", "Lin, Z.", "Han, J.", "Ding, G."), "2025",
                  "YOLOE: Real-time seeing anything", "arXiv preprint", "arXiv:2503.07465",
                  url="https://arxiv.org/abs/2503.07465", kind="misc"),
        Reference("paszke2019pytorch", "PyTorch", "Runs the neural networks (CPU or GPU).",
                  ("Paszke, A.", "Gross, S.", "Massa, F."), "2019",
                  "PyTorch: An imperative style, high-performance deep learning library",
                  "Advances in Neural Information Processing Systems 32 (NeurIPS 2019)",
                  url="https://arxiv.org/abs/1912.01703", kind="inproceedings", et_al=True),
    ]),
    ("Medical imaging & DICOM", [
        Reference("nema2026dicom", "DICOM standard", "The rules the annotation files and converted slices follow "
                  "(PS3.3 Segmentation and MR Image IODs, PS3.16 codes).",
                  ("National Electrical Manufacturers Association",), "2026",
                  "NEMA PS3 / ISO 12052, Digital Imaging and Communications in Medicine (DICOM) Standard (2026d)",
                  "National Electrical Manufacturers Association, Rosslyn, VA, USA",
                  url="https://www.dicomstandard.org/current", kind="misc"),
        Reference("mason2011pydicom", "pydicom", "Reads and writes every DICOM file; its copy of DICOM's code "
                  "list gives the standard terms.",
                  ("Mason, D.",), "2011", "SU-E-T-33: Pydicom: An open source DICOM library", "Medical Physics",
                  "38(6), 3493", doi="10.1118/1.3611983"),
        Reference("bridge2022highdicom", "highdicom", "Creates the standard DICOM Segmentation (_seg.dcm) files.",
                  ("Bridge, C. P.", "Gorman, C.", "Pieper, S.", "Doyle, S. W.", "Lennerz, J. K.",
                   "Kalpathy-Cramer, J.", "Clunie, D. A.", "Fedorov, A. Y.", "Herrmann, M. D."), "2022",
                  "Highdicom: A Python library for standardized encoding of image annotations and machine "
                  "learning model outputs in pathology and radiology", "Journal of Digital Imaging",
                  "35(6), 1719–1737", doi="10.1007/s10278-022-00683-y"),
        Reference("ziegler2020ohif", "OHIF Viewer", "Web viewer the annotation files are checked with.",
                  ("Ziegler, E.", "Urban, T.", "Brown, D."), "2020",
                  "Open Health Imaging Foundation Viewer: An extensible open-source framework for building "
                  "web-based imaging applications to support cancer research", "JCO Clinical Cancer Informatics",
                  "4, 336–345", doi="10.1200/CCI.19.00131", et_al=True),
        Reference("lowekamp2013simpleitk", "SimpleITK", "3D volumes, resampling and registration.",
                  ("Lowekamp, B. C.", "Chen, D. T.", "Ibáñez, L.", "Blezek, D."), "2013", "The design of SimpleITK",
                  "Frontiers in Neuroinformatics", "7, 45", doi="10.3389/fninf.2013.00045"),
    ]),
    ("Image processing & scientific computing", [
        Reference("harris2020numpy", "NumPy", "All image and mask arithmetic.",
                  ("Harris, C. R.", "Millman, K. J.", "van der Walt, S. J."), "2020", "Array programming with NumPy",
                  "Nature", "585(7825), 357–362", doi="10.1038/s41586-020-2649-2", et_al=True),
        Reference("virtanen2020scipy", "SciPy", "Boundary gap closing, morphology and connected components.",
                  ("Virtanen, P.", "Gommers, R.", "Oliphant, T. E."), "2020",
                  "SciPy 1.0: Fundamental algorithms for scientific computing in Python", "Nature Methods",
                  "17(3), 261–272", doi="10.1038/s41592-019-0686-2", et_al=True),
        Reference("vanderwalt2014skimage", "scikit-image", "Texture metrics (GLCM) of the lesion.",
                  ("van der Walt, S.", "Schönberger, J. L.", "Nunez-Iglesias, J."), "2014",
                  "scikit-image: Image processing in Python", "PeerJ", "2, e453", doi="10.7717/peerj.453",
                  et_al=True),
        Reference("bradski2000opencv", "OpenCV", "Edges and contours of annotations.",
                  ("Bradski, G.",), "2000", "The OpenCV library", "Dr. Dobb's Journal of Software Tools",
                  url="https://opencv.org", kind="article"),
        Reference("clark2015pillow", "Pillow", "Image file support used by the AI framework.",
                  ("Clark, A.",), "2015", "Pillow (PIL fork) documentation", "Read the Docs",
                  url="https://pillow.readthedocs.io", kind="misc"),
        Reference("unser1999splines", "Cubic B-splines", "Interpolation between slices (B-Spline and "
                  "Through-plane reconstruction in the Multiplanar view).",
                  ("Unser, M.",), "1999", "Splines: A perfect fit for signal and image processing",
                  "IEEE Signal Processing Magazine", "16(6), 22–38", doi="10.1109/79.799930"),
        Reference("irani1991improving", "Iterative back-projection", "Slice-profile step of the Through-plane "
                  "reconstruction.", ("Irani, M.", "Peleg, S."), "1991", "Improving resolution by image registration",
                  "CVGIP: Graphical Models and Image Processing", "53(3), 231–239",
                  doi="10.1016/1049-9652(91)90045-L"),
        Reference("greenspan2002mri", "MRI inter-slice reconstruction", "The same idea applied to MRI slices, "
                  "which the Through-plane reconstruction follows.",
                  ("Greenspan, H.", "Oz, G.", "Kiryati, N.", "Peled, S."), "2002",
                  "MRI inter-slice reconstruction using super-resolution", "Magnetic Resonance Imaging",
                  "20(5), 437–446", doi="10.1016/S0730-725X(02)00511-8"),
    ]),
    ("Visualization & interface", [
        Reference("schroeder2006vtk", "VTK", "3D model and volume rendering.",
                  ("Schroeder, W.", "Martin, K.", "Lorensen, B."), "2006",
                  "The Visualization Toolkit: An object-oriented approach to 3D graphics (4th ed.)", "Kitware",
                  url="https://vtk.org", kind="book"),
        Reference("hunter2007matplotlib", "Matplotlib", "Slice views of the Multiplanar & 3D Viewer.",
                  ("Hunter, J. D.",), "2007", "Matplotlib: A 2D graphics environment",
                  "Computing in Science & Engineering", "9(3), 90–95", doi="10.1109/MCSE.2007.55"),
        Reference("thompson2023pyqt5", "PyQt5", "The user interface.",
                  ("Thompson, P.",), "2023", "PyQt5 (Version 5.15)", "Riverbank Computing [Computer software]",
                  url="https://www.riverbankcomputing.com/software/pyqt/", kind="software"),
    ]),
    ("Ontologies", [
        Reference("noy2009bioportal", "BioPortal", "Standard terms not in DICOM's own code list.",
                  ("Noy, N. F.", "Shah, N. H.", "Whetzel, P. L."), "2009",
                  "BioPortal: Ontologies and integrated data resources at the click of a mouse",
                  "Nucleic Acids Research", "37(Web Server issue), W170–W173", doi="10.1093/nar/gkp440",
                  et_al=True),
        Reference("vendetti2025bioportal", "BioPortal (2025)", "The current BioPortal service and API.",
                  ("Vendetti, J.", "Harris, N. L.", "Dorf, M. V."), "2025",
                  "BioPortal: An open community resource for sharing, searching, and utilizing biomedical "
                  "ontologies", "Nucleic Acids Research", "53(W1), W84–W94", doi="10.1093/nar/gkaf402",
                  et_al=True),
    ]),
    ("Optional components", [
        Reference("isensee2019hdbet", "HD-BET", "Skull stripping in the Multiplanar & 3D Viewer, when installed.",
                  ("Isensee, F.", "Schell, M.", "Pflueger, I."), "2019",
                  "Automated brain extraction of multisequence MRI using artificial neural networks",
                  "Human Brain Mapping", "40(17), 4952–4964", doi="10.1002/hbm.24750", et_al=True),
        Reference("isensee2021nnunet", "nnU-Net", "The network framework HD-BET runs on.",
                  ("Isensee, F.", "Jaeger, P. F.", "Kohl, S. A. A.", "Petersen, J.", "Maier-Hein, K. H."), "2021",
                  "nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation",
                  "Nature Methods", "18(2), 203–211", doi="10.1038/s41592-020-01008-z"),
        Reference("avants2011ants", "ANTs", "ANTsPy registration engine, when installed.",
                  ("Avants, B. B.", "Tustison, N. J.", "Song, G."), "2011",
                  "A reproducible evaluation of ANTs similarity metric performance in brain image registration",
                  "NeuroImage", "54(3), 2033–2044", doi="10.1016/j.neuroimage.2010.09.025", et_al=True),
    ]),
]


def all_references() -> List[Reference]:
    return [ref for _title, refs in SECTIONS for ref in refs]


def bibtex_file() -> str:
    entries = [app_reference()] + all_references()
    return "\n\n".join(ref.bibtex() for ref in entries) + "\n"


def plain_text() -> str:
    lines = ["How to cite DicomDoodle", app_reference().text(), software_note(), ""]
    for title, refs in SECTIONS:
        lines.append(title)
        lines += [f"  • {ref.component}: {ref.text()}" for ref in refs]
        lines.append("")
    return "\n".join(lines)
