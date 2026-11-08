"""
Coded concepts — the standard, machine-readable meaning of a segment.

DICOM describes what a segment shows with two coded concepts (PS3.3
C.8.20.2, Segment Description): a *category* from CID 7150 "Segmentation
Property Categories" (Tissue, Anatomical Structure, Morphologically
Abnormal Structure …) and a *type* from CID 7151 "Segmentation Property
Types" and its sub-groups (Neoplasm, Edema, Brain …).

pydicom ships the code dictionary of the DICOM Content Mapping Resource
(PS3.16: about 15,000 SNOMED CT, DICOM, NCI Thesaurus … codes and the
context groups that list them), so ``DicomCodeDictionary`` finds most terms
offline. The Ontology Search asks BioPortal (core/ontology.py) only when the
dictionary has nothing, or when the user asks for more.

Coding scheme designators are written as PS3.16 Table 8-1 spells them
("SCT" for SNOMED CT, "NCIt" for the NCI Thesaurus …); ``dicom_scheme``
translates BioPortal's ontology acronyms. A scheme that is not in that
table is a local scheme, and ``scheme_identification`` describes it in the
Coding Scheme Identification Sequence (PS3.3 C.12.1).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

Code = Tuple[str, str, str]          # (code value, coding scheme designator, code meaning)

CATEGORY_CID = 7150                  # Segmentation Property Categories
TYPE_CID = 7151                      # Segmentation Property Types

# The concepts DicomDoodle writes itself (category from CID 7150, type from CID 7151)
ANATOMICAL_STRUCTURE: Code = ("91723000", "SCT", "Anatomical Structure")
MORPHOLOGICALLY_ABNORMAL_STRUCTURE: Code = ("49755003", "SCT", "Morphologically Abnormal Structure")
BRAIN: Code = ("12738006", "SCT", "Brain")
EDEMA: Code = ("79654002", "SCT", "Edema")
LESION: Code = ("52988006", "SCT", "Lesion")
NECROSIS: Code = ("6574001", "SCT", "Necrosis")
NEOPLASM: Code = ("108369006", "SCT", "Neoplasm")

# BioPortal ontology acronym -> DICOM coding scheme designator (PS3.16 Table 8-1)
BIOPORTAL_TO_DICOM: Dict[str, str] = {
    "SNOMEDCT": "SCT", "NCIT": "NCIt", "RADLEX": "RADLEX", "LOINC": "LN", "FMA": "FMA",
    "UBERON": "UBERON", "MESH": "MSH", "ICD10": "I10", "ICD10CM": "I10C", "ICD10PCS": "I10P",
    "ICD9CM": "I9C", "RXNORM": "RXNORM", "DCM": "DCM", "MA": "MA", "UCUM": "UCUM", "CPT": "C4",
    "RCD": "CTV3", "SNMI": "SNM3",
}
# ... and back, to look up definitions of dictionary terms on BioPortal: (acronym, class IRI prefix)
DICOM_TO_BIOPORTAL: Dict[str, Tuple[str, str]] = {
    "SCT": ("SNOMEDCT", "http://purl.bioontology.org/ontology/SNOMEDCT/"),
    "SRT": ("SNOMEDCT", "http://purl.bioontology.org/ontology/SNOMEDCT/"),
    "DCM": ("DCM", "http://dicom.nema.org/resources/ontology/DCM/"),
    "NCIt": ("NCIT", "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"),
    "RADLEX": ("RADLEX", "http://www.radlex.org/RID/"),
}

# Designators of PS3.16 Table 8-1 (2026d); any other designator is a local coding scheme.
STANDARD_SCHEMES = frozenset("""
    ACR ASTM-sigpurpose BARI BI C4 C5 caDSR CD2 CTV3 DC DCM DCMUID FMA HPC I10 I10C I10P I11 I9 I9C
    ICDO3 IBSI IETF4646 ISO639_1 ISO639_2 ISO3166_1 ISO5218_1 ISO_OID ITIS_TSN LN MA MAYOASRG MDC MDNS
    MGI MSH NBD NBG NCDR NCIt NDC NEU NICIP NPI NYUMCCG PATHLEX POS PUBCHEM_CID RADLEX RADELEMENT
    RFC3066 RFC-3881 RFC5646 RO RRID RXNORM 99SDM SCPECG SNM3 SCT SRT UBERON UCUM UMLS UNS UPC
""".split())

# Schemes of pydicom's dictionary that never name what a segment shows (units, devices, features).
_SKIPPED_SCHEMES = frozenset({"UCUM", "MDC", "IBSI", "LN", "PUBCHEM_CID", "NDC", "RXNORM", "ITIS_TSN"})

# Everyday words for terms the dictionary spells differently.
_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "tumor": ("neoplasm",), "tumors": ("neoplasm",), "tumour": ("neoplasm",), "tumours": ("neoplasm",),
    "cancer": ("malignant neoplasm",), "bleed": ("hemorrhage",), "bleeding": ("hemorrhage",),
    "enhancing tumor": ("enhancing lesion",), "enhancing tumour": ("enhancing lesion",),
    "non-enhancing tumor": ("non-enhancing lesion",), "non-enhancing tumour": ("non-enhancing lesion",),
    "necrotic": ("necrosis",), "oedematous": ("edema",), "edematous": ("edema",),
}


@dataclass(frozen=True)
class CodedTerm:
    """A coded concept found in the DICOM dictionary or on BioPortal."""

    value: str                              # Code Value
    scheme: str                             # Coding Scheme Designator, DICOM spelling (e.g. SCT)
    meaning: str                            # Code Meaning
    source: str = "DICOM"                   # "DICOM" (built-in dictionary) or "BioPortal"
    context_groups: Tuple[int, ...] = ()    # CIDs that list the code (DICOM dictionary only)
    synonyms: Tuple[str, ...] = ()
    ontology: str = ""                      # BioPortal ontology acronym (e.g. SNOMEDCT)
    term_id: str = ""                       # BioPortal class IRI
    self_link: str = ""                     # BioPortal API link of the class (definitions, synonyms)
    ontology_link: str = ""                 # BioPortal API link of the ontology

    # Labels as the original tool's ontology window used them.
    @property
    def label(self) -> str:
        return self.meaning

    @property
    def code_value(self) -> str:
        return self.value

    @property
    def code(self) -> Code:
        return self.value, self.scheme, self.meaning

    @property
    def is_category(self) -> bool:
        """Listed in CID 7150 — the categories the standard asks for."""
        return CATEGORY_CID in self.context_groups

    @property
    def is_standard_scheme(self) -> bool:
        return self.scheme in STANDARD_SCHEMES

    def coded_concept(self):
        from highdicom.sr.coding import CodedConcept
        return CodedConcept(value=self.value, scheme_designator=self.scheme, meaning=self.meaning)

    def context_group_names(self) -> List[str]:
        return [f"CID {cid} {context_group_name(cid)}" for cid in self.context_groups]


def coded_concept(code: Code):
    """highdicom CodedConcept for a (value, scheme, meaning) triple."""
    from highdicom.sr.coding import CodedConcept
    value, scheme, meaning = code
    return CodedConcept(value=value, scheme_designator=scheme, meaning=meaning)


def dicom_scheme(bioportal_acronym: str) -> str:
    """DICOM coding scheme designator for a BioPortal ontology (the acronym itself if DICOM has none)."""
    return BIOPORTAL_TO_DICOM.get(bioportal_acronym.upper(), bioportal_acronym)


def scheme_identification(term: CodedTerm):
    """Coding Scheme Identification item for a local (non PS3.16) scheme, else None."""
    if term.is_standard_scheme:
        return None
    from pydicom.dataset import Dataset
    item = Dataset()
    item.CodingSchemeDesignator = term.scheme
    item.CodingSchemeName = f"{term.ontology or term.scheme} (BioPortal)"[:64]
    acronym = term.ontology or term.scheme
    resource = Dataset()
    resource.CodingSchemeURLType = "DOC"
    resource.CodingSchemeURL = f"https://bioportal.bioontology.org/ontologies/{acronym}"
    from pydicom.sequence import Sequence as DicomSequence
    item.CodingSchemeResourcesSequence = DicomSequence([resource])
    return item


def context_group_name(cid: int) -> str:
    """'Lesion Segmentation Type' for 7159 (pydicom stores the names in CamelCase)."""
    from pydicom.sr._cid_dict import name_for_cid
    name = name_for_cid.get(int(cid), "")
    name = re.sub(r"(?<=[a-z])of(?=[A-Z])", " of", name)          # "SourceofVascularFinding"
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)


def _is_segmentation_group(cid: int) -> bool:
    return cid == 4 or 7150 <= cid < 7200      # anatomic regions, segmentation categories and types


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _spellings(query: str) -> List[str]:
    """The query, its American spelling and its everyday synonyms (all casefolded)."""
    base = " ".join(query.casefold().split())
    american = re.sub(r"(?<=[a-z])our\b", "or", base)            # tumour -> tumor
    american = re.sub(r"\b(f?)oe", r"\1e", american)               # oedema -> edema, foetal -> fetal
    american = american.replace("ae", "e")                         # haemorrhage -> hemorrhage
    forms = [base] + ([american] if american != base else [])
    for form in list(forms):
        forms += [s for s in _SYNONYMS.get(form, ()) if s not in forms]
    return forms


class DicomCodeDictionary:
    """Searchable copy of pydicom's DICOM code dictionary (built on first use)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._terms: Optional[List[CodedTerm]] = None
        self._by_code: Dict[Tuple[str, str], CodedTerm] = {}

    def _build(self) -> List[CodedTerm]:
        from pydicom.sr._concepts_dict import concepts
        entries: Dict[Tuple[str, str], List[Tuple[str, Tuple[int, ...]]]] = {}
        for scheme, keywords in concepts.items():
            if scheme in _SKIPPED_SCHEMES:
                continue
            for codes in keywords.values():
                for value, (meaning, cids) in codes.items():
                    entries.setdefault((scheme, value), []).append((meaning, tuple(sorted(set(cids)))))
        terms = []
        for (scheme, value), variants in entries.items():
            # the meaning used by the segmentation context groups (else by most groups) is the DICOM
            # one; the others are synonyms (e.g. 108369006: "Neoplasm", also "Tumor")
            variants.sort(key=lambda v: (not any(_is_segmentation_group(c) for c in v[1]), -len(v[1]), len(v[0])))
            meaning = variants[0][0]
            cids = tuple(sorted({cid for _, groups in variants for cid in groups}))
            synonyms = tuple(dict.fromkeys(m for m, _ in variants[1:] if m.casefold() != meaning.casefold()))
            if len(value) > 16 or len(meaning) > 64:      # DICOM limits of Code Value and Code Meaning
                continue
            term = CodedTerm(value, scheme, meaning, "DICOM", cids, synonyms)
            terms.append(term)
            self._by_code[(scheme, value)] = term
        return terms

    @property
    def terms(self) -> List[CodedTerm]:
        with self._lock:
            if self._terms is None:
                self._terms = self._build()
            return self._terms

    def lookup(self, scheme: str, value: str) -> Optional[CodedTerm]:
        self.terms          # noqa: B018 — builds the index
        return self._by_code.get((scheme, value))

    def context_group(self, cid: int) -> List[CodedTerm]:
        """Every term of a context group, e.g. the CID 7150 categories, sorted by meaning."""
        return sorted((t for t in self.terms if cid in t.context_groups), key=lambda t: t.meaning.casefold())

    def search(self, query: str, limit: int = 200) -> List[CodedTerm]:
        """Terms whose meaning (or a synonym, or the code value) matches every word of ``query``.

        Exact matches come first, then meanings that start with the query,
        then the rest; within each, terms used for segmentation (CID 7150-7199
        and anatomic regions) before other listed terms, before unlisted ones.
        """
        forms = _spellings(query)
        if not forms or not forms[0]:
            return []
        scored = []
        for term in self.terms:
            best = None
            for k, form in enumerate(forms):
                quality = self._match(term, form)
                if quality is not None:
                    quality += 0.5 if k else 0          # other spellings / everyday words after the literal query
                    best = quality if best is None else min(best, quality)
            if best is None:
                continue
            relevance = 0 if any(_is_segmentation_group(c) for c in term.context_groups) else \
                1 if term.context_groups else 2
            scored.append(((best, relevance, len(term.meaning), term.meaning.casefold()), term))
        scored.sort(key=lambda item: item[0])
        return [term for _, term in scored[:limit]]

    @staticmethod
    def _match(term: CodedTerm, form: str) -> Optional[int]:
        """0/1 exact (meaning/synonym), 2/3 starts with the query, 4/5 every query word starts a word."""
        if form == term.value.casefold():
            return 0
        joined = " ".join(_words(form))
        best = None
        for k, text in enumerate((term.meaning,) + term.synonyms):
            folded = " ".join(_words(text))
            if folded == joined:
                quality = 0
            elif folded.startswith(joined):
                quality = 2
            else:
                words = folded.split()
                if not all(any(w.startswith(q) for w in words) for q in joined.split()):
                    continue
                quality = 4
            quality += 1 if k else 0                                     # a synonym after the DICOM meaning
            best = quality if best is None else min(best, quality)
        return best


_dictionary = DicomCodeDictionary()


def dicom_dictionary() -> DicomCodeDictionary:
    """The shared dictionary (indexed once per run, in about a second)."""
    return _dictionary


def search_dicom(query: str, limit: int = 200) -> List[CodedTerm]:
    return _dictionary.search(query, limit)


def categories() -> List[CodedTerm]:
    """The CID 7150 segmentation property categories."""
    return _dictionary.context_group(CATEGORY_CID)


def unique_terms(terms: Iterable[CodedTerm]) -> List[CodedTerm]:
    """Drop repeated (scheme, value) pairs, keeping the first."""
    seen, out = set(), []
    for term in terms:
        key = (term.scheme, term.value)
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out

