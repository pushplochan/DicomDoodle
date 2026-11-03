"""
BioPortal ontology search — the second source of coded concepts, used when
DICOM's own code dictionary (core/coding.py) has no matching term.

Search results come from the first page of BioPortal's answer and are
filtered as in the original tool (labels up to 64 characters, code values
up to 16 characters — DICOM limits). Each result is a ``CodedTerm`` whose
coding scheme designator is the DICOM one (SNOMEDCT -> SCT, NCIT -> NCIt …,
PS3.16 Table 8-1), so annotation files stay readable by other DICOM software.

Requests go through the ``ontoportal_client`` package when it is installed
and straight to the BioPortal REST API otherwise, so the feature works
without it. Only the package's ``get_json`` is used, because it is the same
in every version: its ``search()`` returned one page up to 0.0.4 but yields
every result of every page since 0.0.5, which broke the original tool.
"""
from __future__ import annotations

import inspect
import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .coding import DICOM_TO_BIOPORTAL, CodedTerm, dicom_scheme

BIOPORTAL_API = "https://data.bioontology.org"


class OntologyError(RuntimeError):
    """A BioPortal request failed; the message is written for the user."""


def _first(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def parse_results(payload: Dict[str, Any]) -> List[CodedTerm]:
    """Turn a BioPortal /search response into terms usable in DICOM."""
    terms = []
    for item in payload.get("collection", []):
        label = _first(item.get("prefLabel", "[no label]"))
        if len(label) > 64:
            parts = [s.strip() for s in label.split(",")]
            label = next((s for s in parts if len(s) <= 64), "")
            if not label:
                continue
        code_value = _first(item.get("notation")) or item.get("@id", "").split("/")[-1].split("#")[-1]
        if len(code_value) > 16:
            continue
        links = item.get("links", {}) or {}
        ontology_link = links.get("ontology", "")
        acronym = ontology_link.split("/")[-1] if ontology_link else ""
        scheme = dicom_scheme(acronym)
        if not (label and code_value and acronym) or len(scheme) > 16 or "\\" in scheme:
            continue
        terms.append(CodedTerm(code_value, scheme, label, source="BioPortal", ontology=acronym,
                               term_id=item.get("@id", ""), self_link=links.get("self", ""),
                               ontology_link=ontology_link))
    return terms


def _describe(exc: Exception) -> str:
    """A failed request in words (never its URL: that contains the API key)."""
    status = getattr(getattr(exc, "response", None), "status_code", None) or getattr(exc, "code", None)
    if status in (401, 403):
        return "BioPortal did not accept the API key — check it in File ▸ Preferences ▸ Ontology."
    if status == 429:
        return "BioPortal received too many requests — wait a few seconds and try again."
    if isinstance(status, int):
        return f"BioPortal could not answer (HTTP {status}) — please try again later."
    if isinstance(exc, ValueError):
        return "BioPortal sent an answer that could not be read — please try again later."
    if "timed out" in str(exc).lower():
        return "BioPortal did not answer in time — check the internet connection and try again."
    return "Could not reach BioPortal — check the internet connection (some hospital networks block it)."


def _package_client(api_key: str):
    """The ``ontoportal_client`` package's BioPortal client (the original dependency), or None."""
    try:
        from ontoportal_client import BioPortalClient as PackageClient
    except ImportError:
        return None
    return PackageClient(api_key=api_key)


class BioPortalClient:
    def __init__(self, api_key: str, timeout: float = 25.0):
        self.api_key = api_key
        self.timeout = timeout
        self.package = _package_client(api_key)
        # since 0.0.6 the package takes a timeout, and its default (5 s) is short for BioPortal
        takes_timeout = (self.package is not None
                         and "timeout" in inspect.signature(self.package.get_response).parameters)
        self._package_options = {"timeout": timeout} if takes_timeout else {}

    def get_json(self, path: str, params: Optional[Dict[str, str]] = None) -> Any:
        """GET an API path (e.g. "/search") or a full API URL and decode the JSON answer.
        Network, HTTP and JSON errors raise OntologyError."""
        try:
            if self.package is not None:
                return self.package.get_json(path, params=dict(params or {}), **self._package_options)
            return self._direct_get_json(path, params or {})
        except (OSError, ValueError) as exc:       # (the errors of requests are OSErrors too)
            raise OntologyError(_describe(exc)) from exc

    def _direct_get_json(self, path: str, params: Dict[str, str]) -> Any:
        url = path if path.startswith("http") else f"{BIOPORTAL_API}/{path.lstrip('/')}"
        query = dict(params, apikey=self.api_key)
        separator = "&" if "?" in url else "?"
        request = urllib.request.Request(url + separator + urllib.parse.urlencode(query),
                                         headers={"Accept": "application/json",
                                                  "Authorization": f"apikey token={self.api_key}"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def search(self, term: str) -> List[CodedTerm]:
        """Terms usable in DICOM, from the first page of results (as the original tool)."""
        return parse_results(self.get_json("/search", {"q": term}))

    def details(self, term: CodedTerm) -> Tuple[List[str], List[str]]:
        """(definitions, synonyms) for a BioPortal result or a DICOM dictionary term."""
        link = term.self_link or class_link(term)
        if not link:
            raise OntologyError(f"BioPortal has no entry for codes of the {term.scheme} coding scheme.")
        payload = self.get_json(link)
        definitions = payload.get("definition") or []
        synonyms = payload.get("synonym") or []
        as_list = lambda v: [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]  # noqa: E731
        return as_list(definitions), as_list(synonyms)


def class_link(term: CodedTerm) -> Optional[str]:
    """BioPortal API path of a dictionary term's class (SNOMED CT, DICOM, NCIt and RadLex codes)."""
    known = DICOM_TO_BIOPORTAL.get(term.scheme)
    if known is None:
        return None
    acronym, prefix = known
    return f"/ontologies/{acronym}/classes/{urllib.parse.quote(prefix + term.value, safe='')}"
