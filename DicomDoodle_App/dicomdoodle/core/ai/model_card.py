"""
The model card: one file that holds DicomDoodle's AI models.

DicomDoodle's installer has no AI models; they come as a separate download, the
model card — a file whose name starts with "dicomdoodle" and ends in .ddcard
(a zip archive of modelcard.json, MODEL_CARD.md and the model files, which may
have any names). Without one, DicomDoodle is for manual annotation only.

At start, DicomDoodle looks for card files where a download lands (Downloads,
Desktop, Documents), next to the program, and in the folder chosen in the
preferences; a card chosen with File ▸ Load Model Card… is installed at once.
A card found for the first time is installed: unpacked into DicomDoodle's data
folder (%LOCALAPPDATA%\\DicomDoodle\\ModelCard on Windows), so later starts
just load it and the downloaded file may be deleted. A newer card found later
replaces the installed one. Remove deletes it, and DicomDoodle then stops
looking for cards until one is loaded with File ▸ Load Model Card….

What each file of the card is for is read from the file itself (model_files.py).
For every purpose the newest model wins, in the fastest format this computer
can run on the chosen device — ONNX (ONNX Runtime) before PyTorch, unless only
PyTorch can reach the graphics card. scripts/make_model_card.py makes cards.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ... import APP_NAME, resources
from ..progress import Progress, ensure_progress
from . import devices
from .model_files import (MODEL_EXTENSIONS, SAM, SAM2_DECODER, SAM2_ENCODER, ModelFile, NotAModel, inspect,
                          torch_loadable)

log = logging.getLogger(__name__)

CARD_PREFIX = "dicomdoodle"
CARD_EXTENSION = ".ddcard"
CARD_EXTENSIONS = (CARD_EXTENSION, ".zip")      # a card renamed (or re-zipped) to .zip still counts
MANIFEST = "modelcard.json"
CARD_FORMAT = "dicomdoodle-model-card"
INSTALL_FOLDER = "ModelCard"
INDEX = "installed.json"                        # in the installed folder: where the card came from
DATA_DIR_ENV = "DICOMDOODLE_DATA_DIR"           # another data folder (tests, the self-test)
SPARE_BYTES = 200 * 1024 * 1024                 # free space left over after installing
INSTALLING = "Unpacking the model card (first time only)"     # install()'s progress messages start so


class CardError(RuntimeError):
    """A file is not a usable model card, or installing it failed (message in the user's words)."""


# --------------------------------------------------------------------------- #
# Folders
# --------------------------------------------------------------------------- #

def data_dir() -> str:
    """DicomDoodle's folder for large data (the installed model card)."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return override
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, APP_NAME)


def install_dir() -> str:
    return os.path.join(data_dir(), INSTALL_FOLDER)


def _windows_user_folders() -> List[str]:
    """Downloads, Desktop and Documents where Windows really keeps them (they may have been moved,
    e.g. to OneDrive)."""
    import winreg

    names = ("{374DE290-123F-4565-9164-39C4925E467B}", "Desktop", "Personal")   # Downloads, Desktop, Documents
    found = []
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            for name in names:
                try:
                    found.append(os.path.expandvars(winreg.QueryValueEx(key, name)[0]))
                except OSError:
                    pass
    except OSError:
        pass
    return found


def user_folders() -> List[str]:
    """Where a downloaded card is likely to be."""
    home = os.path.expanduser("~")
    folders = []
    if sys.platform.startswith("win"):
        try:
            folders += _windows_user_folders()
        except Exception:
            pass
    return folders + [os.path.join(home, name) for name in ("Downloads", "Desktop", "Documents")]


def search_folders(extra_dirs: Iterable[str] = ()) -> List[str]:
    folders = [d for d in extra_dirs if d]
    if getattr(sys, "frozen", False):
        folders.append(os.path.dirname(sys.executable))        # next to DicomDoodle.exe
    else:
        folders += [resources.APP_DIR, resources.LEGACY_DIR]    # running from the source folder
    folders += user_folders() + [os.getcwd()]
    unique, seen = [], set()
    for folder in folders:
        key = os.path.normcase(os.path.abspath(folder))
        if key not in seen and os.path.isdir(folder):
            seen.add(key)
            unique.append(folder)
    return unique


# --------------------------------------------------------------------------- #
# Card files
# --------------------------------------------------------------------------- #

def is_card_name(filename: str) -> bool:
    name = os.path.basename(filename).lower()
    return name.startswith(CARD_PREFIX) and name.endswith(CARD_EXTENSIONS)


def _version_key(version: str) -> Tuple:
    parts = []
    for piece in str(version).replace("-", ".").split("."):
        parts.append((0, int(piece), "") if piece.isdigit() else (1, 0, piece))
    return tuple(parts)


@dataclass
class CardFile:
    """A model card file (not installed yet)."""

    path: str
    size: int
    mtime: float
    manifest: Dict
    members: List[Tuple[str, int, int]]       # (name in the archive, size, CRC-32), files only

    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or os.path.splitext(os.path.basename(self.path))[0])

    @property
    def version(self) -> str:
        return str(self.manifest.get("version") or "")

    @property
    def label(self) -> str:
        return f"{self.name} {self.version}".strip()

    @property
    def total_bytes(self) -> int:
        return sum(size for _name, size, _crc in self.members)

    @property
    def identity(self) -> str:
        """Same for every copy of one card, wherever it is and whatever its file is called."""
        text = json.dumps([self.name, self.version, sorted(self.members)])
        return hashlib.sha1(text.encode()).hexdigest()[:16]


def _safe_member(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return not (name.startswith(("/", "\\")) or ":" in parts[0] or ".." in parts)


def read_card(path: str) -> CardFile:
    """The card file at ``path``; raises CardError when it is not one."""
    if not os.path.isfile(path):
        raise CardError(f"{path} does not exist.")
    if not zipfile.is_zipfile(path):
        raise CardError(f"{os.path.basename(path)} is not a DicomDoodle model card.")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = [i for i in archive.infolist() if not i.is_dir() and _safe_member(i.filename)]
            names = {i.filename for i in infos}
            manifest = json.loads(archive.read(MANIFEST).decode("utf-8")) if MANIFEST in names else {}
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise CardError(f"{os.path.basename(path)} could not be read ({exc}); download it again.") from exc
    if manifest and manifest.get("format") not in (None, CARD_FORMAT):
        raise CardError(f"{os.path.basename(path)} is not a DicomDoodle model card.")
    if not any(i.filename.lower().endswith(MODEL_EXTENSIONS) for i in infos):
        raise CardError(f"{os.path.basename(path)} holds no model files.")
    stat = os.stat(path)
    members = [(i.filename, i.file_size, i.CRC) for i in infos]
    return CardFile(os.path.abspath(path), stat.st_size, stat.st_mtime, manifest if isinstance(manifest, dict) else {},
                    members)


def find_cards(extra_dirs: Iterable[str] = ()) -> List[CardFile]:
    """Every model card file in the search folders, newest version first."""
    cards: Dict[str, CardFile] = {}
    for folder in search_folders(extra_dirs):
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            if not is_card_name(entry.name):
                continue
            try:
                if entry.is_file():
                    card = read_card(entry.path)
                    cards.setdefault(os.path.normcase(os.path.realpath(card.path)), card)
            except (CardError, OSError) as exc:
                log.debug("Not a model card: %s (%s)", entry.path, exc)
    return sorted(cards.values(), key=lambda c: (_version_key(c.version), c.mtime), reverse=True)


# --------------------------------------------------------------------------- #
# The installed card
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    """One way to provide a model: a file (or SAM 2's encoder + decoder) and its engine."""

    kind: str
    engine: str
    files: Tuple[ModelFile, ...]

    @property
    def title(self) -> str:
        title = self.files[0].title
        return title.replace(" image encoder", "") if self.kind == SAM else title

    @property
    def date(self) -> str:
        return max(f.date[:19] for f in self.files)


@dataclass
class InstalledCard:
    folder: str
    name: str
    version: str
    identity: str
    source: str                       # the card file it was installed from
    installed: str                    # date (ISO 8601)
    manifest: Dict
    models: List[ModelFile] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)       # files in the card that are no usable model

    @property
    def label(self) -> str:
        return f"{self.name} {self.version}".strip()

    def candidates(self, kind: str) -> List[Candidate]:
        found = [Candidate(kind, m.engine, (m,)) for m in self.models if m.kind == kind]
        if kind == SAM:                                     # SAM 2 in ONNX: an encoder with its decoder
            decoders = [m for m in self.models if m.kind == SAM2_DECODER]
            for encoder in (m for m in self.models if m.kind == SAM2_ENCODER):
                decoder = next((d for d in decoders if d.family == encoder.family), None)
                if decoder is not None:
                    found.append(Candidate(SAM, "onnx", (encoder, decoder)))
        return found


def have_onnxruntime() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


def usable(candidate: Candidate) -> Tuple[bool, str]:
    """Whether this computer can run the candidate at all (its engine is installed)."""
    if candidate.engine == "onnx":
        return (True, "") if have_onnxruntime() else (False, "ONNX Runtime (onnxruntime) is not installed")
    if importlib.util.find_spec("ultralytics") is None:
        return False, "Ultralytics is not installed"
    return torch_loadable(candidate.files[0])


def engine_preference(engine: str, device: str) -> int:
    """ONNX before PyTorch: faster on the CPU and the way to DirectML; PyTorch first only where it
    alone reaches the chosen graphics card (CUDA PyTorch without a GPU build of ONNX Runtime)."""
    gpu = devices.known_gpu()
    if devices.normalize(device) == devices.GPU and gpu is not None and gpu.torch_cuda and not gpu.onnx_provider:
        return 2 if engine == "torch" else 1
    return 2 if engine == "onnx" else 1


def ranked(card: Optional[InstalledCard], kind: str, device: str) -> List[Tuple[Candidate, str]]:
    """The candidates for ``kind``, best first, each with why it cannot run ('' = it can):
    the newest model, then the preferred engine."""
    if card is None:
        return []
    order = sorted(card.candidates(kind), key=lambda c: (c.date, engine_preference(c.engine, device), c.title),
                   reverse=True)
    return [(c, usable(c)[1]) for c in order]


def _inspect_folder(folder: str) -> Tuple[List[ModelFile], List[str]]:
    models, skipped = [], []
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if not name.lower().endswith(MODEL_EXTENSIONS):
                continue
            path = os.path.join(root, name)
            try:
                models.append(inspect(path))
            except NotAModel as exc:
                skipped.append(str(exc))
    return models, skipped


def installed() -> Optional[InstalledCard]:
    """The installed card, None when there is none (or it is damaged)."""
    folder = install_dir()
    index_path = os.path.join(folder, INDEX)
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
    except (OSError, ValueError):
        return None
    models, skipped = _inspect_folder(folder)
    if not models:
        return None
    return InstalledCard(folder, index.get("name", ""), index.get("version", ""), index.get("identity", ""),
                         index.get("source", ""), index.get("installed", ""), index.get("manifest", {}),
                         models, skipped)


def install(card: CardFile, progress: Optional[Progress] = None) -> InstalledCard:
    """Unpack ``card`` into the data folder (replacing the installed card) and check its files."""
    step = ensure_progress(progress)
    root = data_dir()
    target, partial, old = install_dir(), install_dir() + ".partial", install_dir() + ".old"
    try:
        os.makedirs(root, exist_ok=True)
        shutil.rmtree(partial, ignore_errors=True)
        free = shutil.disk_usage(root).free
    except OSError as exc:
        raise CardError(f"The model card cannot be installed in {root}: {exc}") from exc
    total = card.total_bytes
    if free < total + SPARE_BYTES:
        raise CardError(f"Not enough free disk space for the model card: it needs {total / 1e9:.1f} GB "
                        f"in {root}, and {free / 1e9:.1f} GB are free.")
    expected = {str(item.get("file")): str(item.get("sha256", "")).lower()
                for item in card.manifest.get("files", []) if isinstance(item, dict)}
    done, started = 0, time.time()
    try:
        with zipfile.ZipFile(card.path) as archive:
            for name, _size, _crc in card.members:
                destination = os.path.join(partial, *name.replace("\\", "/").split("/"))
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(name) as source, open(destination, "wb") as out:
                    while True:
                        chunk = source.read(4 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        step(done >> 20, total >> 20,          # in MiB: Qt's progress counts are 32-bit
                             f"{INSTALLING}… {done / 1e6:,.0f} of {total / 1e6:,.0f} MB")
                if expected.get(name) and digest.hexdigest() != expected[name]:
                    raise CardError(f"{os.path.basename(card.path)} is damaged ({name} differs from the card's "
                                    "list); download it again.")
    except zipfile.BadZipFile as exc:
        shutil.rmtree(partial, ignore_errors=True)
        raise CardError(f"{os.path.basename(card.path)} is damaged ({exc}); download it again.") from exc
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    models, skipped = _inspect_folder(partial)
    if not models:
        shutil.rmtree(partial, ignore_errors=True)
        raise CardError(f"{os.path.basename(card.path)} holds no model DicomDoodle can use.\n\n" + "\n".join(skipped))
    index = {"name": card.name, "version": card.version, "identity": card.identity, "source": card.path,
             "installed": time.strftime("%Y-%m-%dT%H:%M:%S"), "manifest": card.manifest}
    with open(os.path.join(partial, INDEX), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    _swap(partial, target, old)
    log.info("Model card %s installed from %s (%.0f MB, %.0f s)", card.label, card.path, total / 1e6,
             time.time() - started)
    return installed()


def prepare(chosen: str = "", extra_dirs: Iterable[str] = (), search: bool = True,
            progress: Optional[Progress] = None) -> Tuple[Optional[InstalledCard], str]:
    """The card to use, installing one when needed, and what happened: "installed" (``chosen``,
    or the first card found), "updated" (a newer card replaced the installed one), "ready" (the
    installed card stays) or "none" (no card: manual annotation only).

    chosen: a card file the user picked (File ▸ Load Model Card…): installed unless it is the
        installed card already. Raises CardError when it is no card.
    search: look for card files (off after the user removed the card, and in the self-test).
    """
    step = ensure_progress(progress)
    current = installed()
    if chosen:
        card = read_card(chosen)
        if current is not None and current.identity == card.identity:
            return current, "ready"
        return install(card, step), "installed"
    step(0, 0, "Looking for the model card…")
    found = find_cards(extra_dirs) if search else []
    if found:
        newest = found[0]
        if current is None:
            return install(newest, step), "installed"
        if newest.identity != current.identity and _version_key(newest.version) > _version_key(current.version):
            return install(newest, step), "updated"
    return current, ("ready" if current is not None else "none")


def _retry(action, *args) -> None:
    """Windows: a virus scanner may hold a file just written for a moment."""
    for attempt in range(6):
        try:
            return action(*args)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(1.0)


def _swap(partial: str, target: str, old: str) -> None:
    """The unpacked card replaces the installed one; on failure the installed one stays."""
    shutil.rmtree(old, ignore_errors=True)
    moved = False
    try:
        if os.path.isdir(target):
            _retry(os.replace, target, old)
            moved = True
        _retry(os.replace, partial, target)
    except OSError as exc:
        if moved and not os.path.isdir(target):
            try:
                os.replace(old, target)
            except OSError:
                pass
        shutil.rmtree(partial, ignore_errors=True)
        raise CardError(f"The model card could not be installed in {target}: {exc}") from exc
    shutil.rmtree(old, ignore_errors=True)


def remove_installed() -> None:
    folder = install_dir()
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        log.info("Model card removed (%s)", folder)
