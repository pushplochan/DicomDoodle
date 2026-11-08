#!/usr/bin/env python3
"""
Uploads a model card (the AI models of DicomDoodle, made with
DicomDoodle_App/scripts/make_model_card.py) to the GitHub release
"model-card-<version>" of pushplochan/DicomDoodle. Users download it from there,
and the Windows build's self-test uses it (.github/workflows/windows-installer.yml).

    python DicomDoodle_App/packaging/upload_model_card.py [CARD]

CARD defaults to the newest DicomDoodle-ModelCard-*.ddcard in the repository
root. The version comes from the card itself (its modelcard.json). It asks for
a GitHub access token with "Contents: Read and write" access to the
repository (typed invisibly, never saved), creates the release if it does not
exist yet (never marked "latest", so the app's newest release stays the one
the download page shows) and skips the file when it is already there.
Standard library only.
"""
import getpass
import glob
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

REPO = "pushplochan/DicomDoodle"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
HINTS = {
    401: "The token is wrong or has expired: make a new one.",
    403: "The token lacks access: it needs 'Contents: Read and write' for the DicomDoodle repository.",
    404: "The token cannot see the repository: choose DicomDoodle under 'Repository access'.",
}


def release_notes(card: str, manifest: dict) -> str:
    name = os.path.basename(card)
    rows = "\n".join(f"| {f.get('purpose', '')} | {f.get('title', '')} |" for f in manifest.get("files", []))
    return (f"**The AI models of DicomDoodle** — download `{name}` below (under *Assets*). This is not an app "
            "release: get DicomDoodle itself from the newest `v…` release.\n\n"
            "Put the file in your **Downloads** folder and start DicomDoodle (2.1 or newer): it finds the card, "
            "installs it once and loads the models (a notification at the bottom right shows the progress). Or "
            "open it with *File ▸ Load Model Card…*. Without a model card DicomDoodle works for manual "
            "annotation only.\n\n| Purpose | Model |\n|---|---|\n" + rows + "\n\nAll models are in ONNX format "
            "and run on the processor or, when chosen, on a graphics card. Not a medical device.")


class UploadReader:
    """Reads a file for the upload and shows how much has been sent."""

    def __init__(self, path: str):
        self.name, self.size = os.path.basename(path), os.path.getsize(path)
        self.file, self.sent, self.shown = open(path, "rb"), 0, -1

    def read(self, size: int = -1) -> bytes:
        chunk = self.file.read(size if size and size > 0 else 1 << 20)
        self.sent += len(chunk)
        percent = 100 * self.sent // max(self.size, 1)
        if percent != self.shown:
            self.shown = percent
            print(f"\r  {self.name}: {percent:3d} % of {self.size / 1e6:.0f} MB", end="", flush=True)
        if not chunk:
            self.file.close()
        return chunk


def call(method: str, url: str, token: str, body=None, headers=None):
    """One GitHub API request; JSON answer (None for 404 on GET or an empty answer)."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", **(headers or {})}
    if isinstance(body, dict):
        body = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            answer = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and method == "GET":
            return None
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"\nGitHub refused it ({exc.code} {exc.reason}): {detail}\n{HINTS.get(exc.code, '')}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"\nCould not reach GitHub: {exc.reason}")
    return json.loads(answer) if answer else None


def main(argv) -> int:
    cards = argv[:1] or sorted(glob.glob(os.path.join(REPO_DIR, "DicomDoodle-ModelCard-*.ddcard")),
                               key=os.path.getmtime)[-1:]
    if not cards or not os.path.isfile(cards[0]):
        raise SystemExit("No model card found: give its path (make one with DicomDoodle_App/scripts/make_model_card.py).")
    card = cards[0]
    with zipfile.ZipFile(card) as archive:
        manifest = json.loads(archive.read("modelcard.json"))
    version = str(manifest.get("version") or "").strip()
    if not version:
        raise SystemExit(f"{card}: its modelcard.json has no version.")
    tag, name, size = f"model-card-{version}", os.path.basename(card), os.path.getsize(card)
    print(f"Uploading {name} ({size / 1e6:.0f} MB) to the release '{tag}' of github.com/{REPO}.")
    token = getpass.getpass("GitHub token (paste it, nothing is shown, then press Enter): ").strip()
    if not token:
        raise SystemExit("No token given.")

    release = call("GET", f"{API}/repos/{REPO}/releases/tags/{tag}", token)
    if release is None:
        release = call("POST", f"{API}/repos/{REPO}/releases", token,
                       {"tag_name": tag, "name": f"{manifest.get('name', 'DicomDoodle Model Card')} {version}",
                        "body": release_notes(card, manifest), "make_latest": "false"})
        print(f"Created the release '{tag}'.")
    old = {asset["name"]: asset for asset in release.get("assets", [])}.get(name)
    if old and old.get("state") == "uploaded" and old.get("size") == size:
        print(f"  {name}: already there")
    else:
        if old:                                       # an interrupted or different earlier upload
            call("DELETE", f"{API}/repos/{REPO}/releases/assets/{old['id']}", token)
        upload_url = release["upload_url"].split("{")[0]
        call("POST", f"{upload_url}?name={urllib.parse.quote(name)}", token, UploadReader(card),
             {"Content-Type": "application/octet-stream", "Content-Length": str(size)})
        print()
    print(f"Done: https://github.com/{REPO}/releases/tag/{tag}")
    print(f"Download link: https://github.com/{REPO}/releases/download/{tag}/{urllib.parse.quote(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
