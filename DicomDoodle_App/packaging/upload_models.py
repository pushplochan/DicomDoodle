#!/usr/bin/env python3
"""
Uploads the three AI model files to the GitHub release "models-v1" of
pushplochan/DicomDoodle; the Windows build downloads them from there
(.github/workflows/windows-installer.yml). Run it once, from any folder
(again only when a model file changes):

    python DicomDoodle_App/packaging/upload_models.py

It asks for a GitHub access token with "Contents: Read and write" access to
the repository (typed invisibly, never saved), creates the release if it does
not exist yet and skips files that are already there. Standard library only.
"""
import getpass
import json
import os
import urllib.error
import urllib.parse
import urllib.request

REPO = "pushplochan/DicomDoodle"
TAG = "models-v1"
MODELS = ["best_flair.pt", "yoloe-11l-seg-pf.pt", "sam2.1_l.pt"]
MODELS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
RELEASE_NOTES = ("The AI model files of DicomDoodle 2. The Windows build downloads them from here and puts "
                 "them into the installer. This is not an app release: get DicomDoodle from the newest v… release.")
HINTS = {
    401: "The token is wrong or has expired: make a new one.",
    403: "The token lacks access: it needs 'Contents: Read and write' for the DicomDoodle repository.",
    404: "The token cannot see the repository: choose DicomDoodle under 'Repository access'.",
}


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
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and method == "GET":
            return None
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"\nGitHub refused it ({exc.code} {exc.reason}): {detail}\n{HINTS.get(exc.code, '')}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"\nCould not reach GitHub: {exc.reason}")
    return json.loads(answer) if answer else None


def main() -> int:
    paths = [os.path.join(MODELS_DIR, name) for name in MODELS]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise SystemExit("Model files not found: " + ", ".join(missing))
    total = sum(os.path.getsize(path) for path in paths)
    print(f"Uploading {', '.join(MODELS)} ({total / 1e6:.0f} MB) to the release '{TAG}' of github.com/{REPO}.")
    token = getpass.getpass("GitHub token (paste it, nothing is shown, then press Enter): ").strip()
    if not token:
        raise SystemExit("No token given.")

    release = call("GET", f"{API}/repos/{REPO}/releases/tags/{TAG}", token)
    if release is None:
        release = call("POST", f"{API}/repos/{REPO}/releases", token,
                       {"tag_name": TAG, "name": "AI model files (for the Windows build)",
                        "body": RELEASE_NOTES, "make_latest": "false"})
        print(f"Created the release '{TAG}'.")
    assets = {asset["name"]: asset for asset in release.get("assets", [])}
    upload_url = release["upload_url"].split("{")[0]

    for path in paths:
        name, size = os.path.basename(path), os.path.getsize(path)
        old = assets.get(name)
        if old and old.get("state") == "uploaded" and old.get("size") == size:
            print(f"  {name}: already there")
            continue
        if old:                                       # an interrupted or different earlier upload
            call("DELETE", f"{API}/repos/{REPO}/releases/assets/{old['id']}", token)
        call("POST", f"{upload_url}?name={urllib.parse.quote(name)}", token, UploadReader(path),
             {"Content-Type": "application/octet-stream", "Content-Length": str(size)})
        print()
    print(f"Done: https://github.com/{REPO}/releases/tag/{TAG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
