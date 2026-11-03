# Windows installer

GitHub builds `DicomDoodle-<version>-Setup.exe` on one of its Windows machines
whenever a version tag (`v…`) is pushed, and attaches it to that version's
release. The recipe is [`.github/workflows/windows-installer.yml`](../../.github/workflows/windows-installer.yml);
it installs the packages, builds `DicomDoodle.exe`, runs its self-test, makes
the installer and publishes it (about an hour).

| File | What it is |
|---|---|
| `requirements-windows.txt` | the packages of the build, exact versions (PyTorch CPU-only, installed first) |
| `constraints-windows.txt` | the version of every package they pull in, as tested (`pip freeze`) |
| `DicomDoodle.spec` | PyInstaller recipe: `DicomDoodle.exe` + its `_internal/` folder, the AI models included |
| `DicomDoodle.iss` | Inno Setup script: the installer (Start menu, optional desktop icon, uninstaller, no administrator rights needed) |
| `DicomDoodle.ico` | the icon of `DicomDoodle.exe` and of the installer, in Windows' icon format (16–256 px) |
| `upload_models.py` | uploads the three `.pt` files to the release `models-v1`, where the build gets them |

## Once: upload the AI models

The model files are too large for git (GitHub refuses files over 100 MB), so
`.gitignore` keeps them out and the build downloads them from the release
`models-v1` instead. From the repository folder:

    python DicomDoodle_App/packaging/upload_models.py

It asks for a GitHub token with *Contents: Read and write* access to the
repository. Run it again only when a model file changes (it replaces only
what differs).

## Make a release

1. Raise `APP_VERSION` in `dicomdoodle/__init__.py`, commit and push.
2. Tag the version and push the tag: `git tag v2.0.1 && git push origin v2.0.1`
3. Watch the **Actions** tab. When the run is green, the release `v2.0.1`
   on the **Releases** page has the installer under *Assets*.

A failed run can be re-run from its page. **Actions ▸ Windows installer ▸ Run
workflow** builds without a tag: the installer is then kept for 7 days on the
run's page and no release is made.

## Good to know

- `DicomDoodle.exe --self-test report.txt` checks an installed copy: modules,
  SEG writing, DICOM codes, the three AI models and HD-BET (exit code 0 = OK).
- The installed app writes its messages to `%APPDATA%\DicomDoodle\dicomdoodle.log`.
- HD-BET downloads its network to `%USERPROFILE%\hd-bet_params` the first time
  skull stripping runs (needs internet once).
- The installer is not code-signed, so Windows SmartScreen asks once:
  *More info ▸ Run anyway*.
- `file.ico` is really a PNG picture, which Inno Setup refuses ("Icon file is
  invalid"); `DicomDoodle.ico` is the same picture as a real Windows icon. After
  changing the picture, remake it from the repository root:
  `python -c "from PIL import Image; Image.open('file.ico').save('DicomDoodle_App/packaging/DicomDoodle.ico', sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])"`
- Test the recipe on Linux in a fresh virtual environment (the PyTorch line
  in `requirements-windows.txt`, then `pip install -r requirements-windows.txt -c constraints-windows.txt`),
  from the repository root:
  `pyinstaller --noconfirm --distpath build/dist --workpath build/work DicomDoodle_App/packaging/DicomDoodle.spec`,
  then `build/dist/DicomDoodle/DicomDoodle --self-test`. With Anaconda's Python
  the HTTPS check fails there (PyInstaller mixes Anaconda's and the system's
  OpenSSL); the Windows build is not affected.
