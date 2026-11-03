# Windows installer

GitHub builds `DicomDoodle-<version>-Setup.exe` on one of its Windows machines
whenever a version tag (`v…`) is pushed, and attaches it to that version's
release. The recipe is [`.github/workflows/windows-installer.yml`](../../.github/workflows/windows-installer.yml);
it installs the packages, builds `DicomDoodle.exe`, runs its self-test, makes
the installer and publishes it (about an hour).

The installer has **no AI models**. They are a separate download, the **model
card** (`DicomDoodle-ModelCard-<version>.ddcard`, about 860 MB), attached to
its own release `model-card-<version>`. DicomDoodle finds the card at start
(in Downloads, on the Desktop, in Documents or next to the program), installs
it once and loads the models; without it, it is for manual annotation only.
The AI runs on the processor, or — when the user switches — on a graphics card
through DirectML (`onnxruntime-directml`).

| File | What it is |
|---|---|
| `requirements-windows.txt` | the packages of the build, exact versions (PyTorch CPU-only, installed first) |
| `constraints-windows.txt` | the version of every package they pull in, as tested (`pip freeze`) |
| `DicomDoodle.spec` | PyInstaller recipe: `DicomDoodle.exe` + its `_internal/` folder, the AI models included |
| `DicomDoodle.iss` | Inno Setup script: the installer (Start menu, optional desktop icon, uninstaller, no administrator rights needed) |
| `DicomDoodle.ico` | the icon of `DicomDoodle.exe` and of the installer, in Windows' icon format (16–256 px) |
| `upload_model_card.py` | uploads a model card to the release `model-card-<version>`, where users and the build's self-test get it |

## The model card

Make it with [`scripts/make_model_card.py`](../scripts/make_model_card.py): it
converts each model to ONNX, checks every conversion against the original, and
packs them with `modelcard.json` (files, purposes, SHA-256) and a readable
`MODEL_CARD.md`. The app reads what each file is from its content, so the
names inside do not matter. For the card 1.0 (best.pt needs Ultralytics 8.4
for its YOLO26 head; the other two were converted with the app's Ultralytics 8.3.99):

    python DicomDoodle_App/scripts/make_model_card.py convert best.pt --out work --check-dicom "<a DICOM series>"
    python DicomDoodle_App/scripts/make_model_card.py convert yoloe-11l-seg-pf.pt sam2.1_l.pt --out work --check-dicom "<a DICOM series>"
    python DicomDoodle_App/scripts/make_model_card.py pack "work/*.onnx" --version 1.0 --out DicomDoodle-ModelCard-1.0.ddcard

Then upload it once (`.gitignore` keeps cards and model files out of git):

    python DicomDoodle_App/packaging/upload_model_card.py DicomDoodle-ModelCard-1.0.ddcard

It asks for a GitHub token with *Contents: Read and write* access to the
repository and prints the card's download link. A new card version gets its
own release (`model-card-1.1` …); change `MODEL_CARD_RELEASE` in the workflow
to test the build with it. Installed apps replace an older card with a newer one
they find.

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
  SEG writing, DICOM codes, the model card's AI models and HD-BET (exit code 0 = OK).
  The models come from the card named in the environment variable
  `DICOMDOODLE_MODEL_CARD` (installed into a temporary folder), else from the
  installed card; with neither, that check says SKIP. An `INFO  AI runtime` line
  shows ONNX Runtime's providers and the graphics card the AI could use.
- The installed app keeps the model card in `%LOCALAPPDATA%\DicomDoodle\ModelCard`
  and its settings in `%APPDATA%\DicomDoodle\settings.json`.
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
