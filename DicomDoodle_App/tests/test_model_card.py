"""
Tests for the model card (core/ai/): what a model file is from its content,
finding / installing / replacing / removing cards, which model and format is
chosen, and the SAM prompts — with tiny synthetic models, so no download is
needed. (The real card is checked by scripts/make_model_card.py when it is made.)

    python -m pytest tests          (or simply: python tests/test_model_card.py)
"""
import collections
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dicomdoodle.core.ai import devices, model_card, model_files  # noqa: E402
from dicomdoodle.core.ai.model_files import SAM, SAM2_DECODER, SAM2_ENCODER, YOLO, YOLOE  # noqa: E402


def _onnx(path, inputs, outputs, metadata=None):
    """A tiny valid ONNX file (Identity nodes) with these input / output names and metadata."""
    from onnx import TensorProto, helper, save
    values = lambda names: [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 4]) for n in names]  # noqa: E731
    nodes = [helper.make_node("Identity", [inputs[0]], [o]) for o in outputs]
    model = helper.make_model(helper.make_graph(nodes, "tiny", values(inputs), values(outputs)),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    for key, value in (metadata or {}).items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    save(model, path)
    return path


def _yolo_onnx(path, names=None, date="2026-11-02T20:12:19", description="Ultralytics YOLO26x-seg model"):
    return _onnx(path, ["images"], ["output0", "output1"],
                 {"description": description, "author": "Ultralytics", "task": "segment", "stride": 32,
                  "names": repr(names or {0: "tumor"}), "imgsz": "[480, 480]", "dicomdoodle_source_date": date})


def _sam_onnx(folder, family="sam2.1_l:abc"):
    meta = {"dicomdoodle_family": family, "dicomdoodle_title": "SAM 2.1 large"}
    encoder = _onnx(os.path.join(folder, "a.onnx"), ["image"], sorted(model_files.SAM2_ENCODER_OUTPUTS),
                    {**meta, "dicomdoodle_model": SAM2_ENCODER})
    decoder = _onnx(os.path.join(folder, "b.onnx"), sorted(model_files.SAM2_DECODER_INPUTS),
                    ["masks", "iou_predictions"], {**meta, "dicomdoodle_model": SAM2_DECODER})
    return encoder, decoder


def _sam_weights(path, blocks=48, sam21=True):
    import torch
    weights = collections.OrderedDict()
    for i in range(blocks):
        weights[f"image_encoder.trunk.blocks.{i}.attn.qkv.weight"] = torch.zeros(2)
    if sam21:
        weights["no_obj_embed_spatial"] = torch.zeros(1)
    torch.save({"model": weights}, path)
    return path


def _card(path, files, version="1.0", name="DicomDoodle Model Card"):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(model_card.MANIFEST, json.dumps({"format": model_card.CARD_FORMAT, "name": name,
                                                          "version": version}))
        for name_in_card, source in files.items():
            archive.write(source, name_in_card)
    return path


class _Folders:
    """A temporary data folder and download folder; the card search looks only there."""

    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="dicomdoodle_cardtest_")
        self.data, self.downloads, self.models = (os.path.join(self.root, n) for n in ("data", "dl", "models"))
        for folder in (self.downloads, self.models):
            os.makedirs(folder)
        self._env = os.environ.get(model_card.DATA_DIR_ENV)
        os.environ[model_card.DATA_DIR_ENV] = self.data
        self._search = model_card.search_folders
        model_card.search_folders = lambda extra=(): [self.downloads]
        return self

    def __exit__(self, *exc):
        model_card.search_folders = self._search
        if self._env is None:
            os.environ.pop(model_card.DATA_DIR_ENV, None)
        else:
            os.environ[model_card.DATA_DIR_ENV] = self._env
        shutil.rmtree(self.root, ignore_errors=True)

    def models_card(self, filename="DicomDoodle-ModelCard-1.0.ddcard", version="1.0"):
        yolo = _yolo_onnx(os.path.join(self.models, "detector.onnx"))
        encoder, decoder = _sam_onnx(self.models)
        return _card(os.path.join(self.downloads, filename),
                     {"models/x1.onnx": yolo, "models/x2.onnx": encoder, "models/x3.onnx": decoder}, version)


def test_onnx_files_are_recognized_by_content():
    with _Folders() as f:
        yolo = model_files.inspect(_yolo_onnx(os.path.join(f.models, "whatever-name.onnx")))
        assert (yolo.kind, yolo.engine) == (YOLO, "onnx") and yolo.details["imgsz"] == [480, 480]
        assert yolo.title.startswith("YOLO26x-seg") and "tumor" in yolo.title and yolo.date.startswith("2026-11-02")
        vocabulary = {i: f"thing {i}" for i in range(2000)}
        fallback = model_files.inspect(_yolo_onnx(os.path.join(f.models, "y.onnx"), vocabulary,
                                                  description="Ultralytics YOLOe-11l-seg model"))
        assert fallback.kind == YOLOE and "2,000 classes" in fallback.title
        encoder, decoder = (model_files.inspect(p) for p in _sam_onnx(f.models))
        assert (encoder.kind, decoder.kind) == (SAM2_ENCODER, SAM2_DECODER) and encoder.family == decoder.family
        # without DicomDoodle's metadata, the input and output names tell
        bare = model_files.inspect(_onnx(os.path.join(f.models, "e.onnx"), ["image"],
                                         sorted(model_files.SAM2_ENCODER_OUTPUTS)))
        assert bare.kind == SAM2_ENCODER
        other = _onnx(os.path.join(f.models, "other.onnx"), ["x"], ["y"])
        for path in (other, __file__):
            try:
                model_files.inspect(path)
                raise AssertionError(f"{path} was taken for a model")
            except model_files.NotAModel:
                pass


def test_pytorch_files_are_recognized_without_running_them():
    with _Folders() as f:
        large = model_files.inspect(_sam_weights(os.path.join(f.models, "weights.pt")))
        assert (large.kind, large.engine, large.title) == (SAM, "torch", "SAM 2.1 large")
        assert large.details["builder"] == "build_sam2_l"
        tiny = model_files.inspect(_sam_weights(os.path.join(f.models, "t.pt"), blocks=12, sam21=False))
        assert tiny.title == "SAM 2 tiny" and tiny.details["builder"] == "build_sam2_t"
        # a pickle that would run code when loaded normally: inspected, never executed
        evil = os.path.join(f.models, "evil.pt")
        marker = os.path.join(f.models, "ran")
        import torch

        class Payload:
            def __reduce__(self):
                return (open, (marker, "w"))
        torch.save({"model": Payload()}, evil)
        try:
            model_files.inspect(evil)
        except model_files.NotAModel:
            pass
        assert not os.path.exists(marker), "inspect() executed code from the file"


def test_ultralytics_checkpoint_needs_its_classes():
    try:
        from ultralytics.nn.tasks import DetectionModel
    except Exception:
        print("skipped: Ultralytics is not installed")
        return
    import torch
    with _Folders() as f:
        model = DetectionModel("yolo11n.yaml", nc=1, verbose=False)
        model.names = {0: "tumor"}
        path = os.path.join(f.models, "my-detector.pt")
        torch.save({"model": model, "date": "2026-01-02T03:04:05", "train_args": {"imgsz": 256}}, path)
        found = model_files.inspect(path)
        assert (found.kind, found.engine) == (YOLO, "torch") and found.details["imgsz"] == [256, 256]
        assert found.title.startswith("YOLO11n") and model_files.torch_loadable(found) == (True, "")
        found.needs = found.needs + (("ultralytics.nn.modules.head", "HeadOfTheFuture"),)
        loadable, why = model_files.torch_loadable(found)
        assert not loadable and "HeadOfTheFuture" in why


def test_card_is_found_installed_and_kept():
    with _Folders() as f:
        card_path = f.models_card()
        assert model_card.is_card_name(card_path) and not model_card.is_card_name("DicomDoodle-main.tar")
        card, event = model_card.prepare()
        assert event == "installed" and card.label == "DicomDoodle Model Card 1.0"
        assert {m.kind for m in card.models} == {YOLO, SAM2_ENCODER, SAM2_DECODER}
        assert [c.engine for c in card.candidates(SAM)] == ["onnx"]       # the encoder found its decoder
        assert model_card.prepare()[1] == "ready"                          # the next start: nothing unpacked
        os.remove(card_path)
        assert model_card.prepare()[1] == "ready"                          # the download may be deleted
        newer = f.models_card("DicomDoodle-ModelCard-1.1.ddcard", version="1.1")
        card, event = model_card.prepare()
        assert event == "updated" and card.version == "1.1"
        os.remove(newer)
        f.models_card("dicomdoodle-modelcard-0.9.zip", version="0.9")    # an older card does not replace it
        assert model_card.prepare()[0].version == "1.1"
        model_card.remove_installed()
        assert model_card.installed() is None
        f.models_card("DicomDoodle-ModelCard-1.1.ddcard", version="1.1")
        assert model_card.prepare(search=False)[1] == "none"               # after Remove: no search
        assert model_card.prepare()[0].version == "1.1"                     # searching: the newest card found


def test_damaged_or_wrong_files_are_refused():
    with _Folders() as f:
        not_a_card = os.path.join(f.downloads, "DicomDoodle-ModelCard-x.ddcard")
        with open(not_a_card, "wb") as fh:
            fh.write(b"not a zip")
        empty = _card(os.path.join(f.downloads, "DicomDoodle-empty.ddcard"), {})
        for path in (not_a_card, empty):
            try:
                model_card.read_card(path)
                raise AssertionError(f"{path} was taken for a card")
            except model_card.CardError:
                pass
        assert model_card.prepare()[1] == "none"
        good = f.models_card()
        card = model_card.read_card(good)
        card.manifest["files"] = [{"file": "models/x1.onnx", "sha256": "0" * 64}]
        try:
            model_card.install(card)
            raise AssertionError("a card whose file differs from its list was installed")
        except model_card.CardError as exc:
            assert "damaged" in str(exc)
        assert model_card.installed() is None and not os.path.exists(model_card.install_dir() + ".partial")


def test_newest_model_then_fastest_format():
    with _Folders() as f:
        old = _yolo_onnx(os.path.join(f.models, "old.onnx"), date="2025-02-07T06:54:56",
                         description="Ultralytics YOLO11x-seg model")
        new = _yolo_onnx(os.path.join(f.models, "new.onnx"), date="2026-11-02T20:12:19")
        weights = _sam_weights(os.path.join(f.models, "sam.pt"))
        encoder, decoder = _sam_onnx(f.models)
        card = model_card.InstalledCard(f.models, "test", "1", "id", "", "", {},
                                        [model_files.inspect(p) for p in (old, new, weights, encoder, decoder)])
        detectors = model_card.ranked(card, YOLO, devices.CPU)
        assert [c.files[0].path for c, _problem in detectors] == [new, old]          # newest first
        segmenters = model_card.ranked(card, SAM, devices.CPU)
        assert [c.engine for c, _problem in segmenters] == ["onnx", "torch"]         # ONNX before PyTorch
        assert model_card.ranked(card, YOLOE, devices.CPU) == []


def test_sam_prompts_as_ultralytics_makes_them():
    from dicomdoodle.core.ai.onnx_models import OnnxSam2
    sam = OnnxSam2.__new__(OnnxSam2)
    sam.size = 1024
    coords, labels = sam._prompts((512, 256), [10, 20, 30, 40], None, None)       # one box, image 512 x 256
    assert coords.shape == (1, 2, 2) and np.allclose(coords[0], [[20, 40], [60, 80]]) and labels.tolist() == [[2, 3]]
    coords, labels = sam._prompts((1024, 1024), None, [[5, 6]], [1])               # one click
    assert coords.shape == (1, 1, 2) and labels.tolist() == [[1]]
    coords, labels = sam._prompts((1024, 1024), None, [[5, 6], [7, 8]], [1, 0])    # two objects, a point each
    assert coords.shape == (2, 1, 2) and labels.tolist() == [[1], [0]]
    coords, labels = sam._prompts((1024, 1024), [0, 0, 9, 9], [[[5, 6], [7, 8]]], [[1, 0]])   # box + points
    assert coords.shape == (1, 4, 2) and labels.tolist() == [[2, 3, 1, 0]]


def test_a_failing_graphics_card_hands_over_to_the_processor():
    from dicomdoodle.core.ai.onnx_models import OnnxModel
    with _Folders() as f:
        model = OnnxModel(_onnx(os.path.join(f.models, "m.onnx"), ["x"], ["y"]), devices.CPU)

        class LostDevice:
            def run(self, *args):
                raise RuntimeError("the graphics card was removed")
        model.session, model.provider = LostDevice(), devices.DML      # as if DirectML had been running it
        out = model.run({"x": np.ones((1, 4), np.float32)})
        assert np.allclose(out[0], 1.0) and not model.on_gpu and model.gpu_failed


def test_device_preference():
    assert devices.normalize("auto") == devices.normalize("cpu") == devices.CPU      # 2.0's default: the CPU
    assert devices.normalize("cuda") == devices.normalize("gpu") == devices.GPU
    previous = devices.preferred()
    try:
        devices.set_preferred("cpu")
        assert devices.torch_device() == "cpu" and devices.onnx_provider("cpu") == ""
    finally:
        devices.set_preferred(previous)
    assert devices._ranked([("Microsoft Basic Render Driver", "PCI\\VEN_1414&DEV_008C"),
                            ("Intel(R) UHD Graphics", "PCI\\VEN_8086&DEV_9A49"),
                            ("NVIDIA GeForce RTX 3060 Laptop GPU", "PCI\\VEN_10DE&DEV_2520"),
                            ("NVIDIA GeForce RTX 3060 Laptop GPU", "pci\\ven_10de&dev_2520")]) == \
        ["NVIDIA GeForce RTX 3060 Laptop GPU", "Intel(R) UHD Graphics"]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
