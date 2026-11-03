"""
Runs HD-BET on one image file and writes its brain mask — started as a
separate Python process by skull_strip.brain_mask():

    python hdbet_runner.py IMAGE MASK
    DicomDoodle.exe --hdbet-runner IMAGE MASK     (the Windows build, via main.py)

IMAGE is any file SimpleITK reads; MASK gets the brain mask (uint8, 1 =
brain) on the same grid. The network, its settings and its weights are
HD-BET's own (HD_BET.hd_bet_prediction.get_hdbet_predictor, weights
downloaded on first use as by the hd-bet command). The prediction runs in
memory (nnU-Net's predict_single_npy_array): the hd-bet command goes through
nnU-Net's multiprocessing instead, whose Unix sockets fail when the temp
folder path is long.

It runs on the device in DICOMDOODLE_TORCH_DEVICE ("cpu" or "cuda", set by
skull_strip.py from the preferences: the CPU unless the graphics card is
chosen); without it, on the GPU whenever PyTorch can use CUDA. On a GPU,
test-time augmentation is used (HD-BET's default); on the CPU it is left out,
as HD-BET recommends there.

Kept free of dicomdoodle imports, so it runs from its file path alone.
"""
import os
import sys

FLAG = "--hdbet-runner"        # main.py's option that runs this file inside DicomDoodle.exe
DEVICE_ENV = "DICOMDOODLE_TORCH_DEVICE"     # = core/ai/devices.py TORCH_DEVICE_ENV


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    image_path, mask_path = argv[1:]

    import numpy as np
    import SimpleITK as sitk
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor

    device = "cuda" if os.environ.get(DEVICE_ENV) != "cpu" and torch.cuda.is_available() else "cpu"
    print(f"HD-BET on {device}", flush=True)
    maybe_download_parameters()
    predictor = get_hdbet_predictor(use_tta=device == "cuda", device=torch.device(device))
    image = sitk.ReadImage(image_path)
    # the axis order and spacing nnU-Net's SimpleITK reader gives: (z, y, x)
    array = sitk.GetArrayFromImage(image).astype(np.float32)[None]
    spacing = list(image.GetSpacing())[::-1]
    segmentation = predictor.predict_single_npy_array(array, {"spacing": spacing}, None, None, False)
    mask = sitk.GetImageFromArray((np.asarray(segmentation) > 0).astype(np.uint8))
    mask.CopyInformation(image)
    sitk.WriteImage(mask, mask_path)
    print(f"brain mask: {int(np.count_nonzero(segmentation))} voxels", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
