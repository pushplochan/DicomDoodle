"""
The AI models' plumbing behind core/ai_models.py (ModelManager):

    devices.py       processor (CPU, the default) or graphics card (GPU): which cards and
                     runtimes this computer has, and the device the user chose
    model_files.py   what a model file is, read from its content (never from its name)
    model_card.py    the model card: one file with the AI models, found at start,
                     installed once, and the best model and format chosen for each purpose
    onnx_models.py   the models in ONNX format, run with ONNX Runtime (faster on a CPU,
                     and the way to a graphics card on Windows); results identical in
                     form to Ultralytics' own
    torch_models.py  the models as PyTorch checkpoints (.pt), run by Ultralytics
"""
