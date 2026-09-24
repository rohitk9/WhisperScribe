# PyInstaller spec — build with:  python packaging/build.py
# One-folder build (a single-file exe would unpack gigabytes of CUDA libraries on every launch).
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [(os.path.join(ROOT, "icon.ico"), ".")]
datas += collect_data_files("customtkinter")
datas += collect_data_files("faster_whisper")      # Silero VAD model
datas += collect_data_files("tkinterdnd2")         # tkdnd Tcl extension
datas += collect_data_files("transformers", include_py_files=False)

binaries = collect_dynamic_libs("ctranslate2")
for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cuda_nvrtc"):
    try:
        binaries += collect_dynamic_libs(pkg, destdir=pkg.replace(".", os.sep) + os.sep + "bin")
    except Exception:
        pass

hiddenimports = collect_submodules("transformers.models.qwen2") + collect_submodules("transformers.models.qwen3")
hiddenimports += collect_submodules("transformers.models.phi3") + collect_submodules("transformers.models.wavlm")
hiddenimports += ["sklearn.cluster", "sklearn.metrics", "bitsandbytes"]

a = Analysis(
    [os.path.join(ROOT, "local_transcriber_app.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "IPython", "jupyter", "pytest", "tensorflow", "whisper", "torchvision", "tensorboard"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperScribe",
    icon=os.path.join(ROOT, "icon.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="WhisperScribe", upx=False)
