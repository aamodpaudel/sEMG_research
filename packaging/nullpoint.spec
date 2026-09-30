# PyInstaller spec for Afono Nullpoint (Windows, one-folder build).
#   packaging\.venv\Scripts\python -m PyInstaller packaging\nullpoint.spec --noconfirm --distpath dist --workpath build
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

REPO = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [
    (os.path.join(REPO, "app", "static"), os.path.join("app", "static")),
    (os.path.join(REPO, "app", "demo_pack"), os.path.join("app", "demo_pack")),
    (os.path.join(REPO, "site", "assets"), os.path.join("site", "assets")),
    (os.path.join(REPO, "vocab"), "vocab"),
    (os.path.join(REPO, "extension"), "extension"),
    (os.path.join(REPO, "packaging", "models"), "models"),
    (os.path.join(REPO, "site", "demo", "DATA_LICENSE.txt"), "."),
]
hidden = [
    "app.server", "app.paths",
    "semg.calibrate", "semg.compare", "semg.train", "semg.data", "semg.model", "semg.decode", "semg.signal",
    "semg.text", "semg.user", "semg.registry", "semg.metrics", "semg.voice",
    "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto", "uvicorn.protocols", "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto", "uvicorn.lifespan", "uvicorn.lifespan.on", "multipart", "python_multipart",
]
# Whisper mode: transformers imports model code lazily and checks package versions at import time.
hidden += collect_submodules("transformers.models.whisper") + ["tokenizers", "safetensors"]
datas += collect_data_files("transformers", include_py_files=False)
for pkg in ("transformers", "tokenizers", "safetensors", "huggingface_hub", "tqdm", "regex", "requests", "packaging",
            "filelock", "numpy", "pyyaml", "torch"):
    try:
        datas += copy_metadata(pkg)
    except Exception:  # noqa: BLE001 - metadata of a package that is not installed
        pass

a = Analysis(
    [os.path.join(REPO, "app", "desktop.py")],
    pathex=[REPO],
    datas=datas,
    hiddenimports=hidden,
    excludes=["matplotlib", "tkinter", "onnx", "onnxruntime", "pytest", "IPython", "notebook", "pandas", "PIL"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Afono Nullpoint",
    icon=os.path.join(REPO, "app", "assets", "afono.ico"),
    console=False,
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Afono Nullpoint")
