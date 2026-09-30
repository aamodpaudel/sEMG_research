"""Build the Afono Nullpoint desktop app.

    python packaging/build.py

1. Creates packaging/.venv with the CPU build of PyTorch (runs on any Windows PC).
2. Copies the current sEMG base model and Whisper base.en (Whisper mode, works offline) into the bundle.
3. Runs PyInstaller -> dist/Afono Nullpoint/Afono Nullpoint.exe
4. Zips it -> dist/Afono-Nullpoint-<version>-windows.zip
Optional: compile packaging/installer.iss with Inno Setup for a setup.exe.
"""
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV = os.path.join(REPO, "packaging", ".venv")
PY = os.path.join(VENV, "Scripts", "python.exe")


def run(*cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO)


def bundle_whisper(dest, model_id="openai/whisper-base.en"):
    """Save Whisper base.en with float16 weights (about 145 MB) so Whisper mode works offline."""
    if os.path.exists(os.path.join(dest, "model.safetensors")):
        return
    code = ("import sys, torch\n"
            "from transformers import WhisperForConditionalGeneration, WhisperProcessor\n"
            f"m = WhisperForConditionalGeneration.from_pretrained({model_id!r}, dtype=torch.float16)\n"
            "m.save_pretrained(sys.argv[1], safe_serialization=True)\n"
            f"WhisperProcessor.from_pretrained({model_id!r}).save_pretrained(sys.argv[1])\n")
    run(PY, "-c", code, dest)


def main():
    if not os.path.exists(PY):
        run(sys.executable, "-m", "venv", VENV)
        run(PY, "-m", "pip", "install", "--upgrade", "pip")
    run(PY, "-m", "pip", "install", "-r", os.path.join("packaging", "requirements-desktop.txt"))
    models = os.path.join(REPO, "packaging", "models")
    os.makedirs(models, exist_ok=True)
    shutil.copyfile(os.path.join(REPO, "runs", "base", "best.pt"), os.path.join(models, "semg-base.pt"))
    bundle_whisper(os.path.join(models, "whisper-base.en"))
    run(PY, "-m", "PyInstaller", os.path.join("packaging", "nullpoint.spec"), "--noconfirm",
        "--distpath", "dist", "--workpath", "build")
    with open(os.path.join(REPO, "app", "server.py"), encoding="utf-8") as f:
        version = re.search(r'VERSION = "([^"]+)"', f.read()).group(1)
    zip_base = os.path.join(REPO, "dist", f"Afono-Nullpoint-{version}-windows")
    shutil.make_archive(zip_base, "zip", os.path.join(REPO, "dist"), "Afono Nullpoint")
    print("built", os.path.join("dist", "Afono Nullpoint", "Afono Nullpoint.exe"), "and", zip_base + ".zip")


if __name__ == "__main__":
    main()
