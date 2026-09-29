"""Build the Afono Nullpoint desktop app.

    python packaging/build.py

1. Creates packaging/.venv with the CPU build of PyTorch (runs on any Windows PC).
2. Copies the current sEMG base model into the bundle.
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


def main():
    if not os.path.exists(PY):
        run(sys.executable, "-m", "venv", VENV)
        run(PY, "-m", "pip", "install", "--upgrade", "pip")
    run(PY, "-m", "pip", "install", "-r", os.path.join("packaging", "requirements-desktop.txt"))
    models = os.path.join(REPO, "packaging", "models")
    os.makedirs(models, exist_ok=True)
    shutil.copyfile(os.path.join(REPO, "runs", "base", "best.pt"), os.path.join(models, "semg-base.pt"))
    run(PY, "-m", "PyInstaller", os.path.join("packaging", "nullpoint.spec"), "--noconfirm",
        "--distpath", "dist", "--workpath", "build")
    with open(os.path.join(REPO, "app", "server.py"), encoding="utf-8") as f:
        version = re.search(r'VERSION = "([^"]+)"', f.read()).group(1)
    zip_base = os.path.join(REPO, "dist", f"Afono-Nullpoint-{version}-windows")
    shutil.make_archive(zip_base, "zip", os.path.join(REPO, "dist"), "Afono Nullpoint")
    print("built", os.path.join("dist", "Afono Nullpoint", "Afono Nullpoint.exe"), "and", zip_base + ".zip")


if __name__ == "__main__":
    main()
