"""Where Afono Nullpoint finds its bundled files and keeps the user's data.

Development: everything lives in the repository.
Packaged app: bundled files are read from the install folder; recordings, profiles,
calibrated models and the run log go to %LOCALAPPDATA%\\Afono Nullpoint.
"""
import os
import sys

FROZEN = bool(getattr(sys, "frozen", False))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = getattr(sys, "_MEIPASS", REPO) if FROZEN else REPO

if FROZEN:
    DATA = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Afono Nullpoint")
else:
    DATA = REPO


def res(*p):
    return os.path.join(RES, *p)


def setup():
    """Create the data folders and point the semg package at them (call before importing semg)."""
    os.makedirs(os.path.join(DATA, "runs"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "data", "user"), exist_ok=True)
    os.environ.setdefault("AFONO_REGISTRY", os.path.join(DATA, "runs", "registry.jsonl"))
    os.environ.setdefault("AFONO_USER_ROOT", os.path.join(DATA, "data", "user"))
    # Whisper mode: use the speech model bundled with the app, and never reach for the network.
    if os.path.isdir(WHISPER_MODEL):
        os.environ.setdefault("AFONO_WHISPER_MODEL", WHISPER_MODEL)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.chdir(DATA)


BASE_MODEL = res("models", "semg-base.pt") if FROZEN else os.path.join(REPO, "runs", "base", "best.pt")
WHISPER_MODEL = res("models", "whisper-base.en") if FROZEN else os.path.join(REPO, "packaging", "models", "whisper-base.en")
