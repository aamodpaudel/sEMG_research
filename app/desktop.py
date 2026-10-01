"""Afono Nullpoint desktop launcher (the entry point of Afono Nullpoint.exe).

Starts the local server, opens it in its own app window (Microsoft Edge or Chrome in
app mode, with a separate profile so camera and sensor permissions stay with Nullpoint),
and exits when that window has been closed.
"""
import multiprocessing
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

PORT = int(os.environ.get("AFONO_PORT", "8765"))
HEARTBEAT_TIMEOUT_S = 180
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]


def _quiet_streams(data_dir):
    """A windowed exe has no console; send output to a log file instead."""
    if sys.stdout is None or sys.stderr is None or getattr(sys, "frozen", False):
        log = open(os.path.join(data_dir, "nullpoint.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log


def _running(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1) as r:
            return b"Nullpoint" in r.read()
    except Exception:
        return False


def _port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def open_window(url, profile_dir):
    """True if Nullpoint got its own app window (Edge or Chrome with a separate profile)."""
    for b in BROWSERS:
        if os.path.exists(b):
            subprocess.Popen([b, f"--app={url}", f"--user-data-dir={profile_dir}", "--window-size=1440,920",
                              "--no-first-run", "--no-default-browser-check"])
            return True
    webbrowser.open(url)
    return False


def window_open(profile_dir):
    """Whether a browser is using Nullpoint's window profile. Chromium holds <profile>/lockfile open exclusively
    while it runs. (The launched process itself is no guide: Edge hands the window to another process and exits.)"""
    try:
        fd = os.open(os.path.join(profile_dir, "lockfile"), os.O_RDWR)
    except FileNotFoundError:
        return False
    except PermissionError:
        return True
    os.close(fd)
    return False


def main():
    multiprocessing.freeze_support()
    from app import paths
    paths.setup()
    _quiet_streams(paths.DATA)
    if "--serve" in sys.argv:                 # no window: for testing and headless use
        import uvicorn
        from app import server
        uvicorn.run(server.app, host="127.0.0.1", port=PORT, log_level="warning", log_config=None)
        return
    url = f"http://127.0.0.1:{PORT}/"
    profile_dir = os.path.join(paths.DATA, "window")
    if _running(PORT):                       # already open: just show another window
        open_window(url, profile_dir)        # the running instance keeps serving it
        return
    if not _port_free(PORT):
        print(f"Port {PORT} is used by another program; Nullpoint needs it for the browser extension.")
        webbrowser.open("https://github.com/aamodpaudel/sEMG_research#afono-nullpoint")
        return
    import uvicorn
    from app import server
    cfg = uvicorn.Config(server.app, host="127.0.0.1", port=PORT, log_level="warning", log_config=None)
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(200):
        if _running(PORT):
            break
        time.sleep(0.1)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "Afono Nullpoint started", url, "data:", paths.DATA)
    own_window = open_window(url, profile_dir)
    # Stay alive while Nullpoint's window profile is in use. Heartbeats are only the fallback (e.g. when the
    # default browser had to be used): a background or minimised window has its timers throttled to about one a
    # minute, which is what used to shut the server down behind an open window, so the timeout must be longer.
    start, seen = time.time(), False
    while True:
        time.sleep(2)
        if own_window and window_open(profile_dir):
            seen = True
            continue
        beat = server.LAST_BEAT[0]
        if seen and (beat == 0 or time.time() - beat > 10):      # the window was there and has been closed
            break
        if beat == 0 and time.time() - start > 120:
            break
        if beat and time.time() - beat > HEARTBEAT_TIMEOUT_S:
            break
    srv.should_exit = True
    t.join(timeout=5)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "Afono Nullpoint closed")


if __name__ == "__main__":
    main()
