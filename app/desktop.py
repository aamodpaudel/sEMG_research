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
    for b in BROWSERS:
        if os.path.exists(b):
            subprocess.Popen([b, f"--app={url}", f"--user-data-dir={profile_dir}", "--window-size=1440,920",
                              "--no-first-run", "--no-default-browser-check"])
            return True
    webbrowser.open(url)
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
        open_window(url, profile_dir)
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
    open_window(url, profile_dir)
    # stay alive while the window sends heartbeats; give it a minute to open the first time
    start = time.time()
    while True:
        time.sleep(2)
        beat = server.LAST_BEAT[0]
        if beat == 0 and time.time() - start > 90:
            break
        if beat and time.time() - beat > 25:
            break
    srv.should_exit = True
    t.join(timeout=5)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "Afono Nullpoint closed")


if __name__ == "__main__":
    main()
