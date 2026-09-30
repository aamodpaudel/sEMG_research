"""EMG preprocessing and augmentation.

Filters are *causal* (sosfilt, not filtfilt) so the exact same processing can run
on a live stream from cheap hardware in the browser later.
"""
import numpy as np
from scipy import signal as sps


def design_filters(fs: float = 1000.0, mains_hz: float = 60.0, highpass_hz: float = 2.0,
                   lowpass_hz: float | None = 450.0):
    sections = [sps.butter(4, highpass_hz, "highpass", fs=fs, output="sos")]
    h = mains_hz
    while h < fs / 2 - 1:
        b, a = sps.iirnotch(h, 30.0, fs)
        sections.append(sps.tf2sos(b, a))
        h += mains_hz
    if lowpass_hz and lowpass_hz < fs / 2:
        sections.append(sps.butter(4, lowpass_hz, "lowpass", fs=fs, output="sos"))
    return np.concatenate(sections, axis=0)


_FILTER_CACHE: dict = {}


def preprocess(emg: np.ndarray, fs: float = 1000.0, mains_hz: float = 60.0,
               highpass_hz: float = 2.0) -> np.ndarray:
    """(T, C) raw EMG -> (T, C) filtered, per-channel standardised float32."""
    key = (fs, mains_hz, highpass_hz)
    if key not in _FILTER_CACHE:
        _FILTER_CACHE[key] = design_filters(fs, mains_hz, highpass_hz)
    sos = _FILTER_CACHE[key]
    x = emg.astype(np.float64)
    x = x - x[:50].mean(axis=0, keepdims=True)  # causal-friendly offset removal
    x = sps.sosfilt(sos, x, axis=0)
    x = x / (x.std(axis=0, keepdims=True) + 1e-6)
    return np.clip(x, -20, 20).astype(np.float32)


def augment(x: np.ndarray, rng: np.random.Generator, return_shift: bool = False):
    """Augmentations that mimic electrode/session variability of cheap hardware."""
    x = x.copy()
    T, C = x.shape
    x *= np.exp(rng.normal(0, 0.25, size=(1, C))).astype(np.float32)  # per-channel gain
    x += rng.normal(0, 0.05, size=x.shape).astype(np.float32)  # sensor noise
    if rng.random() < 0.5:  # drop one electrode
        x[:, rng.integers(C)] = 0
    for _ in range(rng.integers(0, 4)):  # short time masks
        w = int(rng.integers(20, 100))
        s = int(rng.integers(0, max(1, T - w)))
        x[s:s + w] = 0
    shift = int(rng.integers(0, 40))  # small start jitter
    return (x[shift:], shift) if return_shift else x[shift:]


def augment_v2(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stronger augmentation (run r4): v1 plus random time stretch (0.9-1.1), up to two dropped
    channels, more and longer time masks, and a random per-channel DC-free baseline wander."""
    T, C = x.shape
    if rng.random() < 0.8:  # time stretch by linear interpolation (speaking-rate variation)
        r = float(rng.uniform(0.9, 1.1))
        n = max(200, int(T * r))
        src = np.linspace(0, T - 1, n)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, T - 1)
        f = (src - i0)[:, None].astype(np.float32)
        x = x[i0] * (1 - f) + x[i1] * f
        T = n
    x = x.astype(np.float32, copy=True)
    x *= np.exp(rng.normal(0, 0.25, size=(1, C))).astype(np.float32)
    x += rng.normal(0, 0.05, size=x.shape).astype(np.float32)
    for _ in range(rng.choice([0, 1, 1, 2])):
        x[:, rng.integers(C)] = 0
    for _ in range(rng.integers(0, 7)):
        w = int(rng.integers(20, 150))
        s = int(rng.integers(0, max(1, T - w)))
        x[s:s + w] = 0
    shift = int(rng.integers(0, 40))
    return x[shift:]
