"""Isolate the guitar from a full mix before pitch tracking.

Uses Demucs (htdemucs_6s), which splits music into vocals, drums, bass, piano, guitar and
"other"; the analyzer then tracks the guitar stem only. Note that the guitar stem holds every
guitar in the mix: it removes the band around a solo, not a second guitar.

The model needs PyTorch, far too big to put in the app itself. So it runs either:
  - in-process, when torch and demucs are importable (running from source), or
  - as the separate "guitar isolation pack": a small program (built by CI with the model
    weights included) that the app downloads once and runs as a subprocess.

The pack's program is this file:  separation.py INPUT.wav OUTPUT.wav [--models DIR]
It prints "PROGRESS <0-1>" lines while it works.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import soundfile as sf

MODEL = "htdemucs_6s"
STEM = "guitar"
MODEL_RATE = 44100
OUTPUT_RATE = 22050          # what the analyzer works at
CHUNK_SECONDS = 20.0         # separated in chunks so progress shows and cancel works
OVERLAP_SECONDS = 1.0        # chunks overlap and cross-fade

REPO = "mscottwhit12-hub/Solo_Analyzer"
PACK_NAME = {"win32": "GuitarSeparator-windows.zip", "darwin": "GuitarSeparator-macos.zip"}.get(
    sys.platform, "GuitarSeparator-linux.zip")
PACK_URL = os.environ.get("SOLO_ANALYZER_PACK_URL",
                          f"https://github.com/{REPO}/releases/latest/download/{PACK_NAME}")
PACK_APPROX_MB = 300
PACK_EXE = "GuitarSeparator.exe" if sys.platform == "win32" else "GuitarSeparator"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class SeparationCancelled(Exception):
    pass


class SeparationUnavailable(Exception):
    pass


# --- Where things live -------------------------------------------------------

def data_dir():
    """Per-user folder for the downloaded pack (no Qt needed, so the pack can use it too)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(os.environ.get("SOLO_ANALYZER_DATA", base), "Solo Analyzer")


def pack_dir():
    return os.path.join(data_dir(), "GuitarSeparator")


def pack_executable():
    """Path of the installed pack's program, or None."""
    for root, _dirs, files in os.walk(pack_dir()):
        if PACK_EXE in files:
            return os.path.join(root, PACK_EXE)
    return None


def in_process_available():
    """True when torch and demucs are installed in this Python (running from source)."""
    import importlib.util
    try:
        return all(importlib.util.find_spec(name) is not None for name in ("torch", "demucs"))
    except (ImportError, ValueError):
        return False


def available():
    """'in-process', 'pack', or None (needs the pack downloaded)."""
    if in_process_available():
        return "in-process"
    if pack_executable():
        return "pack"
    return None


# --- Downloading the pack ------------------------------------------------------

def download_pack(progress=None, cancelled=None, url=None):
    """Download and unpack the guitar isolation pack. progress(done_mb, total_mb)."""
    url = url or PACK_URL
    os.makedirs(data_dir(), exist_ok=True)
    fd, zip_path = tempfile.mkstemp(suffix=".zip", dir=data_dir())
    os.close(fd)
    try:
        try:
            response = urllib.request.urlopen(url, timeout=60)
        except Exception as e:
            raise SeparationUnavailable(
                f"Couldn't download the guitar isolation pack ({e}).\n"
                f"It's published with each release at github.com/{REPO}/releases.") from e
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with response, open(zip_path, "wb") as out:
            while True:
                if cancelled and cancelled():
                    raise SeparationCancelled
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if progress:
                    progress(done / 1e6, total / 1e6)
        staging = pack_dir() + ".partial"
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():  # refuse paths that escape the folder
                target = os.path.realpath(os.path.join(staging, member))
                if not target.startswith(os.path.realpath(staging) + os.sep):
                    raise SeparationUnavailable("The downloaded pack is malformed.")
            archive.extractall(staging)
        shutil.rmtree(pack_dir(), ignore_errors=True)
        os.replace(staging, pack_dir())
        exe = pack_executable()
        if not exe:
            raise SeparationUnavailable("The downloaded pack doesn't contain the separator program.")
        if sys.platform != "win32":
            for root, _dirs, files in os.walk(pack_dir()):  # zip doesn't keep exec bits
                for name in files:
                    path = os.path.join(root, name)
                    if name == PACK_EXE or name.endswith((".so", ".dylib")):
                        os.chmod(path, 0o755)
        return exe
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


# --- Separation ------------------------------------------------------------------

def load_model(models_dir=None):
    from demucs.pretrained import get_model
    from pathlib import Path
    if not models_dir:
        return get_model(MODEL)  # downloads (or reuses) the official weights
    # The bundled checkpoint stores model classes, which PyTorch 2.6+ refuses to unpickle by
    # default. It's Meta's official file (its checksum is in the name and demucs verifies it),
    # so allow it for this one load only.
    import torch
    original = torch.load
    torch.load = lambda *args, **kwargs: original(*args, **{**kwargs, "weights_only": False})
    try:
        return get_model(MODEL, repo=Path(models_dir))
    finally:
        torch.load = original


def separate(stereo, model, progress=None, cancelled=None):
    """Guitar stem (mono, OUTPUT_RATE) from stereo float32 audio (2, n) at MODEL_RATE."""
    import torch
    from demucs.apply import apply_model
    from scipy.signal import resample_poly

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    wav = torch.from_numpy(np.ascontiguousarray(stereo, dtype=np.float32))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    stem_index = model.sources.index(STEM)

    n = wav.shape[1]
    chunk, overlap = int(CHUNK_SECONDS * MODEL_RATE), int(OVERLAP_SECONDS * MODEL_RATE)
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    start = 0
    while start < n:
        if cancelled and cancelled():
            raise SeparationCancelled
        end = min(n, start + chunk)
        with torch.no_grad():
            sources = apply_model(model, wav[None, :, start:end], shifts=0, split=True,
                                  overlap=0.25, progress=False, device="cpu")
        guitar = (sources[0, stem_index] * std + mean).mean(0).numpy()
        fade = np.ones(end - start, dtype=np.float32)
        if start > 0:
            ramp = min(overlap, end - start)
            fade[:ramp] = np.linspace(0, 1, ramp)
        if end < n:
            ramp = min(overlap, end - start)
            fade[-ramp:] = np.minimum(fade[-ramp:], np.linspace(1, 0, ramp))
        out[start:end] += guitar * fade
        weight[start:end] += fade
        if progress:
            progress(end / n)
        if end == n:
            break
        start = end - overlap
    out /= np.maximum(weight, 1e-6)
    return resample_poly(out, OUTPUT_RATE, MODEL_RATE).astype(np.float32)


def stereo_from_file(path):
    """Decode any audio/video file to stereo float32 (2, n) at MODEL_RATE with bundled ffmpeg."""
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-f", "f32le",
           "-ac", "2", "-ar", str(MODEL_RATE), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    if raw.returncode != 0:
        raise RuntimeError("Failed to decode audio for guitar isolation.")
    return np.frombuffer(raw.stdout, dtype=np.float32).reshape(-1, 2).T.copy()


def isolate_file(media_path, progress=None, cancelled=None):
    """Guitar stem of a media file: mono float32 at OUTPUT_RATE. Uses the cache if possible."""
    cached = load_cached_isolation(media_path)
    if cached is not None:
        return cached
    stereo = stereo_from_file(media_path)
    mode = available()
    if mode == "in-process":
        guitar = separate(stereo, load_model(), progress, cancelled)
    elif mode == "pack":
        guitar = run_pack(stereo, progress, cancelled)
    else:
        raise SeparationUnavailable("The guitar isolation pack isn't installed.")
    save_cached_isolation(media_path, guitar)
    return guitar


def run_pack(stereo, progress=None, cancelled=None, exe=None):
    """Run the downloaded separator program on stereo audio; returns the guitar stem."""
    exe = exe or pack_executable()
    work = tempfile.mkdtemp()
    try:
        source, target = os.path.join(work, "in.wav"), os.path.join(work, "guitar.wav")
        sf.write(source, stereo.T, MODEL_RATE, subtype="FLOAT")
        proc = subprocess.Popen([exe, source, target], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, creationflags=NO_WINDOW)
        tail = []
        for line in proc.stdout:
            if cancelled and cancelled():
                proc.kill()
                proc.wait()
                raise SeparationCancelled
            if line.startswith("PROGRESS ") and progress:
                progress(float(line.split()[1]))
            else:
                tail = (tail + [line.rstrip()])[-15:]
        if proc.wait() != 0 or not os.path.exists(target):
            raise RuntimeError("Guitar isolation failed:\n" + "\n".join(tail))
        guitar, rate = sf.read(target, dtype="float32")
        if rate != OUTPUT_RATE:
            raise RuntimeError(f"Guitar isolation returned {rate} Hz audio.")
        return guitar
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- Cache (separation is slow: keep the result) ------------------------------------

def _cache_file(media_path):
    stat = os.stat(media_path)
    key = f"{MODEL}|{CHUNK_SECONDS}|{os.path.abspath(media_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return os.path.join(data_dir(), "isolated", hashlib.sha1(key.encode()).hexdigest() + ".npy")


def load_cached_isolation(media_path):
    try:
        return np.load(_cache_file(media_path)).astype(np.float32)
    except (OSError, ValueError):
        return None


def save_cached_isolation(media_path, guitar):
    try:
        path = _cache_file(media_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, guitar.astype(np.float16))
    except OSError:
        pass


# --- The pack's program ------------------------------------------------------------

def main(argv):
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    source, target = argv[0], argv[1]
    models_dir = argv[argv.index("--models") + 1] if "--models" in argv else None
    if models_dir is None and getattr(sys, "frozen", False):  # weights shipped inside the pack
        models_dir = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "models")
    stereo, rate = sf.read(source, dtype="float32", always_2d=True)
    if rate != MODEL_RATE:
        from scipy.signal import resample_poly
        stereo = resample_poly(stereo, MODEL_RATE, rate, axis=0)
    if stereo.shape[1] == 1:
        stereo = np.repeat(stereo, 2, axis=1)
    model = load_model(models_dir)
    guitar = separate(stereo[:, :2].T, model, progress=lambda f: print(f"PROGRESS {f:.4f}", flush=True))
    sf.write(target, guitar, OUTPUT_RATE)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
