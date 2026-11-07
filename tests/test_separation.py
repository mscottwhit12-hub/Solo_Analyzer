"""Tests for 'Isolate guitar' around the model: download/install, the pack protocol, caching,
and analysis on an isolated stem. A stand-in separator program replaces the real pack (the
real model is exercised in CI by benchmark/separation_eval.py and pack_smoke_test.py).
"""

import http.server
import io
import os
import sys
import threading
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import soundfile as sf

import analysis as an
import separation

FAKE_SEPARATOR = r'''#!{python}
"""Stand-in for the pack: speaks the same protocol, 'isolates' by keeping the left channel."""
import sys, numpy as np, soundfile as sf
from scipy.signal import resample_poly
if "{mode}" == "fail":
    print("model exploded"); sys.exit(3)
x, rate = sf.read(sys.argv[1], dtype="float32", always_2d=True)
for step in (0.25, 0.5, 0.75, 1.0):
    print(f"PROGRESS {{step}}", flush=True)
    if "{mode}" == "slow":
        import time; time.sleep(0.5)
open("{counter}", "a").write("x")
sf.write(sys.argv[2], resample_poly(x[:, 0], 22050, rate).astype(np.float32), 22050)
print("DONE")
'''


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLO_ANALYZER_DATA", str(tmp_path / "data"))
    return tmp_path


def fake_pack_zip(tmp_path, mode="ok"):
    counter = tmp_path / "runs.txt"
    script = FAKE_SEPARATOR.format(python=sys.executable, mode=mode, counter=counter)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"GuitarSeparator/{separation.PACK_EXE}", script)
        z.writestr("GuitarSeparator/_internal/models/readme.txt", "weights go here")
    return buf.getvalue(), counter


@pytest.fixture
def server():
    """Tiny local HTTP server; set server.payload to the bytes to serve."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/pack.zip":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(srv.payload)))
            self.end_headers()
            for i in range(0, len(srv.payload), 65536):
                self.wfile.write(srv.payload[i:i + 65536])

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.payload = b""
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}/pack.zip"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


@pytest.mark.skipif(sys.platform == "win32", reason="stand-in pack is a script with a shebang")
def test_download_installs_the_pack(data_dir, server):
    server.payload, _ = fake_pack_zip(data_dir)
    seen = []
    exe = separation.download_pack(progress=lambda done, total: seen.append((done, total)), url=server.url)
    assert exe == separation.pack_executable() and os.access(exe, os.X_OK)
    assert seen and abs(seen[-1][0] - seen[-1][1]) < 1e-9          # progress reached the total
    assert separation.available() in ("pack", "in-process")


def test_download_failure_explains_itself(data_dir, server):
    with pytest.raises(separation.SeparationUnavailable, match="releases"):
        separation.download_pack(url=server.url.replace("pack.zip", "missing.zip"))
    assert separation.pack_executable() is None


def test_download_can_be_cancelled(data_dir, server):
    server.payload = b"x" * (5 << 20)
    with pytest.raises(separation.SeparationCancelled):
        separation.download_pack(url=server.url, cancelled=lambda: True)
    assert separation.pack_executable() is None
    assert not [f for f in os.listdir(separation.data_dir()) if f.endswith(".zip")]  # temp file cleaned up


def test_malicious_zip_is_refused(data_dir, server):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.txt", "nope")
    server.payload = buf.getvalue()
    with pytest.raises(separation.SeparationUnavailable, match="malformed"):
        separation.download_pack(url=server.url)
    assert not (data_dir / "evil.txt").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="stand-in pack is a script with a shebang")
def test_pack_protocol_progress_and_output(data_dir, server):
    server.payload, _ = fake_pack_zip(data_dir)
    exe = separation.download_pack(url=server.url)
    stereo = np.stack([np.sin(np.arange(44100) / 10), np.zeros(44100)]).astype(np.float32)
    steps = []
    guitar = separation.run_pack(stereo, progress=steps.append, exe=exe)
    assert steps == [0.25, 0.5, 0.75, 1.0]
    assert abs(len(guitar) - 22050) < 10


@pytest.mark.skipif(sys.platform == "win32", reason="stand-in pack is a script with a shebang")
def test_pack_failure_reports_its_output(data_dir, server):
    server.payload, _ = fake_pack_zip(data_dir, mode="fail")
    exe = separation.download_pack(url=server.url)
    with pytest.raises(RuntimeError, match="model exploded"):
        separation.run_pack(np.zeros((2, 4410), np.float32), exe=exe)


@pytest.mark.skipif(sys.platform == "win32", reason="stand-in pack is a script with a shebang")
def test_pack_run_can_be_cancelled(data_dir, server):
    server.payload, _ = fake_pack_zip(data_dir, mode="slow")
    exe = separation.download_pack(url=server.url)
    calls = []
    with pytest.raises(separation.SeparationCancelled):
        separation.run_pack(np.zeros((2, 4410), np.float32), exe=exe,
                            cancelled=lambda: calls.append(1) or len(calls) > 1)


@pytest.mark.skipif(sys.platform == "win32", reason="stand-in pack is a script with a shebang")
def test_isolation_is_cached_per_file(data_dir, server, monkeypatch):
    server.payload, counter = fake_pack_zip(data_dir)
    separation.download_pack(url=server.url)
    monkeypatch.setattr(separation, "in_process_available", lambda: False)
    media = data_dir / "take.wav"
    sf.write(media, np.sin(np.arange(44100 * 2) / 7).astype(np.float32) * 0.5, 44100)
    first = separation.isolate_file(str(media))
    second = separation.isolate_file(str(media))
    assert counter.read_text() == "x"                                  # separated once
    assert np.allclose(first, second, atol=1e-3)                       # (cache stores float16)
    sf.write(media, np.zeros(44100, np.float32), 44100)                # file changed -> separate again
    separation.isolate_file(str(media))
    assert counter.read_text() == "xx"


def test_isolate_without_pack_says_so(data_dir, monkeypatch):
    monkeypatch.setattr(separation, "in_process_available", lambda: False)
    media = data_dir / "take.wav"
    sf.write(media, np.zeros(4410, np.float32), 44100)
    with pytest.raises(separation.SeparationUnavailable):
        separation.isolate_file(str(media))


def test_analysis_tracks_the_isolated_stem():
    sr = an.SAMPLE_RATE
    t = np.arange(int(sr * 3.2)) / sr
    lead = np.zeros(len(t))
    for k, m in enumerate([69, 72, 74, 76, 79, 76, 74, 72]):
        a, b = int(k * 0.4 * sr), int((k + 1) * 0.4 * sr)
        tt = t[a:b] - t[a]
        f = an.midi_to_hz(m)
        lead[a:b] = np.exp(-tt * 3) * (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(4 * np.pi * f * tt))
    band = np.sin(2 * np.pi * an.midi_to_hz(57) * t) * 2.0 + np.sin(2 * np.pi * an.midi_to_hz(64) * t) * 2.0
    mix = (lead + band).astype(np.float32)
    plain = an.analyze_pitch(mix, sr)
    isolated = an.analyze_pitch(mix, sr, isolated=lead.astype(np.float32)[:-100])  # decoders differ slightly
    lead_share = lambda n: np.mean(n[n >= 0] >= 69) if np.any(n >= 0) else 0
    assert lead_share(isolated.notes) > 0.9 > lead_share(plain.notes)
    assert isolated.isolated and not plain.isolated
    assert np.array_equal(isolated.audio, mix)          # the original is kept for the Diagnostics tab


def test_cache_keeps_isolated_and_plain_apart(tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    assert an.cache_path(str(media)) != an.cache_path(str(media), isolated=True)
    assert an.cache_path(str(media), band_mode=True) != an.cache_path(str(media), band_mode=True, isolated=True)
