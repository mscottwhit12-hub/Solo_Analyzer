import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

import solo_analyzer as sa

SR = sa.SAMPLE_RATE


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("Solo Analyzer Tests")
    application.setOrganizationName("Solo Analyzer Tests")
    return application


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def melody(midis, note_seconds=0.4, gap=0.05):
    """Plucked-string-ish synthetic melody with a little vibrato."""
    parts = []
    for m in midis:
        t = np.arange(int(SR * note_seconds)) / SR
        f = hz(m + 0.1 * np.sin(2 * np.pi * 5 * t))
        phase = 2 * np.pi * np.cumsum(f) / SR
        parts.append(np.exp(-t * 3) * (np.sin(phase) + 0.4 * np.sin(2 * phase)))
        parts.append(np.zeros(int(SR * gap)))
    return np.concatenate(parts).astype(np.float32)


# --- theory ----------------------------------------------------------------

def test_breakdown_uses_correct_root():
    # Regression: A used to count as ♭3 in A minor (9-semitone offset).
    assert sa.degree_breakdown(np.array([hz(69)]), "A", "Minor") == {"1": 100.0}
    assert sa.degree_breakdown(np.array([hz(72)]), "A", "Minor") == {"♭3": 100.0}
    assert sa.degree_breakdown(np.array([hz(76)]), "A", "Minor") == {"5": 100.0}
    assert sa.degree_breakdown(np.array([hz(70)]), "A", "Minor") == {"Out": 100.0}


@pytest.mark.parametrize("scale", list(sa.SCALES))
@pytest.mark.parametrize("root", sa.NOTE_NAMES)
def test_readout_and_breakdown_agree(root, scale):
    rng = np.random.default_rng(0)
    for f in hz(rng.uniform(40, 90, 50)):
        note, _ = sa.note_info(f)
        assert sa.degree_breakdown(np.array([f]), root, scale) == {sa.degree_of(note, root, scale): 100.0}


def test_note_info_cents():
    assert sa.note_info(hz(69.18)) == ("A", 18)
    assert sa.note_info(hz(68.7)) == ("A", -30)
    assert sa.note_info(0.0) == (None, 0)


def test_blues_scale_has_flat_five():
    assert sa.degree_of("D#", "A", "Blues") == "♭5"
    assert sa.degree_of("B", "A", "Minor Pentatonic") == "Out"


def test_detect_key():
    a_minor = np.repeat(hz(np.array([69, 72, 76, 74, 69, 67, 69, 64, 72, 69])), 50)
    assert sa.detect_key(a_minor) == ("A", "Minor")
    c_major = np.repeat(hz(np.array([60, 64, 67, 65, 62, 60, 67, 71, 72, 60])), 50)
    assert sa.detect_key(c_major) == ("C", "Major")
    assert sa.detect_key(np.zeros(100)) is None


# --- analysis --------------------------------------------------------------

def test_tracks_notes_up_to_e6():
    y = melody([88, 86, 84], note_seconds=0.6)  # E6, D6, C6
    f0, _ = sa.analyze_pitch(y, SR)
    found = {sa.note_info(f)[0] for f in f0 if f > 0}
    assert {"E", "D", "C"} <= found


def test_chunked_tracking_matches_whole_file():
    y = melody([57, 60, 62, 64, 67, 69, 72, 74, 76, 74, 72, 69] * 4)  # ~22 s, three chunks
    assert len(y) / SR > 2 * sa.CHUNK_SECONDS
    chunked, chunked_voiced = sa.track_pitch(y, SR)
    whole, whole_voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz(sa.PITCH_FLOOR),
                                          fmax=librosa.note_to_hz(sa.PITCH_CEILING),
                                          sr=SR, hop_length=sa.HOP_LENGTH)
    assert len(chunked) == len(whole)
    assert np.mean(chunked_voiced == whole_voiced) > 0.98
    both = chunked_voiced & whole_voiced
    assert np.mean(np.abs(sa.hz_to_midi(chunked[both]) - sa.hz_to_midi(whole[both])) < 0.5) > 0.99


def test_cancel_stops_analysis():
    with pytest.raises(sa.AnalysisCancelled):
        sa.analyze_pitch(melody([60] * 10), SR, cancelled=lambda: True)


def test_progress_reaches_the_end():
    seen = []
    sa.analyze_pitch(melody([60, 64]), SR, progress=lambda fraction, message: seen.append(fraction))
    assert seen == sorted(seen) and seen[-1] >= 0.95


def test_drift_filter_keeps_slides_and_holds_jitter():
    quiet = np.zeros(8)
    # Slide: pitch already rising, then a 2-semitone step in the same direction -> kept
    slide = hz(np.array([60, 60, 60.5, 61, 63, 63.5, 64, 64]))
    assert np.allclose(sa.remove_drift(slide, quiet, quiet + 1), slide)
    # Jitter: steady pitch, sudden 3-semitone jump with no attack -> held at previous pitch
    jitter = hz(np.array([60, 60, 60, 60, 63, 60, 60, 60]))
    assert np.allclose(sa.remove_drift(jitter, quiet, quiet + 1)[4], hz(60))
    # A real picked note (onset) is kept
    onset = quiet.copy()
    onset[4] = 1.0
    assert np.allclose(sa.remove_drift(jitter, onset, quiet + 1)[4], hz(63))


def test_cache_roundtrip(app, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")
    f0, times = np.array([0.0, 440.0]), np.array([0.0, 0.01])
    assert sa.load_cached(str(media)) is None
    sa.save_cached(str(media), f0, times)
    cached = sa.load_cached(str(media))
    assert np.array_equal(cached[0], f0) and np.array_equal(cached[1], times)
    media.write_bytes(b"changed!")  # different size -> different cache entry
    assert sa.load_cached(str(media)) is None


# --- widgets ---------------------------------------------------------------

def test_breakdown_legend_wraps_when_narrow(app):
    widget = sa.ScaleBreakdownWidget()
    f0 = hz(np.repeat(np.arange(57, 69), 10))  # every pitch class -> 8 labels
    widget.show()
    widget.resize(1200, 50)
    app.processEvents()
    widget.update_breakdown(f0, "A", "Minor")
    wide = widget.height()
    widget.resize(250, wide)
    app.processEvents()
    assert widget.height() > wide


def test_graph_click_requests_seek(app):
    graph = sa.PitchGraphWidget()
    graph.resize(1000, 300)
    times = np.arange(0, 60, 0.01)
    graph.set_data(times, np.full(len(times), 440.0))
    graph.set_time(40.0)  # visible window is 10..40 s
    got = []
    graph.seek_requested.connect(got.append)
    pos = QPointF(500, 150)
    graph.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress, pos, pos,
                                      Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                      Qt.KeyboardModifier.NoModifier))
    assert got and abs(got[0] - 25.0) < 0.1
