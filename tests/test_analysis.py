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

def frames(*notes_and_counts):
    """Per-frame note array, e.g. frames((69, 10), (-1, 3), (72, 5))."""
    return np.concatenate([np.full(n, note) for note, n in notes_and_counts])


def test_breakdown_uses_correct_root():
    # Regression: A used to count as ♭3 in A minor (9-semitone offset).
    assert sa.degree_breakdown(frames((69, 5)), "A", "Minor") == {"1": 100.0}
    assert sa.degree_breakdown(frames((72, 5)), "A", "Minor") == {"♭3": 100.0}
    assert sa.degree_breakdown(frames((76, 5)), "A", "Minor") == {"5": 100.0}
    assert sa.degree_breakdown(frames((70, 5)), "A", "Minor") == {"Out": 100.0}
    assert sa.degree_breakdown(frames((-1, 5)), "A", "Minor") == {}


@pytest.mark.parametrize("scale", list(sa.SCALES))
@pytest.mark.parametrize("root", sa.NOTE_NAMES)
def test_readout_and_breakdown_agree(root, scale):
    for midi in range(40, 90):
        expected = sa.degree_of(sa.note_name(midi), root, scale)
        assert sa.degree_breakdown(frames((midi, 3)), root, scale) == {expected: 100.0}


def test_cents_are_relative_to_tuning():
    assert sa.cents_off(hz(69.18), 69) == 18
    assert sa.cents_off(hz(68.9), 69, tuning=-10) == 0   # in tune for a guitar tuned 10 cents flat
    assert sa.cents_off(0.0, -1) == 0
    assert sa.note_name(69) == "A" and sa.note_name(-1) is None


def test_blues_scale_has_flat_five():
    assert sa.degree_of("D#", "A", "Blues") == "♭5"
    assert sa.degree_of("B", "A", "Minor Pentatonic") == "Out"


def test_detect_key():
    a_minor = np.repeat([69, 72, 76, 74, 69, 67, 69, 64, 72, 69], 50)
    assert sa.detect_key(a_minor) == ("A", "Minor")
    c_major = np.repeat([60, 64, 67, 65, 62, 60, 67, 71, 72, 60], 50)
    assert sa.detect_key(c_major) == ("C", "Major")
    assert sa.detect_key(np.full(100, -1)) is None


# --- note tracking ---------------------------------------------------------

def test_tuning_estimate():
    f0 = hz(np.repeat([69, 72, 74, 76], 20) - 0.12)
    assert abs(sa.estimate_tuning(f0) - -12) < 1


def test_vibrato_stays_on_its_note():
    t = np.arange(200) * sa.HOP_LENGTH / SR
    f0 = hz(69 + 0.45 * np.sin(2 * np.pi * 5.5 * t))  # wide ±45 cent vibrato
    assert set(sa.quantize_notes(f0, 0.0)) == {69}


def test_bend_transit_counts_as_target_and_held_note_is_kept():
    hold, rise, top = 15, 20, 30
    pitch = np.concatenate([np.full(hold, 74.0), np.linspace(74, 76, rise), np.full(top, 76.0)])
    notes = sa.quantize_notes(hz(pitch), 0.0)
    assert set(notes[:hold]) == {74}                      # D held before the bend
    assert 75 not in notes                                # the bend is never counted as D#
    left_d = pitch - 74 > sa.SETTLED_SEMITONES
    assert set(notes[left_d]) == {76}                     # once it has left D, it counts as E


def test_fast_legato_notes_are_not_swallowed():
    pitch = np.repeat([72, 74] * 6, 9).astype(float)  # 105 ms C/D trill
    assert np.array_equal(sa.quantize_notes(hz(pitch), 0.0), pitch.astype(int))


def test_double_stop_subharmonic_is_folded_back():
    # A4 phrase with a passage reported two octaves low (what a fourth double stop produces)
    notes = frames((69, 40), (72, 40), (50, 30), (74, 40), (72, 40))
    f0 = hz(notes.astype(float))
    fixed_f0, fixed = sa.fix_octave_errors(f0, notes)
    assert set(fixed[80:110]) == {74}
    assert np.allclose(fixed_f0[80:110], hz(74))


def test_real_octave_leap_is_kept():
    notes = frames((69, 60), (81, 60), (79, 60), (76, 60))  # A4 up to A5, then down: a real leap
    _, fixed = sa.fix_octave_errors(hz(notes.astype(float)), notes)
    assert np.array_equal(fixed, notes)


def test_islands_and_gaps():
    notes = frames((-1, 10), (60, 2), (-1, 10), (69, 20), (-1, 3), (69, 20))
    f0 = np.where(notes >= 0, hz(np.maximum(notes, 0).astype(float)), 0.0)
    f0, notes = sa.remove_islands(f0, notes)
    assert 60 not in notes                     # 2-frame blip dropped
    f0, notes = sa.fill_gaps(f0, notes)
    assert set(notes[22:65]) == {69}           # 3-frame dropout inside the held A bridged


def test_display_range_ignores_outliers():
    f0 = np.concatenate([hz(np.repeat(np.arange(60, 80), 50).astype(float)), hz(np.array([30.0, 100.0]))])
    low, high = sa.display_range(f0)
    assert np.log2(high / low) < 2.5


# --- analysis --------------------------------------------------------------

def test_tracks_notes_up_to_e6():
    y = melody([88, 86, 84], note_seconds=0.6)  # E6, D6, C6
    result = sa.analyze_pitch(y, SR)
    assert {88, 86, 84} <= set(result.notes)


def test_chunked_tracking_matches_whole_file():
    y = melody([57, 60, 62, 64, 67, 69, 72, 74, 76, 74, 72, 69] * 4)  # ~22 s, three chunks
    assert len(y) / SR > 2 * sa.CHUNK_SECONDS
    chunked, chunked_voiced = sa.track_pitch(y, SR)
    whole, whole_voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz(sa.PITCH_FLOOR),
                                          fmax=librosa.note_to_hz(sa.PITCH_CEILING),
                                          sr=SR, hop_length=sa.HOP_LENGTH, frame_length=sa.FRAME_LENGTH)
    assert len(chunked) == len(whole)
    assert np.mean(chunked_voiced == whole_voiced) > 0.98
    both = chunked_voiced & whole_voiced
    assert np.mean(np.abs(sa.hz_to_midi(chunked[both]) - sa.hz_to_midi(whole[both])) < 0.5) > 0.99


def test_full_analysis_on_a_melody():
    played = [69, 72, 74, 76, 79, 76, 74, 72]
    result = sa.analyze_pitch(melody(played, note_seconds=0.35), SR)
    assert len(result.f0) == len(result.times) == len(result.notes)
    assert set(result.notes[result.notes >= 0]) == set(played)
    assert abs(result.tuning) < 5


def test_cancel_stops_analysis():
    with pytest.raises(sa.AnalysisCancelled):
        sa.analyze_pitch(melody([60] * 10), SR, cancelled=lambda: True)


def test_progress_reaches_the_end():
    seen = []
    sa.analyze_pitch(melody([60, 64]), SR, progress=lambda fraction, message: seen.append(fraction))
    assert seen == sorted(seen) and seen[-1] >= 0.95


def test_cache_roundtrip(app, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")
    result = sa.AnalysisResult(np.array([0.0, 440.0]), np.array([0.0, 0.01]), np.array([-1, 69]), -7.5)
    assert sa.load_cached(str(media)) is None
    sa.save_cached(str(media), result)
    cached = sa.load_cached(str(media))
    assert np.array_equal(cached.f0, result.f0) and np.array_equal(cached.notes, result.notes)
    assert cached.tuning == -7.5
    media.write_bytes(b"changed!")  # different size -> different cache entry
    assert sa.load_cached(str(media)) is None


# --- widgets ---------------------------------------------------------------

def test_breakdown_legend_wraps_when_narrow(app):
    widget = sa.ScaleBreakdownWidget()
    notes = np.repeat(np.arange(57, 69), 10)  # every pitch class -> 8 labels
    widget.show()
    widget.resize(1200, 50)
    app.processEvents()
    widget.update_breakdown(notes, "A", "Minor")
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
