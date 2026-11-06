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

import analysis as an
import solo_analyzer as sa

SR = an.SAMPLE_RATE


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
    assert an.degree_breakdown(frames((69, 5)), "A", "Minor") == {"1": 100.0}
    assert an.degree_breakdown(frames((72, 5)), "A", "Minor") == {"♭3": 100.0}
    assert an.degree_breakdown(frames((76, 5)), "A", "Minor") == {"5": 100.0}
    assert an.degree_breakdown(frames((70, 5)), "A", "Minor") == {"Out": 100.0}
    assert an.degree_breakdown(frames((-1, 5)), "A", "Minor") == {}


@pytest.mark.parametrize("scale", list(an.SCALES))
@pytest.mark.parametrize("root", an.NOTE_NAMES)
def test_readout_and_breakdown_agree(root, scale):
    for midi in range(40, 90):
        expected = an.degree_of(an.note_name(midi), root, scale)
        assert an.degree_breakdown(frames((midi, 3)), root, scale) == {expected: 100.0}


def test_cents_are_relative_to_tuning():
    assert an.cents_off(hz(69.18), 69) == 18
    assert an.cents_off(hz(68.9), 69, tuning=-10) == 0   # in tune for a guitar tuned 10 cents flat
    assert an.cents_off(0.0, -1) == 0
    assert an.note_name(69) == "A" and an.note_name(-1) is None


def test_blues_scale_has_flat_five():
    assert an.degree_of("D#", "A", "Blues") == "♭5"
    assert an.degree_of("B", "A", "Minor Pentatonic") == "Out"


def test_detect_key():
    a_minor = np.repeat([69, 72, 76, 74, 69, 67, 69, 64, 72, 69], 50)
    assert an.detect_key(a_minor) == ("A", "Minor")
    c_major = np.repeat([60, 64, 67, 65, 62, 60, 67, 71, 72, 60], 50)
    assert an.detect_key(c_major) == ("C", "Major")
    assert an.detect_key(np.full(100, -1)) is None


# --- note tracking ---------------------------------------------------------

def test_tuning_estimate():
    f0 = hz(np.repeat([69, 72, 74, 76], 20) - 0.12)
    assert abs(an.estimate_tuning(f0) - -12) < 1


def test_vibrato_stays_on_its_note():
    t = np.arange(200) * an.HOP_LENGTH / SR
    f0 = hz(69 + 0.45 * np.sin(2 * np.pi * 5.5 * t))  # wide ±45 cent vibrato
    assert set(an.quantize_notes(f0, 0.0)) == {69}


def test_bend_transit_counts_as_target_and_held_note_is_kept():
    hold, rise, top = 15, 20, 30
    pitch = np.concatenate([np.full(hold, 74.0), np.linspace(74, 76, rise), np.full(top, 76.0)])
    notes = an.quantize_notes(hz(pitch), 0.0)
    assert set(notes[:hold]) == {74}                      # D held before the bend
    assert 75 not in notes                                # the bend is never counted as D#
    left_d = pitch - 74 > an.SETTLED_SEMITONES
    assert set(notes[left_d]) == {76}                     # once it has left D, it counts as E


def test_fast_legato_notes_are_not_swallowed():
    pitch = np.repeat([72, 74] * 6, 9).astype(float)  # 105 ms C/D trill
    assert np.array_equal(an.quantize_notes(hz(pitch), 0.0), pitch.astype(int))


def test_double_stop_subharmonic_is_folded_back():
    # A4 phrase with a passage reported two octaves low (what a fourth double stop produces)
    notes = frames((69, 40), (72, 40), (50, 30), (74, 40), (72, 40))
    f0 = hz(notes.astype(float))
    fixed_f0, fixed = an.fix_octave_errors(f0, notes)
    assert set(fixed[80:110]) == {74}
    assert np.allclose(fixed_f0[80:110], hz(74))


def test_real_octave_leap_is_kept():
    notes = frames((69, 60), (81, 60), (79, 60), (76, 60))  # A4 up to A5, then down: a real leap
    _, fixed = an.fix_octave_errors(hz(notes.astype(float)), notes)
    assert np.array_equal(fixed, notes)


def test_islands_and_gaps():
    notes = frames((-1, 10), (60, 2), (-1, 10), (69, 20), (-1, 3), (69, 20))
    f0 = np.where(notes >= 0, hz(np.maximum(notes, 0).astype(float)), 0.0)
    f0, notes = an.remove_islands(f0, notes)
    assert 60 not in notes                     # 2-frame blip dropped
    f0, notes = an.fill_gaps(f0, notes)
    assert set(notes[22:65]) == {69}           # 3-frame dropout inside the held A bridged


def test_display_range_ignores_outliers():
    f0 = np.concatenate([hz(np.repeat(np.arange(60, 80), 50).astype(float)), hz(np.array([30.0, 100.0]))])
    low, high = an.display_range(f0)
    assert np.log2(high / low) < 2.5


# --- analysis --------------------------------------------------------------

def test_tracks_notes_up_to_e6():
    y = melody([88, 86, 84], note_seconds=0.6)  # E6, D6, C6
    result = an.analyze_pitch(y, SR)
    assert {88, 86, 84} <= set(result.notes)


def test_chunked_tracking_matches_whole_file():
    y = melody([57, 60, 62, 64, 67, 69, 72, 74, 76, 74, 72, 69] * 4)  # ~22 s, three chunks
    assert len(y) / SR > 2 * an.CHUNK_SECONDS
    chunked, chunked_voiced, _ = an.track_pitch(y, SR)
    whole, whole_voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz(an.PITCH_FLOOR),
                                          fmax=librosa.note_to_hz(an.PITCH_CEILING),
                                          sr=SR, hop_length=an.HOP_LENGTH, frame_length=an.FRAME_LENGTH)
    assert len(chunked) == len(whole)
    assert np.mean(chunked_voiced == whole_voiced) > 0.98
    both = chunked_voiced & whole_voiced
    assert np.mean(np.abs(an.hz_to_midi(chunked[both]) - an.hz_to_midi(whole[both])) < 0.5) > 0.99


def test_full_analysis_on_a_melody():
    played = [69, 72, 74, 76, 79, 76, 74, 72]
    result = an.analyze_pitch(melody(played, note_seconds=0.35), SR)
    assert len(result.f0) == len(result.times) == len(result.notes)
    assert set(result.notes[result.notes >= 0]) == set(played)
    assert abs(result.tuning) < 5


def test_cancel_stops_analysis():
    with pytest.raises(an.AnalysisCancelled):
        an.analyze_pitch(melody([60] * 10), SR, cancelled=lambda: True)


def test_progress_reaches_the_end():
    seen = []
    an.analyze_pitch(melody([60, 64]), SR, progress=lambda fraction, message: seen.append(fraction))
    assert seen == sorted(seen) and seen[-1] >= 0.95


def test_cache_roundtrip(app, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")
    result = an.AnalysisResult(np.array([0.0, 440.0]), np.array([0.0, 0.01]), np.array([-1, 69]), -7.5)
    assert an.load_cached(str(media)) is None
    an.save_cached(str(media), result)
    cached = an.load_cached(str(media))
    assert np.array_equal(cached.f0, result.f0) and np.array_equal(cached.notes, result.notes)
    assert cached.tuning == -7.5
    media.write_bytes(b"changed!")  # different size -> different cache entry
    assert an.load_cached(str(media)) is None


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


# --- diagnostics -----------------------------------------------------------

import diagnostics as dg  # noqa: E402


def test_spectrogram_puts_a_tone_on_its_row():
    t = np.arange(SR) / SR
    image = dg.log_spectrogram(np.sin(2 * np.pi * 440 * t).astype(np.float32))
    assert image.shape == (1 + SR // an.HOP_LENGTH, len(dg.spectrogram_rows()))
    loudest_row = int(np.median(np.argmax(image[10:-10], axis=1)))
    assert abs(dg.spectrogram_rows()[loudest_row] - 440) < 440 * 0.03


def test_spectrogram_chunks_line_up(monkeypatch):
    y = melody([60, 67, 72, 76] * 3)
    whole = dg.log_spectrogram(y)
    monkeypatch.setattr(dg, "SPEC_CHUNK_FRAMES", 97)  # many chunks, odd boundaries
    assert np.array_equal(dg.log_spectrogram(y), whole)


def fake_result():
    n = 12
    notes = np.array([69] * 4 + [76] * 4 + [-1] * 4)
    r = an.AnalysisResult(an.midi_to_hz(np.maximum(notes, 0)).astype(float) * (notes >= 0), np.arange(n) * 0.0116,
                          notes, -8.0)
    r.raw_f0 = r.f0.copy()
    r.raw_f0[4:8] = an.midi_to_hz(52)        # tracker reported E3; octave fix moved it to E5
    r.confidence = np.full(n, 0.9)
    r.held_notes = np.array([69, 69, 68, 69, 52, 52, 52, 52, -1, -1, -1, -1])
    r.swept_notes = r.held_notes.copy()
    r.octave_shift = np.array([0] * 4 + [24] * 4 + [0] * 4)
    r.removed = np.zeros(n, bool)
    r.filled = np.zeros(n, bool)
    return r


def test_explanations_name_each_decision():
    r = fake_result()
    octave = " ".join(dg.explain_frame(r, 5, "A", "Minor"))
    assert "Octave fix" in octave and "two octaves" in octave and "E5" in octave and "degree <b>5</b>" in octave
    plain = " ".join(dg.explain_frame(r, 0, "A", "Minor"))
    assert "A4" in plain and "degree <b>1</b>" in plain and "Octave fix" not in plain
    assert "no note" in " ".join(dg.explain_frame(r, 10, "A", "Minor"))


def test_analysis_records_every_stage():
    result = an.analyze_pitch(melody([69, 72, 74, 76]), SR)
    n = len(result.notes)
    for name in an.AnalysisResult.DIAGNOSTIC_FIELDS:
        assert len(getattr(result, name)) == n, name
    assert result.audio is not None and result.tracked_audio is not None


def test_diagnostics_panel_renders(app):
    from PyQt6.QtGui import QImage
    r = an.analyze_pitch(melody([69, 72, 74, 76, 79]), SR)
    panel = dg.DiagnosticsPanel()
    panel.resize(1100, 700)
    panel.set_result(r, None)
    panel._on_ready({"tracked": dg.log_spectrogram(r.tracked_audio), "original": dg.log_spectrogram(r.audio)})
    panel.set_time(1.0)
    panel.signal_input.setCurrentText(dg.SIGNAL_ORIGINAL)
    image = QImage(panel.size(), QImage.Format.Format_ARGB32)
    panel.render(image)
    assert "Result" in panel.explanation.text()


# --- double stops, attacks, key detection -----------------------------------

def overdriven(*voices, seconds=1.5, drive=6.0):
    """Sum of harmonic tones (MIDI numbers) through overdrive, like an amped double stop."""
    t = np.arange(int(SR * seconds)) / SR
    x = sum(sum(a * np.sin(2 * np.pi * k * hz(m) * t) for k, a in enumerate([0.45, 1, 0.8, 0.55, 0.45], 1))
            for m in voices)
    x = np.tanh(drive * x / np.abs(x).max()) * np.exp(-t * 0.8)
    return x.astype(np.float32)


def test_attack_gap_goes_to_following_note():
    notes = frames((69, 10), (-1, 8), (72, 20), (-1, 30), (74, 10))
    f0 = np.where(notes >= 0, hz(np.maximum(notes, 0).astype(float)), 0.0)
    level = np.zeros(len(notes))
    level[48:58] = -40                      # the long gap is quiet partway through
    _, filled_notes, filled = an.fill_attack_gaps(f0, notes, level)
    assert set(filled_notes[10:18]) == {72} and filled[10:18].all()   # short loud gap -> next note
    assert set(filled_notes[38:68]) == {-1}                            # long, partly quiet gap untouched


def test_phantom_pitch_vs_real_low_note():
    double_stop = overdriven(69, 64)        # A4 + E4: repeats at A2, which nobody plays
    assert an.is_phantom_pitch(double_stop, SR, slice(20, 100), hz(45))
    low_note = overdriven(57)               # a real A3
    assert not an.is_phantom_pitch(low_note, SR, slice(20, 100), hz(57))


def test_long_double_stop_reads_as_a_note_in_register():
    y = np.concatenate([melody([72, 74, 72], note_seconds=0.5), overdriven(69, 64, seconds=1.6),
                        melody([72, 74, 72], note_seconds=0.5)])
    result = an.analyze_pitch(y, SR)
    start = int(1.6 * SR / an.HOP_LENGTH)
    middle = result.notes[start + 20:start + 120]
    assert np.mean(np.isin(middle, [69, 64])) > 0.8   # not two octaves down at A2


def test_key_uses_phrase_endings():
    # A minor pentatonic phrases that each resolve on a long A
    phrase = frames((72, 10), (74, 10), (76, 10), (79, 10), (76, 10), (69, 40), (-1, 30))
    notes = np.concatenate([phrase] * 6)
    assert an.detect_key(notes) == ("A", "Minor")
    same_notes_resolving_to_c = np.concatenate([frames((69, 10), (74, 10), (76, 10), (79, 10), (76, 10),
                                                       (72, 40), (-1, 30))] * 6)
    assert an.detect_key(same_notes_resolving_to_c) == ("C", "Major")


def test_close_alternatives_flag_ambiguity():
    ambiguous = np.repeat([69, 72, 74, 76, 79], 20)        # pentatonic, no resolution
    assert an.close_alternatives(ambiguous)
    clear = np.concatenate([frames((69, 60), (72, 15), (76, 30), (-1, 30))] * 5)
    assert an.detect_key(clear) == ("A", "Minor")


# --- fast notes, bass, octave guard (realistic-benchmark fixes) ----------------

def test_fast_scale_run_is_not_mistaken_for_a_slide():
    rng = np.random.default_rng(0)
    run_notes = [76, 78, 79, 81, 83, 84, 86, 88]               # E5 up to E6, 16th notes at 150 bpm
    pitch = np.repeat(run_notes, 8).astype(float) + rng.normal(0, 0.04, 64)  # tracker jitter
    notes = an.quantize_notes(hz(pitch), 0.0)
    for k, want in enumerate(run_notes):
        assert np.mean(notes[k * 8 + 2:k * 8 + 7] == want) >= 0.8, (k, want)


def test_confident_short_notes_survive_blip_removal():
    notes = frames((-1, 10), (84, 3), (-1, 10), (60, 3), (-1, 10))
    f0 = np.where(notes >= 0, hz(np.maximum(notes, 0).astype(float)), 0.0)
    confidence = np.where(np.arange(len(notes)) < 15, 0.9, 0.1)   # the 84 is confident, the 60 isn't
    level = np.zeros(len(notes))
    _, kept = an.remove_islands(f0, notes, confidence, level)
    assert 84 in kept and 60 not in kept


def plucked_bass(length, level_rms):
    """Bass guitar in eighth notes on A2/D2-ish roots, decaying each note."""
    t = np.arange(length) / SR
    eighth = 60 / 92 / 2
    out = np.zeros(length)
    for k in range(int(t[-1] / eighth) + 1):
        a = int(k * eighth * SR)
        m = min(length - a, int(eighth * SR))
        tt = np.arange(m) / SR
        f = hz([45, 50, 45, 52][(k // 8) % 4])
        out[a:a + m] = (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(4 * np.pi * f * tt)
                        + 0.25 * np.sin(6 * np.pi * f * tt)) * np.exp(-tt * 4)
    return out / np.sqrt(np.mean(out ** 2)) * level_rms


def lead_share(notes):
    played = notes[notes >= 0]
    return np.mean(played >= 64) if len(played) else 0.0


def test_bass_below_the_lead_does_not_take_over():
    lead = melody([69, 72, 74, 76, 79, 76, 74, 72], note_seconds=0.45)
    mix = lead + plucked_bass(len(lead), np.sqrt(np.mean(lead ** 2)) * 10 ** (-6 / 20))  # bass 6 dB down
    assert lead_share(an.analyze_pitch(mix.astype(np.float32), SR).notes) > 0.8


def test_band_mode_survives_a_bass_as_loud_as_the_lead():
    lead = melody([69, 72, 74, 76, 79, 76, 74, 72], note_seconds=0.45)
    mix = (lead + plucked_bass(len(lead), np.sqrt(np.mean(lead ** 2)))).astype(np.float32)
    assert lead_share(an.analyze_pitch(mix, SR).notes) < 0.2               # normal mode loses to it...
    result = an.analyze_pitch(mix, SR, band_mode=True)
    assert lead_share(result.notes) > 0.5 and result.band_mode              # ...band mode doesn't


def test_single_octave_fix_needs_spectral_evidence():
    # A real A4 whose surroundings (e.g. a rhythm guitar) sit an octave higher: keep A4
    y = overdriven(69, seconds=1.0)
    notes = frames((81, 60), (69, 40), (81, 60))
    f0 = hz(notes.astype(float))
    audio = np.concatenate([overdriven(81, seconds=0.7), y[:int(0.465 * SR)], overdriven(81, seconds=0.7)])
    _, fixed = an.fix_octave_errors(f0, notes, audio, SR)
    assert set(fixed[60:100]) == {69}


def test_unpitched_double_stop_is_read_from_the_spectrum():
    # A4 + E4 double stop the tracker can't pitch, between A minor notes
    y = np.concatenate([melody([72, 74], note_seconds=0.4), overdriven(69, 64, seconds=0.5),
                        melody([72, 69], note_seconds=0.4)]).astype(np.float32)
    s, e = int(0.9 * SR / an.HOP_LENGTH), int(1.3 * SR / an.HOP_LENGTH)
    notes = an.quantize_notes(an.midi_to_hz(np.full(len(y) // an.HOP_LENGTH + 1, 72.0)), 0.0)
    notes[s:e] = -1                                   # tracker gave up on the double stop
    f0 = np.where(notes >= 0, hz(72.0), 0.0)
    level = np.zeros(len(notes))                      # loud throughout
    _, filled, estimated = an.estimate_unpitched_notes(f0, notes, level, y, SR)
    assert estimated[s:e].all()
    assert np.bincount(filled[s:e]).argmax() == 69    # the top note, A4


def test_quiet_gaps_are_not_filled_from_the_spectrum():
    notes = frames((72, 30), (-1, 20), (72, 30))
    f0 = np.where(notes >= 0, hz(72.0), 0.0)
    level = np.full(len(notes), -40.0)                # a rest
    _, filled, estimated = an.estimate_unpitched_notes(f0, notes, level, melody([72] * 3), SR)
    assert not estimated.any() and set(filled[30:50]) == {-1}
