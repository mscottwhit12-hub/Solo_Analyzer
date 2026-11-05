"""Pitch analysis pipeline and music theory for Solo Analyzer (no UI)."""

import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

# Numba (used by librosa) caches compiled functions next to their source files,
# which don't exist inside a frozen executable. Point the cache somewhere writable.
if getattr(sys, "frozen", False):
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "solo_analyzer_numba"))

import imageio_ffmpeg
import librosa
import numpy as np
from PyQt6.QtCore import QStandardPaths
from scipy.signal import medfilt

# --- Constants -------------------------------------------------------------

SAMPLE_RATE = 22050      # Hz, audio is resampled to this before analysis
HOP_LENGTH = 256         # samples between analysis frames
MIN_PITCH_HZ = 20.0      # anything at or below this is treated as "no pitch"
PITCH_FLOOR = 'C2'       # pitch tracking range: below a drop-C low string...
PITCH_CEILING = 'E6'     # ...up to the 24th fret of the high E string
FRAME_LENGTH = 1024      # samples per pitch window (46 ms): short enough for fast runs
USE_HPSS = True          # strip drums/percussive sound before tracking
ANALYSIS_VERSION = 6     # bump whenever analysis output changes, to invalidate cached results

# Pitch is tracked in chunks so progress can be reported and analysis cancelled.
# Each chunk gets extra context on both sides, which is discarded after tracking.
CHUNK_SECONDS = 10.0
CHUNK_CONTEXT_SECONDS = 1.0

# Note tracking (see quantize_notes)
NOTE_HYSTERESIS = 0.7        # semitones the pitch must move before the note changes
SWEEP_MIN_SEMITONES = 0.6    # a short run that travels at least this far...
SWEEP_STRAIGHTNESS = 0.7     # ...mostly in one direction...
SWEEP_MAX_SECONDS = 0.3      # ...within this time, still moving at its end, is a bend/slide
STEADY_STEP_SEMITONES = 0.03 # frame-to-frame change below this is "holding"...
STEADY_MIN_FRAMES = 3        # ...and a run that holds this long is a note, not a transition
SETTLED_SEMITONES = 0.35     # a frame further than this from its note has left it

# Clean-up (see fix_octave_errors, remove_islands, fill_gaps)
OCTAVE_SUSPECT_SEMITONES = 10  # a passage this far from its neighbours may be an octave error...
SUBHARMONIC_SUSPECT_SEMITONES = 14  # ...this far from the local register, almost certainly is
OCTAVE_NEIGHBOUR_GAP = 0.3     # seconds of silence after which notes aren't neighbours
OCTAVE_FIT_SEMITONES = 7       # ...if shifting it by octaves brings it this close to them
OCTAVE_FIX_MAX_SECONDS = 1.0   # longer passages are trusted as real leaps
OCTAVE_CONTEXT_SECONDS = 2.0   # the register is judged from this much playing either side
# Semitone corrections for locking onto the wrong harmonic: 1/2, 1/3, 1/4, 1/5 of the true
# frequency (sub-harmonics) or 2x, 3x (overtones)
HARMONIC_SHIFTS = (12, 19, 24, 28, -12, -19)
SALIENCE_FFT = 4096
SALIENCE_HARMONICS = 5
SALIENCE_DECAY = 0.8
PHANTOM_MAX_RATIO = 0.25          # below -12 dB at its own pitch and octave, it isn't really sounding
                                  # (overdrive leaves difference tones there around -17 dB)
MIN_NOTE_FRAMES = 4            # ~46 ms

# Attack gaps (see fill_attack_gaps)
ATTACK_GAP_MAX_SECONDS = 0.2   # a "no pitch" stretch this short...
ATTACK_GAP_LEVEL_DB = -20.0    # ...that stays this loud, right before a note, belongs to it
MAX_GAP_FRAMES = 4

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Scale name (as shown in the UI) -> {semitones above root: degree label}
SCALES = {
    "Minor":            {0: "1", 2: "2", 3: "♭3", 5: "4", 7: "5", 8: "♭6", 10: "♭7"},
    "Major":            {0: "1", 2: "2", 4: "3", 5: "4", 7: "5", 9: "6", 11: "7"},
    "Minor Pentatonic": {0: "1", 3: "♭3", 5: "4", 7: "5", 10: "♭7"},
    "Major Pentatonic": {0: "1", 2: "2", 4: "3", 7: "5", 9: "6"},
    "Blues":            {0: "1", 3: "♭3", 5: "4", 6: "♭5", 7: "5", 10: "♭7"},
    "Dorian":           {0: "1", 2: "2", 3: "♭3", 5: "4", 7: "5", 9: "6", 10: "♭7"},
    "Mixolydian":       {0: "1", 2: "2", 4: "3", 5: "4", 7: "5", 9: "6", 10: "♭7"},
}
DEFAULT_SCALE = "Minor"

# Albrecht & Shanahan (2013) key profiles, measured on popular music: how much each pitch
# class is used in a major / minor key. (They beat Krumhansl-Kessler on our key benchmark.)
MAJOR_PROFILE = np.array([0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081])
MINOR_PROFILE = np.array([0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052])
PHRASE_REST_SECONDS = 0.25     # a note followed by this much silence ends a phrase
PHRASE_END_WEIGHT = 0.25       # extra weight for phrase-ending notes (solos resolve to the tonic)
PHRASE_QUIET_DB = -20.0        # quieter than this = resting (playing is above -5 dB, rests around -20 to -30)
KEY_ALTERNATIVE_MARGIN = 0.08  # other keys this close to the best are shown (clear solos: gap > 0.085)


# --- Music theory ----------------------------------------------------------

def midi_to_hz(midi):
    """Convert a (fractional) MIDI note number to Hz. Works on scalars and arrays."""
    return 440.0 * 2.0 ** ((np.asarray(midi, float) - 69) / 12)


def hz_to_midi(freq):
    """Convert frequency in Hz to a (fractional) MIDI note number. Works on scalars and arrays."""
    return 69 + 12 * np.log2(freq / 440.0)


def get_intervals(scale):
    return SCALES.get(scale, SCALES[DEFAULT_SCALE])


def note_name(midi):
    """Note name for a MIDI number, None for -1 (no note)."""
    return NOTE_NAMES[int(midi) % 12] if midi >= 0 else None


def cents_off(freq, midi, tuning=0.0):
    """How far a frequency is from a note, in cents, relative to the instrument's tuning."""
    if midi < 0 or not freq > MIN_PITCH_HZ:
        return 0
    return int(round((hz_to_midi(freq) - tuning / 100 - midi) * 100))


def degree_of(note_name, root, scale):
    """Scale degree label of a note in a key, "Out" if it isn't in the scale, "--" if no note."""
    if note_name is None or root not in NOTE_NAMES:
        return "--"
    semitones = (NOTE_NAMES.index(note_name) - NOTE_NAMES.index(root)) % 12
    return get_intervals(scale).get(semitones, "Out")


def degree_breakdown(notes, root, scale):
    """{degree label: % of time with a note}, from per-frame notes (the same ones the readout shows)."""
    if notes is None or root not in NOTE_NAMES:
        return {}
    pitched = notes[notes >= 0]
    if len(pitched) == 0:
        return {}
    semitones = (pitched - NOTE_NAMES.index(root)) % 12
    counts = np.bincount(semitones, minlength=12)
    percentages = {}
    for semitone, label in get_intervals(scale).items():
        if counts[semitone]:
            percentages[label] = counts[semitone] / len(pitched) * 100
    out = len(pitched) - sum(counts[s] for s in get_intervals(scale))
    if out:
        percentages["Out"] = out / len(pitched) * 100
    return percentages


def key_candidates(notes, level_db=None):
    """[(root, "Major"/"Minor", score)] best first, from per-frame notes; [] if too little to go on.

    Pitch classes are weighted by how long they're played, plus extra weight for notes that
    end a phrase, since solos tend to resolve there. Scores are correlations with key profiles.
    With per-frame loudness, quiet frames (a note's reverb tail ringing through a rest) count
    as silence, so phrase endings are found even though the pitch tracker still hears the tail.
    """
    if level_db is not None:
        notes = np.where(level_db[:len(notes)] < PHRASE_QUIET_DB, -1, notes)
    segments = [(s, e, n) for s, e, n in runs(notes) if n >= 0]
    if not segments:
        return []
    rest = int(PHRASE_REST_SECONDS * SAMPLE_RATE / HOP_LENGTH)
    duration, phrase_end = np.zeros(12), np.zeros(12)
    for k, (s, e, n) in enumerate(segments):
        duration[n % 12] += e - s
        next_start = segments[k + 1][0] if k + 1 < len(segments) else len(notes)
        if next_start - e >= rest:
            phrase_end[n % 12] += e - s
    if np.count_nonzero(duration) < 2:
        return []
    weights = duration / duration.sum()
    if phrase_end.sum():
        weights = weights + PHRASE_END_WEIGHT * phrase_end / phrase_end.sum()
    scores = [(NOTE_NAMES[root], mode, float(np.corrcoef(weights, np.roll(profile, root))[0, 1]))
              for mode, profile in (("Major", MAJOR_PROFILE), ("Minor", MINOR_PROFILE))
              for root in range(12)]
    return sorted(scores, key=lambda c: c[2], reverse=True)


def detect_key(notes, level_db=None):
    """Best-guess (root, "Major"/"Minor") for per-frame notes, or None."""
    candidates = key_candidates(notes, level_db)
    return (candidates[0][0], candidates[0][1]) if candidates else None


def close_alternatives(notes, level_db=None):
    """Other keys that fit almost as well as the best one (ambiguity worth showing)."""
    candidates = key_candidates(notes, level_db)
    if not candidates:
        return []
    best = candidates[0][2]
    return [(r, m) for r, m, sc in candidates[1:3] if best - sc <= KEY_ALTERNATIVE_MARGIN]


# --- Analysis --------------------------------------------------------------

class AnalysisCancelled(Exception):
    pass


def extract_audio(path):
    """Decode any audio/video file to mono float samples at SAMPLE_RATE using bundled ffmpeg."""
    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", path,
               "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", temp_wav]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode != 0:
            raise RuntimeError("Failed to extract audio from the file.")
        y, _ = librosa.load(temp_wav, sr=SAMPLE_RATE)
        return y
    finally:
        try:
            os.remove(temp_wav)
        except OSError:
            pass


def track_pitch(y, sr, progress=None, cancelled=None):
    """pYIN pitch tracking in chunks.

    Returns (f0 with nan where unvoiced, voiced flags, voicing probability).
    """
    n_frames = 1 + len(y) // HOP_LENGTH
    f0 = np.full(n_frames, np.nan)
    voiced = np.zeros(n_frames, dtype=bool)
    probability = np.zeros(n_frames)
    chunk = int(CHUNK_SECONDS * sr / HOP_LENGTH)
    context = int(CHUNK_CONTEXT_SECONDS * sr / HOP_LENGTH)

    for start in range(0, n_frames, chunk):
        if cancelled and cancelled():
            raise AnalysisCancelled
        end = min(start + chunk, n_frames)
        first = max(0, start - context)
        last = min(n_frames, end + context)
        segment = y[first * HOP_LENGTH:min(len(y), last * HOP_LENGTH + HOP_LENGTH)]
        seg_f0, seg_voiced, seg_prob = librosa.pyin(
            segment, fmin=librosa.note_to_hz(PITCH_FLOOR), fmax=librosa.note_to_hz(PITCH_CEILING),
            sr=sr, hop_length=HOP_LENGTH, frame_length=FRAME_LENGTH, fill_na=None,
        )
        keep = slice(start - first, end - first)
        f0[start:end] = np.where(seg_voiced[keep], seg_f0[keep], np.nan)
        voiced[start:end] = seg_voiced[keep]
        probability[start:end] = seg_prob[keep]
        if progress:
            progress(end / n_frames)
    return f0, voiced, probability


@dataclass
class AnalysisResult:
    f0: np.ndarray      # pitch contour for the graph, Hz (0 = no pitch)
    times: np.ndarray   # seconds, one per frame
    notes: np.ndarray   # note being played per frame (MIDI number), -1 = none
    tuning: float       # how far the instrument is from A440, in cents

    # How each frame's note was decided (shown in the Diagnostics tab)
    raw_f0: np.ndarray = None        # what the pitch tracker reported, Hz (0 = no pitch)
    confidence: np.ndarray = None    # tracker's probability that a pitch is present, 0-1
    attack_filled: np.ndarray = None # loud "no pitch" gap just before a note, given to that note
    level_db: np.ndarray = None      # loudness per frame, dB relative to the file's loud passages
    held_notes: np.ndarray = None    # after rounding with hysteresis
    swept_notes: np.ndarray = None   # after crediting bends/slides to where they land
    octave_shift: np.ndarray = None  # semitones the octave fix moved each frame (0 = untouched)
    removed: np.ndarray = None       # dropped as a too-short blip
    filled: np.ndarray = None        # bridged as a brief dropout

    # Audio, kept in memory only (not cached): original and what the tracker analyzed
    audio: np.ndarray = field(default=None, repr=False)
    tracked_audio: np.ndarray = field(default=None, repr=False)

    DIAGNOSTIC_FIELDS = ("raw_f0", "confidence", "attack_filled", "level_db", "held_notes", "swept_notes",
                         "octave_shift", "removed", "filled")


def estimate_tuning(f0):
    """Cents offset of the playing from A440, from where pitches sit between semitones."""
    pitched = f0[f0 > MIN_PITCH_HZ]
    if len(pitched) < 10:
        return 0.0
    angle = 2 * np.pi * hz_to_midi(pitched)  # semitone position as a point on a circle
    return float(np.angle(np.mean(np.exp(1j * angle))) / (2 * np.pi) * 100)


def frame_levels(y, n_frames):
    """Loudness of each frame in dB relative to the file's loud passages (95th percentile)."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms = np.pad(rms, (0, max(0, n_frames - len(rms))))[:n_frames]
    reference = np.percentile(rms, 95) if np.any(rms > 0) else 1.0
    return 20 * np.log10(rms / reference + 1e-9)


def fill_attack_gaps(f0, notes, level_db):
    """Give short, loud "no pitch" stretches just before a note to that note.

    After a new attack the tracker needs ~100-150 ms to commit, longer for a double stop,
    though the guitar is plainly sounding. A short gap that stays loud and is followed by a
    note belongs to that note.
    """
    f0, notes = f0.copy(), notes.copy()
    filled = np.zeros(len(notes), dtype=bool)
    max_len = int(ATTACK_GAP_MAX_SECONDS * SAMPLE_RATE / HOP_LENGTH)
    for s, e, note in runs(notes):
        if note >= 0 or e >= len(notes) or e - s > max_len or s == 0:
            continue
        if np.all(level_db[s:e] >= ATTACK_GAP_LEVEL_DB):
            notes[s:e] = notes[e]
            f0[s:e] = f0[e]
            filled[s:e] = True
    return f0, notes, filled


def runs(values):
    """[(start, end, value)] for runs of equal consecutive values."""
    if len(values) == 0:
        return []
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(values)]))
    return [(s, e, values[s]) for s, e in zip(starts, ends)]


def quantize_notes(f0, tuning):
    """Which note is being played in each frame (MIDI number, -1 = none).

    - Tuning-corrected, so a guitar tuned slightly flat doesn't drift into neighbouring notes.
    - Hysteresis: the note only changes once the pitch moves NOTE_HYSTERESIS semitones away,
      so wide vibrato stays on its note.
    - Frames where the pitch is sweeping through notes on its way somewhere (bends, slides,
      releases) are credited to the note it arrives at.
    """
    midi = tuned_midi(f0, tuning)
    return credit_sweeps(midi, hold_notes(midi, f0 > MIN_PITCH_HZ))


def tuned_midi(f0, tuning):
    """Fractional MIDI pitch per frame relative to the instrument's tuning (0 where no pitch)."""
    pitched = f0 > MIN_PITCH_HZ
    midi = np.zeros(len(f0))
    midi[pitched] = hz_to_midi(f0[pitched]) - tuning / 100
    return midi


def hold_notes(midi, pitched):
    """Round to notes with hysteresis: the note only changes once the pitch moves well away."""
    notes = np.full(len(midi), -1, dtype=int)
    current = None
    for i in range(len(midi)):
        if not pitched[i]:
            current = None
            continue
        if current is None or abs(midi[i] - current) > NOTE_HYSTERESIS:
            current = int(round(midi[i]))
        notes[i] = current
    return notes


def credit_sweeps(midi, notes):
    """Give frames that are sweeping between notes (bends, slides) to the note they land on."""
    notes = notes.copy()
    # Credit sweeps to their destination. A short run that moves steadily one way and is
    # still moving toward the next note when it ends is a transition (e.g. the D# a D->E
    # bend passes through): it goes to the next note. Runs are processed backwards so a
    # slide through several notes resolves all the way to where it lands.
    for s, e, note in reversed(runs(notes)):
        nxt = notes[e] if e < len(notes) else -1
        if note < 0 or nxt < 0 or nxt == note:
            continue
        if (e - s) * HOP_LENGTH / SAMPLE_RATE > SWEEP_MAX_SECONDS:
            continue
        lo, hi = max(0, s - 1), min(len(notes), e + 1)
        if np.any(notes[lo:hi] < 0):
            lo, hi = s, e  # don't measure motion across silence
        path = midi[lo:hi]
        if len(path) < 2:
            continue
        net = path[-1] - path[0]
        steps = np.abs(np.diff(path))
        heading = np.sign(nxt - note)
        end_motion = midi[e - 1] - midi[max(s, e - 3)]
        held = longest_run(steps < STEADY_STEP_SEMITONES) >= STEADY_MIN_FRAMES
        if (not held and abs(net) >= SWEEP_MIN_SEMITONES and abs(net) >= SWEEP_STRAIGHTNESS * steps.sum()
                and np.sign(net) == heading and np.sign(end_motion) == heading):
            notes[s:e] = nxt  # the whole run is a transition
        elif heading:
            # A held note that then bends/slides away: frames that have already left the
            # note, heading for the next one, belong to the next one.
            k = e
            while k > s and np.sign(midi[k - 1] - note) == heading and abs(midi[k - 1] - note) > SETTLED_SEMITONES:
                k -= 1
            notes[k:e] = nxt
    return notes


def longest_run(mask):
    best = count = 0
    for value in mask:
        count = count + 1 if value else 0
        best = max(best, count)
    return best


def harmonic_salience(y, sr, frames, f_hz):
    """How strongly a frequency and its first few harmonics are present over some frames."""
    half = SALIENCE_FFT // 2
    start = max(0, frames.start * HOP_LENGTH - half)
    excerpt = y[start:frames.stop * HOP_LENGTH + half]
    if len(excerpt) < SALIENCE_FFT:
        excerpt = np.pad(excerpt, (0, SALIENCE_FFT - len(excerpt)))
    band = np.abs(librosa.stft(excerpt, n_fft=SALIENCE_FFT, hop_length=HOP_LENGTH, center=False)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=SALIENCE_FFT)
    total = 0.0
    for k in range(1, SALIENCE_HARMONICS + 1):
        if k * f_hz >= freqs[-1]:
            break
        b = int(np.searchsorted(freqs, k * f_hz))
        total += band[max(0, b - 1):b + 2].max() * SALIENCE_DECAY ** (k - 1)
    return total


def fix_octave_errors(f0, notes, y=None, sr=SAMPLE_RATE):
    """Fold back short passages that sit whole octaves away from the notes on both sides.

    Pitch trackers lock onto a harmonic or sub-harmonic when the fundamental is weak, and a
    double stop a fourth apart repeats exactly two octaves below its top note.
    """
    f0, notes = f0.copy(), notes.copy()
    context = int(OCTAVE_CONTEXT_SECONDS * SAMPLE_RATE / HOP_LENGTH)
    max_gap = int(OCTAVE_NEIGHBOUR_GAP * SAMPLE_RATE / HOP_LENGTH)
    max_len = int(OCTAVE_FIX_MAX_SECONDS * SAMPLE_RATE / HOP_LENGTH)
    for _ in range(2):  # a second pass judges against the already-corrected context
        for s, e, note in runs(notes):
            if note < 0:
                continue
            around = np.concatenate((notes[max(0, s - context):s], notes[e:e + context]))
            around = around[around >= 0]
            if len(around) < MIN_NOTE_FRAMES:
                continue
            register = np.median(around)
            distance = abs(note - register)
            if distance >= SUBHARMONIC_SUSPECT_SEMITONES and (
                    e - s <= max_len or (y is not None and is_phantom_pitch(y, sr, slice(s, e), np.median(f0[s:e])))):
                # Short passages this far out of register are errors. Long ones only if nothing
                # is actually sounding at that pitch (a double stop's common sub-harmonic).
                candidates = HARMONIC_SHIFTS
            elif (distance >= OCTAVE_SUSPECT_SEMITONES and e - s <= max_len
                  and disagrees_with_neighbours(notes, s, e, max_gap)):
                candidates = (12, -12)  # a real solo can span 2 octaves: need local evidence too
            else:
                continue
            fitting = [sh for sh in candidates if abs(note + sh - register) <= OCTAVE_FIT_SEMITONES]
            if not fitting:
                continue
            if y is not None and len(fitting) > 1:
                # The real note has energy at all its harmonics; a wrong guess misses some.
                base = np.median(f0[s:e])
                shift = max(fitting, key=lambda sh: harmonic_salience(
                    y, sr, slice(s, e), base * 2.0 ** (sh / 12)))
            else:
                shift = min(fitting, key=lambda sh: abs(note + sh - register))
            notes[s:e] = note + shift
            f0[s:e] *= 2.0 ** (shift / 12)
    return f0, notes


def is_phantom_pitch(y, sr, frames, f_hz):
    """True if a reported pitch has no energy of its own (at itself or its octave).

    A double stop repeats at a common sub-harmonic of its two notes, so the tracker can report
    a pitch nobody is playing. A real low note always has energy at its fundamental or 2nd
    harmonic.
    """
    half = SALIENCE_FFT // 2
    start = max(0, frames.start * HOP_LENGTH - half)
    excerpt = y[start:frames.stop * HOP_LENGTH + half]
    if len(excerpt) < SALIENCE_FFT:
        excerpt = np.pad(excerpt, (0, SALIENCE_FFT - len(excerpt)))
    band = np.abs(librosa.stft(excerpt, n_fft=SALIENCE_FFT, hop_length=HOP_LENGTH, center=False)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=SALIENCE_FFT)

    def energy(hz):
        b = int(np.searchsorted(freqs, hz))
        return band[max(0, b - 1):b + 2].max() if b < len(band) else 0.0

    return max(energy(f_hz), energy(2 * f_hz)) < PHANTOM_MAX_RATIO * band.max()


def disagrees_with_neighbours(notes, s, e, max_gap):
    """True if the notes just before and after (within max_gap frames) are both far from notes[s]."""
    neighbours = []
    before = notes[max(0, s - max_gap):s]
    after = notes[e:e + max_gap]
    if np.any(before >= 0):
        neighbours.append(before[before >= 0][-1])
    if np.any(after >= 0):
        neighbours.append(after[after >= 0][0])
    # Both sides must disagree: a phrase that starts on a note and leaps an octave is real.
    return len(neighbours) == 2 and all(abs(notes[s] - n) >= OCTAVE_SUSPECT_SEMITONES for n in neighbours)


def remove_islands(f0, notes):
    """Drop pitched blips too short to be played notes (with nothing either side)."""
    f0, notes = f0.copy(), notes.copy()
    pitched = notes >= 0
    for s, e, on in runs(pitched):
        if on and e - s < MIN_NOTE_FRAMES:
            notes[s:e] = -1
            f0[s:e] = 0.0
    return f0, notes


def fill_gaps(f0, notes):
    """Bridge brief dropouts inside a held note."""
    f0, notes = f0.copy(), notes.copy()
    segments = runs(notes)
    for k in range(1, len(segments) - 1):
        s, e, note = segments[k]
        before, after = segments[k - 1][2], segments[k + 1][2]
        if note < 0 and e - s <= MAX_GAP_FRAMES and before == after >= 0:
            notes[s:e] = before
            f0[s:e] = np.geomspace(f0[s - 1], f0[e], e - s + 2)[1:-1]
    return f0, notes


def display_range(f0):
    """(low, high) Hz for the graph: robust to a few stray detections far from the rest."""
    pitched = f0[f0 > MIN_PITCH_HZ]
    if len(pitched) == 0:
        return 80.0, 1300.0
    low, high = np.percentile(pitched, [2, 98])
    return max(40.0, low * 0.9), max(high * 1.1, low * 2)


def analyze_pitch(y, sr, progress=None, cancelled=None):
    """Pitch contour, per-frame notes and tuning for an audio signal.

    progress(fraction, message) is called as work completes; cancelled() is polled between chunks.
    """
    report = progress or (lambda fraction, message: None)

    if USE_HPSS:
        report(0.10, "Separating guitar from drums...")
        # Aggressive harmonic/percussive separation to pull the guitar away from drums
        y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
    else:
        y_harmonic = y
    if cancelled and cancelled():
        raise AnalysisCancelled

    report(0.20, "Tracking pitch...")
    f0, voiced, confidence = track_pitch(
        y_harmonic, sr, cancelled=cancelled, progress=lambda f: report(0.20 + 0.75 * f, "Tracking pitch..."))

    report(0.96, "Cleaning up...")
    f0[~voiced] = 0.0
    raw_f0 = np.nan_to_num(f0)

    f0_smooth = medfilt(raw_f0, kernel_size=3)
    times = librosa.times_like(f0_smooth, sr=sr, hop_length=HOP_LENGTH)
    tuning = estimate_tuning(f0_smooth)
    midi = tuned_midi(f0_smooth, tuning)
    held = hold_notes(midi, f0_smooth > MIN_PITCH_HZ)
    swept = credit_sweeps(midi, held)
    f0_octave, octave_notes = fix_octave_errors(f0_smooth, swept, y_harmonic, sr)
    f0_islands, island_notes = remove_islands(f0_octave, octave_notes)
    f0_gaps, gap_notes = fill_gaps(f0_islands, island_notes)
    level_db = frame_levels(y_harmonic, len(gap_notes))
    f0_clean, notes, attack_filled = fill_attack_gaps(f0_gaps, gap_notes, level_db)
    return AnalysisResult(
        f0_clean, times, notes, tuning,
        raw_f0=raw_f0, confidence=confidence, attack_filled=attack_filled, level_db=level_db,
        held_notes=held, swept_notes=swept,
        octave_shift=np.where((octave_notes >= 0) & (swept >= 0), octave_notes - swept, 0),
        removed=(octave_notes >= 0) & (island_notes < 0),
        filled=(island_notes < 0) & (gap_notes >= 0),
        audio=y, tracked_audio=y_harmonic,
    )


# --- Result cache ----------------------------------------------------------

def cache_path(media_path):
    """Where analysis results for this exact file (path, size, modified time) are stored."""
    stat = os.stat(media_path)
    key = f"{ANALYSIS_VERSION}|{os.path.abspath(media_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    folder = os.path.join(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation),
                          "analysis")
    return os.path.join(folder, hashlib.sha1(key.encode()).hexdigest() + ".npz")


def load_cached(media_path):
    try:
        with np.load(cache_path(media_path)) as data:
            extras = {name: data[name] for name in AnalysisResult.DIAGNOSTIC_FIELDS if name in data}
            return AnalysisResult(data["f0"], data["times"], data["notes"], float(data["tuning"]), **extras)
    except (OSError, KeyError, ValueError):
        return None


def save_cached(media_path, result):
    try:
        path = cache_path(media_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        extras = {name: getattr(result, name) for name in AnalysisResult.DIAGNOSTIC_FIELDS
                  if getattr(result, name) is not None}
        np.savez_compressed(path, f0=result.f0, times=result.times, notes=result.notes,
                            tuning=result.tuning, **extras)
    except OSError:
        pass  # caching is an optimization; never fail analysis over it
