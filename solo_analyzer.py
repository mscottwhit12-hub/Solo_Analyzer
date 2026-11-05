"""Solo Analyzer: tracks the pitch of a guitar solo and maps it to scale degrees."""

import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Numba (used by librosa) caches compiled functions next to their source files,
# which don't exist inside a frozen executable. Point the cache somewhere writable.
if getattr(sys, "frozen", False):
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "solo_analyzer_numba"))

import imageio_ffmpeg
import librosa
import numpy as np
from PyQt6.QtCore import QByteArray, QLine, QRectF, QSettings, QStandardPaths, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QFontMetrics, QKeySequence, QPainter, QPen, QShortcut
from PyQt6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget,
)
from scipy.signal import medfilt

import recording

# --- Constants -------------------------------------------------------------

SAMPLE_RATE = 22050      # Hz, audio is resampled to this before analysis
HOP_LENGTH = 256         # samples between analysis frames
MIN_PITCH_HZ = 20.0      # anything at or below this is treated as "no pitch"
PITCH_FLOOR = 'C2'       # pitch tracking range: below a drop-C low string...
PITCH_CEILING = 'E6'     # ...up to the 24th fret of the high E string
FRAME_LENGTH = 1024      # samples per pitch window (46 ms): short enough for fast runs
USE_HPSS = True          # strip drums/percussive sound before tracking
ANALYSIS_VERSION = 3     # bump whenever analysis output changes, to invalidate cached results

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
MIN_NOTE_FRAMES = 4            # ~46 ms
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

# Krumhansl-Kessler key profiles (how strongly each pitch class implies a key).
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

DEGREE_COLORS = {
    "1": (255, 100, 100),
    "2": (255, 165, 0),
    "3": (100, 180, 255),
    "♭3": (100, 180, 255),
    "4": (255, 105, 180),
    "♭5": (0, 200, 200),
    "5": (180, 100, 255),
    "6": (255, 165, 0),
    "♭6": (255, 165, 0),
    "♭7": (255, 165, 0),
    "7": (255, 215, 0),
    "Out": (80, 80, 90),
}

MEDIA_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg")
SPEEDS = ["25%", "50%", "75%", "100%"]
SEEK_STEP_MS = 5000
FINE_SEEK_STEP_MS = 1000

BACKGROUND = QColor(15, 15, 20)
BORDER = QColor(40, 40, 50)
PLACEHOLDER_TEXT = QColor(100, 100, 100)
UI_FONT = "Segoe UI"


# --- Music theory ----------------------------------------------------------

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


def detect_key(notes):
    """Best-guess (root, "Major"/"Minor") from how long each pitch class is played, or None."""
    pitched = notes[notes >= 0]
    if len(pitched) == 0:
        return None
    histogram = np.bincount(pitched % 12, minlength=12)
    if np.count_nonzero(histogram) < 2:
        return None
    best = None
    for mode, profile in (("Major", MAJOR_PROFILE), ("Minor", MINOR_PROFILE)):
        for root in range(12):
            score = np.corrcoef(histogram, np.roll(profile, root))[0, 1]
            if best is None or score > best[0]:
                best = (score, NOTE_NAMES[root], mode)
    return best[1], best[2]


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
    """pYIN pitch tracking in chunks. Returns (f0 with nan where unvoiced, voiced flags)."""
    n_frames = 1 + len(y) // HOP_LENGTH
    f0 = np.full(n_frames, np.nan)
    voiced = np.zeros(n_frames, dtype=bool)
    chunk = int(CHUNK_SECONDS * sr / HOP_LENGTH)
    context = int(CHUNK_CONTEXT_SECONDS * sr / HOP_LENGTH)

    for start in range(0, n_frames, chunk):
        if cancelled and cancelled():
            raise AnalysisCancelled
        end = min(start + chunk, n_frames)
        first = max(0, start - context)
        last = min(n_frames, end + context)
        segment = y[first * HOP_LENGTH:min(len(y), last * HOP_LENGTH + HOP_LENGTH)]
        seg_f0, seg_voiced, _ = librosa.pyin(
            segment, fmin=librosa.note_to_hz(PITCH_FLOOR), fmax=librosa.note_to_hz(PITCH_CEILING),
            sr=sr, hop_length=HOP_LENGTH, frame_length=FRAME_LENGTH,
        )
        keep = slice(start - first, end - first)
        f0[start:end] = seg_f0[keep]
        voiced[start:end] = seg_voiced[keep]
        if progress:
            progress(end / n_frames)
    return f0, voiced


@dataclass
class AnalysisResult:
    f0: np.ndarray      # pitch contour for the graph, Hz (0 = no pitch)
    times: np.ndarray   # seconds, one per frame
    notes: np.ndarray   # note being played per frame (MIDI number), -1 = none
    tuning: float       # how far the instrument is from A440, in cents


def estimate_tuning(f0):
    """Cents offset of the playing from A440, from where pitches sit between semitones."""
    pitched = f0[f0 > MIN_PITCH_HZ]
    if len(pitched) < 10:
        return 0.0
    angle = 2 * np.pi * hz_to_midi(pitched)  # semitone position as a point on a circle
    return float(np.angle(np.mean(np.exp(1j * angle))) / (2 * np.pi) * 100)


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
    notes = np.full(len(f0), -1, dtype=int)
    pitched = f0 > MIN_PITCH_HZ
    midi = np.zeros(len(f0))
    midi[pitched] = hz_to_midi(f0[pitched]) - tuning / 100

    current = None
    for i in range(len(f0)):
        if not pitched[i]:
            current = None
            continue
        if current is None or abs(midi[i] - current) > NOTE_HYSTERESIS:
            current = int(round(midi[i]))
        notes[i] = current

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
            if note < 0 or e - s > max_len:
                continue
            around = np.concatenate((notes[max(0, s - context):s], notes[e:e + context]))
            around = around[around >= 0]
            if len(around) < MIN_NOTE_FRAMES:
                continue
            register = np.median(around)
            distance = abs(note - register)
            if distance >= SUBHARMONIC_SUSPECT_SEMITONES:
                candidates = HARMONIC_SHIFTS
            elif distance >= OCTAVE_SUSPECT_SEMITONES and disagrees_with_neighbours(notes, s, e, max_gap):
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
    f0, voiced = track_pitch(y_harmonic, sr, cancelled=cancelled,
                             progress=lambda f: report(0.20 + 0.75 * f, "Tracking pitch..."))

    report(0.96, "Cleaning up...")
    f0[~voiced] = 0.0
    f0 = np.nan_to_num(f0)

    f0_smooth = medfilt(f0, kernel_size=3)
    times = librosa.times_like(f0_smooth, sr=sr, hop_length=HOP_LENGTH)
    tuning = estimate_tuning(f0_smooth)
    notes = quantize_notes(f0_smooth, tuning)
    f0_clean, notes = fix_octave_errors(f0_smooth, notes, y_harmonic, sr)
    f0_clean, notes = remove_islands(f0_clean, notes)
    f0_clean, notes = fill_gaps(f0_clean, notes)
    return AnalysisResult(f0_clean, times, notes, tuning)


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
            return AnalysisResult(data["f0"], data["times"], data["notes"], float(data["tuning"]))
    except (OSError, KeyError, ValueError):
        return None


def save_cached(media_path, result):
    try:
        path = cache_path(media_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, f0=result.f0, times=result.times, notes=result.notes, tuning=result.tuning)
    except OSError:
        pass  # caching is an optimization; never fail analysis over it


class AudioAnalyzerThread(QThread):
    """Extracts the audio track and runs pitch detection off the GUI thread."""
    analysis_done = pyqtSignal(object)  # AnalysisResult
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, media_path):
        super().__init__()
        self.media_path = media_path
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            if not os.path.exists(self.media_path):
                self.error.emit("Selected file does not exist.")
                return

            cached = load_cached(self.media_path)
            if cached is not None:
                self.analysis_done.emit(cached)
                return

            self.progress.emit(0, "Extracting audio...")
            y = extract_audio(self.media_path)
            if len(y) == 0:
                self.error.emit("The file contains no audio track to analyze.")
                return

            result = analyze_pitch(
                y, SAMPLE_RATE,
                progress=lambda fraction, message: self.progress.emit(int(fraction * 100), message),
                cancelled=lambda: self.cancel_requested,
            )
            save_cached(self.media_path, result)
            self.analysis_done.emit(result)
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(f"An unexpected error occurred:\n{e}")


# --- Widgets ---------------------------------------------------------------

def degree_color(degree, fallback, alpha=255):
    """Color for a scale degree label, or `fallback` if the label has no assigned color."""
    rgb = DEGREE_COLORS.get(degree)
    return QColor(*rgb, alpha) if rgb else fallback


def paint_background(painter, width, height):
    painter.fillRect(0, 0, width, height, BACKGROUND)
    painter.setPen(BORDER)
    painter.drawRect(0, 0, width - 1, height - 1)


class ScaleBreakdownWidget(QWidget):
    """Bar showing the share of pitched frames spent on each scale degree."""

    BAR_TOP, BAR_HEIGHT, LINE_HEIGHT, MARGIN, LABEL_GAP = 12, 16, 18, 10, 15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.percentages = {}
        self.font = QFont(UI_FONT, 9)
        self.setFixedHeight(50)

    def update_breakdown(self, notes, root, scale):
        self.percentages = degree_breakdown(notes, root, scale)
        self._fit_height()
        self.update()

    def _ranked(self):
        return sorted(self.percentages.items(), key=lambda x: x[1], reverse=True)

    def _legend_layout(self, width):
        """[(text, degree, x, row)] with labels wrapped onto new rows when they don't fit."""
        metrics = QFontMetrics(self.font)
        items, x, row = [], self.MARGIN, 0
        for deg, pct in self._ranked():
            text = f"{deg}: {pct:.1f}%"
            text_w = metrics.horizontalAdvance(text)
            if x > self.MARGIN and x + text_w > width - self.MARGIN:
                x, row = self.MARGIN, row + 1
            items.append((text, deg, x, row))
            x += text_w + self.LABEL_GAP
        return items

    def _fit_height(self):
        rows = 1 + max((row for *_, row in self._legend_layout(self.width())), default=0)
        self.setFixedHeight(self.BAR_TOP + self.BAR_HEIGHT + rows * self.LINE_HEIGHT + 8)

    def resizeEvent(self, event):
        self._fit_height()
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        paint_background(painter, width, height)

        if not self.percentages:
            painter.setPen(PLACEHOLDER_TEXT)
            painter.setFont(QFont(UI_FONT, 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Scale percentage breakdown will appear here")
            return

        fallback = QColor(150, 150, 150)
        bar_w = width - 2 * self.MARGIN
        current_x = self.MARGIN
        painter.setPen(Qt.PenStyle.NoPen)
        for deg, pct in self._ranked():
            segment_w = max(4, int(pct / 100.0 * bar_w))
            painter.setBrush(degree_color(deg, fallback))
            painter.drawRoundedRect(QRectF(current_x, self.BAR_TOP, segment_w, self.BAR_HEIGHT), 3, 3)
            current_x += segment_w

        painter.setFont(self.font)
        baseline = self.BAR_TOP + self.BAR_HEIGHT + 16
        for text, deg, x, row in self._legend_layout(width):
            painter.setPen(degree_color(deg, fallback))
            painter.drawText(x, baseline + row * self.LINE_HEIGHT, text)


class PitchGraphWidget(QWidget):
    """Scrolling pitch contour drawn against the selected scale's degree lines. Click to seek."""
    seek_requested = pyqtSignal(float)  # seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.times = None
        self.f0 = None
        self.current_time = 0.0
        self.window_duration = 30.0  # seconds visible at once
        self.f_min = 65.0
        self.f_max = 1300.0
        self.root_note = "A"
        self.scale_type = DEFAULT_SCALE
        self.setMinimumHeight(200)

    def set_data(self, times, f0):
        self.times = times
        self.f0 = f0
        self.f_min, self.f_max = display_range(f0)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.update()

    def set_scale_info(self, root_note, scale_type):
        self.root_note = root_note
        self.scale_type = scale_type
        self.update()

    def set_time(self, current_time):
        self.current_time = current_time
        self.update()

    def _visible_start(self):
        return max(0.0, self.current_time - self.window_duration)

    def mousePressEvent(self, event):
        if self.times is None or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        fraction = event.position().x() / max(1, self.width())
        self.seek_requested.emit(self._visible_start() + fraction * self.window_duration)

    def get_reference_frequencies(self):
        """(frequency, degree label) for every scale note inside the visible range."""
        if self.root_note not in NOTE_NAMES:
            return []
        root_idx = NOTE_NAMES.index(self.root_note)
        intervals = get_intervals(self.scale_type)
        lines = []
        for midi in range(12, 109):
            semitone_diff = (midi % 12 - root_idx) % 12
            if semitone_diff in intervals:
                freq = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
                if self.f_min <= freq <= self.f_max:
                    lines.append((freq, intervals[semitone_diff]))
        return lines

    def _freq_to_y(self, freq, height, log_min, log_max):
        norm_f = np.clip((np.log2(freq) - log_min) / (log_max - log_min), 0.0, 1.0)
        return (height - norm_f * height).astype(int) if isinstance(norm_f, np.ndarray) \
            else int(height - norm_f * height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        paint_background(painter, width, height)

        log_min, log_max = np.log2(self.f_min), np.log2(self.f_max)

        # Scale degree guide lines
        fallback = QColor(200, 200, 200, 180)
        painter.setFont(QFont(UI_FONT, 9))
        for freq, label in self.get_reference_frequencies():
            y = self._freq_to_y(freq, height, log_min, log_max)
            color = degree_color(label, fallback, alpha=180)
            painter.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
            painter.drawLine(0, y, width, y)
            painter.setPen(color)
            painter.drawText(5, y - 3, label)

        if self.times is None or self.f0 is None:
            painter.setPen(PLACEHOLDER_TEXT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Open, drop or record a file to see the pitch graph")
            return

        t_start = self._visible_start()
        mask = (self.times >= t_start) & (self.times <= self.current_time)
        if not np.any(mask):
            return

        sub_times = self.times[mask]
        sub_f0 = self.f0[mask]
        pitched = sub_f0 > MIN_PITCH_HZ

        xs = ((sub_times - t_start) / self.window_duration * width).astype(int)
        ys = np.zeros(len(sub_f0), dtype=int)
        if np.any(pitched):
            ys[pitched] = self._freq_to_y(sub_f0[pitched], height, log_min, log_max)

        # Pitch contour: connect consecutive frames that both have a pitch
        joined = np.flatnonzero(pitched[:-1] & pitched[1:])
        painter.setPen(QPen(QColor(0, 255, 128), 2))
        painter.drawLines([QLine(xs[i], ys[i], xs[i + 1], ys[i + 1]) for i in joined])

        # Playhead
        painter.setPen(QPen(QColor(255, 100, 100), 1, Qt.PenStyle.DashLine))
        painter.drawLine(width - 1, 0, width - 1, height)


# --- Main window -----------------------------------------------------------

def is_media_file(path):
    return path.lower().endswith(MEDIA_EXTENSIONS)


class GuitarTrackerApp(QMainWindow):
    """Main window: controls, readout, breakdown bar, pitch graph and video."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solo Analyzer")
        self.setAcceptDrops(True)
        self.settings = QSettings()

        self.video_path = ""
        self.result = None  # AnalysisResult of the loaded file
        self.analyzer = None
        self.is_slider_dragging = False

        self.init_ui()
        self.init_shortcuts()
        self.restore_settings()

    # -- setup --

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Row 1: file and playback
        file_row = QHBoxLayout()
        self.file_btn = QPushButton("Open File")
        self.file_btn.clicked.connect(self.select_file)
        file_row.addWidget(self.file_btn)

        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.open_recorder)
        file_row.addWidget(self.record_btn)

        self.file_label = QLabel("No file selected. Open, record, or drop a file here.")
        file_row.addWidget(self.file_label, 1)

        file_row.addWidget(QLabel("Speed:"))
        self.speed_input = QComboBox()
        self.speed_input.addItems(SPEEDS)
        self.speed_input.setCurrentText("100%")
        self.speed_input.setToolTip("Slow down playback without changing pitch")
        self.speed_input.currentTextChanged.connect(self.on_speed_changed)
        file_row.addWidget(self.speed_input)

        self.play_btn = QPushButton("Play / Pause")
        self.play_btn.setToolTip("Space: play/pause    ←/→: skip 5 s    Shift+←/→: skip 1 s")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        file_row.addWidget(self.play_btn)
        main_layout.addLayout(file_row)

        # Row 2: key
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("Root:"))
        self.root_input = QComboBox()
        self.root_input.addItems(NOTE_NAMES)
        self.root_input.setCurrentText("A")
        self.root_input.currentTextChanged.connect(self.on_scale_changed)
        key_row.addWidget(self.root_input)

        key_row.addWidget(QLabel("Scale:"))
        self.scale_input = QComboBox()
        self.scale_input.addItems(SCALES)
        self.scale_input.currentTextChanged.connect(self.on_scale_changed)
        key_row.addWidget(self.scale_input)

        self.auto_key_input = QCheckBox("Auto-detect key")
        self.auto_key_input.setChecked(True)
        self.auto_key_input.setToolTip("Set the root and scale from the notes played when a file finishes analyzing")
        key_row.addWidget(self.auto_key_input)

        self.detected_label = QLabel("")
        self.detected_label.setStyleSheet("color: #888;")
        key_row.addWidget(self.detected_label)
        key_row.addStretch(1)
        main_layout.addLayout(key_row)

        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setEnabled(False)
        self.scrubber.sliderPressed.connect(self.on_slider_pressed)
        self.scrubber.sliderReleased.connect(self.on_slider_released)
        self.scrubber.sliderMoved.connect(self.on_slider_moved)
        main_layout.addWidget(self.scrubber)

        self.info_panel = QLabel("Note: --   |   Scale Degree: --")
        self.info_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_panel.setStyleSheet(
            "background-color: #111111; color: #00FF00; font-size: 32px; "
            "font-weight: bold; border: 2px solid #333333; border-radius: 8px; padding: 12px;"
        )
        main_layout.addWidget(self.info_panel)

        # Analysis progress (hidden when idle)
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_row.addWidget(self.progress_bar, 1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        progress_row.addWidget(self.cancel_btn)
        self.progress_widget = QWidget()
        self.progress_widget.setLayout(progress_row)
        self.progress_widget.setVisible(False)
        main_layout.addWidget(self.progress_widget)

        self.breakdown_widget = ScaleBreakdownWidget()
        main_layout.addWidget(self.breakdown_widget)

        self.pitch_graph = PitchGraphWidget()
        self.pitch_graph.seek_requested.connect(lambda s: self.seek_to(int(s * 1000)))
        main_layout.addWidget(self.pitch_graph, 3)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(120)
        main_layout.addWidget(self.video_widget, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Keyboard shortcuts belong to the window, so keep controls from grabbing focus.
        for widget in (self.file_btn, self.record_btn, self.play_btn, self.cancel_btn, self.speed_input,
                       self.root_input, self.scale_input, self.auto_key_input, self.scrubber):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        if hasattr(self.media_player, "setPitchCompensation"):
            self.media_player.setPitchCompensation(True)

        # Qt locks playback to whatever the default output was at startup. Follow the
        # system default instead, so e.g. Bluetooth headphones connected later are used.
        self.media_devices = QMediaDevices(self)
        self.media_devices.audioOutputsChanged.connect(self.follow_default_output)

    def init_shortcuts(self):
        bindings = [
            ("Space", self.toggle_playback),
            ("Left", lambda: self.seek_by(-SEEK_STEP_MS)),
            ("Right", lambda: self.seek_by(SEEK_STEP_MS)),
            ("Shift+Left", lambda: self.seek_by(-FINE_SEEK_STEP_MS)),
            ("Shift+Right", lambda: self.seek_by(FINE_SEEK_STEP_MS)),
            ("Home", lambda: self.seek_to(0)),
        ]
        for key, action in bindings:
            QShortcut(QKeySequence(key), self, activated=action)

    def restore_settings(self):
        geometry = self.settings.value("geometry")
        if isinstance(geometry, QByteArray) and self.restoreGeometry(geometry):
            pass
        else:
            available = QApplication.primaryScreen().availableGeometry()
            self.resize(min(1100, int(available.width() * 0.9)), min(1000, int(available.height() * 0.9)))

        for key, combo in (("root", self.root_input), ("scale", self.scale_input), ("speed", self.speed_input)):
            value = self.settings.value(key)
            if value and combo.findText(value) >= 0:
                combo.setCurrentText(value)
        self.auto_key_input.setChecked(self.settings.value("auto_key", True, type=bool))
        self.on_scale_changed()

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("root", self.root_input.currentText())
        self.settings.setValue("scale", self.scale_input.currentText())
        self.settings.setValue("speed", self.speed_input.currentText())
        self.settings.setValue("auto_key", self.auto_key_input.isChecked())

    # -- loading files --

    def _set_controls_enabled(self, file_btn, playback):
        self.file_btn.setEnabled(file_btn)
        self.record_btn.setEnabled(file_btn)
        self.play_btn.setEnabled(playback)
        self.scrubber.setEnabled(playback)

    def is_busy(self):
        return self.analyzer is not None and self.analyzer.isRunning()

    def select_file(self):
        start_dir = self.settings.value("last_dir", "")
        patterns = " ".join(f"*{ext}" for ext in MEDIA_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Open Video or Audio File", start_dir,
                                              f"Media Files ({patterns})")
        if path:
            self.load_file(path)

    def open_recorder(self):
        if recording.AUDIO_IMPORT_ERROR:
            QMessageBox.critical(self, "Record", f"Audio recording is unavailable:\n{recording.AUDIO_IMPORT_ERROR}")
            return
        self.media_player.pause()
        dialog = recording.RecordDialog(self)
        if dialog.exec() and dialog.output_path:
            self.load_file(dialog.output_path)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if not self.is_busy() and any(u.isLocalFile() and is_media_file(u.toLocalFile()) for u in urls):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and is_media_file(url.toLocalFile()):
                self.load_file(url.toLocalFile())
                break

    def load_file(self, path):
        """Start playback and background pitch analysis for a video or audio file."""
        if self.is_busy():
            return
        self.video_path = path
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.file_label.setText(os.path.basename(path))
        self.info_panel.setText("Analyzing audio...")
        self.detected_label.setText("")
        self._set_controls_enabled(file_btn=False, playback=False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.progress_widget.setVisible(True)

        self.media_player.setSource(QUrl.fromLocalFile(path))

        self.analyzer = AudioAnalyzerThread(path)
        self.analyzer.analysis_done.connect(self.on_analysis_finished)
        self.analyzer.progress.connect(self.on_analysis_progress)
        self.analyzer.error.connect(self.on_analysis_error)
        self.analyzer.cancelled.connect(self.on_analysis_cancelled)
        self.analyzer.start()

    def on_analysis_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{message}  %p%")

    def cancel_analysis(self):
        if self.analyzer:
            self.analyzer.cancel()
            self.cancel_btn.setEnabled(False)
            self.progress_bar.setFormat("Cancelling...")

    def _analysis_ended(self):
        self.progress_widget.setVisible(False)
        self.cancel_btn.setEnabled(True)

    def on_analysis_finished(self, result):
        self._analysis_ended()
        self.result = result
        self._set_controls_enabled(file_btn=True, playback=True)
        self.pitch_graph.set_data(result.times, result.f0)

        details = [f"Tuning: {result.tuning:+.0f}¢ from A440"]
        key = detect_key(result.notes)
        if key:
            root, mode = key
            details.insert(0, f"Detected key: {root} {mode.lower()}")
            if self.auto_key_input.isChecked():
                for combo, value in ((self.root_input, root), (self.scale_input, mode)):
                    combo.blockSignals(True)
                    combo.setCurrentText(value)
                    combo.blockSignals(False)
        self.detected_label.setText("   ·   ".join(details))
        self.detected_label.setToolTip("Notes are judged relative to the detected tuning, "
                                       "and cents in the readout are measured from it.")
        self.on_scale_changed()
        self.info_panel.setText("Note: --   |   Scale Degree: --")

    def on_analysis_error(self, err_msg):
        self._analysis_ended()
        QMessageBox.critical(self, "Error", err_msg)
        self.info_panel.setText("Load a file to begin")
        self._set_controls_enabled(file_btn=True, playback=False)

    def on_analysis_cancelled(self):
        self._analysis_ended()
        self.media_player.setSource(QUrl())
        self.file_label.setText("Analysis cancelled")
        self.info_panel.setText("Load a file to begin")
        self._set_controls_enabled(file_btn=True, playback=False)

    # -- key / scale / speed --

    def on_scale_changed(self):
        root = self.root_input.currentText()
        scale = self.scale_input.currentText()
        self.pitch_graph.set_scale_info(root, scale)
        self.breakdown_widget.update_breakdown(self.result.notes if self.result else None, root, scale)
        self.position_changed(self.media_player.position())

    def on_speed_changed(self, text):
        self.media_player.setPlaybackRate(int(text.rstrip("%")) / 100.0)

    # -- playback --

    def toggle_playback(self):
        if not self.play_btn.isEnabled():
            return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.follow_default_output()  # catch default-device changes Qt didn't announce
            self.media_player.play()

    def follow_default_output(self):
        default = QMediaDevices.defaultAudioOutput()
        if not default.isNull() and default.id() != self.audio_output.device().id():
            self.audio_output.setDevice(default)

    def seek_to(self, ms):
        if not self.play_btn.isEnabled():
            return
        duration = self.media_player.duration()
        self.media_player.setPosition(int(max(0, min(ms, duration if duration > 0 else ms))))

    def seek_by(self, delta_ms):
        self.seek_to(self.media_player.position() + delta_ms)

    def position_changed(self, position):
        """Keep the scrubber, graph playhead and note readout in sync with playback."""
        if not self.is_slider_dragging:
            self.scrubber.setValue(position)

        if self.result is None:
            return

        current_time = position / 1000.0
        self.pitch_graph.set_time(current_time)

        r = self.result
        idx = np.clip(np.searchsorted(r.times, current_time, side="left"), 0, len(r.times) - 1)
        note = note_name(r.notes[idx])
        cents = cents_off(r.f0[idx], r.notes[idx], r.tuning)
        degree = degree_of(note, self.root_input.currentText(), self.scale_input.currentText())
        note_text = f"{note} {cents:+d}¢" if note else "--"
        self.info_panel.setText(f"Note: {note_text}   |   Scale Degree: {degree}")

    def duration_changed(self, duration):
        self.scrubber.setRange(0, duration)

    def on_slider_pressed(self):
        self.is_slider_dragging = True

    def on_slider_released(self):
        self.is_slider_dragging = False
        self.media_player.setPosition(self.scrubber.value())

    def on_slider_moved(self, position):
        if self.is_slider_dragging:
            self.media_player.setPosition(position)

    def closeEvent(self, event):
        self.save_settings()
        if self.is_busy():
            self.analyzer.cancel()
            self.analyzer.wait()  # stops at the next chunk; destroying a running QThread crashes
        self.media_player.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Solo Analyzer")
    app.setOrganizationName("Solo Analyzer")
    window = GuitarTrackerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
