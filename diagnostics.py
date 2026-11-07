"""Diagnostics tab: see the audio the way the pitch tracker does, and why each note was chosen.

- Spectrogram: frequency (log, note-labelled) over time, with the tracker's raw pitch, the
  final notes, a lane marking every correction, and the tracker's confidence.
- Spectrum at the playhead: the frequency content of that instant, with the chosen note's
  harmonics marked.
- Explanation: the decision chain for the frame under the playhead, step by step.
"""

import numpy as np
import librosa
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, qRgb
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QSizePolicy, QToolTip, QVBoxLayout, QWidget,
)

import analysis as an
import separation
from ui_common import BACKGROUND, BORDER, PLACEHOLDER_TEXT, UI_FONT, paint_background

# Spectrogram geometry: rows are log-spaced, ROWS_PER_SEMITONE per semitone from C2 to C8
SPEC_LOW_MIDI = 36
SPEC_HIGH_MIDI = 108
ROWS_PER_SEMITONE = 4
SPEC_FFT = 4096
SPEC_DYNAMIC_RANGE_DB = 80.0
SPEC_CHUNK_FRAMES = 2000
SPECTRUM_FFT = 4096
WINDOW_SECONDS = 8.0          # spectrogram width; the playhead sits in the middle
SPECTRUM_HARMONICS = 8

SIGNAL_TRACKED = "What the tracker hears (bass range cut)"
SIGNAL_TRACKED_ISOLATED = "What the tracker hears (isolated guitar)"
SIGNAL_ORIGINAL = "Original audio"

# Decision-lane colours
COLOR_OCTAVE = QColor(255, 150, 40)
COLOR_SWEEP = QColor(60, 200, 230)
COLOR_HOLD = QColor(150, 150, 160)
COLOR_REMOVED = QColor(255, 70, 70)
COLOR_FILLED = QColor(190, 110, 255)
COLOR_TRACKER = QColor(255, 255, 255, 200)
COLOR_NOTE = QColor(0, 255, 128)

LEGEND = [
    ("Tracker pitch", COLOR_TRACKER), ("Final note", COLOR_NOTE), ("Octave fix", COLOR_OCTAVE),
    ("Bend/slide credited to target", COLOR_SWEEP), ("Held by hysteresis", COLOR_HOLD),
    ("Blip removed", COLOR_REMOVED), ("Gap filled (dropout, attack, or read from spectrum)", COLOR_FILLED),
]


def _magma():
    anchors = [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66),
               (245, 125, 21), (250, 193, 39), (252, 255, 164)]
    table = []
    for i in range(256):
        pos = i / 255 * (len(anchors) - 1)
        a, frac = int(pos), pos - int(pos)
        b = min(a + 1, len(anchors) - 1)
        table.append(qRgb(*(int(anchors[a][c] + (anchors[b][c] - anchors[a][c]) * frac) for c in range(3))))
    return table


COLOR_TABLE = _magma()


# --- Computation -----------------------------------------------------------

def spectrogram_rows():
    """Centre frequency (Hz) of each spectrogram row, bottom to top."""
    count = (SPEC_HIGH_MIDI - SPEC_LOW_MIDI) * ROWS_PER_SEMITONE + 1
    return an.midi_to_hz(SPEC_LOW_MIDI + np.arange(count) / ROWS_PER_SEMITONE)


def log_spectrogram(y, sr=an.SAMPLE_RATE, cancelled=None):
    """uint8 image (frames x rows), one column per analysis frame, log-frequency rows.

    Rows wide enough to contain FFT bins take the loudest bin (so harmonics stay crisp);
    narrower rows interpolate between neighbouring bins.
    """
    n_frames = 1 + len(y) // an.HOP_LENGTH
    rows = spectrogram_rows()
    bins = librosa.fft_frequencies(sr=sr, n_fft=SPEC_FFT)
    half_row = 2 ** (0.5 / ROWS_PER_SEMITONE / 12)
    lo = np.searchsorted(bins, rows / half_row)
    hi = np.searchsorted(bins, rows * half_row)
    pooled = hi > lo
    pool_idx = np.ravel(np.column_stack((lo[pooled], hi[pooled])))
    interp_pos = np.interp(rows[~pooled], bins, np.arange(len(bins)))
    i0 = np.clip(np.floor(interp_pos).astype(int), 0, len(bins) - 2)
    w = interp_pos - i0

    padded = np.pad(y, SPEC_FFT // 2)
    db = np.empty((n_frames, len(rows)), dtype=np.float32)
    for start in range(0, n_frames, SPEC_CHUNK_FRAMES):
        if cancelled and cancelled():
            raise an.AnalysisCancelled
        end = min(n_frames, start + SPEC_CHUNK_FRAMES)
        segment = padded[start * an.HOP_LENGTH:(end - 1) * an.HOP_LENGTH + SPEC_FFT]
        if len(segment) < SPEC_FFT + (end - start - 1) * an.HOP_LENGTH:
            segment = np.pad(segment, (0, SPEC_FFT + (end - start - 1) * an.HOP_LENGTH - len(segment)))
        mag = np.abs(librosa.stft(segment, n_fft=SPEC_FFT, hop_length=an.HOP_LENGTH, center=False))
        mag = mag[:, :end - start]
        out = np.empty((len(rows), mag.shape[1]), dtype=np.float32)
        if pooled.any():
            out[pooled] = np.maximum.reduceat(mag, pool_idx, axis=0)[::2]
        out[~pooled] = mag[i0] * (1 - w)[:, None] + mag[i0 + 1] * w[:, None]
        db[start:end] = 20 * np.log10(out.T + 1e-10)
    top = db.max() if db.size else 0.0
    scaled = (db - (top - SPEC_DYNAMIC_RANGE_DB)) / SPEC_DYNAMIC_RANGE_DB * 255
    return np.clip(scaled, 0, 255).astype(np.uint8)


def spectrum_at(y, frame, sr=an.SAMPLE_RATE):
    """(frequencies, dB relative to the loudest bin) of a Hann window centred on a frame."""
    centre = frame * an.HOP_LENGTH
    half = SPECTRUM_FFT // 2
    excerpt = y[max(0, centre - half):centre + half]
    if len(excerpt) < SPECTRUM_FFT:
        excerpt = np.pad(excerpt, (0, SPECTRUM_FFT - len(excerpt)))
    mag = np.abs(np.fft.rfft(excerpt * np.hanning(SPECTRUM_FFT)))
    db = 20 * np.log10(mag + 1e-10)
    return np.fft.rfftfreq(SPECTRUM_FFT, 1 / sr), db - db.max()


def note_label(midi):
    """'A4' style name for a MIDI number."""
    return f"{an.NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


SHIFT_DESCRIPTIONS = {
    12: "up one octave (the tracker had locked onto half the real frequency)",
    -12: "down one octave (the tracker had locked onto the 2nd harmonic)",
    24: "up two octaves (a quarter of the real frequency, typical of a double stop a fourth apart)",
    -24: "down two octaves",
    19: "up an octave and a fifth (the tracker had locked onto a third of the real frequency)",
    -19: "down an octave and a fifth (the tracker had locked onto the 3rd harmonic)",
    28: "up two octaves and a third (a fifth of the real frequency)",
}


def explain_frame(result, i, root, scale):
    """Plain-English steps explaining how frame i ended up with its note. Returns a list of strings."""
    steps = []
    raw = result.raw_f0[i] if result.raw_f0 is not None else result.f0[i]
    conf = result.confidence[i] if result.confidence is not None else None
    conf_text = f", confidence {conf:.0%}" if conf is not None else ""
    if raw > an.MIN_PITCH_HZ:
        midi = an.hz_to_midi(raw)
        nearest = int(round(midi))
        steps.append(f"<b>Tracker</b> heard {raw:.1f} Hz: {note_label(nearest)} "
                     f"{(midi - nearest) * 100:+.0f}¢{conf_text}.")
    else:
        steps.append(f"<b>Tracker</b> found no clear pitch here{conf_text}.")

    if result.held_notes is None:
        steps.append("(Detailed decision data isn't available for this file. Re-analyze it to see it.)")
    else:
        held, swept = int(result.held_notes[i]), int(result.swept_notes[i])
        if held >= 0:
            smooth = an.midi_to_hz(held) if raw <= an.MIN_PITCH_HZ else raw
            tuned = an.hz_to_midi(smooth) - result.tuning / 100
            steps.append(f"<b>Tuning</b>: the guitar is {result.tuning:+.0f}¢ from A440, so relative to "
                         f"it this is {tuned:.2f} (MIDI).")
            if int(round(tuned)) != held:
                steps.append(f"<b>Hysteresis</b> kept it on {note_label(held)}: it's "
                             f"{tuned - held:+.2f} semitones away, inside the ±{an.NOTE_HYSTERESIS} band, "
                             "so vibrato doesn't flip notes.")
            else:
                steps.append(f"<b>Nearest note</b>: {note_label(held)}.")
        if swept != held and swept >= 0:
            steps.append(f"<b>Bend/slide</b> in progress, heading for {note_label(swept)}: "
                         f"counted as {note_label(swept)}, where it lands.")
        shift = int(result.octave_shift[i]) if result.octave_shift is not None else 0
        if shift:
            desc = SHIFT_DESCRIPTIONS.get(shift, f"by {shift:+d} semitones")
            steps.append(f"<b>Octave fix</b>: moved {desc}. This passage sat far from the notes around "
                         f"it, and the spectrum's harmonics fit {note_label(swept + shift)} best.")
        if result.removed is not None and result.removed[i]:
            steps.append("<b>Removed</b>: a blip shorter than 45 ms, too short to be a played note.")
        if result.filled is not None and result.filled[i]:
            steps.append(f"<b>Filled</b>: a brief dropout inside a held {note_label(result.notes[i])}.")
        if result.estimated is not None and result.estimated[i]:
            steps.append(f"<b>Read from the spectrum</b>: the tracker gave up here (two strings at once often "
                         f"do that) but the guitar was loud, so the note is the one whose harmonics are "
                         f"strongest near the surrounding notes: {note_label(result.notes[i])}. If two notes "
                         f"were equally strong (a double stop), the higher one counts.")
        if result.attack_filled is not None and result.attack_filled[i]:
            steps.append(f"<b>Attack</b>: the tracker hadn't locked on yet (it takes ~0.1 s after a new note, "
                         f"longer for a double stop), but the guitar was clearly sounding, so this counts as "
                         f"the note that follows, {note_label(result.notes[i])}.")

    final = int(result.notes[i])
    if final >= 0:
        degree = an.degree_of(an.note_name(final), root, scale)
        steps.append(f"<b>Result</b>: {note_label(final)}, scale degree <b>{degree}</b> in {root} {scale.lower()}.")
    else:
        steps.append("<b>Result</b>: no note.")
    return steps


class DiagnosticsThread(QThread):
    """Prepares spectrograms (and re-reads the audio if the analysis came from the cache)."""
    ready = pyqtSignal(object)   # {"tracked": image, "original": image}
    progress = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, result, media_path):
        super().__init__()
        self.result = result
        self.media_path = media_path
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        stop = lambda: self.cancel_requested  # noqa: E731
        try:
            r = self.result
            if r.audio is None:
                self.progress.emit("Reading audio...")
                r.audio = an.extract_audio(self.media_path)
                r.tracked_audio = None
            if r.tracked_audio is None:
                self.progress.emit("Preparing audio...")
                source = r.audio
                if r.isolated:  # the separated guitar is cached; re-separate only if it was cleared
                    guitar = separation.isolate_file(self.media_path, cancelled=stop)
                    source = np.zeros(len(r.audio), dtype=np.float32)
                    source[:min(len(r.audio), len(guitar))] = guitar[:len(r.audio)]
                r.tracked_audio = an.tracking_signal(source, band_mode=r.band_mode)
            self.progress.emit("Building spectrogram...")
            tracked = log_spectrogram(r.tracked_audio, cancelled=stop)
            original = tracked if r.tracked_audio is r.audio else log_spectrogram(r.audio, cancelled=stop)
            self.ready.emit({"tracked": tracked, "original": original})
        except (an.AnalysisCancelled, separation.SeparationCancelled):
            pass
        except Exception as e:
            self.failed.emit(str(e))


# --- Widgets ---------------------------------------------------------------

def _runs(values):
    return an.runs(values)


class SpectrogramWidget(QWidget):
    seek_requested = pyqtSignal(float)

    LANE_H = 14
    CONF_H = 34
    LEFT = 44

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.image = None
        self.time = 0.0
        self.show_tracker = True
        self.show_notes = True
        self.message = "Analyze a file, then open this tab"
        self.rows = spectrogram_rows()
        self.setMinimumHeight(260)
        self.setMouseTracking(True)

    def set_data(self, result, image):
        self.result, self.image = result, image
        self.update()

    def set_time(self, t):
        self.time = t
        self.update()

    def _window(self):
        start = self.time - WINDOW_SECONDS / 2
        return start, start + WINDOW_SECONDS

    def _plot_rect(self):
        return QRectF(self.LEFT, 4, self.width() - self.LEFT - 4,
                      self.height() - 8 - self.LANE_H - self.CONF_H - 8)

    def _y_for_midi(self, midi, rect):
        frac = (np.asarray(midi, float) - SPEC_LOW_MIDI) / (SPEC_HIGH_MIDI - SPEC_LOW_MIDI)
        return rect.bottom() - frac * rect.height()

    def _x_for_time(self, t, rect):
        start, _ = self._window()
        return rect.left() + (np.asarray(t, float) - start) / WINDOW_SECONDS * rect.width()

    def paintEvent(self, event):
        p = QPainter(self)
        paint_background(p, self.width(), self.height())
        rect = self._plot_rect()
        p.setFont(QFont(UI_FONT, 8))

        if self.result is None or self.image is None:
            p.setPen(PLACEHOLDER_TEXT)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return

        times = self.result.times
        start, end = self._window()
        f0 = max(0, int(np.searchsorted(times, start)))
        f1 = min(len(times), int(np.searchsorted(times, end)) + 1)

        # Spectrogram image for the visible frames (time -> x, rows -> y, high notes on top)
        if f1 > f0:
            block = np.ascontiguousarray(self.image[f0:f1].T[::-1])
            self._buf = block.tobytes()
            img = QImage(self._buf, block.shape[1], block.shape[0], block.shape[1], QImage.Format.Format_Indexed8)
            img.setColorTable(COLOR_TABLE)
            x0 = float(self._x_for_time(times[f0], rect))
            x1 = float(self._x_for_time(times[f1 - 1], rect))
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(QRectF(x0, rect.top(), max(1.0, x1 - x0), rect.height()), img)

        # Note grid: every C labelled, plus A4 (440 Hz) for reference
        for midi in range(SPEC_LOW_MIDI, SPEC_HIGH_MIDI + 1):
            if midi % 12 != 0 and midi != 69:
                continue
            y = float(self._y_for_midi(midi, rect))
            p.setPen(QPen(QColor(255, 255, 255, 40 if midi % 12 else 70), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            if midi % 12 == 0:  # A4 is only a line: its label would collide with C5 (hover shows notes)
                p.setPen(QColor(170, 170, 180))
                p.drawText(QRectF(0, y - 7, self.LEFT - 4, 14),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, note_label(midi))

        r = self.result
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        sl = slice(f0, f1)
        xs = self._x_for_time(times[sl], rect)

        if self.show_notes:
            p.setPen(QPen(COLOR_NOTE, 2))
            for s, e, note in _runs(r.notes[sl]):
                if note < 0:
                    continue
                y = float(self._y_for_midi(note + r.tuning / 100, rect))
                p.drawLine(QPointF(float(xs[s]), y), QPointF(float(xs[min(e, len(xs)) - 1]) + 1, y))

        if self.show_tracker:
            raw = r.raw_f0[sl] if r.raw_f0 is not None else r.f0[sl]
            has = raw > an.MIN_PITCH_HZ
            ys = self._y_for_midi(an.hz_to_midi(np.where(has, raw, 1.0)), rect)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(COLOR_TRACKER)
            for x, y in zip(xs[has], ys[has]):
                p.drawEllipse(QPointF(float(x), float(y)), 1.3, 1.3)

        # Decision lane
        lane_top = rect.bottom() + 6
        p.setPen(QColor(150, 150, 160))
        p.drawText(QRectF(0, lane_top, self.LEFT - 4, self.LANE_H), Qt.AlignmentFlag.AlignRight, "fixes")
        if r.held_notes is not None:
            midi = an.tuned_midi(r.f0[sl], r.tuning)
            held = r.held_notes[sl]
            categories = [
                (COLOR_HOLD, (held >= 0) & (np.round(midi).astype(int) != held)),
                (COLOR_SWEEP, (r.swept_notes[sl] != held) & (r.swept_notes[sl] >= 0)),
                (COLOR_OCTAVE, r.octave_shift[sl] != 0),
                (COLOR_REMOVED, r.removed[sl]),
                (COLOR_FILLED, r.filled[sl]
                 | (r.attack_filled[sl] if r.attack_filled is not None else np.zeros(f1 - f0, bool))
                 | (r.estimated[sl] if r.estimated is not None else np.zeros(f1 - f0, bool))),
            ]
            width = max(1.0, rect.width() / max(1, WINDOW_SECONDS * an.SAMPLE_RATE / an.HOP_LENGTH))
            p.setPen(Qt.PenStyle.NoPen)
            for color, mask in categories:
                p.setBrush(color)
                for s, e, on in _runs(mask.astype(int)):
                    if on:
                        p.drawRect(QRectF(float(xs[s]), lane_top, float(xs[e - 1] - xs[s]) + width, self.LANE_H))

        # Confidence lane
        conf_top = lane_top + self.LANE_H + 4
        p.setPen(QColor(150, 150, 160))
        p.drawText(QRectF(0, conf_top, self.LEFT - 4, 14), Qt.AlignmentFlag.AlignRight, "conf.")
        p.setPen(QPen(QColor(255, 255, 255, 50), 1, Qt.PenStyle.DashLine))
        mid = conf_top + self.CONF_H / 2
        p.drawLine(QPointF(rect.left(), mid), QPointF(rect.right(), mid))
        if r.confidence is not None and f1 > f0:
            poly = QPolygonF([QPointF(float(x), conf_top + (1 - c) * self.CONF_H)
                              for x, c in zip(xs, r.confidence[sl])])
            p.setPen(QPen(QColor(120, 200, 255), 1))
            p.drawPolyline(poly)

        # Playhead
        x = float(self._x_for_time(self.time, rect))
        p.setPen(QPen(QColor(255, 100, 100), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x, rect.top()), QPointF(x, conf_top + self.CONF_H))
        p.setPen(BORDER)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)

    def _time_and_midi_at(self, pos):
        rect = self._plot_rect()
        start, _ = self._window()
        t = start + (pos.x() - rect.left()) / rect.width() * WINDOW_SECONDS
        midi = SPEC_LOW_MIDI + (rect.bottom() - pos.y()) / rect.height() * (SPEC_HIGH_MIDI - SPEC_LOW_MIDI)
        return t, midi

    def mousePressEvent(self, event):
        if self.result is not None and event.button() == Qt.MouseButton.LeftButton:
            t, _ = self._time_and_midi_at(event.position())
            self.seek_requested.emit(max(0.0, t))

    def mouseMoveEvent(self, event):
        if self.result is None:
            return
        t, midi = self._time_and_midi_at(event.position())
        if SPEC_LOW_MIDI <= midi <= SPEC_HIGH_MIDI and t >= 0:
            QToolTip.showText(event.globalPosition().toPoint(),
                              f"{note_label(round(midi))}  ({an.midi_to_hz(midi):.0f} Hz)   {t:.2f} s", self)


class SpectrumWidget(QWidget):
    """Frequency content at the playhead, with the chosen note's harmonics marked."""

    LOW_HZ = 55.0
    HIGH_HZ = 5000.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.freqs = None
        self.db = None
        self.note_hz = None
        self.tracker_hz = None
        self.setMinimumHeight(200)

    def set_spectrum(self, freqs, db, note_hz, tracker_hz):
        self.freqs, self.db, self.note_hz, self.tracker_hz = freqs, db, note_hz, tracker_hz
        self.update()

    def clear(self):
        self.freqs = None
        self.update()

    def _x(self, hz, rect):
        frac = (np.log2(np.asarray(hz, float)) - np.log2(self.LOW_HZ)) / (np.log2(self.HIGH_HZ) - np.log2(self.LOW_HZ))
        return rect.left() + frac * rect.width()

    def paintEvent(self, event):
        p = QPainter(self)
        paint_background(p, self.width(), self.height())
        p.setFont(QFont(UI_FONT, 8))
        rect = QRectF(36, 18, self.width() - 44, self.height() - 38)
        if self.freqs is None:
            p.setPen(PLACEHOLDER_TEXT)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Spectrum at the playhead appears here")
            return

        # Grid: dB lines and frequency ticks
        for db in range(0, -81, -20):
            y = rect.top() - db / 80 * rect.height()
            p.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(QColor(150, 150, 160))
            p.drawText(QRectF(0, y - 7, 32, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{db}")
        for hz in (100, 200, 500, 1000, 2000, 5000):
            x = float(self._x(hz, rect))
            p.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            p.setPen(QColor(150, 150, 160))
            p.drawText(QRectF(x - 25, rect.bottom() + 3, 50, 14), Qt.AlignmentFlag.AlignCenter,
                       f"{hz // 1000}k" if hz >= 1000 else str(hz))

        # Harmonics of the chosen note (the evidence the note is built on)
        if self.note_hz:
            for k in range(1, SPECTRUM_HARMONICS + 1):
                hz = self.note_hz * k
                if hz > self.HIGH_HZ:
                    break
                x = float(self._x(hz, rect))
                p.setPen(QPen(QColor(0, 255, 128, 200 if k == 1 else 90), 2 if k == 1 else 1))
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                p.setPen(QColor(0, 255, 128))
                p.drawText(QRectF(x - 14, 2, 28, 14), Qt.AlignmentFlag.AlignCenter, f"{k}×")
            sub = self.note_hz / 2
            if sub >= self.LOW_HZ:
                x = float(self._x(sub, rect))
                p.setPen(QPen(QColor(200, 200, 200, 90), 1, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                p.setPen(QColor(170, 170, 170))
                p.drawText(QRectF(x - 14, 2, 28, 14), Qt.AlignmentFlag.AlignCenter, "½×")

        # Tracker's raw pick, if it differs from the final note
        if self.tracker_hz and (not self.note_hz or abs(np.log2(self.tracker_hz / self.note_hz)) > 0.03):
            x = float(self._x(self.tracker_hz, rect))
            p.setPen(QPen(COLOR_TRACKER, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            p.drawText(QRectF(x + 3, rect.top() + 2, 60, 14), Qt.AlignmentFlag.AlignLeft, "tracker")

        # The spectrum itself
        keep = (self.freqs >= self.LOW_HZ) & (self.freqs <= self.HIGH_HZ)
        xs = self._x(self.freqs[keep], rect)
        ys = rect.top() - np.clip(self.db[keep], -80, 0) / 80 * rect.height()
        fill = QPolygonF([QPointF(float(xs[0]), rect.bottom())] +
                         [QPointF(float(x), float(y)) for x, y in zip(xs, ys)] +
                         [QPointF(float(xs[-1]), rect.bottom())])
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 190, 60), 1.2))
        p.setBrush(QColor(255, 190, 60, 50))
        p.drawPolygon(fill)
        p.setPen(QColor(150, 150, 160))
        p.drawText(QRectF(0, 2, 32, 14), Qt.AlignmentFlag.AlignRight, "dB")
        p.drawText(QRectF(0, rect.bottom() + 3, 32, 14), Qt.AlignmentFlag.AlignRight, "Hz")


class DiagnosticsPanel(QWidget):
    """The Diagnostics tab."""
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.media_path = None
        self.images = None
        self.thread = None
        self.time = 0.0
        self.root, self.scale = "A", an.DEFAULT_SCALE

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Signal:"))
        self.signal_input = QComboBox()
        self.signal_input.addItems([SIGNAL_TRACKED, SIGNAL_ORIGINAL])
        self.signal_input.setToolTip("The pitch tracker analyzes the audio with everything below the "
                                     "lead guitar's range (bass, kick drum) cut; switch to compare with the original.")
        self.signal_input.currentTextChanged.connect(self._refresh)
        controls.addWidget(self.signal_input)
        self.tracker_input = QCheckBox("Tracker pitch")
        self.notes_input = QCheckBox("Final notes")
        for box in (self.tracker_input, self.notes_input):
            box.setChecked(True)
            box.toggled.connect(self._refresh)
            controls.addWidget(box)
        controls.addStretch(1)
        self.status = QLabel("")
        self.status.setStyleSheet("color: #888;")
        controls.addWidget(self.status)
        layout.addLayout(controls)

        legend = QLabel("   ".join(f'<span style="color:{c.name()}">■</span> {name}' for name, c in LEGEND)
                        + '   <span style="color:#78c8ff">―</span> Tracker confidence')
        legend.setStyleSheet(f"color: #aaa; font-size: 11px; background: {BACKGROUND.name()}; padding: 4px;")
        legend.setWordWrap(True)
        layout.addWidget(legend)

        self.spectrogram = SpectrogramWidget()
        self.spectrogram.seek_requested.connect(self.seek_requested)
        layout.addWidget(self.spectrogram, 3)

        bottom = QHBoxLayout()
        self.spectrum = SpectrumWidget()
        bottom.addWidget(self.spectrum, 3)
        self.explanation = QLabel("")
        self.explanation.setWordWrap(True)
        self.explanation.setTextFormat(Qt.TextFormat.RichText)
        self.explanation.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.explanation.setStyleSheet(
            f"background: {BACKGROUND.name()}; color: #ddd; border: 1px solid {BORDER.name()}; padding: 8px;")
        self.explanation.setMinimumWidth(320)
        self.explanation.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        bottom.addWidget(self.explanation, 2)
        layout.addLayout(bottom, 2)

        for w in (self.signal_input, self.tracker_input, self.notes_input):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    # -- data --

    def set_result(self, result, media_path):
        self.stop()
        self.result, self.media_path, self.images = result, media_path, None
        self.spectrogram.set_data(None, None)
        self.spectrogram.message = "Open this tab to prepare diagnostics"
        self.spectrum.clear()
        self.explanation.setText("")
        self.signal_input.setEnabled(bool(an.USE_HPSS or an.LOW_CUT_HZ or (result and result.isolated)))
        self.signal_input.setItemText(0, SIGNAL_TRACKED_ISOLATED if result is not None and result.isolated
                                      else SIGNAL_TRACKED)

    def ensure_ready(self):
        """Start preparing spectrograms for the current result, if not already done."""
        if self.result is None or self.images is not None or (self.thread and self.thread.isRunning()):
            return
        self.spectrogram.message = "Preparing diagnostics..."
        self.spectrogram.update()
        self.thread = DiagnosticsThread(self.result, self.media_path)
        self.thread.progress.connect(self._on_progress)
        self.thread.ready.connect(self._on_ready)
        self.thread.failed.connect(lambda msg: self._on_progress(f"Couldn't prepare diagnostics: {msg}"))
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()

    def _on_progress(self, message):
        self.status.setText(message)
        self.spectrogram.message = message
        self.spectrogram.update()

    def _on_ready(self, images):
        self.images = images
        self.status.setText("")
        self._refresh()

    # -- view --

    def set_key(self, root, scale):
        self.root, self.scale = root, scale
        self._refresh()

    def set_time(self, t):
        self.time = t
        self._refresh()

    def _signal(self):
        tracked = self.signal_input.currentIndex() == 0
        return ("tracked" if tracked else "original",
                self.result.tracked_audio if tracked else self.result.audio)

    def _refresh(self):
        if self.result is None or self.images is None:
            return
        key, audio = self._signal()
        self.spectrogram.show_tracker = self.tracker_input.isChecked()
        self.spectrogram.show_notes = self.notes_input.isChecked()
        self.spectrogram.set_data(self.result, self.images[key])
        self.spectrogram.set_time(self.time)

        r = self.result
        i = int(np.clip(np.searchsorted(r.times, self.time), 0, len(r.times) - 1))
        freqs, db = spectrum_at(audio, i)
        note = int(r.notes[i])
        note_hz = an.midi_to_hz(note + r.tuning / 100) if note >= 0 else None
        raw = r.raw_f0[i] if r.raw_f0 is not None else r.f0[i]
        self.spectrum.set_spectrum(freqs, db, note_hz, raw if raw > an.MIN_PITCH_HZ else None)

        steps = explain_frame(r, i, self.root, self.scale)
        self.explanation.setText(f"<div style='color:#888'>At {r.times[i]:.2f} s</div><ol style='margin-left:-20px'>"
                                 + "".join(f"<li style='margin-bottom:4px'>{s}</li>" for s in steps) + "</ol>")
