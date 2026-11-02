"""Solo Analyzer: tracks the pitch of a guitar solo in a video and maps it to scale degrees."""

import os
import subprocess
import sys
import tempfile

# Numba (used by librosa) caches compiled functions next to their source files,
# which don't exist inside a frozen executable. Point the cache somewhere writable.
if getattr(sys, "frozen", False):
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "solo_analyzer_numba"))

import imageio_ffmpeg
import librosa
import numpy as np
from PyQt6.QtCore import QLine, QRectF, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget,
)
from scipy.signal import medfilt

# --- Constants -------------------------------------------------------------

SAMPLE_RATE = 22050      # Hz, audio is resampled to this before analysis
HOP_LENGTH = 256         # samples between analysis frames
MIN_PITCH_HZ = 20.0      # anything at or below this is treated as "no pitch"
DEGREE_MARGIN = 0.45     # semitone tolerance for bends and vibrato in the breakdown

# Semitone offsets from the root -> scale degree label
SCALE_INTERVALS = {
    "minor": {0: "1", 2: "2", 3: "♭3", 5: "4", 7: "5", 8: "6", 10: "♭7"},
    "major": {0: "1", 2: "2", 4: "3", 5: "4", 7: "5", 9: "6", 11: "7"},
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

DEGREE_COLORS = {
    "1": (255, 100, 100),
    "2": (255, 165, 0),
    "3": (100, 180, 255),
    "♭3": (100, 180, 255),
    "4": (255, 105, 180),
    "5": (180, 100, 255),
    "6": (255, 165, 0),
    "♭7": (255, 165, 0),
    "7": (255, 215, 0),
    "Out": (80, 80, 90),
}

BACKGROUND = QColor(15, 15, 20)
BORDER = QColor(40, 40, 50)
PLACEHOLDER_TEXT = QColor(100, 100, 100)


# --- Helpers ---------------------------------------------------------------

def hz_to_midi(freq):
    """Convert frequency in Hz to a (fractional) MIDI note number. Works on scalars and arrays."""
    return 69 + 12 * np.log2(freq / 440.0)


def degree_color(degree, fallback, alpha=255):
    """Color for a scale degree label, or `fallback` if the label has no assigned color."""
    rgb = DEGREE_COLORS.get(degree)
    return QColor(*rgb, alpha) if rgb else fallback


def get_intervals(scale_type):
    return SCALE_INTERVALS.get(scale_type.lower(), SCALE_INTERVALS["minor"])


def freq_to_note(freq):
    """Convert a frequency (Hz) to its note name and MIDI number."""
    if freq < MIN_PITCH_HZ or np.isnan(freq):
        return None, -1
    midi_num = round(hz_to_midi(freq))
    return NOTE_NAMES[midi_num % 12], midi_num


def get_scale_degree(note_name, root_note, scale_type):
    """Return the scale degree of a note in the given key, "Out" if it isn't in the scale."""
    if note_name is None:
        return "--"
    try:
        root_idx = NOTE_NAMES.index(root_note.upper())
        note_idx = NOTE_NAMES.index(note_name)
    except ValueError:
        return "--"
    semitone_diff = (note_idx - root_idx) % 12
    return get_intervals(scale_type).get(semitone_diff, "Out")


def paint_background(painter, width, height):
    painter.fillRect(0, 0, width, height, BACKGROUND)
    painter.setPen(BORDER)
    painter.drawRect(0, 0, width - 1, height - 1)


# --- Widgets ---------------------------------------------------------------

class ScaleBreakdownWidget(QWidget):
    """Bar showing the share of pitched frames spent on each scale degree."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.percentages = {}
        self.setMinimumHeight(50)

    def _clear(self):
        self.percentages = {}
        self.update()

    def update_breakdown(self, f0, root_note, scale_type):
        if f0 is None or len(f0) == 0:
            return self._clear()
        try:
            root_idx = NOTE_NAMES.index(root_note.upper())
        except ValueError:
            return self._clear()

        valid_f0 = f0[f0 > MIN_PITCH_HZ]
        if len(valid_f0) == 0:
            return self._clear()

        intervals = get_intervals(scale_type)
        semitone_from_root = (hz_to_midi(valid_f0) - (69 + root_idx)) % 12

        counts = {}
        unmatched = np.ones(len(valid_f0), dtype=bool)
        for target, deg_name in intervals.items():
            delta = semitone_from_root - target
            diff = np.minimum.reduce([np.abs(delta), np.abs(delta - 12), np.abs(delta + 12)])
            hit = unmatched & (diff <= DEGREE_MARGIN)
            counts[deg_name] = int(hit.sum())
            unmatched &= ~hit
        counts["Out"] = int(unmatched.sum())

        total = len(valid_f0)
        self.percentages = {deg: cnt / total * 100 for deg, cnt in counts.items() if cnt > 0}
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        paint_background(painter, width, height)

        if not self.percentages:
            painter.setPen(PLACEHOLDER_TEXT)
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Scale percentage breakdown will appear here")
            return

        fallback = QColor(150, 150, 150)
        ranked = sorted(self.percentages.items(), key=lambda x: x[1], reverse=True)
        bar_x, bar_y, bar_w, bar_h = 10, 12, width - 20, 16

        current_x = bar_x
        painter.setPen(Qt.PenStyle.NoPen)
        for deg, pct in ranked:
            segment_w = max(4, int(pct / 100.0 * bar_w))
            painter.setBrush(degree_color(deg, fallback))
            painter.drawRoundedRect(QRectF(current_x, bar_y, segment_w, bar_h), 3, 3)
            current_x += segment_w

        painter.setFont(QFont("Segoe UI", 9))
        label_x = 10
        for deg, pct in ranked:
            painter.setPen(degree_color(deg, fallback))
            text = f"{deg}: {pct:.1f}%"
            painter.drawText(label_x, bar_y + bar_h + 16, text)
            label_x += len(text) * 9 + 15


class PitchGraphWidget(QWidget):
    """Scrolling pitch contour drawn against the selected scale's degree lines."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.times = None
        self.f0 = None
        self.current_time = 0.0
        self.window_duration = 30.0  # seconds visible at once
        self.f_min = 65.0
        self.f_max = 1300.0
        self.root_note = "A"
        self.scale_type = "Minor"
        self.setMinimumHeight(600)

    def set_data(self, times, f0):
        self.times = times
        self.f0 = f0
        valid_freqs = f0[f0 > MIN_PITCH_HZ]
        if len(valid_freqs) > 0:
            self.f_min = max(40.0, float(np.min(valid_freqs)) * 0.9)
            self.f_max = max(float(np.max(valid_freqs)) * 1.1, 400.0)
        else:
            self.f_min, self.f_max = 80.0, 1300.0
        self.update()

    def set_scale_info(self, root_note, scale_type):
        self.root_note = root_note
        self.scale_type = scale_type
        self.update()

    def set_time(self, current_time):
        self.current_time = current_time
        self.update()

    def get_reference_frequencies(self):
        """(frequency, degree label) for every scale note inside the visible range."""
        try:
            root_idx = NOTE_NAMES.index(self.root_note.upper())
        except ValueError:
            return []
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
        painter.setFont(QFont("Segoe UI", 9))
        for freq, label in self.get_reference_frequencies():
            y = self._freq_to_y(freq, height, log_min, log_max)
            color = degree_color(label, fallback, alpha=180)
            painter.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
            painter.drawLine(0, y, width, y)
            painter.setPen(color)
            painter.drawText(5, y - 3, label)

        if self.times is None or self.f0 is None:
            painter.setPen(PLACEHOLDER_TEXT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load video to view pitch graph")
            return

        t_end = self.current_time
        t_start = max(0.0, t_end - self.window_duration)
        mask = (self.times >= t_start) & (self.times <= t_end)
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


# --- Analysis --------------------------------------------------------------

class AudioAnalyzerThread(QThread):
    """Extracts the audio track and runs pitch detection off the GUI thread."""
    analysis_done = pyqtSignal(np.ndarray, np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        temp_wav = None
        try:
            if not os.path.exists(self.video_path):
                self.error.emit("Selected file does not exist.")
                return

            temp_fd, temp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)

            # Extract mono audio with the bundled ffmpeg (no console window on Windows)
            cmd = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", self.video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", temp_wav,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode != 0:
                self.error.emit("Failed to extract audio from video.")
                return

            y, sr = librosa.load(temp_wav, sr=SAMPLE_RATE)
            if len(y) == 0:
                self.error.emit("The file contains no audio track to analyze.")
                return

            self.analysis_done.emit(*analyze_pitch(y, sr))
        except Exception as e:
            self.error.emit(f"An unexpected error occurred:\n{e}")
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass


def analyze_pitch(y, sr):
    """Return (f0, times) for an audio signal. f0 is 0 where no pitch was detected."""
    # Aggressive harmonic/percussive separation to pull the guitar away from drums
    y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))

    f0, voiced_flag, _ = librosa.pyin(
        y_harmonic,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    rms = librosa.feature.rms(y=y_harmonic, frame_length=2048, hop_length=HOP_LENGTH)[0]
    onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr, hop_length=HOP_LENGTH)

    f0[~voiced_flag] = 0.0
    corrected_f0 = np.nan_to_num(f0)

    # Drift check: a big pitch jump with no pick attack and no smooth slide is
    # treated as detection jitter and held at the previous pitch.
    if len(corrected_f0) > 2:
        midi_track = np.zeros_like(corrected_f0)
        valid = corrected_f0 > MIN_PITCH_HZ
        midi_track[valid] = hz_to_midi(corrected_f0[valid])

        for i in range(2, len(corrected_f0) - 1):
            if not (corrected_f0[i] > MIN_PITCH_HZ and corrected_f0[i - 1] > MIN_PITCH_HZ):
                continue
            step = midi_track[i] - midi_track[i - 1]
            if abs(step) <= 1.5:
                continue

            has_transient = onset_env[i] > 0.3 or rms[i] > rms[i - 1] * 1.3

            is_smooth_slide = False
            if corrected_f0[i - 2] > MIN_PITCH_HZ:
                prev_step = midi_track[i - 1] - midi_track[i - 2]
                is_smooth_slide = (abs(prev_step) < 1.0 and abs(step) < 1.5
                                   and np.sign(prev_step) == np.sign(step))

            if not has_transient and not is_smooth_slide:
                corrected_f0[i] = corrected_f0[i - 1]

    f0_smooth = medfilt(corrected_f0, kernel_size=3)
    times = librosa.times_like(f0_smooth, sr=sr, hop_length=HOP_LENGTH)
    return f0_smooth, times


# --- Main window -----------------------------------------------------------

class GuitarTrackerApp(QMainWindow):
    """Main window: controls, readout, breakdown bar, pitch graph and video."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guitar Scale Tracker")
        self.resize(900, 1100)

        self.video_path = ""
        self.f0 = None
        self.times = None
        self.analyzer = None
        self.is_slider_dragging = False

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        controls_layout = QHBoxLayout()

        self.file_btn = QPushButton("Select Video")
        self.file_btn.clicked.connect(self.select_file)
        controls_layout.addWidget(self.file_btn)

        self.file_label = QLabel("No file selected")
        controls_layout.addWidget(self.file_label)

        controls_layout.addWidget(QLabel("Root:"))
        self.root_input = QComboBox()
        self.root_input.addItems(NOTE_NAMES)
        self.root_input.setCurrentText("A")
        self.root_input.currentTextChanged.connect(self.on_scale_changed)
        controls_layout.addWidget(self.root_input)

        controls_layout.addWidget(QLabel("Scale:"))
        self.scale_input = QComboBox()
        self.scale_input.addItems(["Minor", "Major"])
        self.scale_input.currentTextChanged.connect(self.on_scale_changed)
        controls_layout.addWidget(self.scale_input)

        self.play_btn = QPushButton("Play / Pause")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)

        main_layout.addLayout(controls_layout)

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

        self.breakdown_widget = ScaleBreakdownWidget()
        main_layout.addWidget(self.breakdown_widget)

        self.pitch_graph = PitchGraphWidget()
        main_layout.addWidget(self.pitch_graph)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(200)
        main_layout.addWidget(self.video_widget)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)

    def _set_controls_enabled(self, file_btn, playback):
        self.file_btn.setEnabled(file_btn)
        self.play_btn.setEnabled(playback)
        self.scrubber.setEnabled(playback)

    def on_scale_changed(self):
        root = self.root_input.currentText()
        scale = self.scale_input.currentText()
        self.pitch_graph.set_scale_info(root, scale)
        if self.f0 is not None:
            self.breakdown_widget.update_breakdown(self.f0, root, scale)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "",
                                              "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self.video_path = path
        self.file_label.setText(os.path.basename(path))
        self.info_panel.setText("Analyzing audio...")
        self._set_controls_enabled(file_btn=False, playback=False)

        self.media_player.setSource(QUrl.fromLocalFile(path))

        self.analyzer = AudioAnalyzerThread(path)
        self.analyzer.analysis_done.connect(self.on_analysis_finished)
        self.analyzer.error.connect(self.on_analysis_error)
        self.analyzer.start()

    def on_analysis_finished(self, f0, times):
        self.f0 = f0
        self.times = times
        self._set_controls_enabled(file_btn=True, playback=True)
        self.pitch_graph.set_data(times, f0)
        self.breakdown_widget.update_breakdown(f0, self.root_input.currentText(),
                                               self.scale_input.currentText())
        self.info_panel.setText("Note: --   |   Scale Degree: --")

    def on_analysis_error(self, err_msg):
        QMessageBox.critical(self, "Error", err_msg)
        self.info_panel.setText("Load a video to begin")
        self._set_controls_enabled(file_btn=True, playback=False)

    def toggle_playback(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def position_changed(self, position):
        """Keep the scrubber, graph playhead and note readout in sync with playback."""
        if not self.is_slider_dragging:
            self.scrubber.setValue(position)

        if self.f0 is None or self.times is None:
            return

        current_time = position / 1000.0
        self.pitch_graph.set_time(current_time)

        idx = np.clip(np.searchsorted(self.times, current_time, side="left"), 0, len(self.times) - 1)
        note, _ = freq_to_note(self.f0[idx])
        degree = get_scale_degree(note, self.root_input.currentText(), self.scale_input.currentText())
        self.info_panel.setText(f"Note: {note if note else '--'}   |   Scale Degree: {degree}")

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
        self.media_player.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = GuitarTrackerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
