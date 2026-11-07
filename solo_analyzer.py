"""Solo Analyzer: tracks the pitch of a guitar solo and maps it to scale degrees."""

import os
import sys

import numpy as np
from PyQt6.QtCore import QByteArray, QLine, QRectF, QSettings, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QFontMetrics, QKeySequence, QPainter, QPen, QShortcut
from PyQt6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSlider, QTabWidget, QVBoxLayout, QWidget,
)

from analysis import (  # noqa: E402  (analysis sets up numba before librosa loads)
    DEFAULT_SCALE, MIN_PITCH_HZ, NOTE_NAMES, SAMPLE_RATE, SCALES,
    AnalysisCancelled, analyze_pitch, cents_off, close_alternatives, degree_breakdown, degree_of, detect_key,
    display_range, extract_audio, get_intervals, load_cached, note_name, save_cached,
)
import recording
import separation
from diagnostics import DiagnosticsPanel
from ui_common import PLACEHOLDER_TEXT, UI_FONT, degree_color, paint_background

MEDIA_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg")
SPEEDS = ["25%", "50%", "75%", "100%"]
SEEK_STEP_MS = 5000
FINE_SEEK_STEP_MS = 1000


# --- Analysis thread -------------------------------------------------------

ISOLATION_SHARE = 0.7  # share of the progress bar for guitar isolation (it's the slow part)

class AudioAnalyzerThread(QThread):
    """Extracts the audio track and runs pitch detection off the GUI thread."""
    analysis_done = pyqtSignal(object)  # AnalysisResult
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, media_path, band_mode=False, isolate=False):
        super().__init__()
        self.media_path = media_path
        self.band_mode = band_mode
        self.isolate = isolate
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            if not os.path.exists(self.media_path):
                self.error.emit("Selected file does not exist.")
                return

            cached = load_cached(self.media_path, self.band_mode, self.isolate)
            if cached is not None:
                self.analysis_done.emit(cached)
                return

            self.progress.emit(0, "Extracting audio...")
            y = extract_audio(self.media_path)
            if len(y) == 0:
                self.error.emit("The file contains no audio track to analyze.")
                return

            stop = lambda: self.cancel_requested  # noqa: E731
            guitar, start = None, 0.0
            if self.isolate:
                if separation.available() is None:
                    separation.download_pack(
                        progress=lambda done, total: self.progress.emit(
                            int(done / total * 100) if total else 0,
                            f"Downloading guitar isolation pack ({done:.0f} of {total:.0f} MB)..."),
                        cancelled=stop)
                guitar = separation.isolate_file(
                    self.media_path, cancelled=stop,
                    progress=lambda f: self.progress.emit(int(f * ISOLATION_SHARE * 100), "Isolating the guitar..."))
                start = ISOLATION_SHARE

            result = analyze_pitch(
                y, SAMPLE_RATE,
                progress=lambda fraction, message: self.progress.emit(
                    int((start + (1 - start) * fraction) * 100), message),
                cancelled=stop,
                band_mode=self.band_mode,
                isolated=guitar,
            )
            save_cached(self.media_path, result)
            self.analysis_done.emit(result)
        except (AnalysisCancelled, separation.SeparationCancelled):
            self.cancelled.emit()
        except separation.SeparationUnavailable as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"An unexpected error occurred:\n{e}")


# --- Widgets ---------------------------------------------------------------

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

        self.band_mode_input = QCheckBox("Over a band")
        self.band_mode_input.setToolTip(
            "Ignore the bass range (below C4) so a loud bass guitar can't be mistaken for the lead.\n"
            "Leave off for solo guitar, or if the lead plays notes below middle C.")
        self.band_mode_input.toggled.connect(self.on_band_mode_changed)
        key_row.addWidget(self.band_mode_input)

        self.isolate_input = QCheckBox("Isolate guitar")
        self.isolate_input.setToolTip(
            "Separate the guitar from drums, bass, keys and vocals with an AI model before tracking.\n"
            "Slower, and a one-time download of about %d MB. It can't split two guitars apart."
            % separation.PACK_APPROX_MB)
        self.isolate_input.toggled.connect(self.on_isolate_changed)
        key_row.addWidget(self.isolate_input)

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
        self.diagnostics = DiagnosticsPanel()
        self.diagnostics.seek_requested.connect(lambda s: self.seek_to(int(s * 1000)))
        self.tabs = QTabWidget()
        self.tabs.addTab(self.pitch_graph, "Pitch")
        self.tabs.addTab(self.diagnostics, "Diagnostics")
        self.tabs.setTabToolTip(1, "See the audio the way the pitch tracker does, and why each note was chosen")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        main_layout.addWidget(self.tabs, 3)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(120)
        main_layout.addWidget(self.video_widget, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Keyboard shortcuts belong to the window, so keep controls from grabbing focus.
        for widget in (self.file_btn, self.record_btn, self.play_btn, self.cancel_btn, self.speed_input,
                       self.root_input, self.scale_input, self.auto_key_input, self.band_mode_input,
                       self.isolate_input, self.scrubber):
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
        self.band_mode_input.blockSignals(True)
        self.band_mode_input.setChecked(self.settings.value("band_mode", False, type=bool))
        self.isolate_input.blockSignals(True)
        self.isolate_input.setChecked(self.settings.value("isolate", False, type=bool)
                                      and separation.available() is not None)
        self.isolate_input.blockSignals(False)
        self.band_mode_input.blockSignals(False)
        self.on_scale_changed()

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("root", self.root_input.currentText())
        self.settings.setValue("scale", self.scale_input.currentText())
        self.settings.setValue("speed", self.speed_input.currentText())
        self.settings.setValue("auto_key", self.auto_key_input.isChecked())
        self.settings.setValue("band_mode", self.band_mode_input.isChecked())
        self.settings.setValue("isolate", self.isolate_input.isChecked())

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
        self.diagnostics.set_result(None, None)
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.file_label.setText(os.path.basename(path))
        self.info_panel.setText("Analyzing audio...")
        self.detected_label.setText("")
        self._set_controls_enabled(file_btn=False, playback=False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.progress_widget.setVisible(True)

        self.media_player.setSource(QUrl.fromLocalFile(path))

        self.analyzer = AudioAnalyzerThread(path, self.band_mode_input.isChecked(), self.isolate_input.isChecked())
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
        self.diagnostics.set_result(result, self.video_path)
        if self.tabs.currentWidget() is self.diagnostics:
            self.diagnostics.ensure_ready()

        details = [f"Tuning: {result.tuning:+.0f}¢ from A440"] + (["guitar isolated"] if result.isolated else [])
        key = detect_key(result.notes, result.level_db)
        if key:
            root, mode = key
            alternatives = close_alternatives(result.notes, result.level_db)
            also = f" (close: {', '.join(f'{r} {m.lower()}' for r, m in alternatives)})" if alternatives else ""
            details.insert(0, f"Detected key: {root} {mode.lower()}{also}")
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
        self.diagnostics.set_key(root, scale)
        self.breakdown_widget.update_breakdown(self.result.notes if self.result else None, root, scale)
        self.position_changed(self.media_player.position())

    def on_band_mode_changed(self, _checked):
        """Re-analyze the loaded file with the new tracking range (cached per setting)."""
        if self.video_path and not self.is_busy():
            self.load_file(self.video_path)

    def on_isolate_changed(self, checked):
        """Turning isolation on the first time needs the pack: ask before downloading it."""
        if checked and separation.available() is None and not self.confirm_pack_download():
            self.isolate_input.blockSignals(True)
            self.isolate_input.setChecked(False)
            self.isolate_input.blockSignals(False)
            return
        if self.video_path and not self.is_busy():
            self.load_file(self.video_path)

    def confirm_pack_download(self):
        answer = QMessageBox.question(
            self, "Isolate guitar",
            "Isolating the guitar uses an AI model (Demucs) that separates the guitar from drums, bass, "
            "keys and vocals before the notes are tracked.\n\n"
            f"It needs a one-time download of about {separation.PACK_APPROX_MB} MB, and analysis takes "
            "noticeably longer. It can't separate two guitars from each other.\n\nDownload it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

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
        if self.tabs.currentWidget() is self.diagnostics:
            self.diagnostics.set_time(current_time)

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

    def on_tab_changed(self, _index):
        if self.tabs.currentWidget() is self.diagnostics:
            self.diagnostics.ensure_ready()
            self.diagnostics.set_time(self.media_player.position() / 1000.0)

    def closeEvent(self, event):
        self.save_settings()
        self.diagnostics.stop()
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
