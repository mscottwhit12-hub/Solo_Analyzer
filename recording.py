"""Record audio (and optionally webcam video) straight into Solo Analyzer.

Audio is captured with PortAudio (sounddevice) rather than Qt so the input can be
opened raw: on Windows the stream asks WASAPI for AUDCLNT_STREAMOPTIONS_RAW, which
bypasses the system's audio effects (automatic gain, noise suppression, "enhancements").
macOS Core Audio input is already unprocessed unless an app opts into voice processing.
Webcam video is recorded separately by Qt and muxed with the audio afterwards.
"""

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PyQt6.QtCore import (
    QCameraPermission, QMicrophonePermission, QStandardPaths, Qt, QThread, QTimer, QUrl, pyqtSignal,
)
from PyQt6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QMediaFormat, QMediaRecorder
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QVBoxLayout,
)

try:
    import sounddevice as sd
    AUDIO_IMPORT_ERROR = None
except (ImportError, OSError) as e:  # sounddevice or its PortAudio library missing
    sd = None
    AUDIO_IMPORT_ERROR = str(e)

# Host API whose devices we list per platform. WASAPI is required for raw mode on Windows.
PREFERRED_HOST_API = {"win32": "Windows WASAPI", "darwin": "Core Audio"}.get(sys.platform)
MAX_CHANNELS = 2
METER_FLOOR_DB = -60.0
CLIP_LEVEL = 0.999
MAX_SYNC_OFFSET = 2.0  # seconds; anything larger is treated as a bad measurement
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

MODE_AUDIO = "Audio only"
MODE_VIDEO = "Webcam + audio"


# --- Audio -----------------------------------------------------------------

def list_input_devices():
    """Return ([(device index, name)], default index) for devices that can record."""
    host_apis = sd.query_hostapis()
    devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]

    preferred = [(i, d) for i, d in devices if host_apis[d["hostapi"]]["name"] == PREFERRED_HOST_API]
    if preferred:
        devices = preferred
        api = next(a for a in host_apis if a["name"] == PREFERRED_HOST_API)
        default = api["default_input_device"]
    else:
        default = sd.default.device[0]

    return [(i, d["name"]) for i, d in devices], default


def input_settings(raw):
    """Host-specific stream settings. On Windows: let WASAPI convert channel count/format,
    and optionally request the raw capture path. None elsewhere."""
    if sys.platform != "win32":
        return None
    settings = sd.WasapiSettings(auto_convert=True)
    if raw:
        # sounddevice doesn't expose this field; PortAudio applies it via
        # IAudioClient2::SetClientProperties (AUDCLNT_STREAMOPTIONS_RAW).
        settings._streaminfo.streamOption = getattr(sd._lib, "eStreamOptionRaw", 1)
    return settings


class AudioRecorder:
    """Keeps an input stream open for metering and writes it to a WAV file on demand."""

    def __init__(self, device, raw=True):
        info = sd.query_devices(device)
        self.samplerate = int(info["default_samplerate"])
        self.channels = min(MAX_CHANNELS, int(info["max_input_channels"]))
        self.peak = 0.0
        self.clipped = False
        self.recording = False
        self.first_sample_time = None  # perf_counter() time of the first recorded sample

        self._queue = queue.Queue()
        self._writer = None
        self._stream = sd.InputStream(
            device=device, channels=self.channels, samplerate=self.samplerate,
            dtype="float32", callback=self._callback,
            extra_settings=input_settings(raw),
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        block_peak = float(np.abs(indata).max()) if frames else 0.0
        self.peak = max(self.peak, block_peak)
        if not self.recording:
            return
        if block_peak >= CLIP_LEVEL:
            self.clipped = True
        if self.first_sample_time is None:
            age = time_info.currentTime - time_info.inputBufferAdcTime
            if not 0 < age < 1:
                age = frames / self.samplerate
            self.first_sample_time = time.perf_counter() - age
        self._queue.put(indata.copy())

    def take_peak(self):
        """Highest level since the last call (0..1)."""
        peak, self.peak = self.peak, 0.0
        return peak

    def start_recording(self, path):
        audio_file = sf.SoundFile(path, "w", samplerate=self.samplerate,
                                  channels=self.channels, subtype="PCM_24")
        self._queue = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, args=(audio_file, self._queue), daemon=True)
        self._writer.start()
        self.clipped = False
        self.first_sample_time = None
        self.recording = True

    @staticmethod
    def _write_loop(audio_file, blocks):
        with audio_file:
            while (block := blocks.get()) is not None:
                audio_file.write(block)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self._queue.put(None)
        self._writer.join()

    def close(self):
        self.stop_recording()
        self._stream.stop()
        self._stream.close()


# --- Muxing ----------------------------------------------------------------

def mux_command(video_path, audio_path, out_path, offset):
    """ffmpeg command combining webcam video with the separately recorded audio.

    offset > 0: audio started `offset` seconds before the video, so trim it.
    offset < 0: audio started late, so delay it.
    """
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", video_path]
    if offset > 0:
        cmd += ["-ss", f"{offset:.3f}"]
    cmd += ["-i", audio_path, "-map", "0:v:0", "-map", "1:a:0"]
    if offset < 0:
        cmd += ["-af", f"adelay={round(-offset * 1000)}:all=1"]  # pad real silence at the start
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "320k", out_path]
    return cmd


class MuxThread(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, video_path, audio_path, out_path, offset):
        super().__init__()
        self.args = (video_path, audio_path, out_path, offset)

    def run(self):
        video_path, audio_path, out_path, _ = self.args
        result = subprocess.run(mux_command(*self.args), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, creationflags=NO_WINDOW)
        for path in (video_path, audio_path):
            try:
                os.remove(path)
            except OSError:
                pass
        if result.returncode == 0:
            self.done.emit(out_path)
        else:
            self.failed.emit(result.stderr.decode(errors="replace")[-500:])


# --- Permissions (macOS asks the user; elsewhere these are granted) --------

def with_permission(permission, on_granted, on_denied):
    app = QApplication.instance()
    status = app.checkPermission(permission)
    if status == Qt.PermissionStatus.Granted:
        on_granted()
    elif status == Qt.PermissionStatus.Denied:
        on_denied()
    else:
        app.requestPermission(permission, lambda p: with_permission(p, on_granted, on_denied))


# --- Dialog ----------------------------------------------------------------

def output_path(extension):
    location = QStandardPaths.StandardLocation.MusicLocation if extension == "wav" \
        else QStandardPaths.StandardLocation.MoviesLocation
    folder = os.path.join(QStandardPaths.writableLocation(location), "Solo Analyzer")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"Recording {datetime.now():%Y-%m-%d %H-%M-%S}.{extension}")


class RecordDialog(QDialog):
    """Pick inputs, watch the level, record. `self.output_path` is set when accepted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Record")
        self.resize(640, 560)

        self.output_path = None
        self.audio = None
        self.camera = None
        self.session = QMediaCaptureSession()
        self.recorder = QMediaRecorder()
        self.session.setRecorder(self.recorder)
        self.video_start_time = None
        self.temp_audio = None
        self.mux_thread = None
        self.record_started = 0.0

        self._build_ui()
        self.recorder.durationChanged.connect(self._on_video_duration)
        # PyQt6 can't connect recorderStateChanged/errorOccurred (enum argument types),
        # so errors come via errorChanged and the stopped state is polled in _tick.
        self.recorder.errorChanged.connect(self._on_recorder_error)
        self.finishing_video = False

        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self._tick)
        self.meter_timer.start(50)

        with_permission(QMicrophonePermission(), self._open_audio,
                        lambda: self._set_status("Microphone access denied. Allow it in system settings."))

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.mode_input = QComboBox()
        self.mode_input.addItems([MODE_AUDIO, MODE_VIDEO])
        self.mode_input.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Mode:", self.mode_input)

        self.audio_input = QComboBox()
        devices, default = list_input_devices()
        for index, name in devices:
            self.audio_input.addItem(name, index)
        default_row = self.audio_input.findData(default)
        if default_row >= 0:
            self.audio_input.setCurrentIndex(default_row)
        self.audio_input.currentIndexChanged.connect(self._open_audio)
        form.addRow("Audio input:", self.audio_input)

        self.camera_input = QComboBox()
        for cam in QMediaDevices.videoInputs():
            self.camera_input.addItem(cam.description(), cam)
        self.camera_input.currentIndexChanged.connect(self._open_camera)
        form.addRow("Camera:", self.camera_input)

        self.raw_input = QCheckBox("Raw input: bypass system audio processing (recommended)")
        self.raw_input.setChecked(True)
        if sys.platform == "win32":
            self.raw_input.setToolTip("Turns off Windows audio effects such as automatic gain control, "
                                      "noise suppression and 'enhancements' for this recording.")
            self.raw_input.toggled.connect(self._open_audio)
        else:
            self.raw_input.setEnabled(False)
            self.raw_input.setToolTip("Input on this system is already recorded unprocessed.")
        form.addRow("", self.raw_input)
        layout.addLayout(form)

        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(14)
        self.clip_label = QLabel("")
        self.clip_label.setStyleSheet("color: #ff5050; font-weight: bold;")
        meter_row = QHBoxLayout()
        meter_row.addWidget(QLabel("Level:"))
        meter_row.addWidget(self.meter, 1)
        meter_row.addWidget(self.clip_label)
        layout.addLayout(meter_row)

        self.preview = QVideoWidget()
        self.preview.setMinimumHeight(240)
        self.session.setVideoOutput(self.preview)
        layout.addWidget(self.preview, 1)

        self.status = QLabel("Ready")
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        self.record_btn = QPushButton("● Record")
        self.record_btn.clicked.connect(self._toggle_recording)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.record_btn)
        layout.addLayout(buttons)

        self._on_mode_changed(self.mode_input.currentText())

    # -- device handling --

    def _set_status(self, text):
        self.status.setText(text)

    def _open_audio(self, *_):
        if self._is_recording():
            return
        if self.audio:
            self.audio.close()
            self.audio = None
        device = self.audio_input.currentData()
        if device is None:
            self._set_status("No audio input devices found.")
            self.record_btn.setEnabled(False)
            return
        try:
            self.audio = AudioRecorder(device, raw=self.raw_input.isChecked())
        except Exception as e:
            self._set_status(f"Couldn't open audio input: {e}")
            self.record_btn.setEnabled(False)
            return
        self.record_btn.setEnabled(True)
        self._set_status(f"{self.audio.samplerate} Hz, {self.audio.channels} channel(s). Ready.")

    def _on_mode_changed(self, mode):
        video = mode == MODE_VIDEO
        self.camera_input.setEnabled(video)
        self.preview.setVisible(video)
        if video:
            with_permission(QCameraPermission(), self._open_camera,
                            lambda: self._set_status("Camera access denied. Allow it in system settings."))
        elif self.camera:
            self.camera.stop()
            self.camera = None
            self.session.setCamera(None)

    def _open_camera(self, *_):
        if self._is_recording() or self.mode_input.currentText() != MODE_VIDEO:
            return
        device = self.camera_input.currentData()
        if device is None:
            self._set_status("No camera found.")
            return
        if self.camera:
            self.camera.stop()
        self.camera = QCamera(device)
        self.session.setCamera(self.camera)
        self.camera.start()

    # -- recording --

    def _is_recording(self):
        return self.audio is not None and self.audio.recording

    def _toggle_recording(self):
        if self._is_recording():
            self._stop()
        else:
            self._start()

    def _start(self):
        video = self.mode_input.currentText() == MODE_VIDEO
        if video and not (self.camera and self.camera.isActive()):
            QMessageBox.warning(self, "Record", "The camera isn't running.")
            return

        if video:
            fd, self.temp_audio = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            fd, temp_video = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            fmt = QMediaFormat(QMediaFormat.FileFormat.MPEG4)
            fmt.setVideoCodec(QMediaFormat.VideoCodec.H264)
            self.recorder.setMediaFormat(fmt)
            self.recorder.setQuality(QMediaRecorder.Quality.HighQuality)
            self.recorder.setOutputLocation(QUrl.fromLocalFile(temp_video))
            self.video_start_time = None
            self.audio.start_recording(self.temp_audio)
            self.recorder.record()
        else:
            self.output_path = output_path("wav")
            self.audio.start_recording(self.output_path)

        self.record_started = time.perf_counter()
        self.record_btn.setText("■ Stop")
        self.mode_input.setEnabled(False)
        self.audio_input.setEnabled(False)
        self.camera_input.setEnabled(False)
        self.raw_input.setEnabled(False)

    def _stop(self):
        self.audio.stop_recording()
        self.record_btn.setEnabled(False)
        if self.mode_input.currentText() == MODE_VIDEO:
            self.cancel_btn.setEnabled(False)  # can't abandon an in-progress mux
            self._set_status("Finishing video...")
            self.finishing_video = True
            self.recorder.stop()  # continues in _on_video_stopped once the file is finalized
        else:
            self.accept()

    def _on_video_duration(self, ms):
        if self.video_start_time is None and ms > 0:
            self.video_start_time = time.perf_counter() - ms / 1000.0

    def _on_recorder_error(self):
        if self.recorder.error() != QMediaRecorder.Error.NoError:
            self._fail(f"Webcam recording failed:\n{self.recorder.errorString()}")

    def _on_video_stopped(self):
        self.finishing_video = False
        video_path = self.recorder.actualLocation().toLocalFile()
        offset = 0.0
        if self.video_start_time and self.audio.first_sample_time:
            offset = self.video_start_time - self.audio.first_sample_time
            if abs(offset) > MAX_SYNC_OFFSET:
                offset = 0.0
        self._set_status("Combining audio and video...")
        self.mux_thread = MuxThread(video_path, self.temp_audio, output_path("mp4"), offset)
        self.temp_audio = None
        self.mux_thread.done.connect(self._on_muxed)
        self.mux_thread.failed.connect(lambda err: self._fail(f"Couldn't combine audio and video:\n{err}"))
        self.mux_thread.start()

    def _on_muxed(self, path):
        self.output_path = path
        self.accept()

    def _fail(self, message):
        if self.audio:
            self.audio.stop_recording()
        QMessageBox.critical(self, "Recording failed", message)
        self.reject()

    def _tick(self):
        if self.finishing_video and self.recorder.recorderState() == QMediaRecorder.RecorderState.StoppedState:
            self._on_video_stopped()
        if not self.audio:
            return
        peak = self.audio.take_peak()
        db = 20 * np.log10(peak) if peak > 0 else METER_FLOOR_DB
        self.meter.setValue(int(np.clip((db - METER_FLOOR_DB) / -METER_FLOOR_DB * 100, 0, 100)))
        if self.audio.recording:
            elapsed = int(time.perf_counter() - self.record_started)
            self._set_status(f"Recording  {elapsed // 60}:{elapsed % 60:02d}")
            self.clip_label.setText("CLIPPING: lower input gain" if self.audio.clipped else "")

    # -- teardown --

    def reject(self):
        if self.mux_thread and self.mux_thread.isRunning():
            return  # the window close button lands here too
        super().reject()

    def done(self, result):
        self.meter_timer.stop()
        self.finishing_video = False
        if self.temp_audio:  # cancelled mid-recording: drop the partial audio, skip muxing
            if self.audio:
                self.audio.stop_recording()
            try:
                os.remove(self.temp_audio)
            except OSError:
                pass
            self.temp_audio = None
        if self.audio:
            self.audio.close()
            self.audio = None
        if self.recorder.recorderState() != QMediaRecorder.RecorderState.StoppedState:
            self.recorder.stop()
        if self.camera:
            self.camera.stop()
        if result != QDialog.DialogCode.Accepted:
            self.output_path = None
        super().done(result)

