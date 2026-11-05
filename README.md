# Solo Analyzer

Open a video or audio file of a guitar solo, or record one straight into the app, and see what the notes are doing against a chosen key. The app tracks the pitch over time and shows:

- **Live readout** of the current note, how many cents sharp or flat it is (handy for checking bends), and its scale degree
- **Breakdown bar** showing how much of the solo sits on each scale degree, and how much is outside the scale
- **Scrolling pitch graph** of the last 30 seconds, drawn against the scale-degree lines. Click it to jump to that moment.

Scales: major, minor, minor and major pentatonic, blues, Dorian and Mixolydian, in all 12 keys. The key is detected automatically when a file finishes analyzing. Untick **Auto-detect key** to set it yourself.

### Practice controls

- **Speed:** play at 25%, 50%, 75% or 100% without changing pitch.
- **Keyboard:** <kbd>Space</kbd> play/pause, <kbd>←</kbd>/<kbd>→</kbd> skip 5 s, <kbd>Shift</kbd>+<kbd>←</kbd>/<kbd>→</kbd> skip 1 s, <kbd>Home</kbd> back to the start.
- **Drag and drop** a video or audio file onto the window to open it.
- Analysis shows progress and can be cancelled. Results are cached, so reopening a file is instant.
- The app remembers your key, scale, speed and window size.

## Recording

Click **Record** to capture from your own hardware:

- **Audio only:** saves a 24-bit WAV to `Music/Solo Analyzer`.
- **Webcam + audio:** saves an MP4 to `Videos/Solo Analyzer` (`Movies` on macOS).

Pick any input: an audio interface, USB guitar cable, mic or webcam mic. A level meter shows your signal and warns if it clips. When you stop, the recording loads into the analyzer automatically.

**Raw input is on by default.** On Windows this bypasses the system's audio processing (automatic gain, noise suppression and "enhancements"), so the analyzer gets the tone as the device sends it. Raw mode only takes effect if the device's driver supports it, which most current interfaces and built-in mics do. macOS records input unprocessed anyway. Some USB webcams and headsets apply gain control inside the device itself, and no software can switch that off. For the cleanest signal, use an audio interface or guitar cable.

![Screenshot](docs/screenshot.png)

## Download

Get the latest build from [Releases](../../releases):

- **Windows:** `SoloAnalyzer-windows.exe`. Windows SmartScreen may warn because the app isn't signed. Click *More info → Run anyway*.
- **macOS (Apple Silicon):** `SoloAnalyzer-macos.zip`. Unzip it, then right-click `SoloAnalyzer.app` → *Open* the first time, because the app isn't signed.
  macOS asks for microphone and camera access the first time you record.

On Windows, each launch takes a few seconds while the app unpacks itself.

## Run from source

Requires Python 3.13.

```bash
pip install -r requirements.txt
python solo_analyzer.py
```

## How it works

1. ffmpeg (bundled through `imageio-ffmpeg`) extracts mono audio at 22.05 kHz.
   Recordings capture audio through PortAudio (`sounddevice`) and webcam video through Qt, then ffmpeg combines them, correcting for the difference in start times.
2. Harmonic/percussive separation removes most of the drums.
3. librosa's pYIN tracks the fundamental frequency between C2 and E6 (24th fret on the high E string), in 10-second chunks so progress can be shown.
4. A drift filter treats a large pitch jump with no pick attack as detection jitter, unless the pitch was already sliding that way. A median filter then smooths the result.
5. Each pitch is rounded to the nearest note to get its scale degree. The readout and the breakdown bar use the same rule. The key is estimated by matching how long each note is played against standard major and minor key profiles (Krumhansl–Kessler).

The analysis works best on isolated or lightly backed guitar.

## Tests

```bash
pip install pytest
python -m pytest tests
```

## Building

Every push to `main` builds Windows and macOS executables with PyInstaller, using `SoloAnalyzer.spec`. You can download them from the run's page in the **Actions** tab. To publish a release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

## License

MIT. See [LICENSE](LICENSE).
