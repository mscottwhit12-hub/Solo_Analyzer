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
3. librosa's pYIN tracks pitch between C2 and E6 (24th fret on the high E string), using 46 ms windows so fast runs stay sharp. It works in 10-second chunks so progress can be shown.
4. **Tuning:** the app measures how far the guitar is from A440 and judges notes relative to that. The offset is shown next to the detected key.
5. **Note tracking** turns the raw pitch into the notes being played:
   - **Vibrato** stays on its note. The note only changes once the pitch moves 0.7 semitones away.
   - **Bends and slides** count as the note they land on, so a D bent up to E never registers as D♯.
   - **Octave errors** get corrected. Distorted guitars and double stops can fool the tracker into reporting a note one or two octaves (or an octave and a fifth) too low. A passage that sits far outside the surrounding register is folded back. The spectrum decides which correction is right.
   - Blips shorter than 45 ms are dropped, and brief dropouts inside a held note are filled.
6. The readout, the breakdown bar and key detection all use these notes, so they always agree. The key is estimated with Krumhansl–Kessler key profiles.

The analysis works best on isolated or lightly backed guitar. With a full band, a loud bass can still be mistaken for the guitar while it rests.

## Accuracy benchmark

`benchmark/` renders synthetic solos where the true note at every moment is known. They include bends, wide vibrato, slides, legato, double stops, a guitar tuned 10¢ flat, overdrive, room reverb, noise and AAC compression, with and without drums and bass. To score the current analysis:

```bash
python benchmark/run.py
```

On 5 solos that weren't used to tune it, the note-tracking work changed results as follows:

| | Before | After |
|---|---|---|
| Steady notes identified correctly | 86.5% | 93.4% |
| "Out" share (ideal ≈ 0) | 3.9% | 1.9% |
| Breakdown bar error | 7.7% | 4.6% |
| Graph range beyond what's played | +2.0 octaves | +0.4 octaves |

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
