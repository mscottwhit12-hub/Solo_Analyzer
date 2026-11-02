# PyInstaller build spec. Build with:  pyinstaller SoloAnalyzer.spec
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = collect_data_files("librosa")  # lazy-loader .pyi stubs and data registry
binaries = []
hiddenimports = []
for pkg in ("imageio_ffmpeg",):  # bundled ffmpeg binary
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["solo_analyzer.py"],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
)
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    # macOS: a regular .app bundle (onefile .app bundles break permission prompts
    # and are deprecated). It still downloads as one zipped app.
    exe = EXE(pyz, a.scripts, exclude_binaries=True, name="SoloAnalyzer", console=False, upx=False)
    bundle_input = COLLECT(exe, a.binaries, a.datas, name="SoloAnalyzer", upx=False)
else:
    # Windows/Linux: one self-contained executable.
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, name="SoloAnalyzer", console=False, upx=False)
    bundle_input = None

if bundle_input is not None:
    # The usage descriptions are required: without them macOS silently denies
    # microphone and camera access.
    app = BUNDLE(
        bundle_input,
        name="SoloAnalyzer.app",
        bundle_identifier="io.github.mscottwhit12-hub.soloanalyzer",
        info_plist={
            "NSMicrophoneUsageDescription": "Solo Analyzer records your guitar to analyze its pitch.",
            "NSCameraUsageDescription": "Solo Analyzer can record video of your playing alongside the audio.",
            "NSHighResolutionCapable": True,
        },
    )
