# PyInstaller spec for the guitar isolation pack (downloaded by the app on first use).
# Build with:  pyinstaller GuitarSeparator.spec   (expects the model files in build/models)
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ["separation.py"],
    datas=collect_data_files("demucs") + [("build/models", "models")],
    hiddenimports=collect_submodules("demucs"),
    excludes=["PyQt6", "librosa", "numba", "matplotlib", "tkinter", "IPython", "imageio_ffmpeg"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, exclude_binaries=True, name="GuitarSeparator", console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="GuitarSeparator", upx=False)
