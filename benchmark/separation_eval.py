"""Does isolating the guitar (Demucs) help? Needs torch + demucs installed (CI runs this).

    python benchmark/separation_eval.py [--stems DIR]

Compares notes found without isolation, with it, and with it plus "Over a band", on the
realistic recordings (held-out renders) and the original benchmark, plus two stress mixes.
"""

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import soundfile as sf

import analysis as an
import realistic as rl
import run
import separation

SEEDS = (11, 12, 13, 14, 15)


def found(result, notes, truth, steady):
    ev = rl.evaluate(result, notes, truth, steady)["notes"]
    return np.mean([x["found"] for x in ev]) * 100, ev


def main():
    stems_dir = sys.argv[sys.argv.index("--stems") + 1] if "--stems" in sys.argv else None
    if stems_dir:
        os.makedirs(stems_dir, exist_ok=True)
    t0 = time.time()
    model = separation.load_model()
    print(f"model {separation.MODEL} loaded in {time.time() - t0:.1f}s; sources {model.sources}", flush=True)
    seconds_audio = seconds_sep = 0.0

    def isolate(path):
        nonlocal seconds_audio, seconds_sep
        stereo = separation.stereo_from_file(path)
        start = time.time()
        guitar = separation.separate(stereo, model)
        seconds_sep += time.time() - start
        seconds_audio += stereo.shape[1] / separation.MODEL_RATE
        return guitar

    print("\nREALISTIC RECORDINGS (held-out renders), notes found")
    print(f"{'':36s}{'no isolation':>14s}{'isolated':>12s}{'isolated+band':>15s}")
    totals = {k: [] for k in ("plain", "iso", "iso_band")}
    for i, seed in enumerate(SEEDS):
        p, paths, truth, steady = rl.render_profile(rl.PROFILES[i], seed)
        for label in ("noisy", "clean"):
            y = an.extract_audio(paths[label])
            guitar = isolate(paths[label])
            if stems_dir and label == "noisy":
                sf.write(os.path.join(stems_dir, f"{i + 1}_isolated_guitar.wav"), guitar, separation.OUTPUT_RATE)
            plain, e1 = found(an.analyze_pitch(y, an.SAMPLE_RATE), p["notes"], truth, steady)
            iso, e2 = found(an.analyze_pitch(y, an.SAMPLE_RATE, isolated=guitar), p["notes"], truth, steady)
            iso_band, e3 = found(an.analyze_pitch(y, an.SAMPLE_RATE, isolated=guitar, band_mode=True),
                                 p["notes"], truth, steady)
            if label == "noisy":
                totals["plain"] += e1
                totals["iso"] += e2
                totals["iso_band"] += e3
            print(f"  {p['name'][:26] + ' ' + label:34s}{plain:13.1f}%{iso:11.1f}%{iso_band:14.1f}%", flush=True)
    pct = lambda rows: np.mean([x["found"] for x in rows]) * 100
    print(f"  {'ALL noisy':34s}{pct(totals['plain']):13.1f}%{pct(totals['iso']):11.1f}%{pct(totals['iso_band']):14.1f}%")

    print("\nSTRESS MIXES (lead + one instrument)")
    folder = tempfile.mkdtemp()
    for seed, which, label in ((2, 1, "lead + bass at equal level"), (4, 2, "lead + rhythm guitar, same register")):
        rng = np.random.default_rng(seed)
        p = rl.PROFILES[seed - 1](rng)
        raw, truth, steady = rl.lead_audio(p["notes"], p["seconds"], rng, tuning_cents=p["lead"]["tuning"])
        guitar_track = rl.room(rl.amp_chain(raw, p["lead"]["drive"], rng, p["lead"]["sympathetic"]), p["lead"]["wet"], rng)
        part = p["mix"](rng)[which][0][:len(guitar_track)]
        mix = guitar_track + part / rl.rms(part) * rl.rms(guitar_track[truth >= 0])
        wav = os.path.join(folder, f"stress_{seed}.wav")
        sf.write(wav, (mix / np.abs(mix).max() * 0.9).astype(np.float32), rl.SR)
        y = an.extract_audio(wav)
        guitar = isolate(wav)
        plain, _ = found(an.analyze_pitch(y, an.SAMPLE_RATE), p["notes"], truth, steady)
        band, _ = found(an.analyze_pitch(y, an.SAMPLE_RATE, band_mode=True), p["notes"], truth, steady)
        iso, _ = found(an.analyze_pitch(y, an.SAMPLE_RATE, isolated=guitar), p["notes"], truth, steady)
        print(f"  {label:38s} no isolation {plain:5.1f}%  band mode {band:5.1f}%  isolated {iso:5.1f}%", flush=True)

    print("\nORIGINAL BENCHMARK, note accuracy")
    for name, path, tn, ts, td in run.load_cases():
        y = an.extract_audio(path)
        guitar = isolate(path)
        r1 = an.analyze_pitch(y, an.SAMPLE_RATE)
        r2 = an.analyze_pitch(y, an.SAMPLE_RATE, isolated=guitar)
        s1 = run.score(r1.times, r1.notes, an.display_range(r1.f0), tn, ts, td)
        s2 = run.score(r2.times, r2.notes, an.display_range(r2.f0), tn, ts, td)
        print(f"  {name:18s} no isolation {s1['note_acc']:5.1f}% (false notes {s1['false_notes']:5.1f}%)   "
              f"isolated {s2['note_acc']:5.1f}% (false notes {s2['false_notes']:5.1f}%)", flush=True)

    print(f"\nSPEED: separation took {seconds_sep / seconds_audio:.2f} s per second of audio "
          f"on this machine ({os.cpu_count()} CPU threads)")


if __name__ == "__main__":
    main()
