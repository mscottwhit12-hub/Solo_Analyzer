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


VARIANTS = {
    "guitar stem": ("guitar",),
    "minus drums+bass": ("guitar", "other", "vocals", "piano"),
    "guitar+other": ("guitar", "other"),
}


def confident_pick(a, b, smooth_frames=9):
    """Per frame, take the notes of whichever analysis the tracker was more confident in."""
    kernel = np.ones(smooth_frames) / smooth_frames
    ca = np.convolve(a.confidence, kernel, mode="same")
    cb = np.convolve(b.confidence, kernel, mode="same")
    use_a = ca >= cb
    return an.AnalysisResult(np.where(use_a, a.f0, b.f0), a.times, np.where(use_a, a.notes, b.notes), a.tuning,
                             raw_f0=np.where(use_a, a.raw_f0, b.raw_f0))


def analyses(y, stems):
    """Every variant's analysis of one recording."""
    out = {"no isolation": an.analyze_pitch(y, an.SAMPLE_RATE)}
    for name, keep in VARIANTS.items():
        out[name] = an.analyze_pitch(y, an.SAMPLE_RATE, isolated=sum(stems[k] for k in keep))
    out["pick: plain|guitar"] = confident_pick(out["no isolation"], out["guitar stem"])
    out["pick: minus-rhythm|guitar"] = confident_pick(out["minus drums+bass"], out["guitar stem"])
    return out


def main():
    stems_dir = sys.argv[sys.argv.index("--stems") + 1] if "--stems" in sys.argv else None
    if stems_dir:
        os.makedirs(stems_dir, exist_ok=True)
    t0 = time.time()
    model = separation.load_model()
    print(f"model {separation.MODEL} loaded in {time.time() - t0:.1f}s; sources {model.sources}", flush=True)
    timing = [0.0, 0.0]

    def stems_of(path):
        stereo = separation.stereo_from_file(path)
        start = time.time()
        stems = separation.separate_stems(stereo, model)
        timing[0] += time.time() - start
        timing[1] += stereo.shape[1] / separation.MODEL_RATE
        return stems

    names = ["no isolation", *VARIANTS, "pick: plain|guitar", "pick: minus-rhythm|guitar"]
    header = f"{'':32s}" + "".join(f"{n[:16]:>17s}" for n in names)
    print("\nREALISTIC RECORDINGS (held-out renders), notes found\n" + header)
    totals = {lab: {n: [] for n in names} for lab in ("noisy", "clean")}
    for i, seed in enumerate(SEEDS):
        p, paths, truth, steady = rl.render_profile(rl.PROFILES[i], seed)
        for label in ("noisy", "clean"):
            stems = stems_of(paths[label])
            if stems_dir and label == "noisy":
                for n, keep in VARIANTS.items():
                    sf.write(os.path.join(stems_dir, f"{i + 1}_{n.replace(' ', '_').replace('+', '_')}.wav"),
                             sum(stems[k] for k in keep), separation.OUTPUT_RATE)
            results = analyses(an.extract_audio(paths[label]), stems)
            cells = []
            for n in names:
                score, ev = found(results[n], p["notes"], truth, steady)
                totals[label][n] += ev
                cells.append(f"{score:16.1f}%")
            print(f"  {p['name'][:22] + ' ' + label:30s}" + "".join(cells), flush=True)
    pct = lambda rows: np.mean([x["found"] for x in rows]) * 100
    for label in ("noisy", "clean"):
        print(f"  {'ALL ' + label:30s}" + "".join(f"{pct(totals[label][n]):16.1f}%" for n in names))

    print("\nSTRESS MIXES (lead + one instrument at equal level)\n" + header)
    folder = tempfile.mkdtemp()
    for seed, which, label in ((2, 1, "lead + bass"), (4, 2, "lead + rhythm guitar")):
        rng = np.random.default_rng(seed)
        p = rl.PROFILES[seed - 1](rng)
        raw, truth, steady = rl.lead_audio(p["notes"], p["seconds"], rng, tuning_cents=p["lead"]["tuning"])
        lead = rl.room(rl.amp_chain(raw, p["lead"]["drive"], rng, p["lead"]["sympathetic"]), p["lead"]["wet"], rng)
        part = p["mix"](rng)[which][0][:len(lead)]
        mix = lead + part / rl.rms(part) * rl.rms(lead[truth >= 0])
        wav = os.path.join(folder, f"stress_{seed}.wav")
        sf.write(wav, (mix / np.abs(mix).max() * 0.9).astype(np.float32), rl.SR)
        results = analyses(an.extract_audio(wav), stems_of(wav))
        print(f"  {label:30s}" + "".join(f"{found(results[n], p['notes'], truth, steady)[0]:16.1f}%" for n in names), flush=True)

    print("\nORIGINAL BENCHMARK, note accuracy\n" + header)
    for name, path, tn, ts, td in run.load_cases():
        results = analyses(an.extract_audio(path), stems_of(path))
        cells = []
        for n in names:
            r = results[n]
            cells.append(f"{run.score(r.times, r.notes, an.display_range(r.f0), tn, ts, td)['note_acc']:16.1f}%")
        print(f"  {name:30s}" + "".join(cells), flush=True)

    print(f"\nSPEED: separation took {timing[0] / timing[1]:.2f} s per second of audio "
          f"on this machine ({os.cpu_count()} CPU threads)")


if __name__ == "__main__":
    main()
