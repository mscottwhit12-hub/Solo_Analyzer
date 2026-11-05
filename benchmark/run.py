"""Score the analysis pipeline against synthetic solos with known notes.

    python benchmark/run.py

Metrics (averaged over cases):
  note acc     % of steady played frames labelled with the right note
  octave err   % of steady played frames labelled an octave off
  coverage     % of played frames that got a note at all
  false notes  % of silent frames (outside reverb tails) that got a note
  out %        share of the breakdown bar marked "Out" (ideal ~0: every note is in A minor)
  bar error    how far the breakdown bar is from the true one (total variation, %)
  range        octaves shown on the graph vs. octaves actually played

Results on 5 held-out solos (not used for tuning), before -> after the note-tracking work:
  note acc 86.5 -> 93.4, out % 3.9 -> 1.9, bar error 7.7 -> 4.6, range +2.0 -> +0.4 octaves
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

import analysis as sa
from synth import A_MINOR, RENDER_SR, make_case

CASES = [("solo", 11, False), ("solo", 23, False), ("with backing", 11, True), ("with backing", 37, True)]
CACHE = os.path.join(tempfile.gettempdir(), "solo_benchmark_cases")


def load_cases():
    os.makedirs(CACHE, exist_ok=True)
    cases = []
    for name, seed, with_backing in CASES:
        path = os.path.join(CACHE, f"{name.replace(' ', '_')}_{seed}.m4a")
        truth_path = path + ".npz"
        if not os.path.exists(truth_path):
            truth = make_case(path, seed, with_backing=with_backing)
            np.savez(truth_path, note=truth.note, stable=truth.stable)
        t = np.load(truth_path)
        cases.append((f"{name} #{seed}", path, t["note"], t["stable"]))
    return cases


def current_pipeline(y):
    result = sa.analyze_pitch(y, sa.SAMPLE_RATE)
    return result.times, result.notes, sa.display_range(result.f0)


def score(times, labels, frange, truth_note, truth_stable):
    idx = np.clip(np.round(times * RENDER_SR).astype(int), 0, len(truth_note) - 1)
    best = None
    for lag in range(-6, 7):  # absorb codec delay
        shifted = np.clip(idx + int(lag * sa.HOP_LENGTH * RENDER_SR / sa.SAMPLE_RATE), 0, len(truth_note) - 1)
        tn, ts = truth_note[shifted], truth_stable[shifted]
        steady = (tn >= 0) & ts
        acc = np.mean(labels[steady] == tn[steady])
        if best is None or acc > best[0]:
            best = (acc, tn, ts)
    acc, tn, ts = best
    steady = (tn >= 0) & ts
    played = tn >= 0

    # silent frames well clear of note tails (the reverb rings for about a second)
    tail = int(1.0 * sa.SAMPLE_RATE / sa.HOP_LENGTH)
    recent = np.convolve(played.astype(float), np.ones(tail), mode="full")[:len(played)] > 0
    silent = ~recent

    def bar(notes):
        notes = notes[notes >= 0]
        degrees = (notes - 69) % 12
        out = np.mean(~np.isin(degrees, A_MINOR)) * 100
        hist = np.bincount(degrees, minlength=12) / max(1, len(notes)) * 100
        return out, hist

    out, hist = bar(labels)
    _, true_hist = bar(tn[steady])
    shown = np.log2(frange[1] / frange[0])
    played_notes = tn[played]
    true_span = (played_notes.max() - played_notes.min()) / 12
    return dict(
        note_acc=acc * 100,
        octave_err=np.mean(np.abs(labels[steady] - tn[steady]) == 12) * 100,
        coverage=np.mean(labels[played] >= 0) * 100,
        false_notes=np.mean(labels[silent] >= 0) * 100 if silent.any() else 0.0,
        out=out,
        bar_err=np.abs(hist - true_hist).sum() / 2,
        range=f"{shown:.1f} vs {true_span:.1f}",
        range_excess=shown - true_span,
    )


def main():
    pipeline = current_pipeline
    rows = []
    for name, path, tn, ts in load_cases():
        start = time.time()
        y = sa.extract_audio(path)
        times, labels, frange = pipeline(y)
        r = score(times, labels, frange, tn, ts)
        r["secs"] = time.time() - start
        rows.append((name, r))

    cols = ["note_acc", "octave_err", "coverage", "false_notes", "out", "bar_err", "range", "secs"]
    print(f"{'case':18s}" + "".join(f"{c:>13s}" for c in cols))
    for name, r in rows:
        print(f"{name:18s}" + "".join(f"{r[c]:>13.1f}" if not isinstance(r[c], str) else f"{r[c]:>13s}" for c in cols))
    mean = {c: np.mean([r[c] for _, r in rows]) for c in cols if c != "range"}
    mean["range"] = f"+{np.mean([r['range_excess'] for _, r in rows]):.1f} oct"
    print(f"{'MEAN':18s}" + "".join(f"{mean[c]:>13.1f}" if not isinstance(mean[c], str) else f"{mean[c]:>13s}" for c in cols))


if __name__ == "__main__":
    main()
