"""Score key detection on synthetic solos in every key and common scale type.

    python benchmark/keys.py

Solos are generated as note sequences (no audio), so this isolates key detection from
pitch tracking. Each phrase resolves on a tonic-chord tone, as real solos tend to.
"""

import os
import sys
from collections import Counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import analysis as an
from synth import compose, event_frames

# name: (scale degrees, favoured degrees, tonic chord, expected mode)
MODES = {
    "natural minor":    ([0, 2, 3, 5, 7, 8, 10], [0, 3, 5, 7, 10], (0, 3, 7), "Minor"),
    "major":            ([0, 2, 4, 5, 7, 9, 11], [0, 2, 4, 7, 9], (0, 4, 7), "Major"),
    "minor pentatonic": ([0, 3, 5, 7, 10], [0, 3, 5, 7, 10], (0, 3, 7), "Minor"),
    "major pentatonic": ([0, 2, 4, 7, 9], [0, 2, 4, 7, 9], (0, 4, 7), "Major"),
    "blues":            ([0, 3, 5, 6, 7, 10], [0, 3, 5, 7, 10], (0, 3, 7), "Minor"),
    "dorian":           ([0, 2, 3, 5, 7, 9, 10], [0, 3, 5, 7, 10], (0, 3, 7), "Minor"),
    "mixolydian":       ([0, 2, 4, 5, 7, 9, 10], [0, 2, 4, 7, 10], (0, 4, 7), "Major"),
}
SEEDS = (1, 2, 3)
SECONDS = 40.0


def cases():
    for mode, (degrees, strong, triad, expected) in MODES.items():
        for root in range(12):
            for seed in SEEDS:
                tonic = 60 + root
                events = compose(seed * 100 + root, SECONDS, tonic=tonic, degrees=degrees,
                                 strong=strong, triad=triad, bends=False)
                notes = event_frames(events, SECONDS, an.HOP_LENGTH / an.SAMPLE_RATE)
                yield mode, (an.NOTE_NAMES[tonic % 12], expected), notes


def main(detect=None):
    detect = detect or an.detect_key
    per_mode, mistakes = Counter(), Counter()
    totals = Counter()
    for mode, expected, notes in cases():
        got = detect(notes)
        totals[mode] += 1
        if got == expected:
            per_mode[mode] += 1
        else:
            relation = describe(expected, got)
            mistakes[relation] += 1
    print(f"{'scale type':18s} correct")
    for mode in MODES:
        print(f"{mode:18s} {per_mode[mode]:3d}/{totals[mode]}")
    total = sum(per_mode.values())
    print(f"{'ALL':18s} {total:3d}/{sum(totals.values())}  ({total / sum(totals.values()):.0%})")
    if mistakes:
        print("mistakes:", dict(mistakes.most_common()))
    return total / sum(totals.values())


def describe(expected, got):
    if got is None:
        return "no answer"
    (er, em), (gr, gm) = expected, got
    interval = (an.NOTE_NAMES.index(gr) - an.NOTE_NAMES.index(er)) % 12
    if em == "Minor" and gm == "Major" and interval == 3:
        return "relative major"
    if em == "Major" and gm == "Minor" and interval == 9:
        return "relative minor"
    if interval == 0:
        return "right root, wrong mode"
    return f"{gm.lower()} {interval} semitones up"


if __name__ == "__main__":
    main()
