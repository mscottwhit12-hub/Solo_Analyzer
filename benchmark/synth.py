"""Synthetic guitar solos with frame-exact ground truth, for measuring analysis accuracy.

The renderer imitates the things that make real solos hard to track: picked and legato
notes, whole/half-step bends, wide vibrato, slides, a slightly-off tuning, overdrive,
room reverb, background noise and lossy compression. Optionally adds drums and bass.
"""

import os
import subprocess
from dataclasses import dataclass

import imageio_ffmpeg
import numpy as np
import soundfile as sf

RENDER_SR = 44100
A_MINOR = [0, 2, 3, 5, 7, 8, 10]          # semitones above A
PENTATONIC = [0, 3, 5, 7, 10]
BENDS = {62: 64, 67: 69, 69: 71, 72: 74, 74: 76, 79: 81, 81: 83, 84: 86, 76: 77}  # source -> target


def midi_to_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


def scale_notes(low, high, degrees):
    return [m for m in range(low, high + 1) if (m - 69) % 12 in degrees]


@dataclass
class Truth:
    pitch: np.ndarray    # continuous intended pitch (midi, fractional) per render sample, nan = silent
    note: np.ndarray     # intended note (int midi) per render sample, -1 = silent
    stable: np.ndarray   # False during bend/slide transitions and attacks
    double: np.ndarray   # True where a double stop is sounding (truth is its top note)


def compose(seed, seconds, tonic=69, degrees=A_MINOR, strong=PENTATONIC, triad=(0, 3, 7), bends=True):
    """Random solo as a list of events, built from phrases like a real solo.

    Notes come from `degrees` (semitones above `tonic`), favouring the `strong` ones. Each
    phrase ends on a held tonic-chord tone (`triad`), the way solos resolve.
    """
    rng = np.random.default_rng(seed)
    pool = [m for m in range(57, 87) if (m - tonic) % 12 in degrees]
    weights = np.array([3.0 if (m - tonic) % 12 in strong else 1.0 for m in pool])
    weights *= np.exp(-((np.array(pool) - 72) / 9.0) ** 2)  # centred on the 5th-8th position
    weights /= weights.sum()
    resolutions = [m for m in pool if (m - tonic) % 12 in triad]
    res_weights = np.array([2.0 if (m - tonic) % 12 == 0 else 1.0 for m in resolutions])
    events, t, current = [], 0.3, min(pool, key=lambda m: abs(m - 69))
    phrase_left = rng.integers(4, 11)
    while t < seconds:
        if phrase_left == 0:  # resolve on a held chord tone, then rest
            near = [m for m in resolutions if abs(m - current) <= 7] or resolutions
            w = np.array([res_weights[resolutions.index(m)] for m in near])
            note = int(rng.choice(near, p=w / w.sum()))
            dur = rng.uniform(0.5, 1.2)
            events.append(dict(kind="picked", note=note, start=t, dur=dur, double_stop=False,
                               vibrato=rng.uniform(25, 50), bend_time=0.2, release=False, slide_from=note))
            current = note
            t += dur + rng.uniform(0.3, 0.9)
            phrase_left = rng.integers(4, 11)
            continue
        phrase_left -= 1
        # mostly stepwise motion
        near = [m for m in pool if abs(m - current) <= 5] or pool
        w = np.array([weights[pool.index(m)] for m in near])
        note = int(rng.choice(near, p=w / w.sum()))
        r = rng.random()
        if r < 0.18 and bends and note in BENDS:
            kind, dur = "bend", rng.uniform(0.6, 1.4)
        elif r < 0.26:
            kind, dur = "slide", rng.uniform(0.25, 0.5)
        elif r < 0.50:
            kind, dur = "legato", rng.uniform(0.11, 0.25)
        else:
            kind, dur = "picked", rng.choice([rng.uniform(0.12, 0.25), rng.uniform(0.3, 0.9)])
        events.append(dict(kind=kind, note=note, start=t, dur=dur,
                           double_stop=kind == "picked" and rng.random() < 0.12,
                           vibrato=rng.uniform(25, 50) if dur > 0.45 else 0.0,
                           bend_time=rng.uniform(0.12, 0.4), release=rng.random() < 0.3,
                           slide_from=note - int(rng.choice([2, 3, 5])) * (1 if rng.random() < 0.6 else -1)))
        current = note
        t += dur
    return events


def event_frames(events, seconds, hop_seconds):
    """Per-frame intended note (-1 = silent) for a list of events, without rendering audio."""
    notes = np.full(int(seconds / hop_seconds) + 1, -1, dtype=int)
    for ev in events:
        a, b = int(ev["start"] / hop_seconds), int((ev["start"] + ev["dur"]) / hop_seconds)
        notes[a:b] = ev["note"]
        if ev["kind"] == "bend":
            notes[a + int(0.12 / hop_seconds):b] = BENDS[ev["note"]]
    return notes


def render(events, seconds, tuning_cents=-10.0, drive=6.0, reverb=0.25, noise_db=-42, seed=0):
    rng = np.random.default_rng(seed + 1)
    n = int(seconds * RENDER_SR)
    pitch = np.full(n, np.nan)
    second = np.full(n, np.nan)  # lower note of a double stop (truth stays the top note)
    note = np.full(n, -1, dtype=int)
    stable = np.zeros(n, dtype=bool)
    amp = np.zeros(n)
    plucks = []

    for i, ev in enumerate(events):
        a, b = int(ev["start"] * RENDER_SR), min(n, int((ev["start"] + ev["dur"]) * RENDER_SR))
        if b <= a:
            continue
        t = np.arange(b - a) / RENDER_SR
        p = np.full(b - a, float(ev["note"]))
        lab = np.full(b - a, ev["note"])
        st = t > 0.03
        if ev["kind"] == "bend":
            target, bt = BENDS[ev["note"]], ev["bend_time"]
            hold = 0.12
            rise = np.clip((t - hold) / bt, 0, 1)
            p = ev["note"] + (target - ev["note"]) * (0.5 - 0.5 * np.cos(np.pi * rise))
            lab = np.where(t < hold, ev["note"], target)
            st = (t > 0.03) & ((t < hold) | (t > hold + bt))
            if ev["release"]:
                rel_start = ev["dur"] - 0.2
                fall = np.clip((t - rel_start) / 0.15, 0, 1)
                p = p - (target - ev["note"]) * fall
                lab = np.where(t > rel_start, ev["note"], lab)
                st &= (t < rel_start) | (t > rel_start + 0.15)
        elif ev["kind"] == "slide":
            glide = np.clip(t / 0.08, 0, 1)
            p = ev["slide_from"] + (ev["note"] - ev["slide_from"]) * glide
            st = t > 0.1
        if ev["vibrato"]:
            ramp = np.clip((t - 0.15) / 0.2, 0, 1)
            p = p + ramp * ev["vibrato"] / 100 * np.sin(2 * np.pi * 5.5 * t)
        pitch[a:b], note[a:b], stable[a:b] = p, lab, st
        if ev.get("double_stop"):
            second[a:b] = p - 5  # a fourth below, as in typical rock double stops

        legato = ev["kind"] == "legato" and i > 0 and events[i - 1]["start"] + events[i - 1]["dur"] >= ev["start"] - 1e-6
        if legato:
            level = amp[a - 1] if a > 0 else 0.6
            amp[a:b] = level * np.exp(-t * 1.2)
        else:
            amp[a:b] = np.minimum(1, t / 0.004) * np.exp(-t * 2.2)
            plucks.append(a)

    # Amped/miked guitar: the 2nd harmonic is often stronger than the fundamental,
    # which is what lures pitch trackers into octave errors.
    harmonic_amps = [0.45, 1.0, 0.8, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1]

    def string(p, level):
        voiced = ~np.isnan(p)
        freq = np.where(voiced, midi_to_hz(np.nan_to_num(p, nan=69) + tuning_cents / 100), 0)
        phase = 2 * np.pi * np.cumsum(freq) / RENDER_SR
        out = np.zeros(n)
        for k, h in enumerate(harmonic_amps, start=1):
            inharm = np.sqrt(1 + 1e-4 * k * k)
            partial = np.sin(k * inharm * phase + k)
            partial[(freq * k * inharm > RENDER_SR / 2 - 500) | ~voiced] = 0
            out += h * partial
        return out * level

    x = (string(pitch, 1.0) + string(second, 0.8)) * amp
    for a in plucks:  # pick noise
        m = min(n - a, int(0.01 * RENDER_SR))
        x[a:a + m] += 0.3 * rng.standard_normal(m) * np.exp(-np.arange(m) / (0.002 * RENDER_SR))
    x = np.tanh(drive * x) / np.tanh(drive)
    ir_len = int(1.2 * RENDER_SR)
    ir = rng.standard_normal(ir_len) * np.exp(-np.arange(ir_len) / RENDER_SR * 5.0)
    ir /= np.sqrt(np.sum(ir ** 2))
    x = x + reverb * np.convolve(x, ir)[:n]
    x = x / np.max(np.abs(x)) * 0.8
    x += 10 ** (noise_db / 20) * np.cumsum(rng.standard_normal(n)) / 300  # brownish room noise
    return x, Truth(pitch, note, stable, ~np.isnan(second))


def backing(seconds, seed=0):
    """Drums + bass on Am-G-F at 98 bpm."""
    rng = np.random.default_rng(seed + 7)
    n = int(seconds * RENDER_SR)
    y = np.zeros(n)
    beat = 60 / 98
    for k in range(int(seconds / beat * 2)):
        a = int(k * beat / 2 * RENDER_SR)
        if a >= n:
            break
        m = min(n - a, int(0.25 * RENDER_SR))
        t = np.arange(m) / RENDER_SR
        if k % 4 == 0:
            y[a:a + m] += 0.9 * np.sin(2 * np.pi * (50 + 60 * np.exp(-t * 30)) * t) * np.exp(-t * 12)
        if k % 4 == 2:
            y[a:a + m] += 0.5 * rng.standard_normal(m) * np.exp(-t * 18)
        hat = min(m, int(0.05 * RENDER_SR))
        y[a:a + hat] += 0.12 * np.diff(rng.standard_normal(hat + 1)) * np.exp(-np.arange(hat) / RENDER_SR * 60)
    bar = 4 * beat
    for k, root in enumerate([45, 43, 41, 41] * int(seconds / bar / 4 + 1)):
        a, b = int(k * bar * RENDER_SR), min(n, int((k + 1) * bar * RENDER_SR))
        if a >= n:
            break
        t = np.arange(b - a) / RENDER_SR
        f = midi_to_hz(root)
        y[a:b] += 0.35 * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 0.8)
    return y


def make_case(path, seed, seconds=45.0, with_backing=False):
    """Render a case to an AAC .m4a (like a downloaded video). Returns ground truth."""
    events = compose(seed, seconds - 1)
    guitar, truth = render(events, seconds, seed=seed)
    mix = guitar + (0.7 * backing(seconds, seed) if with_backing else 0)
    mix = mix / np.max(np.abs(mix)) * 0.9
    wav = path + ".wav"
    sf.write(wav, mix.astype(np.float32), RENDER_SR)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", wav, "-c:a", "aac", "-b:a", "128k", path],
                   capture_output=True, check=True)
    os.remove(wav)
    return truth
