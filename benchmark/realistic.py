"""Messy, realistic test recordings with exact note truth.

Five profiles, each a lead guitar part whose every note is known, rendered through an
amp/cab/room chain and mixed with real-world interference: a band with cymbals, rhythm
guitar in the same register, crowd and TV noise, mains hum, hiss, a phone microphone.
Each profile is also rendered as a "clean control" (same lead and amp, nothing else), so
failures can be pinned on the playing (speed, register) or on the interference.

    python benchmark/realistic.py              # evaluate all five
    python benchmark/realistic.py --write DIR  # also keep the rendered audio in DIR
"""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100


def hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69) / 12)


def band(x, lo=None, hi=None, order=4):
    if lo and hi:
        sos = butter(order, [lo, hi], btype="bandpass", fs=SR, output="sos")
    elif lo:
        sos = butter(order, lo, btype="highpass", fs=SR, output="sos")
    else:
        sos = butter(order, hi, btype="lowpass", fs=SR, output="sos")
    return sosfilt(sos, x)


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


# --- Lead guitar -------------------------------------------------------------

@dataclass
class Note:
    start: float
    dur: float
    midi: int
    kind: str = "pick"        # pick, legato, tap, bend, slide, ghost (muted, no pitch)
    target: int = None        # bend target
    vibrato: float = 0.0      # cents
    vel: float = 1.0
    double: int = None        # lower note of a double stop (truth stays the top note)


def lead_audio(notes, seconds, rng, tuning_cents=-6.0, intonation_cents=7.0):
    """Render the lead line. Returns (audio, per-sample truth note, per-sample steady mask)."""
    n = int(seconds * SR)
    out = np.zeros(n)
    truth = np.full(n, -1, dtype=np.int16)
    steady = np.zeros(n, dtype=bool)
    for i, note in enumerate(notes):
        a = int(note.start * SR)
        tail = 0.0 if (i + 1 < len(notes) and notes[i + 1].kind in ("legato", "tap", "slide")
                       and abs(notes[i + 1].start - (note.start + note.dur)) < 1e-3) else 0.06
        length = int((note.dur + tail) * SR)
        if a >= n:
            break
        length = min(length, n - a)
        t = np.arange(length) / SR
        if note.kind == "ghost":  # muted string: pick click and a dull thud, no pitch
            burst = rng.standard_normal(length) * np.exp(-t / 0.012)
            out[a:a + length] += 0.35 * note.vel * band(burst, 150, 2500, 2)
            continue

        detune = rng.normal(0, intonation_cents) + tuning_cents
        pitch = np.full(length, float(note.midi))
        label = np.full(length, note.midi, dtype=np.int16)
        settle = {"pick": 0.02, "legato": 0.012, "tap": 0.015, "bend": 0.02, "slide": 0.06}[note.kind]
        stable = t > settle
        if note.kind == "bend":
            hold, rise = 0.08, min(0.25, note.dur * 0.4)
            frac = np.clip((t - hold) / rise, 0, 1)
            pitch = note.midi + (note.target - note.midi) * (0.5 - 0.5 * np.cos(np.pi * frac))
            pitch -= np.where(frac >= 1, rng.uniform(0, 0.12), 0)  # bends often land a bit flat
            label = np.where(t < hold, note.midi, note.target).astype(np.int16)
            stable &= (t < hold) | (t > hold + rise)
        elif note.kind == "slide":
            glide = np.clip(t / 0.05, 0, 1)
            pitch = note.midi - 3 + 3 * glide
        if note.vibrato:
            ramp = np.clip((t - 0.12) / 0.15, 0, 1)
            pitch = pitch + ramp * note.vibrato / 100 * np.sin(2 * np.pi * rng.uniform(5, 6.5) * t)
        stable &= t < note.dur
        label[t >= note.dur] = -1

        voices = [(pitch, 1.0)] + ([(pitch - (note.midi - note.double), 0.8)] if note.double else [])
        tone = np.zeros(length)
        for vp, level in voices:
            f = hz(vp + detune / 100)
            pick_pos = rng.uniform(0.1, 0.22)
            for k in range(1, 30):
                fk = k * f * np.sqrt(1 + 1.5e-4 * k * k)
                if fk[0] > 10500:
                    break
                amp = abs(np.sin(np.pi * k * pick_pos)) / k ** 0.75
                decay = 1.5 * (f[0] / 220) ** 0.6 * (1 + 0.25 * (k - 1))   # high notes & partials die fast
                tone += level * amp * np.exp(-t * decay) * np.sin(2 * np.pi * np.cumsum(fk) / SR + rng.uniform(0, 6.28))
        attack = {"pick": 1.0, "bend": 1.0, "slide": 0.5, "legato": 0.45, "tap": 0.6}[note.kind]
        env = np.minimum(1, t / 0.003) * np.where(t < note.dur, 1.0, np.exp(-(t - note.dur) / 0.02))
        tone *= env * note.vel * attack ** 0.5
        if note.kind in ("pick", "bend", "tap"):  # pick / tap click
            m = min(length, int(0.008 * SR))
            tone[:m] += 0.25 * note.vel * band(rng.standard_normal(m), 1500, 7000, 2) * np.exp(-np.arange(m) / (0.002 * SR))
        if note.kind == "slide":  # string squeak
            m = min(length, int(0.06 * SR))
            tone[:m] += 0.05 * band(rng.standard_normal(m), 2500, 6000, 2)
        out[a:a + length] += tone
        truth[a:a + length] = np.where(label >= 0, label, truth[a:a + length])
        steady[a:a + length] |= stable
    return out, truth, steady


def amp_chain(x, drive, rng, sympathetic=0.0):
    """Preamp drive, speaker cabinet, open strings ringing sympathetically."""
    x = x / (np.max(np.abs(x)) + 1e-9)
    if sympathetic:
        t = np.arange(len(x)) / SR
        for m in (40, 45, 50, 55):  # E2 A2 D3 G3 open strings, swelling and fading
            swell = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 6))
            x += sympathetic * swell * np.sin(2 * np.pi * hz(m) * t) * (0.6 + 0.4 * np.sin(4 * np.pi * hz(m) * t))
    y = np.tanh(drive * (x + 0.05)) - np.tanh(drive * 0.05)
    y = band(y, 85, None, 2)
    y = band(y, None, 5200, 4) + 0.35 * band(y, 1800, 3200, 2)   # cab rolloff + presence bump
    return y / (np.max(np.abs(y)) + 1e-9)


def room(x, wet, rng, rt=0.8):
    m = int(rt * SR)
    ir = rng.standard_normal(m) * np.exp(-np.arange(m) / SR * 6.9 / rt)
    for d in (0.011, 0.017, 0.029):  # early reflections
        ir[int(d * SR)] += 6.0
    ir /= np.sqrt(np.sum(ir ** 2))
    return x + wet * np.convolve(x, ir)[:len(x)]


# --- Interference ------------------------------------------------------------

def drums(seconds, bpm, rng, ride=False, hats=True, crash_every_bars=2):
    n = int(seconds * SR)
    y = np.zeros(n)
    beat = 60 / bpm
    steps = int(seconds / (beat / 4))
    for k in range(steps):
        a = int(k * beat / 4 * SR)
        m = min(n - a, int(0.4 * SR))
        if m <= 0:
            break
        t = np.arange(m) / SR
        if k % 8 == 0 or (k % 16 == 10 and rng.random() < 0.5):
            y[a:a + m] += 0.9 * np.sin(2 * np.pi * (48 + 70 * np.exp(-t * 35)) * t) * np.exp(-t * 9)
        if k % 16 in (4, 12):
            y[a:a + m] += 0.55 * band(rng.standard_normal(m), 200, 9000, 2) * np.exp(-t * 16)
            y[a:a + m] += 0.3 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25)
        if hats and k % 2 == 0:
            h = min(m, int(0.06 * SR))
            y[a:a + h] += 0.18 * band(rng.standard_normal(h), 6000, 10500, 2) * np.exp(-np.arange(h) / SR * 60)
        if ride and k % 4 == 0:
            r = min(n - a, int(1.2 * SR))
            tr = np.arange(r) / SR
            partials = sum(np.sin(2 * np.pi * f * tr) for f in rng.uniform(2500, 9000, 12)) / 12
            y[a:a + r] += 0.12 * (partials + 0.5 * band(rng.standard_normal(r), 3000, 10000, 2)) * np.exp(-tr * 3)
        if crash_every_bars and k % (16 * crash_every_bars) == 0:
            r = min(n - a, int(2.5 * SR))
            tr = np.arange(r) / SR
            y[a:a + r] += 0.35 * band(rng.standard_normal(r), 1500, 10500, 2) * np.exp(-tr * 1.4)
    return y


def bass(seconds, bpm, roots, rng):
    n = int(seconds * SR)
    y = np.zeros(n)
    eighth = 60 / bpm / 2
    for k in range(int(seconds / eighth)):
        a = int(k * eighth * SR)
        m = min(n - a, int(eighth * SR))
        t = np.arange(m) / SR
        f = hz(roots[(k // 8) % len(roots)])
        y[a:a + m] += (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t) + 0.25 * np.sin(6 * np.pi * f * t)) * np.exp(-t * 4)
    return y


def rhythm_guitar(seconds, bpm, chords, rng, drive=6.0):
    """Distorted chords strummed in eighths; chords are lists of MIDI notes (same register as leads)."""
    n = int(seconds * SR)
    y = np.zeros(n)
    eighth = 60 / bpm / 2
    for k in range(int(seconds / eighth)):
        a = int(k * eighth * SR)
        m = min(n - a, int(eighth * SR))
        t = np.arange(m) / SR
        muted = k % 2 == 1
        for j, note in enumerate(chords[(k // 8) % len(chords)]):
            f = hz(note + rng.normal(0, 5) / 100)
            s = int(j * 0.006 * SR)  # strum spread
            tone = sum(np.sin(2 * np.pi * h * f * t + rng.uniform(0, 6)) / h for h in range(1, 8))
            y[a + s:a + m] += (tone * np.exp(-t * (14 if muted else 3)))[:m - s]
    y = np.tanh(drive * y / (np.max(np.abs(y)) + 1e-9))
    return band(band(y, 90, None, 2), None, 5000, 4)


def crowd(seconds, rng):
    n = int(seconds * SR)
    y = band(np.cumsum(rng.standard_normal(n)) * 0.02 + rng.standard_normal(n), 200, 4000, 2)
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.15 * np.arange(n) / SR + 1.0)
    for _ in range(int(seconds / 3)):  # shouts / whoops
        a = rng.integers(0, max(1, n - SR))
        m = int(rng.uniform(0.3, 0.9) * SR)
        t = np.arange(m) / SR
        f = rng.uniform(250, 600) * (1 + 0.3 * t)
        y[a:a + m] += 3 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) * (1 + band(rng.standard_normal(m), 300, 3000, 2))
    return y * swell


def tv_speech(seconds, rng):
    """Pitched, voice-like babble: glottal pulses through moving formants."""
    n = int(seconds * SR)
    y = np.zeros(n)
    t0 = 0.0
    while t0 < seconds:
        syl = rng.uniform(0.12, 0.3)
        a, m = int(t0 * SR), int(syl * SR)
        if a + m >= n:
            break
        t = np.arange(m) / SR
        f0 = rng.uniform(100, 230) * (1 + 0.15 * np.sin(2 * np.pi * 3 * t))
        pulses = np.sign(np.sin(2 * np.pi * np.cumsum(f0) / SR)) * 0.5
        voiced = sum(band(pulses, fc * 0.8, fc * 1.2, 2) for fc in rng.uniform([300, 900, 2200], [800, 1800, 3000]))
        y[a:a + m] += voiced * np.sin(np.pi * t / syl) ** 2
        t0 += syl + (rng.uniform(0.2, 0.8) if rng.random() < 0.2 else 0.02)
    return y


def hum(seconds, base=60.0):
    t = np.arange(int(seconds * SR)) / SR
    return sum(np.sin(2 * np.pi * base * k * t) / k for k in (1, 2, 3, 4, 5))


def hiss(seconds, rng):
    return band(rng.standard_normal(int(seconds * SR)), 1000, None, 2)


def fan(seconds, rng):
    return band(rng.standard_normal(int(seconds * SR)), 80, 1500, 2)


def phone_mic(x, rng):
    """Small mic, aggressive auto-gain and limiting."""
    x = band(x, 300, 3800, 3)
    env = np.sqrt(np.convolve(x ** 2, np.ones(4410) / 4410, mode="same")) + 1e-4
    x = x / env ** 0.7  # auto gain
    x = np.clip(x / np.percentile(np.abs(x), 99.5), -1, 1)
    return x


# --- Phrase helpers ------------------------------------------------------------

def scale_notes(tonic, degrees, low, high):
    return [m for m in range(low, high + 1) if (m - tonic) % 12 in degrees]


def run_phrase(start, pool, idx_from, idx_to, step_seconds, kind_cycle=("pick",), vel=0.9):
    """Scale run from pool[idx_from] to pool[idx_to] (inclusive), one note per step."""
    direction = 1 if idx_to >= idx_from else -1
    notes, t = [], start
    for j, i in enumerate(range(idx_from, idx_to + direction, direction)):
        notes.append(Note(t, step_seconds, pool[i], kind_cycle[j % len(kind_cycle)], vel=vel))
        t += step_seconds
    return notes, t


def sequence_of_fours(start, pool, i0, groups, step, kinds=("pick",), up=True):
    """The classic '1234 2345 3456' scale sequence."""
    notes, t = [], start
    for g in range(groups):
        for j in range(4):
            i = i0 + g + j if up else i0 - g - j
            notes.append(Note(t, step, pool[i], kinds[(g * 4 + j) % len(kinds)], vel=0.85))
            t += step
    return notes, t


def held(start, midi, dur, vibrato=35, kind="pick", target=None, double=None):
    return [Note(start, dur, midi, kind, target=target, vibrato=vibrato, double=double)], start + dur


# --- The five profiles ---------------------------------------------------------

def profile_shred(rng):
    """Fast 16ths and sextuplets high on the neck (E5-E6) over a full band with ride and crashes."""
    bpm, seconds = 150, 34.0
    pool = scale_notes(64, [0, 2, 3, 5, 7, 8, 10], 64, 88)  # E natural minor, E4..E6
    sixteenth, sextuplet = 60 / bpm / 4, 60 / bpm / 6
    notes, t = [], 1.0
    while t < seconds - 4:
        i0 = rng.integers(pool.index(72), pool.index(78) + 1)  # 4 groups of 4 climb 6 scale steps
        part, t = sequence_of_fours(t, pool, int(i0), 4, sixteenth)
        notes += part
        part, t = run_phrase(t, pool, pool.index(88) - rng.integers(0, 3), pool.index(76), sextuplet)
        notes += part
        part, t = held(t, 86, 0.6, kind="bend", target=88) if rng.random() < 0.5 else held(t, 83, 0.5)
        notes += part
        part, t = run_phrase(t, pool, pool.index(79), pool.index(88), sixteenth)
        notes += part
        part, t = held(t, 88, 0.7, vibrato=40)
        notes += part
        t += 0.4
    lead = dict(drive=7.0, sympathetic=0.01, wet=0.15, tuning=-4)
    roots = [40, 36, 38, 35]
    chords = [[52, 59, 64], [48, 55, 60], [50, 57, 62], [47, 54, 59]]
    def mix(g):
        return [(drums(seconds, bpm, g, ride=True, crash_every_bars=2), -2.0),
                (bass(seconds, bpm, roots, g), -6.0),
                (rhythm_guitar(seconds, bpm, chords, g), -5.0)]
    return dict(name="1. Fast high shred over full band", notes=notes, seconds=seconds, lead=lead,
                mix=mix, lead_over_band_db=3.0, phone=False, bitrate="160k",
                summary="E minor, 150 bpm: 16th-note sequences, sextuplet runs up to E6, bends; drums w/ ride + crashes, bass, rhythm guitar")


def profile_bootleg(rng):
    """Live blues solo filmed on a phone: crowd, big room, auto-gain, low bitrate."""
    bpm, seconds = 92, 34.0
    pool = scale_notes(69, [0, 3, 5, 6, 7, 10], 57, 86)  # A blues
    triplet = 60 / bpm / 3
    notes, t = [], 1.0
    while t < seconds - 4:
        part, t = held(t, 79, 0.9, kind="bend", target=81, vibrato=45)
        notes += part
        part, t = run_phrase(t, pool, pool.index(81), pool.index(69), triplet)
        notes += part
        part, t = held(t, 72, 0.35, kind="bend", target=74)
        notes += part
        part, t = held(t, 69, 0.8, vibrato=50)
        notes += part
        for _ in range(3):  # repeated high lick, fast
            for m, k in ((84, "pick"), (81, "legato"), (79, "legato"), (81, "pick")):
                notes.append(Note(t, triplet, m, k)); t += triplet
        part, t = held(t, 76, 0.7, vibrato=45)
        notes += part
        notes.append(Note(t, 0.08, 0, "ghost")); t += 0.12
        t += 0.5
    lead = dict(drive=4.0, sympathetic=0.0, wet=0.45, tuning=8)
    roots = [45, 50, 45, 52]
    chords = [[57, 61, 64], [62, 66, 69], [57, 61, 64], [64, 68, 71]]
    def mix(g):
        return [(drums(seconds, bpm, g, ride=False, crash_every_bars=4), -4.0),
                (bass(seconds, bpm, roots, g), -6.0),
                (rhythm_guitar(seconds, bpm, chords, g, drive=2.0), -10.0),
                (crowd(seconds, g), -12.0)]
    return dict(name="2. Live phone bootleg (blues)", notes=notes, seconds=seconds, lead=lead,
                mix=mix, lead_over_band_db=1.0, phone=True, bitrate="64k",
                summary="A blues, 92 bpm: big bends, triplet runs, fast repeated high lick; band + crowd, heavy room, phone mic, 64 kbps")


def profile_practice_room(rng):
    """Light overdrive at home: TV talking, a fan, mains hum, open strings ringing, fret noise."""
    bpm, seconds = 100, 32.0
    pool = scale_notes(64, [0, 3, 5, 7, 10], 52, 88)  # E minor pentatonic
    sixteenth = 60 / bpm / 4
    notes, t = [], 1.0
    while t < seconds - 4:
        part, t = run_phrase(t, pool, pool.index(64), pool.index(76), sixteenth * 1.5)
        notes += part
        part, t = held(t, 76, 0.5, vibrato=30)
        notes += part
        notes.append(Note(t, 0.07, 0, "ghost")); t += 0.1
        part, t = run_phrase(t, pool, pool.index(88), pool.index(79), sixteenth, kind_cycle=("pick", "legato"))
        notes += part
        part, t = held(t, 81, 0.6, kind="bend", target=83, vibrato=30)
        notes += part
        part, t = run_phrase(t, pool, pool.index(79), pool.index(67), sixteenth)
        notes += part
        part, t = held(t, 64, 0.8, vibrato=40)
        notes += part
        t += 0.6
    lead = dict(drive=2.5, sympathetic=0.03, wet=0.1, tuning=-12)
    def mix(g):
        return [(tv_speech(seconds, g), -14.0), (fan(seconds, g), -26.0), (hum(seconds), -30.0)]
    return dict(name="3. Practice room: TV, fan, hum", notes=notes, seconds=seconds, lead=lead,
                mix=mix, lead_over_band_db=0.0, absolute_mix=True, phone=False, bitrate="128k",
                summary="E minor pentatonic, 100 bpm, light drive: runs into the high register, ghost notes; TV speech, fan, 60 Hz hum, open strings ringing")


def profile_fuzz_bleed(rng):
    """Heavy fuzz lead with bends and double stops, rhythm guitar as loud as the lead."""
    bpm, seconds = 76, 34.0
    pool = scale_notes(62, [0, 2, 3, 5, 7, 8, 10], 57, 86)  # D minor
    eighth = 60 / bpm / 2
    notes, t = [], 1.0
    while t < seconds - 4:
        part, t = held(t, 74, 1.2, kind="bend", target=76, vibrato=45)
        notes += part
        part, t = held(t, 77, 0.6, double=72)
        notes += part
        part, t = run_phrase(t, pool, pool.index(81), pool.index(69), eighth / 2)
        notes += part
        part, t = held(t, 79, 0.9, kind="bend", target=81, vibrato=40)
        notes += part
        part, t = held(t, 81, 0.5, double=76)
        notes += part
        part, t = held(t, 74, 1.0, vibrato=50)
        notes += part
        t += 0.5
    lead = dict(drive=14.0, sympathetic=0.02, wet=0.2, tuning=0)
    roots = [38, 34, 36, 33]
    chords = [[62, 65, 69], [58, 62, 65], [60, 64, 67], [57, 60, 64]]
    def mix(g):
        return [(drums(seconds, bpm, g, ride=True, crash_every_bars=2), -3.0),
                (bass(seconds, bpm, roots, g), -6.0),
                (rhythm_guitar(seconds, bpm, chords, g, drive=10.0), 0.0)]
    return dict(name="4. Fuzz lead + rhythm guitar bleed", notes=notes, seconds=seconds, lead=lead,
                mix=mix, lead_over_band_db=0.0, phone=False, bitrate="128k",
                summary="D minor, 76 bpm, heavy fuzz: long bends, double stops; rhythm guitar chords in the same octave as loud as the lead, drums")


def profile_legato_tapping(rng):
    """Very fast legato and tapped arpeggios E5-E6, quiet: only amp hiss, hum and hi-hats."""
    bpm, seconds = 120, 30.0
    pool = scale_notes(71, [0, 2, 3, 5, 7, 8, 10], 64, 88)  # B minor
    fast = 60 / bpm / 6        # sextuplets: ~83 ms
    faster = 60 / bpm / 8      # 32nds: ~62 ms
    notes, t = [], 1.0
    arps = [(71, 74, 78, 83), (79, 83, 86, 88), (76, 79, 83, 88), (78, 81, 85, 88)]
    while t < seconds - 4:
        for low, mid, high, tap in arps:  # tapped arpeggios: tap, pull, pull, hammer
            for m, k in ((tap, "tap"), (high, "legato"), (mid, "legato"), (high, "legato")) * 2:
                notes.append(Note(t, faster, m, k, vel=0.8)); t += faster
        part, t = run_phrase(t, pool, pool.index(88), pool.index(71), fast, kind_cycle=("pick", "legato", "legato"))
        notes += part
        part, t = held(t, 83, 0.6, vibrato=35)
        notes += part
        part, t = run_phrase(t, pool, pool.index(74), pool.index(88), faster, kind_cycle=("pick", "legato"))
        notes += part
        part, t = held(t, 88, 0.6, vibrato=35)
        notes += part
        t += 0.4
    lead = dict(drive=10.0, sympathetic=0.0, wet=0.12, tuning=3)
    def mix(g):
        return [(drums(seconds, bpm, g, ride=False, hats=True, crash_every_bars=0) * 0.3, -18.0),
                (hiss(seconds, g), -34.0), (hum(seconds, 50.0), -36.0)]
    return dict(name="5. Fast legato + tapping, quiet", notes=notes, seconds=seconds, lead=lead,
                mix=mix, lead_over_band_db=0.0, absolute_mix=True, phone=False, bitrate="128k",
                summary="B minor, 120 bpm: tapped arpeggios and legato at 62-83 ms per note, E5-E6; only hi-hats, amp hiss, 50 Hz hum")


PROFILES = [profile_shred, profile_bootleg, profile_practice_room, profile_fuzz_bleed, profile_legato_tapping]


def render_profile(make, seed, folder):
    """Render a profile's noisy and clean versions to AAC files. Returns (info, truth per frame)."""
    rng = np.random.default_rng(seed)
    p = make(rng)
    seconds = p["seconds"]
    raw, truth, steady = lead_audio(p["notes"], seconds, rng, tuning_cents=p["lead"]["tuning"])
    guitar = amp_chain(raw, p["lead"]["drive"], rng, p["lead"]["sympathetic"])
    clean = guitar.copy()
    guitar = room(guitar, p["lead"]["wet"], rng)
    playing = truth >= 0
    others = np.zeros(len(guitar))
    for part, db in p["mix"](rng):
        part = part[:len(guitar)]
        if p.get("absolute_mix"):  # level relative to the lead while it plays
            others += part / (rms(part) + 1e-9) * rms(guitar[playing]) * 10 ** (db / 20)
        else:
            others += part / (rms(part) + 1e-9) * 10 ** (db / 20)
    if not p.get("absolute_mix"):  # band parts balanced among themselves, then set against the lead
        others *= rms(guitar[playing]) / rms(others) * 10 ** (-p["lead_over_band_db"] / 20)
    noisy = guitar + others
    if p["phone"]:
        noisy = phone_mic(room(noisy, 0.2, rng, rt=1.4), rng)
    paths = {}
    for label, audio, bitrate in (("noisy", noisy, p["bitrate"]), ("clean", clean, "192k")):
        wav = os.path.join(folder, f"{label}_{seed}.wav")
        sf.write(wav, (audio / np.max(np.abs(audio)) * 0.9).astype(np.float32), SR)
        paths[label] = wav[:-4] + ".m4a"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", wav, "-c:a", "aac", "-b:a", bitrate, paths[label]],
                       capture_output=True, check=True)
        os.remove(wav)
    return p, paths, truth, steady


# --- Evaluation ----------------------------------------------------------------

def evaluate(result, notes, truth, steady):
    """Per-frame and per-note accuracy, aligned for codec delay."""
    import analysis as an
    idx = np.clip(np.round(result.times * SR).astype(int), 0, len(truth) - 1)
    best = None
    for lag in range(-6, 7):
        shifted = np.clip(idx + int(lag * an.HOP_LENGTH * SR / an.SAMPLE_RATE), 0, len(truth) - 1)
        st = steady[shifted] & (truth[shifted] >= 0)
        acc = np.mean(result.notes[st] == truth[shifted][st]) if st.any() else 0
        if best is None or acc > best[0]:
            best = (acc, shifted)
    shifted = best[1]
    frame_truth, frame_steady = truth[shifted], steady[shifted] & (truth[shifted] >= 0)
    raw_notes = np.where(result.raw_f0 > an.MIN_PITCH_HZ,
                         np.round(an.hz_to_midi(np.maximum(result.raw_f0, 1)) - result.tuning / 100), -1).astype(int)

    per_note = []
    hop = an.HOP_LENGTH / an.SAMPLE_RATE
    lag_s = (shifted[0] / SR - result.times[0]) if len(shifted) else 0.0
    for note in notes:
        if note.kind == "ghost":
            continue
        want = note.target if note.kind == "bend" else note.midi
        a = int((note.start - lag_s) / hop)
        b = int((note.start + note.dur - lag_s) / hop)
        frames = np.arange(max(a, 0), min(b, len(result.notes)))
        frames = frames[frame_steady[frames] & (frame_truth[frames] == want)]
        if len(frames) == 0:
            continue
        final, raw = result.notes[frames], raw_notes[frames]
        per_note.append(dict(
            midi=want, dur=note.dur, kind=note.kind,
            found=np.mean(final == want) >= 0.5,
            found_raw=np.mean(raw == want) >= 0.5,
            missing=np.mean(final < 0) >= 0.5,
            octave=np.mean(np.abs(final - want) == 12) >= 0.5,
        ))
    st = frame_steady
    return dict(
        frame_acc=np.mean(result.notes[st] == frame_truth[st]) * 100,
        frame_missing=np.mean(result.notes[st] < 0) * 100,
        notes=per_note,
    )


def bucket_table(rows_by_version, key, buckets):
    lines = []
    for label, test in buckets:
        cells = []
        for rows in rows_by_version:
            sel = [r for r in rows if test(r[key])]
            cells.append(f"{np.mean([r['found'] for r in sel]) * 100:5.0f}% of {len(sel):3d}" if sel else f"{'-':>12s}")
        lines.append(f"    {label:24s}" + "".join(f"{c:>16s}" for c in cells))
    return lines


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import analysis as an

    keep = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
    folder = keep or tempfile.mkdtemp()
    os.makedirs(folder, exist_ok=True)
    everything = {"clean": [], "noisy": []}
    for seed, make in enumerate(PROFILES, start=1):
        p, paths, truth, steady = render_profile(make, seed, folder)
        print(f"\n{p['name']}\n  {p['summary']}")
        res = {}
        for label in ("clean", "noisy"):
            result = an.analyze_pitch(an.extract_audio(paths[label]), an.SAMPLE_RATE)
            res[label] = evaluate(result, p["notes"], truth, steady)
            everything[label] += res[label]["notes"]
        for label in ("clean", "noisy"):
            r = res[label]
            n = r["notes"]
            print(f"  {label:6s} notes found {np.mean([x['found'] for x in n]) * 100:5.1f}%  "
                  f"(tracker alone {np.mean([x['found_raw'] for x in n]) * 100:5.1f}%)  "
                  f"missing {np.mean([x['missing'] for x in n]) * 100:5.1f}%  octave {np.mean([x['octave'] for x in n]) * 100:4.1f}%  "
                  f"| frames right {r['frame_acc']:5.1f}%")

    print("\nALL PROFILES, notes found by register and note length" + " " * 6 + "clean control        noisy")
    rows = [everything["clean"], everything["noisy"]]
    for line in bucket_table(rows, "midi", [("below E4", lambda m: m < 64), ("E4-D#5", lambda m: 64 <= m < 76),
                                            ("E5-B5", lambda m: 76 <= m < 84), ("C6-E6 (top frets)", lambda m: m >= 84)]):
        print(line)
    print()
    for line in bucket_table(rows, "dur", [("< 70 ms", lambda d: d < 0.07), ("70-100 ms", lambda d: 0.07 <= d < 0.1),
                                           ("100-150 ms", lambda d: 0.1 <= d < 0.15), ("150-300 ms", lambda d: 0.15 <= d < 0.3),
                                           ("300 ms +", lambda d: d >= 0.3)]):
        print(line)
    print()
    for line in bucket_table(rows, "kind", [(k, (lambda kk: lambda x: x == kk)(k)) for k in ("pick", "legato", "tap", "bend", "slide")]):
        print(line)
    if keep:
        print(f"\nAudio kept in {folder}")


if __name__ == "__main__":
    main()
