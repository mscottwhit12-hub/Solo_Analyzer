"""Run the BUILT guitar isolation pack on a test mix and check it helps.

    python benchmark/pack_smoke_test.py PATH/TO/GuitarSeparator(.exe)

Renders a lead line with a bass as loud as it, runs the packaged program through the same
code path the app uses, and checks the output: right length and rate, and more lead notes
found than on the raw mix.
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
import separation


def main(exe):
    rng = np.random.default_rng(2)
    p = rl.PROFILES[1](rng)  # blues lead
    p["seconds"] = 14.0
    p["notes"] = [n for n in p["notes"] if n.start + n.dur < 13.5]
    raw, truth, steady = rl.lead_audio(p["notes"], p["seconds"], rng, tuning_cents=p["lead"]["tuning"])
    lead = rl.room(rl.amp_chain(raw, p["lead"]["drive"], rng), p["lead"]["wet"], rng)
    bass = rl.bass(p["seconds"], 92, [45, 50, 45, 52], rng)[:len(lead)]
    mix = lead + bass / rl.rms(bass) * rl.rms(lead[truth >= 0])
    path = os.path.join(tempfile.mkdtemp(), "mix.wav")
    sf.write(path, (mix / np.abs(mix).max() * 0.9).astype(np.float32), rl.SR)

    stereo = separation.stereo_from_file(path)
    steps = []
    start = time.time()
    guitar = separation.run_pack(stereo, progress=steps.append, exe=exe)
    took = time.time() - start
    assert steps and steps[-1] > 0.99, f"no progress reported: {steps[-3:]}"
    expected = int(len(mix) * separation.OUTPUT_RATE / rl.SR)
    assert abs(len(guitar) - expected) < 100, (len(guitar), expected)
    assert np.isfinite(guitar).all() and np.abs(guitar).max() > 0.01

    y = an.extract_audio(path)
    before = rl.evaluate(an.analyze_pitch(y, an.SAMPLE_RATE), p["notes"], truth, steady)["notes"]
    after = rl.evaluate(an.analyze_pitch(y, an.SAMPLE_RATE, isolated=guitar), p["notes"], truth, steady)["notes"]
    pct = lambda ev: np.mean([x["found"] for x in ev]) * 100
    print(f"pack ran in {took:.1f}s for {p['seconds']:.0f}s of audio; {len(steps)} progress updates")
    print(f"lead notes found with a bass as loud as the guitar: raw mix {pct(before):.1f}%  isolated {pct(after):.1f}%")
    assert pct(after) > pct(before), "isolation didn't help on the test mix"
    print("PACK SMOKE TEST PASSED")


if __name__ == "__main__":
    main(sys.argv[1])
