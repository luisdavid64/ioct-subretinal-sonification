from time import sleep

import supriya
from supriya import synthdef
from supriya.ugens import LPF, LFTri, Out, SinOsc


def dbamp(db: float) -> float:
    return 10.0 ** (db / 20.0)


@synthdef()
def dist_pulse(
    out=0,
    freq=440.0,
    pulse_freq=2.0,
    gain_db=-120.0,
    pulse_width=0.5,  # unused (same as your SC version)
):
    carrier = SinOsc.ar(frequency=freq)
    carrier = (carrier * 1.3).tanh()

    pulse = LFTri.kr(frequency=pulse_freq).scale(
        input_minimum=-1.0,
        input_maximum=1.0,
        output_minimum=0.0,
        output_maximum=1.0,
    )

    pulse = pulse ** 3
    pulse = pulse * (pulse > 0.15)

    sig = carrier * pulse * dbamp(gain_db)

    cutoff = (freq * 6).clip(800, 8000)
    sig = LPF.ar(source=sig, frequency=cutoff)

    Out.ar(bus=out, source=[sig, sig])


class DistPulseEngine:
    def __init__(self):
        self.server = None
        self.synth = None

    # -----------------------------
    # Boot + init (like waitForBoot)
    # -----------------------------
    def boot(self):
        self.server = supriya.Server().boot()

        self.server.add_synthdefs(dist_pulse)
        self.server.sync()

        self.synth = self.server.add_synth(dist_pulse)

        print("[Supriya] Ready for sound")

    # -----------------------------
    # Parameter updates (your API)
    # -----------------------------
    def update_params(self, freq=None, pulse_freq=None, gain_db=None):
        if self.synth is None:
            raise RuntimeError("Synth not initialized. Call boot() first.")

        params = {}
        if freq is not None:
            params["freq"] = float(freq)
        if pulse_freq is not None:
            params["pulse_freq"] = float(pulse_freq)
        if gain_db is not None:
            params["gain_db"] = float(gain_db)

        if params:
            self.synth.set(**params)

    # -----------------------------
    # Cleanup
    # -----------------------------
    def stop(self):
        if self.synth is not None:
            self.synth.free()
            self.synth = None

    def shutdown(self):
        self.stop()
        if self.server is not None:
            self.server.quit()
            self.server = None

    # optional: nice context manager
    def __enter__(self):
        self.boot()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

if __name__ == "__main__":
    engine = DistPulseEngine()
    engine.boot()

    engine.update_params(freq=440, pulse_freq=2.0, gain_db=-30)
    sleep(1)
    engine.update_params(freq=220)
    engine.update_params(pulse_freq=8.0, gain_db=-12)

    engine.shutdown()