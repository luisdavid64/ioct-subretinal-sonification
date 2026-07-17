NetAddr.langPort

s.waitForBoot({

    // -----------------------------
    // SynthDefs
    // -----------------------------
	SynthDef(\distPulse, {
		|out = 0,
		freq = 440,
		pulseFreq = 2.0,
		gain = -120,
		pulseWidth = 0.5
		|

		var carrier, pulse, sig;

		// richer-but-still-clean timbre (harmonic stack)
		carrier = SinOsc.ar(freq);

		// very gentle warmth
		carrier = tanh(carrier * 1.3);

		// your pulse (unchanged)
		pulse = LFTri.kr(pulseFreq).range(0, 1);
		pulse = pulse.pow(3);
		pulse = pulse * (pulse > 0.15);

		sig = carrier * pulse * gain.dbamp;

		// gentle smoothing: reduces ear fatigue / “whine”
		sig = LPF.ar(sig, (freq * 6).clip(800, 8000));

		Out.ar(out, sig ! 2);
	}).add;

    // -----------------------------
    // Persistent synth instance
    // -----------------------------
    ~son = Synth(\distPulse);

    // -----------------------------
    // Parameter updates from Python
    // -----------------------------
    OSCdef(\updateParams, { |msg|
        var freq = msg[1];
        var pulseFreq = msg[2];
		var gain = msg[3];
		~son.set(\freq, freq, \pulseFreq, pulseFreq, \gain, gain);
    }, '/son/update');

    // -----------------------------
    // Recording control
    // -----------------------------
	OSCdef(\recStart, {
		fork {
			var path = "/tmp/rec_son.wav";

			"[REC] Start".postln;
			s.prepareForRecord(path);

			s.sync;        // now legal
			s.record;
		};
	}, '/rec/start');

	OSCdef(\recStop, {
		fork {
			"[REC] Stop".postln;

			s.sync;        // now legal
			s.stopRecording;
		};
	}, '/rec/stop');

    "[SC] Ready for sound + recording".postln;
});
