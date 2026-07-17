package miPhysics.Engine;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.ArrayList;
import java.util.logging.Logger;
import java.util.logging.Level;

import miPhysics.Engine.Sound.*;

public class miTappedPhyAudioClient extends miPhyAudioClient {

    private static final int INTERNAL_CH = 16;
    private static final int TEXTURE_CRACKLE_CH = 8; 
    // Texture DSP state
    private float crackleDensity = 0f;
    private float crackleGain = 5f;

    // Per-channel RMS tracking for auto-gain
    private float[] rms = new float[INTERNAL_CH];          // running RMS estimate
    private float[] autoGain = new float[INTERNAL_CH];     // computed gain per channel
    private float targetRMS = 0.15f;                       // desired output RMS
    private float rmsCoeff = 0.999f;                       // ~23 ms @ 44.1 kHz

    // Post-mix warmth:  1-pole LPF (per output channel)
    private float[] lpState;   // filter state
    private float lpCoeff = 0.82f;  // ~4.5 kHz rolloff @ 44.1 kHz

    // Jitter/instability parameters
    private float jitter_amplitude = 0.8f;     // 0.0-1.0, amplitude instability amount
    private float jitter_cutoff = 0.3f;        // 0.0-1.0, filter cutoff instability
    private float jitter_rate = 50f;          // Hz, rate of jitter fluctuations
    private boolean jitter_enabled = false;
    private float jitter_phase = 0f;         // internal phase accumulator
    private float[] jitter_noise = new float[INTERNAL_CH]; // per-channel noise state
    
    // Jitter ramping/crossfading parameters
    private float jitter_ramp_duration = 0.5f;  // seconds for ramp up/down
    private boolean jitter_target_state = false; // target enabled state
    private float jitter_ramp_factor = 0.0f;    // current ramp factor (0.0 = off, 1.0 = full)
    private boolean jitter_ramping = false;      // true if currently ramping
    private float[][] mixMatrix;

    private float[][] internal;

    /** Toggle 2 channel mode **/
    private boolean legacyMode = false;

    private float mixer_gain = 1;

    private boolean listenFrc_local = false;
    private listenerAxis axis_local = listenerAxis.ALL;


    /* ---------------------- TAP INTERFACE ---------------------- */

    public interface AudioTap {
        void onAudio(float[][] data, int nframes, int sampleRate, long timeNanos);
    }

    private final CopyOnWriteArrayList<AudioTap> taps = new CopyOnWriteArrayList<>();

    private final int sampleRateInt;

    /* ---------------------- FACTORIES ---------------------- */

    public static miTappedPhyAudioClient miPhyJack(
            float sampleRate, int bufS, int inputCh, int outputCh, PhysicsContext c
    ) {
        try {
            return new miTappedPhyAudioClient(sampleRate, inputCh, outputCh, c, bufS, "JACK");
        } catch (Exception e) {
            Logger.getLogger(miTappedPhyAudioClient.class.getName())
                    .log(Level.SEVERE, "Could not create JACK multi-client", e);
            return null;
        }
    }

    public static miTappedPhyAudioClient miPhyClassic(
            float sampleRate, int bufS, int inputCh, int outputCh, PhysicsContext c
    ) {
        try {
            return new miTappedPhyAudioClient(sampleRate, inputCh, outputCh, c, bufS, "JavaSound");
        } catch (Exception e) {
            Logger.getLogger(miTappedPhyAudioClient.class.getName())
                    .log(Level.SEVERE, "Could not create JavaSound multi-client", e);
            return null;
        }
    }

    /* ---------------------- CONSTRUCTOR ---------------------- */

    public miTappedPhyAudioClient(
            float sampleRate,
            int inputChannelCount,
            int outputChannelCount,
            PhysicsContext c,
            int bufferSize,
            String serverType
    ) throws Exception {

        super(sampleRate, inputChannelCount, outputChannelCount, c, bufferSize, serverType);

        this.sampleRateInt = Math.round(sampleRate);

        // Allocate internal channel buffers
        internal = new float[INTERNAL_CH][bufferSize];

        // Initialize jitter state
        jitter_noise = new float[INTERNAL_CH];

        // Initialize auto-gain / warmth state
        rms = new float[INTERNAL_CH];
        autoGain = new float[INTERNAL_CH];
        for (int i = 0; i < INTERNAL_CH; i++) autoGain[i] = 1f;
        lpState = new float[outputChannelCount];

        // Default mix: stereo with equal spread
        mixMatrix = new float[outputChannelCount][INTERNAL_CH];
        setDefaultStereoMix();

        Logger.getLogger(miTappedPhyAudioClient.class.getName())
                .info("Initialized multi-channel tapped client (" + INTERNAL_CH + " internal channels)");
    }

    /* ---------------------- MIXER SETUP ---------------------- */

    /** Default equal-energy stereo spread */
    private void setDefaultStereoMix() {
        if (mixMatrix.length < 2) return;

        for (int i = 0; i < INTERNAL_CH; i++) {
            float pan = i / (float)(INTERNAL_CH - 1); // 0..1
            mixMatrix[0][i] = (float)Math.cos(pan * Math.PI * 0.5); // L
            mixMatrix[1][i] = (float)Math.sin(pan * Math.PI * 0.5); // R
            // mixMatrix[0][i] = 0.7071f; // Left
            // mixMatrix[1][i] = 0.7071f; // Right
        }
    }

    public void setMixMatrix(float[][] newMatrix) {
        if (newMatrix != null && newMatrix.length == mixMatrix.length
                && newMatrix[0].length == INTERNAL_CH) {
            this.mixMatrix = newMatrix;
        }
    }

    public void setLegacyMode(boolean enable) {
        legacyMode = enable;
    }

    /* ---------------------- TAP API ---------------------- */

    public void addTap(AudioTap t) { if (t != null) taps.add(t); }
    public void removeTap(AudioTap t) { if (t != null) taps.remove(t); }
    public void clearTaps() {taps.clear();}
    public void setMixerGain(float g){
        mixer_gain = g;
    }

    /** Target RMS loudness (0.05–0.4).  Lower = quieter but safer. */
    public void setTargetRMS(float t) {
        targetRMS = Math.max(0.01f, Math.min(0.5f, t));
    }

    /** High-frequency rolloff amount (0.7–0.95).  Lower = darker / warmer. */
    public void setWarmth(float w) {
        lpCoeff = Math.max(0.6f, Math.min(0.95f, w));
    }

    /* ----------------------- GETTERS & SETTERS ---------------- */
    @Override
    public void listenFrc() {
        super.listenFrc();       // call parent
        listenFrc_local = true;  // mirror
    }

    @Override
    public void listenPos() {
        super.listenPos();
        listenFrc_local = false;
    }

    @Override
    public void setListenerAxis(listenerAxis a) {
        super.setListenerAxis(a);
        axis_local = a;
    }


    /* ---------------------- MAIN AUDIO PROCESS ---------------------- */

    @Override
    public boolean process(long time, List<FloatBuffer> inputs, List<FloatBuffer> outputs, int nframes) {

        // Run the physics step once per sample
        for (int i = 0; i < INTERNAL_CH; i++) {
            for (int s = 0; s < nframes; s++) internal[i][s] = 0f;
        }

        ArrayList<Observer3D> obs = getPhyContext().mdl().getObservers();

        for (int s = 0; s < nframes; s++) {
            getPhyContext().computeSingleStep();
            
            // Update jitter for this sample
            updateJitter(s);

            // Fill 16 internal channels
            int N = Math.min(obs.size(), INTERNAL_CH);
            for (int ch = 0; ch < N; ch++) {
                Observer3D o = obs.get(ch);
                if (o != null) {
                    float v;

                    if (!listenFrc_local) {
                        switch (axis_local) {
                            case X:   v = (float) o.observePos().x; break;
                            case Y:   v = (float) o.observePos().y; break;
                            case Z:   v = (float) o.observePos().z; break;
                            case ALL: v = (float)(o.observePos().x + o.observePos().y + o.observePos().z); break;
                            default:  v = 0f; break;
                        }
                    } else {
                        switch (axis_local) {
                            case X:   v = (float) o.observeFrc().x; break;
                            case Y:   v = (float) o.observeFrc().y; break;
                            case Z:   v = (float) o.observeFrc().z; break;
                            case ALL: v = (float)(o.observeFrc().x + o.observeFrc().y + o.observeFrc().z); break;
                            default:  v = 0f; break;
                        }
                    }
                    // --- Auto-gain: track RMS and normalise ---
                    rms[ch] = rmsCoeff * rms[ch] + (1f - rmsCoeff) * v * v;
                    float curRms = (float) Math.sqrt(rms[ch]);
                    float desiredGain = (curRms > 1e-6f)
                            ? targetRMS / curRms
                            : 1f;
                    // Clamp so we don't explode on silence
                    desiredGain = Math.min(desiredGain, 4f);
                    // Smooth gain changes (~5 ms attack/release)
                    autoGain[ch] += 0.005f * (desiredGain - autoGain[ch]);

                    internal[ch][s] = v * autoGain[ch] * mixer_gain;
                    
                    // Apply amplitude jitter with ramping
                    if (jitter_ramp_factor > 0f && jitter_amplitude > 0f) {
                        float amplitude_variation = 1.0f + (jitter_amplitude * jitter_noise[ch] * 0.3f * jitter_ramp_factor);
                        internal[ch][s] *= amplitude_variation;
                    }
                }
            }
        }

        /* ---------- LEGACY MODE ---------- */
        if (legacyMode) {
            for (int out = 0; out < outputs.size(); out++) {
                FloatBuffer fb = outputs.get(out);
                fb.clear();
                int src = Math.min(out, INTERNAL_CH - 1);
                fb.put(internal[src], 0, nframes);
            }
        }
        else {
            /* ---------- MULTICHANNEL MIXING ---------- */
            for (int out = 0; out < outputs.size(); out++) {
                FloatBuffer fb = outputs.get(out);
                fb.clear();
                float[] mixRow = mixMatrix[out];

                for (int s = 0; s < nframes; s++) {
                    float acc = 0f;
                    for (int ch = 0; ch < INTERNAL_CH; ch++) {
                        acc += internal[ch][s] * mixRow[ch];
                    }

                    // Gentle warmth: tanh saturation + 1-pole low-pass
                    acc = (float) Math.tanh(acc);                           // soft-clip peaks
                    lpState[out] += lpCoeff * (acc - lpState[out]);         // roll off highs
                    acc = lpState[out];

                    fb.put(acc);
                }
            }
        }

        /* ---------- TAPS ---------- */
        if (!taps.isEmpty()) {
            int outCh = outputs.size();
            float[][] snap = new float[outCh][nframes];

            for (int ch = 0; ch < outCh; ch++) {
                FloatBuffer fb = outputs.get(ch).asReadOnlyBuffer();
                fb.position(0);
                fb.get(snap[ch], 0, nframes);
            }

            for (AudioTap t : taps) t.onAudio(snap, nframes, sampleRateInt, time);
        }

        return true;


    }

    public void setCrackle(float density, float gain) {
        this.crackleDensity = density;   // grains per second
        this.crackleGain = gain;         // 0–1
    }

    /* ---------------------- JITTER/INSTABILITY ---------------------- */
    
    public void setJitter(float amplitudeJitter, float cutoffJitter, float rate, boolean enabled) {
        this.jitter_amplitude = Math.max(0f, Math.min(10f, amplitudeJitter));
        this.jitter_cutoff = Math.max(0f, Math.min(1f, cutoffJitter));
        this.jitter_rate = Math.max(0.1f, Math.min(50f, rate));
        enableJitterWithRamp(enabled); // Use the ramped version
    }
    
    public void setJitter(float amplitudeJitter, float cutoffJitter) {
        setJitter(amplitudeJitter, cutoffJitter, jitter_rate, true);
    }
    
    /** Enable/disable jitter immediately without ramping */
    public void enableJitter(boolean enabled) {
        this.jitter_enabled = enabled;
        this.jitter_target_state = enabled;
        this.jitter_ramp_factor = enabled ? 1.0f : 0.0f;
        this.jitter_ramping = false;
    }
    
    /** Enable/disable jitter with smooth ramping/crossfading */
    public void enableJitterWithRamp(boolean enabled) {
        if (jitter_target_state == enabled && !jitter_ramping) {
            return; // Already in target state and not ramping
        }
        
        jitter_target_state = enabled;
        jitter_ramping = true;
    }
    
    /** Set the duration for jitter ramp up/down transitions */
    public void setJitterRampDuration(float durationSeconds) {
        this.jitter_ramp_duration = Math.max(0.01f, Math.min(10.0f, durationSeconds));
    }
    
    /** Get current jitter ramp factor (0.0 = off, 1.0 = full) */
    public float getJitterRampFactor() {
        return jitter_ramp_factor;
    }
    
    /** Check if jitter is currently ramping */
    public boolean isJitterRamping() {
        return jitter_ramping;
    }
    
    private void updateJitter(int sampleIndex) {
        // Update ramp factor if ramping is active
        if (jitter_ramping) {
            float ramp_increment = 1.0f / (jitter_ramp_duration * sampleRateInt);
            
            if (jitter_target_state) {
                // Ramping up
                jitter_ramp_factor += ramp_increment;
                if (jitter_ramp_factor >= 1.0f) {
                    jitter_ramp_factor = 1.0f;
                    jitter_ramping = false;
                    jitter_enabled = true;
                }
            } else {
                // Ramping down
                jitter_ramp_factor -= ramp_increment;
                if (jitter_ramp_factor <= 0.0f) {
                    jitter_ramp_factor = 0.0f;
                    jitter_ramping = false;
                    jitter_enabled = false;
                }
            }
        }
        
        // Only generate noise if jitter is active (ramp factor > 0)
        if (jitter_ramp_factor > 0f) {
            // Update phase accumulator
            jitter_phase += (jitter_rate * 2.0f * (float)Math.PI) / sampleRateInt;
            if (jitter_phase > 2.0f * Math.PI) {
                jitter_phase -= 2.0f * (float)Math.PI;
            }
            
            // Generate per-channel noise (different random values per channel)
            for (int ch = 0; ch < INTERNAL_CH; ch++) {
                // Combine low-frequency sine wave with high-frequency noise
                float sine_component = (float)Math.sin(jitter_phase + ch * 0.3f); // slight phase offset per channel
                float noise_component = (float)(Math.random() - 0.5f) * 2f; // -1 to 1
                jitter_noise[ch] = 0.7f * sine_component + 0.3f * noise_component;
            }
        }
    }
}