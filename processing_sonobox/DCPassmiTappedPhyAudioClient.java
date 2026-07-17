package miPhysics.Engine;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import miPhysics.Engine.PhysicsContext;
import miPhysics.Engine.Sound.*;

import java.util.logging.Logger;
import java.util.logging.Level;


/**
 * This class allows tapping into the audio stream of miPhysics audio client.
 * NOW EXTENDED WITH OPTIONAL AUDIO-DOMAIN HIGH-PASS (DC BLOCKER).
 */
public class DCPassmiTappedPhyAudioClient extends miPhyAudioClient {

    /** Callback fired once per processed audio block (on the audio thread). */
    public interface AudioTap {
        void onAudio(float[][] data, int nframes, int sampleRate, long timeNanos);
    }

    private final CopyOnWriteArrayList<AudioTap> taps = new CopyOnWriteArrayList<>();
    private final int sampleRateInt;

    /* === Added: High-pass filter state === */
    private boolean hpEnabled = true;       // turn on/off
    private HPFilter[] hpFilters = null;    // one per channel
    private float hpAlpha = 0.995f;         // pole coefficient (20–30 Hz cutoff)



    /* ---------------- High-pass filter class ---------------- */

    private static class HPFilter {
        private float prevX = 0f;
        private float prevY = 0f;
        private final float alpha;

        HPFilter(float alpha) {
            this.alpha = alpha;
        }

        float process(float x) {
            float y = x - prevX + alpha * prevY;
            prevX = x;
            prevY = y;
            return y;
        }
    }


    /* ---------- Convenience factories ---------- */

    public static DCPassmiTappedPhyAudioClient miPhyJack(
            float sampleRate, int bufS, int inputChannelCount, int outputChannelCount, PhysicsContext c
    ) {
        try {
            return new DCPassmiTappedPhyAudioClient(sampleRate, inputChannelCount, outputChannelCount, c, bufS, "JACK");
        } catch (Exception e) {
            Logger.getLogger(miTappedPhyAudioClient.class.getName())
                .log(Level.SEVERE, "Could not create a JACK miTappedPhyAudioClient", e);
            return null;
        }
    }

    public static miTappedPhyAudioClient miPhyClassic(
            float sampleRate, int bufS, int inputChannelCount, int outputChannelCount, PhysicsContext c
    ) {
        try {
            return new miTappedPhyAudioClient(sampleRate, inputChannelCount, outputChannelCount, c, bufS, "JavaSound");
        } catch (Exception e) {
            Logger.getLogger(miTappedPhyAudioClient.class.getName())
                .log(Level.SEVERE, "Could not create a JavaSound miTappedPhyAudioClient", e);
            return null;
        }
    }


    /* ---------------- Constructor ---------------- */

    public DCPassmiTappedPhyAudioClient(
            float sampleRate,
            int inputChannelCount,
            int outputChannelCount,
            PhysicsContext c,
            int bufferSize,
            String serverType
    ) throws Exception {
        super(sampleRate, inputChannelCount, outputChannelCount, c, bufferSize, serverType);
        this.sampleRateInt = Math.round(sampleRate);
    }



    /* ---------------- Public API ---------------- */

    public void addTap(AudioTap tap) {
        if (tap != null) taps.add(tap);
    }

    public void removeTap(AudioTap tap) {
        if (tap != null) taps.remove(tap);
    }

    public void clearTaps() {
        taps.clear();
    }

    /** Enable/disable the audio-domain HPF */
    public void enableHPF(boolean enable) {
        this.hpEnabled = enable;
    }

    /** Change HPF alpha (0.95–0.999 recommended) */
    public void setHPFAlpha(float alpha) {
        this.hpAlpha = alpha;
    }



    /* ---------------- Audio processing logic ---------------- */

    @Override
    public boolean process(long time, List<FloatBuffer> inputs, List<FloatBuffer> outputs, int nframes) {
        boolean ok = super.process(time, inputs, outputs, nframes);
        if (!ok) return ok;

        final int chCount = outputs.size();

        // Lazy init filters
        if (hpFilters == null) {
            hpFilters = new HPFilter[chCount];
            for (int ch = 0; ch < chCount; ch++)
                hpFilters[ch] = new HPFilter(hpAlpha);
        }

        // Process audio *in-place* directly on output buffers
        for (int ch = 0; ch < chCount; ch++) {
            FloatBuffer buf = outputs.get(ch);
            int end = buf.position();
            int start = end - nframes;

            if (start < 0) start = 0;
            if (end > buf.limit()) end = buf.limit();

            // Apply HPF in-place:
            for (int i = start; i < end; i++) {
                float x = buf.get(i);
                float y = x;

                if (hpEnabled) {
                    y = hpFilters[ch].process(x);
                }

                buf.put(i, y);   // <-- write filtered sample back into buffer
            }
        }

        // Now call taps with a snapshot if you need
        if (!taps.isEmpty()) {
            float[][] snap = new float[chCount][nframes];
            for (int ch = 0; ch < chCount; ch++) {
                FloatBuffer buf = outputs.get(ch);
                int end = buf.position();
                int start = end - nframes;
                if (start < 0) start = 0;
                if (end > buf.limit()) end = buf.limit();
                FloatBuffer ro = buf.asReadOnlyBuffer();
                ro.position(start);
                ro.limit(end);
                ro.get(snap[ch], 0, nframes);
            }
            for (AudioTap tap : taps)
                tap.onAudio(snap, nframes, sampleRateInt, time);
        }

        return true;
    }

}
