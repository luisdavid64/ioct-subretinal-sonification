import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

PATH = "runs"
PATH = "/Users/luisreyes/Sonify/SonifyOCT/utilities/runs/comparison"

runs = glob.glob(PATH + "/*")

for run in runs:
    path_recording = os.path.join(run, "recording.wav")
    path_enhanced = os.path.join(run, "recording_ace.wav")
    class_changes_log = json.load(open(os.path.join(run, "class_changes_log.json"), "r"))
    if os.path.exists(path_recording) and os.path.exists(path_enhanced):
        # Plot both waveforms for comparison and spectrogram and save figure in run folder
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # Load audio waveforms using librosa
        recording_waveform, sr1 = librosa.load(path_recording, sr=None)
        enhanced_waveform, sr2 = librosa.load(path_enhanced, sr=None)
        
        # Create time axes for waveforms
        time1 = np.linspace(0, len(recording_waveform) / sr1, len(recording_waveform))
        time2 = np.linspace(0, len(enhanced_waveform) / sr2, len(enhanced_waveform))
        
        # Plot waveforms
        axs[0, 0].plot(time1, recording_waveform)
        axs[0, 0].set_title("Original Recording Waveform")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].set_ylabel("Amplitude")
        
        axs[0, 1].plot(time2, enhanced_waveform)
        axs[0, 1].set_title("Enhanced Recording Waveform")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Amplitude")

        # add vertical lines for class changes
        for change in class_changes_log:
            if "time" in change:
                time_change = change["time"]
                axs[0, 0].axvline(x=time_change, color='r', linestyle='--', alpha=0.5)
                axs[0, 1].axvline(x=time_change, color='r', linestyle='--', alpha=0.5)
        
        
        # Plot spectrograms using librosa
        D1 = librosa.amplitude_to_db(np.abs(librosa.stft(recording_waveform)), ref=np.max)
        D2 = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_waveform)), ref=np.max)
        
        librosa.display.specshow(D1, sr=sr1, x_axis='time', y_axis='hz', ax=axs[1, 0])
        axs[1, 0].set_title("Original Recording Spectrogram")
        
        librosa.display.specshow(D2, sr=sr2, x_axis='time', y_axis='hz', ax=axs[1, 1])
        axs[1, 1].set_title("Enhanced Recording Spectrogram")
        # Save figure
        fig.suptitle(f"Comparison of Original and Enhanced Recordings for {os.path.basename(run)}")
        plt.tight_layout()
        plt.savefig(os.path.join(run, "ace_comparison_plot.png"))
        plt.close(fig)
        print(f"Saved comparison plot for run: {run}")
    else:
        print(f"Missing files in run: {run}, skipping comparison plot.")

    
