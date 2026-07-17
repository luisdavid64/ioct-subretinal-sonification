import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import warnings

def generate_and_save_spectrogram(audio_file_path, class_changes_log, sonification_start_time, output_dir="/Users/luisreyes/Sonify/SonifyOCT/utilities", show_plot=True, freq_cutoff=2000):
    """
    Generate and save a spectrogram with class change markers.
    
    Args:
        audio_file_path (str): Path to the audio file to analyze
        class_changes_log (list): List of class change records with timestamps
        sonification_start_time (float): Start time of the sonification session
        output_dir (str): Directory to save the spectrogram image
        show_plot (bool): Whether to display the plot
    
    Returns:
        str: Path to the saved spectrogram image
    """
    import matplotlib.pyplot as plt
    
    # Check if file exists before attempting to load
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    # Load audio and compute spectrogram
    # Suppress known warnings from librosa
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed")
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
        y, sr = librosa.load(audio_file_path, sr=None)
    
    # compute spectrogram
    n_fft = 2048  # number of FFT points
    hop_length = 512  # hop length between adjacent frames
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=1.0)
    
    # Compute spectral centroid and RMS energy
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    rms_energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Create time axis for features
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames, sr=sr)
    
    # Create subplot layout: spectrogram on top, centroid in middle, RMS at bottom
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                        gridspec_kw={'height_ratios': [3, 1, 1]}, 
                                        sharex=True, constrained_layout=True)
    
    # Plot spectrogram with frequency cutoff
    # Use fmax parameter to properly limit frequency range for log scale
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax1, fmax=8000, cmap='magma')
    ax1.set_xlabel('')  # Remove x-axis label for top plot
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f'Spectrogram with Class Change Times (Total: {len(class_changes_log)} changes, Max Freq: 8000Hz)')
    
    # Remove the ylim since fmax handles it properly for log scale
    
    # Add colorbar for spectrogram
    cbar = plt.colorbar(img, ax=ax1, format='%+2.0f dB', shrink=0.8)
    
    # Plot spectral centroid with frequency cutoff applied
    ax2.plot(t, spectral_centroids, color='blue', linewidth=2, alpha=0.8, label='Spectral Centroid')
    ax2.set_ylabel('Centroid (Hz)')
    ax2.set_xlabel('')  # Remove x-axis label for middle plot
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Apply frequency cutoff to spectral centroid plot
    if freq_cutoff:
        ax2.set_ylim(0, freq_cutoff)
    
    # Plot RMS energy
    ax3.plot(t, rms_energy, color='green', linewidth=2, alpha=0.8, label='RMS Energy')
    ax3.set_ylabel('RMS')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)
    
    # Mark class change times on all three plots
    change_times = [change['time_seconds'] for change in class_changes_log]
    for i, t_change in enumerate(change_times):
        change = class_changes_log[i]
        # color = 'red' if i % 2 == 0 else 'orange'
        # One color per class
        color = {0: 'gray',
                 1: 'blue',
                 2: 'green',
                 3: 'cyan',
                 4: 'red'}.get(change['to_class'], 'black')
        
        # Add vertical lines to all plots
        ax1.axvline(x=t_change, color=color, linestyle='--', alpha=0.8, linewidth=1.5,
                   label=f'Class Change {change["from_class"]}→{change["to_class"]}' if i < 3 else "")
        ax2.axvline(x=t_change, color=color, linestyle='--', alpha=0.6, linewidth=1)
        ax3.axvline(x=t_change, color=color, linestyle='--', alpha=0.6, linewidth=1)
    
    if class_changes_log:
        ax1.legend(loc='upper right', fontsize=8)
    
    # Save spectrogram with timestamp
    spec_filename = f"recording_spectrogram_with_features_{int(sonification_start_time)}.png"
    spec_out_path = os.path.join(output_dir, spec_filename)
    plt.savefig(spec_out_path, dpi=300, bbox_inches='tight')
    
    print(f"📈 Enhanced spectrogram with spectral features saved to: {spec_filename}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return spec_out_path