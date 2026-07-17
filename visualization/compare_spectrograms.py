"""
    This script generates side-by-side spectrograms for two samples
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os

def plot_spectrogram_comparison(audio_paths, class_changes_logs, sample_names, 
                                 freq_cutoff=5000, max_duration=16, start_offset=1.0, 
                                 show_bleb=False, bleb_formation_time=None, output_path=None, overlay_feature='rms'):
    """
    Generate side-by-side spectrograms for comparison.
    
    Args:
        audio_paths (list): List of paths to audio files
        class_changes_logs (list): List of class change log dictionaries
        sample_names (list): List of sample names for titles
        freq_cutoff (int): Maximum frequency to display (Hz)
        max_duration (float): Maximum duration in seconds to analyze
        start_offset (float): Start time offset in seconds
        show_bleb (bool): Whether to show bleb formation marker
        output_path (str): Path to save the output image (optional)
        overlay_feature (str): Feature to overlay ('rms', 'spectral_centroid', 'spectral_flux', or None)
    """
    n_samples = len(audio_paths)
    fig, axes = plt.subplots(1, n_samples, figsize=(7 * n_samples, 5))
    
    # Ensure axes is iterable even if only one sample
    if n_samples == 1:
        axes = [axes]
    
    for idx, (audio_path, class_changes_log, sample_name, ax) in enumerate(zip(
            audio_paths, class_changes_logs, sample_names, axes)):
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Apply start offset and max_duration trimming
        start_sample = int(start_offset * sr)
        if max_duration is not None:
            end_sample = start_sample + int(max_duration * sr)
        else:
            end_sample = len(y)
        
        y = y[start_sample:end_sample]
        actual_duration = len(y) / sr
        end_time = start_offset + actual_duration
        
        print(f"🎵 {sample_name}: {start_offset:.1f}s to {end_time:.1f}s (duration: {actual_duration:.2f}s)")
        
        # Compute spectrogram
        n_fft = 2048
        hop_length = 512
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_dB = librosa.amplitude_to_db(np.abs(S), ref=1.0)
        
        # Create time axis
        frames = range(S_dB.shape[1])
        time = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        
        # Plot spectrogram
        img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, 
                                       x_axis='time', y_axis='hz', 
                                       cmap='magma', ax=ax)
        
        # Set frequency limit
        ax.set_ylim([0, freq_cutoff])
        ax.set_xlim([0, actual_duration if max_duration is None else max_duration])
        
        # Styling
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_title(sample_name, fontsize=14, fontweight='bold', pad=10)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Overlay spectral feature if requested
        if overlay_feature:
            if overlay_feature == 'rms':
                feature = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
                feature_label = 'RMS Energy'
                feature_color = 'white'
                feature_units = 'Amplitude'
            elif overlay_feature == 'spectral_centroid':
                feature = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
                feature_label = 'Spectral Centroid'
                feature_color = 'white'
                feature_units = 'Hz'
            elif overlay_feature == 'spectral_flux':
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
                feature = onset_env
                feature_label = 'Spectral Flux'
                feature_color = 'white'
                feature_units = 'Strength'
            else:
                feature = None
            
            if feature is not None and idx == n_samples - 1:  # Only overlay on last sample for clarity
                # Get bleb formation time if it exists
                bleb_time = bleb_formation_time if bleb_formation_time is not None else class_changes_log.get('bleb_formation_time_seconds')
                
                # Normalize feature to fit in frequency range
                feature_norm = (feature - feature.min()) / (feature.max() - feature.min())
                feature_scaled = feature_norm * freq_cutoff * 0.8  # Use 80% of freq range
                
                # Filter feature to only show after bleb formation
                if bleb_time is not None and bleb_time >= start_offset:
                    bleb_time_adjusted = bleb_time - start_offset
                    # Find indices where time >= bleb formation time
                    bleb_mask = time >= bleb_time_adjusted
                    time_filtered = time[bleb_mask]
                    feature_filtered = feature_scaled[bleb_mask]
                else:
                    # Show entire feature if no bleb time
                    time_filtered = time
                    feature_filtered = feature_scaled
                
                # Create twin axis for overlay
                ax2 = ax.twinx()
                ax2.plot(time_filtered, feature_filtered, color=feature_color, linewidth=0.8, 
                        alpha=0.2, label=feature_label, zorder=15)
                ax2.set_ylim([0, freq_cutoff])
                
                # Only show tick labels on the rightmost panel
                if idx == n_samples - 1:
                    # Add units label on right axis
                    ax2.set_ylabel(f'{feature_label}\n(normalized)', 
                                  fontsize=9, color='black', fontweight='bold')
                    ax2.tick_params(axis='y', labelcolor='black', labelsize=8, colors='black')
                    
                    # Set tick labels to show normalized values (0 to 1)
                    y_ticks = np.linspace(0, freq_cutoff * 0.8, 5)
                    ax2.set_yticks(y_ticks)
                    ax2.set_yticklabels([f'{val:.2f}' for val in np.linspace(0, 1, 5)], color='black')
                else:
                    # Hide right axis for other panels
                    ax2.set_yticks([])
                    ax2.set_ylabel('')
                
                # Add feature to legend
                from matplotlib.lines import Line2D
                bleb_color = '#9B59B6'  # Purple
                feature_line = Line2D([0], [0], color=bleb_color, linewidth=0.8, 
                                     alpha=0.2, label=feature_label)
        
        # Define anatomy class labels and colors
        class_labels = {
            0: ('Vitreous', '#95A5A6'),      # Gray
            2: ('ILM', '#2ECC71'),            # Green
            4: ('Retina', '#E74C3C'),         # Red
            3: ('RPE', '#3498DB')             # Blue
        }
        
        # Build timeline of anatomy regions
        changes = class_changes_log.get('changes', [])
        max_time = actual_duration if max_duration is None else max_duration
        
        # Check if this sample has bleb formation (determines RPE treatment)
        has_bleb = class_changes_log.get('bleb_formation_time_seconds') is not None
        
        # Create time periods for each anatomy
        # Note: RPE (class 3) touches are brief events only if sample has bleb, otherwise normal period
        anatomy_periods = []
        current_class = 0  # Start in vitreous
        period_start = 0.0
        in_retina = False  # Track if we've entered retina
        
        for change in changes:
            from_class = change['from_class']
            to_class = change['to_class']
            time_seconds = change['time_seconds']
            
            # Adjust time for start offset
            if time_seconds < start_offset:
                current_class = to_class
                if to_class in [4, 3]:  # Entered retina or RPE
                    in_retina = True
                continue
            
            if time_seconds > end_time:
                break
            
            t_change = time_seconds - start_offset
            
            # Determine display class
            display_class = current_class
            # Only treat RPE as Retina if sample has bleb (brief contacts)
            if has_bleb and (current_class == 3 or in_retina):
                display_class = 4  # Show as Retina
            
            # Add period for current anatomy
            if display_class in class_labels:
                anatomy_periods.append({
                    'class': display_class,
                    'start': period_start,
                    'end': t_change
                })
            
            current_class = to_class
            if to_class in [4, 3]:  # Entering retina or RPE
                in_retina = True
            period_start = t_change
        
        # Add final period
        display_class = current_class
        if has_bleb and (current_class == 3 or in_retina):
            display_class = 4  # Show as Retina
        
        if display_class in class_labels:
            anatomy_periods.append({
                'class': display_class,
                'start': period_start,
                'end': max_time
            })
        
        # Check for bleb formation and split Retina period if needed
        # Use sample-specific bleb time only (not global parameter)
        bleb_time = class_changes_log.get('bleb_formation_time_seconds')
        if bleb_time is not None and start_offset <= bleb_time <= end_time:
            bleb_time_adjusted = bleb_time - start_offset
            
            # Split any Retina period that contains the bleb formation time
            new_periods = []
            for period in anatomy_periods:
                if period['class'] == 4 and period['start'] < bleb_time_adjusted < period['end']:
                    # Split into pre-bleb and post-bleb periods
                    new_periods.append({
                        'class': 4,
                        'start': period['start'],
                        'end': bleb_time_adjusted,
                        'is_bleb': False
                    })
                    new_periods.append({
                        'class': 4,
                        'start': bleb_time_adjusted,
                        'end': period['end'],
                        'is_bleb': True
                    })
                else:
                    period['is_bleb'] = False
                    new_periods.append(period)
            anatomy_periods = new_periods
        else:
            for period in anatomy_periods:
                period['is_bleb'] = False
        
        # Add shaded background regions for each anatomy
        for period in anatomy_periods:
            label, color = class_labels[period['class']]
            # Use purple for bleb periods
            if period.get('is_bleb', False):
                color = '#9B59B6'  # Purple for bleb
            ax.axvspan(period['start'], period['end'], alpha=0.12, 
                      color=color, zorder=0)
        
        # # Add "Needle Location" header on the left side
        # ax.text(0.02, 0.98, 'Needle Location:', 
        #        transform=ax.transAxes,
        #        ha='left', va='top',
        #        fontsize=9, fontweight='bold',
        #        color='black',
        #        bbox=dict(boxstyle='round,pad=0.3', 
        #                 facecolor='white', 
        #                 edgecolor='gray',
        #                 alpha=0.8, linewidth=1),
        #        zorder=18)
        
        # Add anatomy text labels at the top of the spectrogram
        label_y_position = freq_cutoff * 0.92  # Position near top
        bleb_times_for_labels = []  # Collect bleb formation times
        
        for period in anatomy_periods:
            label, color = class_labels[period['class']]
            
            # Collect bleb periods for separate labeling, but still show Retina at top
            if period.get('is_bleb', False):
                bleb_times_for_labels.append((period['start'], period['end']))
                # Continue to show Retina label at top
            
            # Skip ILM periods - we'll show them as instantaneous events
            if period['class'] == 2:  # ILM
                continue
            
            mid_time = (period['start'] + period['end']) / 2
            duration = period['end'] - period['start']
            
            # Show labels for periods (Vitreous and Retina mainly)
            if duration > 0.3:  # At least 0.3 seconds
                ax.text(mid_time, label_y_position, label,
                       ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor=color, 
                                edgecolor='white',
                                alpha=0.85, linewidth=1.5),
                       zorder=15)
        
        # Add bleb label(s) separately at a different vertical position
        for bleb_start, bleb_end in bleb_times_for_labels:
            mid_time = (bleb_start + bleb_end) / 2
            duration = bleb_end - bleb_start
            
            if duration > 0.3:
                ax.text(mid_time, freq_cutoff * 0.78, 'Bleb Formation',
                       ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor='#9B59B6', 
                                edgecolor='white',
                                alpha=0.85, linewidth=1.5),
                       zorder=15)
        
        # Add class change markers (transition lines)
        forward_transitions = {
            (0, 2): '#2ECC71',  # Green - Vitreous→ILM
            (2, 4): '#E74C3C',  # Red - ILM→Retina
            (4, 3): '#3498DB'   # Blue - Retina→RPE
        }
        
        for change in changes:
            from_class = change['from_class']
            to_class = change['to_class']
            time_seconds = change['time_seconds']
            
            # Filter forward transitions
            if (from_class, to_class) not in forward_transitions:
                continue
            
            # Adjust time for start offset
            if time_seconds < start_offset or time_seconds > end_time:
                continue
            
            t_change = time_seconds - start_offset
            color = forward_transitions[(from_class, to_class)]
            
            # Add transition line
            ax.axvline(x=t_change, color=color, linestyle='--', alpha=0.75, 
                      linewidth=1.8)
            
            # For ILM events, add a marker and text label since they're instantaneous
            if (from_class, to_class) == (0, 2):
                # Add a marker at the ILM contact point
                marker_y_position = freq_cutoff * 0.5  # Middle of the frequency range
                ax.plot(t_change, marker_y_position, marker='o', markersize=8, 
                       color='#2ECC71', markeredgecolor='white', 
                       markeredgewidth=1.5, zorder=20)
                
                # Add text label for ILM event
                ax.text(t_change, freq_cutoff * 0.65, 'ILM',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='#2ECC71', 
                                edgecolor='white',
                                alpha=0.9, linewidth=1.2),
                       zorder=16,
                       rotation=0)
            
            # For RPE events, add marker and text label only if sample has bleb (instantaneous)
            # Otherwise let RPE show as a normal period
            if (from_class, to_class) == (4, 3) and has_bleb:
                # Add a marker at the RPE contact point
                marker_y_position = freq_cutoff * 0.5  # Middle of the frequency range
                ax.plot(t_change, marker_y_position, marker='v', markersize=8, 
                       color='#3498DB', markeredgecolor='white', 
                       markeredgewidth=1.5, zorder=20)
                
                # Add text label for RPE event
                ax.text(t_change, freq_cutoff * 0.65, 'RPE',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='#3498DB', 
                                edgecolor='white',
                                alpha=0.9, linewidth=1.2),
                       zorder=16,
                       rotation=0)
        
        # Add bleb formation vertical marker if enabled
        # Use sample-specific bleb time only
        bleb_time = class_changes_log.get('bleb_formation_time_seconds')
        if bleb_time is not None:
            if start_offset <= bleb_time <= end_time:
                bleb_time_adjusted = bleb_time - start_offset
                bleb_color = '#9B59B6'  # Purple
                
                # Add vertical line marker for bleb formation
                ax.axvline(x=bleb_time_adjusted, color=bleb_color, linestyle='-', 
                          alpha=0.8, linewidth=2.2, zorder=10)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📈 Comparison spectrogram saved to: {output_path}")
    
    plt.show()
    return fig


# Sample mappings
map_b_21 = {
    "total_duration_seconds": 21.113133907318115,
    "total_class_changes": 3,
    "changes": [
        {
            "frame": 44,
            "time_seconds": 4.76,
            "from_class": 0,
            "to_class": 2
        },
        {
            "frame": 47,
            "time_seconds": 5.00,
            "from_class": 2,
            "to_class": 4
        },
        {
            "frame": 101,
            "time_seconds": 11.8823529412,
            "from_class": 4,
            "to_class": 3
        }
    ]
}

map_b_i17 = {
    "total_duration_seconds": 20.871707916259766,
    "total_class_changes": 3,
    "changes": [
        {
            "frame": 35,
            "time_seconds": 4.1666666667,
            "from_class": 0,
            "to_class": 2
        },
        {
            "frame": 38,
            "time_seconds": 5.0000000000,
            "from_class": 2,
            "to_class": 4
        },
        {
            "frame": 90,
            "time_seconds": 10.71,
            "from_class": 4,
            "to_class": 3
        }
    ],
    "sample_name": "b_i17"
}

map_006 = {
    "total_duration_seconds": 28.801619052886963,
    "total_class_changes": 5,
    "changes": [
        {
            "frame": 62,
            "time_seconds": 6.02,
            "from_class": 0,
            "to_class": 2
        },
        {
            "frame": 64,
            "time_seconds": 6.21,
            "from_class": 2,
            "to_class": 4
        },
        {
            "frame": 94,
            "time_seconds": 9.13,
            "from_class": 4,
            "to_class": 3
        },

        {
            "frame": 128,
            "time_seconds": 12.93,
            "from_class": 4,
            "to_class": 3
        }
    ],
    "bleb_formation_frame": 125,
    "bleb_formation_time_seconds": 13.2,
    "sample_name": "006"
}


if __name__ == "__main__":
    # Define paths to audio recordings
    PATH_B21 = "/Users/luisreyes/Sonify/SonifyOCT/utilities/runs/physics_based_b_i21_inference_24_02_2026_09_32_39"
    PATH_B21 = "/Users/luisreyes/Sonify/SonifyOCT/utilities/runs/physics_based_b_i21_inference_24_02_2026_11_03_51"
    # PATH_B21 = "/Users/luisreyes/Sonify/SonifyOCT/utilities/runs/physics_based_b_i21_inference_26_02_2026_14_11_04"
    PATH_B17 = "/Users/luisreyes/Sonify/SonifyOCT/utilities/runs/physics_based_b_i17_inference_24_02_2026_10_49_36"
    PATH_006 = "/Users/luisreyes/Sonify/SonifyOCT/utilities/runs/physics_based_injection_006_inference_25_02_2026_14_57_13"
    
    audio_paths = [
        os.path.join(PATH_B17, "recording.wav"),
        os.path.join(PATH_006, "recording.wav")
    ]
    
    class_changes_logs = [map_b_i17, map_006]
    sample_names = ["Porcine Robotic Insertion", "Synthetic Subretinal Injection"]
    
    # Generate comparison
    output_path = "/Users/luisreyes/Sonify/SonifyOCT/utilities/spectrogram_comparison_b21_b17.png"
    
    plot_spectrogram_comparison(
        audio_paths=audio_paths,
        class_changes_logs=class_changes_logs,
        sample_names=sample_names,
        freq_cutoff=5000,
        max_duration=16,
        start_offset=1.0,
        show_bleb=True,
        bleb_formation_time=13.2,
        output_path=output_path,
        overlay_feature=None  # Options: 'rms', 'spectral_centroid', 'spectral_flux', or None
    )
