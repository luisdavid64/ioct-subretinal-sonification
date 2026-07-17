import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time
import shutil

from .util import get_force_at_frame


class SonificationConfig:
    """Configuration class to replace the massive parameter list"""
    def __init__(self, **kwargs):
        # Sonification parameters
        self.num_nodes_x = kwargs.get('num_nodes_x', 5)
        self.num_nodes_y = kwargs.get('num_nodes_y', 40)
        self.extend_roi_to_needle_tip = kwargs.get('extend_roi_to_needle_tip', True)
        self.add_margin_to_roi = kwargs.get('add_margin_to_roi', 100)
        self.sep_f_components = kwargs.get('sep_f_components', True)
        self.remap_with_segmentation = kwargs.get('remap_with_segmentation', False)
        self.use_handle_forces = kwargs.get('use_handle_forces', True)
        self.static_mapping_type = kwargs.get('static_mapping_type', "dClass")
        self.dynamic_detuning = kwargs.get('dynamic_detuning', False)
        self.use_deflection_scaling = kwargs.get('use_deflection_scaling', False)
        self.deflection_debug = kwargs.get('deflection_debug', False)
        self.separate_ilm = kwargs.get('separate_ilm', True)
        self.refine_with_sam = kwargs.get('refine_with_sam', [])
        self.ranges = kwargs.get('ranges', "INCREMENTAL")
        self.simulator_path = kwargs.get('simulator_path', "")
        self.use_dynamic_intensity_mapping = kwargs.get('use_dynamic_intensity_mapping', False)
        
        # Runtime data
        self.has_needle = kwargs.get('has_needle', True)
        self.is_synthetic = kwargs.get('is_synthetic', False)
        self.ROI = kwargs.get('ROI', None)
        self.angle = kwargs.get('angle', 0)
        self.needle_tip_pos = kwargs.get('needle_tip_pos', (None, None))
        self.script_type = kwargs.get('script_type', "")
        
        # Data arrays (will be converted to lists for JSON)
        self.RANGE_PARAMS = kwargs.get('RANGE_PARAMS', None)
        self.masses = kwargs.get('masses', None)
        self.stiffnesses = kwargs.get('stiffnesses', None)
        self.damping = kwargs.get('damping', None)
        self.seg_img_0 = kwargs.get('seg_img_0', None)
        self.patch_centers_class = kwargs.get('patch_centers_class', None)
        self.parameter_evolution = kwargs.get('parameter_evolution', None)

    def to_dict(self):
        """Convert to dictionary with JSON-safe types"""
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return obj

        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = make_serializable(value)
        return result


def draw_tracking_overlay(frame, tip, corr,debug=False):
    if debug:
        cv2.circle(frame, (int(tip[0]), int(tip[1])), 5, (0,255,0), 2)
        cv2.putText(frame, f"corr={corr:.2f}", (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

def draw_information(frame, index, frame_files, label=None, force_data=None, show_force=False,
                     rotated_patch_centers=None, patch_centers_class=None, debug=False,
                     masses=None, stiffnesses=None, damping=None, num_nodes_x=None, num_nodes_y=None,
                     is_synthetic=False):
    """ Draw various information overlays on the frame """
    # Display frame number
    if debug:
        cv2.putText(frame, f"Frame: {index + 1}/{len(frame_files)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display label if available
        if label:
            cv2.putText(frame, f"Label: {label}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display force data if available and enabled
        if show_force and force_data is not None:
            force_info = get_force_at_frame(force_data, index, len(frame_files))
            if force_info:
                y_offset = 90
                # Tip force magnitude
                cv2.putText(frame, f"Tip Force: {force_info['tip_force_norm']:.4f} mN", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Sclera force magnitude
                cv2.putText(frame, f"Sclera Force: {force_info['sclera_force_norm']:.4f} mN", 
                            (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Individual tip force components
                cv2.putText(frame, f"Tip XYZ: ({force_info['tip_force_x']:.4f}, {force_info['tip_force_y']:.4f}, {force_info['tip_force_z']:.4f})", 
                            (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Timestamp and synchronization info
                cv2.putText(frame, f"Time: {force_info['timestamp']:.3f}s (CSV:{force_info['csv_index']}, Δ:{force_info['time_diff']:.3f}s)", 
                            (10, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            else:
                # Show when no force data is available for this frame
                cv2.putText(frame, "No force data available for this frame", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Superimpose positions of patches and use patch centers class for coloring 
    if debug and rotated_patch_centers is not None and patch_centers_class is not None:
        for i, (center, cls) in enumerate(zip(rotated_patch_centers, patch_centers_class)):
            cx, cy = int(center[0]), int(center[1])
            if cls == 0:
                color = (200, 200, 200)
            elif cls == 1:
                color = (0, 255, 255)
            elif cls == 2:
                color = (255, 0, 255)
            elif cls == 3:
                color = (255, 255, 0)
            elif cls == 4:
                color = (255, 0, 0)
            elif cls == 5:
                color = (0, 255, 0)
            else:
                color = (100, 100, 100)
            cv2.circle(frame, (cx, cy), 3, color, -1)
            
            # Render mass, stiffness, and damping values as text next to each node
            if (masses is not None and stiffnesses is not None and damping is not None and 
                num_nodes_x is not None and num_nodes_y is not None and is_synthetic):
                
                # Calculate grid position from linear index
                row = i // num_nodes_x
                col = i % num_nodes_x
                
                # Get values for this node
                mass_val = masses[row, col] if row < masses.shape[0] and col < masses.shape[1] else 0.0
                stiff_val = stiffnesses[row, col] if row < stiffnesses.shape[0] and col < stiffnesses.shape[1] else 0.0
                damp_val = damping[row, col] if row < damping.shape[0] and col < damping.shape[1] else 0.0
                
                # Position text offset from node center
                text_x = cx + 10
                text_y = cy - 10
                
                # Ensure text stays within frame bounds
                if text_x > frame.shape[1] - 150:
                    text_x = cx - 150
                if text_y < 40:
                    text_y = cy + 40
                
                # Render values with small font
                font_size = 0.4
                thickness = 1
                
                # Mass (white text)
                cv2.putText(frame, f"M:{mass_val:.2f}", (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)
                
                # Stiffness (cyan text) 
                cv2.putText(frame, f"S:{stiff_val:.2f}", (text_x, text_y + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 0), thickness)
                
                # Damping (yellow text)
                cv2.putText(frame, f"D:{damp_val:.3f}", (text_x, text_y + 24), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), thickness)

def show_if_debug(title, img, debug):
    if debug:
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plot_parameter_evolution(parameter_evolution, sonification_start_time, plot_path=None):
    """
    Plot the evolution of physical parameters (mass, stiffness, damping) over time for different tissue classes.
    
    Parameters:
    -----------
    parameter_evolution : dict
        Dictionary containing frame indices, timestamps, and class-wise parameter averages
    sonification_start_time : float
        Start time of the sonification process
    """
    if len(parameter_evolution['frame_indices']) == 0:
        print("⚠️ No parameter evolution data to plot")
        return
        
    # Class names for better labeling
    class_names = {
        0: "Background", 
        1: "Needle",
        2: "ILM (Top Layer)", 
        3: "RPE (Bottom Layer)", 
        4: "Retina (Middle Layer)", 
        5: "Other"
    }
    
    # Colors for different classes
    class_colors = {
        0: '#808080',  # Gray for background
        1: '#808080',  # Gray for background
        2: '#1f77b4',  # Blue for ILM
        3: '#ff7f0e',  # Orange for Retina  
        4: '#2ca02c',  # Green for RPE
        5: '#d62728'   # Red for Other
    }
    
    timestamps = np.array(parameter_evolution['timestamps'])
    
    # Get unique classes (excluding background)
    active_classes = []
    for param_name in ['masses', 'stiffnesses', 'damping']:
        for class_id in parameter_evolution['class_averages'][param_name].keys():
            if class_id != 0 and class_id not in active_classes:
                active_classes.append(class_id)
    active_classes = sorted(active_classes)
    
    if not active_classes:
        print("⚠️ No active tissue classes found to plot")
        return
    
    # Create subplots: one row per class, three columns for parameters
    num_classes = len(active_classes)
    fig, axes = plt.subplots(num_classes, 3, figsize=(18, 5 * num_classes))
    if num_classes == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array for single class
    
    fig.suptitle('Dynamic Parameter Evolution Over Time (Per Tissue Class)\n(Real-time Tissue Deformation Response)', 
                 fontsize=16, fontweight='bold')
    
    parameter_names = ['masses', 'stiffnesses', 'damping']
    parameter_labels = ['Mass (kg)', 'Stiffness (N/m)', 'Damping (Ns/m)']
    parameter_descriptions = [
        'Lower mass → Higher frequency response',
        'Higher stiffness → Higher frequency response', 
        'Lower damping → Less frequency attenuation'
    ]
    
    for class_idx, class_id in enumerate(active_classes):
        class_label = class_names.get(class_id, f"Class {class_id}")
        class_color = class_colors.get(class_id, f"C{class_id}")
        
        for param_idx, (param_name, param_label, param_desc) in enumerate(zip(parameter_names, parameter_labels, parameter_descriptions)):
            ax = axes[class_idx, param_idx]
            
            # Get values for this class and parameter
            if class_id in parameter_evolution['class_averages'][param_name]:
                values = np.array(parameter_evolution['class_averages'][param_name][class_id])
                
                if len(values) > 0:
                    # Plot the parameter evolution for this class
                    ax.plot(timestamps, values, 
                           color=class_color,
                           linewidth=2.5,
                           marker='o' if len(values) < 50 else None,
                           markersize=4,
                           alpha=0.8,
                           label=f'{class_label} {param_name}')
                    
                    # Add statistics info
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    
                    # Add statistics as text box
                    stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={min_val:.3f}\nmax={max_val:.3f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                    # Highlight regions of significant change for this class
                    if len(values) > 5:
                        window_size = min(5, len(values) // 3)
                        if window_size > 1:
                            rolling_var = []
                            for i in range(len(values) - window_size + 1):
                                window_vals = values[i:i+window_size]
                                rolling_var.append(np.var(window_vals))
                            
                            if len(rolling_var) > 0:
                                rolling_var = np.array(rolling_var)
                                high_var_threshold = np.percentile(rolling_var, 75) if len(rolling_var) > 1 else rolling_var[0]
                                high_var_indices = np.where(rolling_var > high_var_threshold)[0]
                                
                                # Highlight high variance regions
                                for idx in high_var_indices:
                                    start_time = timestamps[idx] if idx < len(timestamps) else timestamps[-1]
                                    end_time = timestamps[min(idx+window_size, len(timestamps)-1)]
                                    ax.axvspan(start_time, end_time, alpha=0.15, color='red', zorder=0)
                
                else:
                    # No data for this class-parameter combination
                    ax.text(0.5, 0.5, f'No data\nfor {class_label}', transform=ax.transAxes,
                           ha='center', va='center', fontsize=12, style='italic',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
            
            # Set labels and formatting
            ax.set_ylabel(param_label, fontweight='bold')
            ax.set_xlabel('Time (seconds)' if class_idx == num_classes-1 else '')
            ax.grid(True, alpha=0.3)
            
            # Set title for each subplot
            ax.set_title(f'{class_label} - {param_name.capitalize()}', fontweight='bold', fontsize=11)
            
            # Add parameter description as small text at bottom
            if class_idx == 0:  # Only on top row
                ax.text(0.02, 0.02, param_desc, transform=ax.transAxes, 
                       fontsize=8, style='italic', verticalalignment='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.6))
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Leave space for main title
    
    # Save the plot
    plot_filename = f"parameter_evolution_per_class_{int(sonification_start_time)}.png"
    plot_path = os.path.join(plot_path, plot_filename) if plot_path else plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    print(f"\n📈 Parameter Evolution Plot (Per Class):")
    print(f"   Frames tracked: {len(parameter_evolution['frame_indices'])}")
    print(f"   Duration: {timestamps[-1]:.1f}s") 
    print(f"   Classes analyzed: {active_classes}")
    for class_id in active_classes:
        class_name = class_names.get(class_id, f"Class {class_id}")
        print(f"     • {class_name} (ID: {class_id})")
    print(f"   Plot saved: {plot_filename}")


def create_run_folder_and_save_data(
    folder, 
    sonification_start_time, 
    class_changes_log, 
    sound_model_config=None,
    config=None,
    # Backward compatibility parameters
    script_type="", num_nodes_x=5, num_nodes_y=40, extend_roi_to_needle_tip=True, 
    add_margin_to_roi=100, sep_f_components=True, remap_with_segmentation=False, 
    use_handle_forces=True, static_mapping_type="dClass", dynamic_detuning=False, 
    use_deflection_scaling=False, deflection_debug=False, separate_ilm=True, 
    refine_with_sam=[], ranges="INCREMENTAL", simulator_path="", 
    use_dynamic_intensity_mapping=False, has_needle=True, is_synthetic=False, 
    ROI=None, angle=0, needle_tip_pos=(None, None), RANGE_PARAMS=None, masses=None, 
    stiffnesses=None, damping=None, seg_img_0=None, patch_centers_class=None, 
    parameter_evolution=None,
    sample_name=None,
    model_type="physics_based"
):
    """
    Create a unique run folder and save all run data using a configuration object.
    Maintains backward compatibility with old parameter approach.
    
    Args:
        folder (str): Path to the data folder
        sonification_start_time (float): Start time of sonification
        class_changes_log (list): List of class change events
        sound_model_config (dict, optional): Sound model configuration
        config (SonificationConfig): Configuration object with all parameters
        ... (backward compatibility parameters)
        
    Returns:
        str: Path to the created run folder
    """
    # Create config object from individual parameters if not provided
    if config is None:
        config = SonificationConfig(
            script_type=script_type, num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y,
            extend_roi_to_needle_tip=extend_roi_to_needle_tip, add_margin_to_roi=add_margin_to_roi,
            sep_f_components=sep_f_components, remap_with_segmentation=remap_with_segmentation,
            use_handle_forces=use_handle_forces, static_mapping_type=static_mapping_type,
            dynamic_detuning=dynamic_detuning, use_deflection_scaling=use_deflection_scaling,
            deflection_debug=deflection_debug, separate_ilm=separate_ilm,
            refine_with_sam=refine_with_sam, ranges=ranges, simulator_path=simulator_path,
            use_dynamic_intensity_mapping=use_dynamic_intensity_mapping,
            has_needle=has_needle, is_synthetic=is_synthetic, ROI=ROI, angle=angle,
            needle_tip_pos=needle_tip_pos, RANGE_PARAMS=RANGE_PARAMS, masses=masses,
            stiffnesses=stiffnesses, damping=damping, seg_img_0=seg_img_0,
            patch_centers_class=patch_centers_class, parameter_evolution=parameter_evolution
        )
        
    total_duration = time.time() - sonification_start_time
    run_timestamp = int(sonification_start_time)
    folder_basename = os.path.basename(folder)
    
    # Create run folder name with script type, data folder and timestamp
    date_str = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    if config.script_type:
        run_folder_name = f"{model_type}_{config.script_type}_{folder_basename}_{date_str}"
    else:
        run_folder_name = f"{model_type}_{folder_basename}_{date_str}"
    
    runs_base_path = os.path.join(os.path.dirname(folder), "runs")
    os.makedirs(runs_base_path, exist_ok=True)
    run_folder_path = os.path.join(runs_base_path, run_folder_name)
    os.makedirs(run_folder_path, exist_ok=True)
    
    print(f"📁 Created run folder: {run_folder_path}")
    
    # Save run configuration with JSON-safe conversion
    run_config = config.to_dict()
    run_config.update({
        'folder_path': folder,
        'folder_basename': folder_basename,
        'timestamp': run_timestamp,
        'date_created': date_str,
        'total_duration_seconds': total_duration,
        'sound_model_config': sound_model_config
    })
    
    # Save class changes log with JSON-safe conversion
    def make_json_safe(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    safe_class_changes = []
    for change in class_changes_log:
        safe_change = {k: make_json_safe(v) for k, v in change.items()}
        safe_class_changes.append(safe_change)
    
    class_changes_summary = {
        'total_duration_seconds': total_duration,
        'total_class_changes': len(class_changes_log),
        'changes': safe_class_changes
    }

    if sample_name is not None:
        run_config['sample_name'] = sample_name
        class_changes_summary['sample_name'] = sample_name
    
    log_path = os.path.join(run_folder_path, "class_changes_log.json")
    with open(log_path, 'w') as f:
        json.dump(class_changes_summary, f, indent=2, default=str)
    
    # Print summary
    print(f"📊 Session Summary:")
    print(f"   Duration: {total_duration:.3f} seconds")
    print(f"   Class changes: {len(class_changes_log)}")
    print(f"   Data folder: {folder_basename}")
    print(f"   Range parameters: {config.ranges}")
    
    return run_folder_path


def move_recording_and_generate_spectrogram(run_folder_path, class_changes_log, sonification_start_time):
    """
    Move the recording file to the run folder and generate spectrogram.
    
    Args:
        run_folder_path (str): Path to the run folder
        class_changes_log (list): List of class change events
        sonification_start_time (float): Start time of sonification
        
    Returns:
        str: Path to the moved recording file
    """
    # Import here to avoid circular imports
    from .audio_util import generate_and_save_spectrogram
    
    # Move recording file to run folder
    original_recording_path = "/Users/luisreyes/Sonify/SonifyOCT/utilities/inference/recording.wav"
    run_recording_path = os.path.join(run_folder_path, "recording.wav")
    
    # Check if recording exists and move it
    if os.path.exists(original_recording_path):
        shutil.move(original_recording_path, run_recording_path)
        print(f"🎵 Moved recording to: recording.wav")
        
        # Generate and save spectrogram with class change markers in the run folder
        try:
            generate_and_save_spectrogram(
                audio_file_path=run_recording_path,
                class_changes_log=class_changes_log,
                sonification_start_time=sonification_start_time,
                output_dir=run_folder_path,
                show_plot=False
            )
            print(f"📈 Generated spectrogram in run folder")
        except Exception as e:
            print(f"⚠️ Failed to generate spectrogram: {e}")
    else:
        print(f"⚠️ Recording file not found at: {original_recording_path}")
        print(f"⚠️ Skipping spectrogram generation")
        run_recording_path = None
    
    print(f"🗂️  Complete run data saved in: {run_folder_path}")
    
    return run_recording_path