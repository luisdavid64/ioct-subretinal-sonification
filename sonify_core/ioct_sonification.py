import subprocess
import time
from time import sleep
import cv2
import os
import argparse
import numpy as np
import threading
from ioct_sonification_base import BaseIOCTSonification
from utils.util import get_force_sum_for_frame, load_force_data, handle_video_controls
from sonification_main import set_sonification_params, sonify_ILM_RPE, sonify_ascan, send_debug_message, start_recording, stop_recording
from sonification_init import add_ILM_RPE_drivers, sound_model_config_init
from inference import preprocess_and_segment_bscan
from settings_sonify import RANGE_PARAMS_ALL
from needle_tracker import NeedleTracker
from utils.sim_viz import *
from mapping import map_physical_params
from utils.util import *
from forces import compute_node_magnitudes
from extrapolate import extract_line, RPE_LABEL, ILM_LABEL
from utils.processing_utils import kill_process_processing
from sonification_main import transmitter

class VariableRateIOCTSonifier(BaseIOCTSonification):
    def __init__(self, *, sonification_rate, ilm_rpe_rate, **kwargs):
        super().__init__(**kwargs)
        self.sonification_rate = sonification_rate
        self.ilm_rpe_rate = ilm_rpe_rate
        self.sonification_state = {
            'lock': threading.Lock(),
            'latest_node_magnitudes': None,
            'latest_current_class': 0,
            'latest_f_ILM': 0.0,
            'latest_f_RPE': 0.0,
            'smoothed_f_ILM': 0.0,
            'smoothed_f_RPE': 0.0,
            'sonification_ready': False,
            'sep_f_components': False,
            'timer': None,
            'stop_event': threading.Event(),
            'sonification_rate': sonification_rate,
            'ilm_rpe_rate': ilm_rpe_rate,
            'ilm_rpe_timer': None,
            'ilm_rpe_stop_event': threading.Event(),
            'latest_needle_pos': (None, None),
            'roi_bounds': None,
            'latest_dist_to_anatomy': float('inf'),
            'f_ILM_is_active': False,
            'f_ILM_ramp_counter': 0,
            'f_ILM_ramp_duration': 1.0,
        }

    def configure_sonification_state(self):
        state = self.sonification_state
        with state['lock']:
            state['latest_node_magnitudes'] = None
            state['latest_current_class'] = 0
            state['latest_f_ILM'] = 0.0
            state['latest_f_RPE'] = 0.0
            state['smoothed_f_ILM'] = 0.0
            state['smoothed_f_RPE'] = 0.0
            state['sonification_ready'] = False
            state['sep_f_components'] = self.sep_f_components
            state['sonification_rate'] = self.sonification_rate
            state['ilm_rpe_rate'] = self.ilm_rpe_rate
            state['latest_needle_pos'] = (None, None)
            state['roi_bounds'] = None
            state['latest_dist_to_anatomy'] = float('inf')
            state['f_ILM_is_active'] = False
            state['f_ILM_ramp_counter'] = 0
        state['timer'] = None
        state['ilm_rpe_timer'] = None
        state['stop_event'].clear()
        state['ilm_rpe_stop_event'].clear()
        return state

    def calculate_f_ilm_ramp_intensity(self):
        with self.sonification_state['lock']:
            return self.calculate_ramp_intensity(
                is_active=self.sonification_state['f_ILM_is_active'],
                ramp_counter=self.sonification_state['f_ILM_ramp_counter'],
                sonification_rate=self.sonification_state['sonification_rate'],
                ramp_duration=self.sonification_state['f_ILM_ramp_duration'],
            )

    def update_f_ilm_ramp_state(self, f_ilm_active):
        with self.sonification_state['lock']:
            is_active, ramp_counter = self.update_ramp_state(
                self.sonification_state['f_ILM_is_active'],
                self.sonification_state['f_ILM_ramp_counter'],
                f_ilm_active,
            )
            started = f_ilm_active and not self.sonification_state['f_ILM_is_active']
            stopped = (not f_ilm_active) and self.sonification_state['f_ILM_is_active']
            self.sonification_state['f_ILM_is_active'] = is_active
            self.sonification_state['f_ILM_ramp_counter'] = ramp_counter

        if started:
            print("🔊 f_ILM ramping started - beginning soft")
        elif stopped:
            print("🔇 f_ILM ramping stopped - resetting")

    def sonification_timer_callback(self):
        try:
            if not self.sonification_state['sonification_ready']:
                return

            with self.sonification_state['lock']:
                node_magnitudes = self.sonification_state['latest_node_magnitudes'].copy() if self.sonification_state['latest_node_magnitudes'] is not None else None
                sep_f_components = self.sonification_state['sep_f_components']

            if node_magnitudes is None:
                return

            sonify_ascan(forces=node_magnitudes, separate_f_components=sep_f_components)
        except Exception as e:
            print(f'❌ Sonification callback error: {str(e)}')

    def ilm_rpe_timer_callback(self):
        try:
            if not self.sonification_state['sonification_ready']:
                return

            with self.sonification_state['lock']:
                current_class = self.sonification_state['latest_current_class']
                f_ILM = self.sonification_state['smoothed_f_ILM']
                f_RPE = self.sonification_state['smoothed_f_RPE']
                needle_pos = self.sonification_state['latest_needle_pos']
                roi_bounds = self.sonification_state['roi_bounds']
                dist_to_anatomy = self.sonification_state['latest_dist_to_anatomy']

            needle_in_roi = False
            if needle_pos[0] is not None and needle_pos[1] is not None and roi_bounds is not None:
                x0, y0, side_x, side_y = roi_bounds
                needle_in_roi = (x0 <= needle_pos[0] <= x0 + side_x and y0 <= needle_pos[1] <= y0 + side_y)

            needle_close_to_anatomy = dist_to_anatomy < 50.0

            if current_class != 0 and (needle_in_roi or needle_close_to_anatomy):
                f_ILM_should_be_active = abs(f_ILM + f_RPE) > 1
                self.update_f_ilm_ramp_state(f_ILM_should_be_active)

                if f_ILM_should_be_active:
                    ramp_intensity = self.calculate_f_ilm_ramp_intensity()
                    ramped_f_ILM = f_ILM * ramp_intensity
                    sonify_ILM_RPE(f_ilm=ramped_f_ILM * 3 / 5, f_rpe=0)
            else:
                self.update_f_ilm_ramp_state(False)
        except Exception as e:
            print(f'❌ ILM/RPE sonification callback error: {str(e)}')

    def start_sonification_timer(self):
        def timer_worker():
            timer_period = 1.0 / self.sonification_state['sonification_rate']
            print(f'🎵 Sonification timer started at {self.sonification_state["sonification_rate"]}Hz ({timer_period*1000:.1f}ms period)')

            while not self.sonification_state['stop_event'].is_set():
                self.sonification_timer_callback()
                time.sleep(timer_period)

        timer_thread = threading.Thread(target=timer_worker, daemon=True)
        timer_thread.start()
        self.sonification_state['timer'] = timer_thread

        def ilm_rpe_worker():
            ilm_rpe_period = 1.0 / self.sonification_state['ilm_rpe_rate']
            print(f'🎵 ILM/RPE timer started at {self.sonification_state["ilm_rpe_rate"]}Hz ({ilm_rpe_period*1000:.1f}ms period)')
            while not self.sonification_state['ilm_rpe_stop_event'].is_set():
                self.ilm_rpe_timer_callback()
                time.sleep(ilm_rpe_period)

        ilm_rpe_thread = threading.Thread(target=ilm_rpe_worker, daemon=True)
        ilm_rpe_thread.start()
        self.sonification_state['ilm_rpe_timer'] = ilm_rpe_thread
        return timer_thread

    def stop_sonification_timer(self):
        self.sonification_state['stop_event'].set()
        self.sonification_state['ilm_rpe_stop_event'].set()
        if self.sonification_state['timer']:
            self.sonification_state['timer'].join(timeout=1.0)
        if self.sonification_state['ilm_rpe_timer']:
            self.sonification_state['ilm_rpe_timer'].join(timeout=1.0)
        print('🎵 Sonification timers stopped')


def parameterize_and_sonify_oct(
        folder, 
        num_nodes_x=5, 
        num_nodes_y=60, 
        extend_roi_to_needle_tip=False, 
        add_margin_to_roi=80, 
        sep_f_components=False, 
        remap_with_segmentation=True,
        use_handle_forces=False,
        static_mapping_type="dClass",
        dynamic_detuning=False,
        use_deflection_scaling=True,
        deflection_debug=False,
        use_dynamic_intensity_mapping=True,
        separate_ilm=True,
        refine_with_sam=[],
        ranges="INCREMENTAL",
        dynamic_segs=True,
        simulator_path="./scripts/simulator_script.sh",
        save_video=True,
        use_confidence_weights=True,
        include_retina_in_ilm_rpe_drivers=True,
        thickness_statistic="median",
        target_resolution=(512, 512),  # (width, height) - normalize to injection resolution
        sonification_rate=60.0,  # Hz - A-scan sonification frequency
        ilm_rpe_rate=10.0,  # Hz - ILM/RPE boundary sonification frequency
        use_huber_regressor=True,  # Use Huber regressor instead of polyfit for robust fitting
        max_needle_position_jump=100.0,  # Maximum allowed needle position jump in pixels to prevent artifacts
    ):
    print(f"Visualizing folder: {folder}")
    deforming = False

    if "injection" not in folder.lower():
        extend_roi_to_needle_tip = True
        add_margin_to_roi = 120

    assert static_mapping_type in ["intensity", "dRPE", "dClass"], "static_mapping_type must be 'intensity', 'dRPE' or 'dClass'"
    session = VariableRateIOCTSonifier(
        sonification_rate=sonification_rate,
        ilm_rpe_rate=ilm_rpe_rate,
        num_nodes_x=num_nodes_x,
        num_nodes_y=num_nodes_y,
        extend_roi_to_needle_tip=extend_roi_to_needle_tip,
        add_margin_to_roi=add_margin_to_roi,
        sep_f_components=sep_f_components,
        static_mapping_type=static_mapping_type,
        separate_ilm=separate_ilm,
        ranges=ranges,
        simulator_path=simulator_path,
        include_retina_in_ilm_rpe_drivers=include_retina_in_ilm_rpe_drivers,
        thickness_statistic=thickness_statistic,
        use_confidence_weights=use_confidence_weights,
        target_resolution=target_resolution,
        max_needle_position_jump=max_needle_position_jump,
    )
    sonification_state = session.configure_sonification_state()
    print(f"🎵 Sonification rate set to {sonification_rate}Hz, ILM/RPE rate set to {ilm_rpe_rate}Hz")

    # Load force data
    force_data = load_force_data(folder)

    # Get label.json
    label_file = os.path.join(folder, "label.json")
    labels = {}

    # Return if folder name contains _seg already
    if "_seg" in os.path.basename(folder):
        return

    initial_data = session.prepare_offline_initial_data(
        folder,
        add_margin_to_roi,
        fallback_loader=lambda frame_path, frame: preprocess_and_segment_bscan(frame_path),
    )
    if initial_data is None:
        print(f"No frames found in folder {folder}. Skipping...")
        return

    frame_files = initial_data.frame_files
    seg_files_path = initial_data.seg_files_path
    seg_files = initial_data.seg_files

    # Other control variables
    paused = False
    label = ""
    show_force = True
    prev_class = None
    debug = False
 
    # Control information
    print(f"Found {len(frame_files)} frames")
    print("Controls:")
    print("  Space: Pause/Resume")
    print("  B: Jump to beginning")
    print("  Left Arrow: Previous frame")
    print("  Right Arrow: Next frame")
    print("  G: Toggle Gaussian blur")
    print("  F: Toggle force display mode")
    print("  Escape: Exit")

    index = 0

    frame_0 = initial_data.frame_0
    seg_img_0 = initial_data.seg_img_0
    global_confidence = initial_data.initial_confidence
    scale_x = initial_data.scale_x
    scale_y = initial_data.scale_y
    add_margin_to_roi = initial_data.add_margin_to_roi
    if target_resolution is not None and (abs(scale_x - 1.0) > 0.05 or abs(scale_y - 1.0) > 0.05):
        target_width, target_height = target_resolution
        print(f"🔄 Resizing from {initial_data.original_shape[1]}x{initial_data.original_shape[0]} to {target_width}x{target_height}")
        print(f"   Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        print(f"   Scaled add_margin_to_roi: {add_margin_to_roi}")

    geometry = session.prepare_initial_geometry(
        frame_0,
        seg_img_0,
        rpe_thickness_pixels=5 if "injection" in folder.lower() else 3,
        injection_mode="injection" in folder.lower(),
        use_huber_regressor=use_huber_regressor,
        margin=add_margin_to_roi,
        extend_tip=extend_roi_to_needle_tip,
    )
    seg_img_0 = geometry.seg_img_0
    ILM_0 = geometry.ilm_line
    RPE_0 = geometry.rpe_line
    frame_0_rotated = geometry.frame_0_rotated
    seg_0_rotated = geometry.seg_0_rotated
    needle_tip_pos = geometry.needle_tip_pos
    ROI = geometry.roi
    rotated_line_points = geometry.rotated_line_points
    M = session.M
    angle = session.angle

    state = session.state
    target_thickness = session.target_thickness
    prev_thickness = session.prev_thickness.copy()
    prev_ILM_line = session.prev_ILM_line.copy()
    prev_RPE_line = session.prev_RPE_line.copy()

    # === DEFORMATION TRACKING INITIALIZATION ===
    deformation_ref = 5
    JITTER_MAX_HZ = 4.0

    
    
    # seg_img_0[seg_img_0 == 2] = 4
    seg_colors = [(255,255,0), (0,255,255), (255,0,255), (255,255,255)]
    seg_img_colored = frame_0.copy()
    for i, label in enumerate(np.unique(seg_img_0)):
        if label == 0:
            continue  # skip background
        mask = (seg_img_0 == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = seg_colors[i % len(seg_colors)]
        cv2.drawContours(seg_img_colored, contours, -1, color, 1)
    
    # Store ROI dimensions needed for dynamic segmentation
    side_x, side_y = None, None

    print("Initial Needle Position and ROI alignment")

    # Show frame_0 and seg_img_0 with cv2
    if debug:
        cv2.imshow("Initial Position 0", frame_0)
        cv2.imshow("Initial Position 0", seg_img_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    seg_0_rotated_rgb = cv2.warpAffine(seg_img_colored, M, (frame_0.shape[1], frame_0.shape[0]))

    # === INITIALIZE ROTATED FRAME THICKNESS TRACKING AFTER M IS DEFINED ===
    # Apply rotation to initial segmentation for accurate thickness measurements
    ILM_0_rotated = extract_line(seg_0_rotated, ILM_LABEL)
    RPE_0_rotated = extract_line(seg_0_rotated, RPE_LABEL)
    baseline_thickness_rotated = RPE_0_rotated - ILM_0_rotated
    prev_thickness_rotated = baseline_thickness_rotated.copy()
    prev_ILM_line_rotated = ILM_0_rotated.copy()
    prev_RPE_line_rotated = RPE_0_rotated.copy()

    if ROI is None:
        x0, y0, side_x, side_y = 0, 0, 1, 1
    else:
        x0, y0, side_x, side_y = ROI
        # Visualize ROI
        frame_with_roi = frame_0_rotated.copy()
        if debug:
            cv2.rectangle(frame_with_roi, (x0, y0), (x0 + side_x, y0 + side_y), (0, 0, 255), 2)
            cv2.polylines(frame_with_roi, [rotated_line_points], False, (0, 255, 0), 1)
            cv2.imshow("ROI Intersection", frame_with_roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    """ Initialize Sound Model """
    
    model_setup = session.initialize_sound_model_setup(
        geometry,
        classification_mode="median",
        enforce_consistency=False,
        debug=debug,
    )
    frame_roi = model_setup.frame_roi
    seg_roi = model_setup.seg_roi
    rotated_patch_centers = model_setup.rotated_patch_centers
    patch_centers_class = model_setup.patch_centers_class
    sound_model_config = model_setup.sound_model_config
    RANGE_PARAMS = model_setup.range_params
    masses = model_setup.masses
    stiffnesses = model_setup.stiffnesses
    damping = model_setup.damping
    node_info = model_setup.node_info
    scales = session.scales

    """Start simulator with given path using routine"""
    # Start subprocess in new process group to capture all child processes (Java/Processing)
    simulator_process = session.start_simulator(
        suppress_output=True,
        new_process_group=True,
    )
    """ Visualization before Sonification Loop """
    
    print("Starting Sonification. Close the visualization window to end.")
    # Open Viw until user opens sonification window
    session.show_startup_preview(frame_roi, wait_ms=5000)

    """ Set Physical Parameters and get superimposed node positions"""

    set_sonification_params(sound_model_config, masses, stiffnesses, damping, patch_centers_class.reshape((num_nodes_y, num_nodes_x)))
    send_debug_message("Sonification parameters set.")
    
    print("Beginning Sonification Loop")

    # Initialize fixed ROI parameters for deflection scaling (using the existing ROI calculation)
    # Store both the ROI coordinates and rotation matrix for proper frame alignment
    fixed_roi_params = None
    rotation_matrix = M if M is not None else np.eye(2, 3, dtype=np.float32)  # Store rotation matrix or identity

    # Audio recording will start with first frame to sync with video
    prev_frame = frame_0.copy()

    # Initialize timing for class changes
    sonification_start_time = time.time()
    class_changes_log = []

    # Initialize parameter tracking for dynamic intensity mapping
    parameter_evolution = {
        'frame_indices': [],
        'timestamps': [],
        'class_averages': {
            'masses': {},      # {class_id: [values_over_time]}
            'stiffnesses': {},
            'damping': {}
        }
    }

    M_t = M.copy() if M is not None else np.eye(2, 3, dtype=np.float32)
    W,H = frame_0.shape[1], frame_0.shape[0]

    del frame_0_rotated, seg_0_rotated, seg_0_rotated_rgb

    # === INITIALIZE NEEDLE TIP TRACKER ===
    # Always create a NeedleTracker but configure it based on dynamic_segs parameter
    tracker = NeedleTracker(
        init_frame=frame_0, 
        tip_pos=get_needle_tip_pos_from_seg(seg_img_0 == 1),
        alpha=0.6,  # Same smoothing as tissue_reorient
        init_segmentation=seg_img_0 if "injection" not in folder.lower() else None
    )
    
    print(f"Using {'segmentation' if dynamic_segs else 'template'}-based needle tracking")
    baseline_frame = frame_0.copy()  # Store first frame as baseline for temporal comparison

    # === INITIALIZE SPLINE-BASED LINE FITTING ===
    spline_prior_lines = [ILM_0.copy(), RPE_0.copy()]
    print(f"🔧 Initialized spline-based fitting:")
    print(f"   RPE line: {np.sum(~np.isnan(RPE_0))}/{len(RPE_0)} valid points")
    print(f"   ILM line: {np.sum(~np.isnan(ILM_0))}/{len(ILM_0)} valid points")

    # === INITIALIZE BACKGROUND SPLINE FITTING ===
    session.start_spline_worker(debug=False)
    print("🚀 Started background spline fitting thread")
    
    # === INITIALIZE VIDEO WRITER ===
    video_writer = None
    video_frames_buffer = []  # Buffer frames with timestamps
    video_start_time = None
    if save_video:
        print(f"📹 Dynamic framerate video recording enabled")

    frame_last = None

    ex_scale = 1
    audio_started = False  # Flag to track audio recording state
    frame_deformation_history = []
    deformation_norm = 0
    f_ILM, f_RPE = 0.0, 0.0
    
    # Smoothing parameters for f_ILM and f_RPE
    f_smoothing_alpha = 0.2  # Lower = more smoothing (0.1 = heavy smoothing, 0.5 = light smoothing)
    
    def smooth_f_values(new_f_ILM, new_f_RPE):
        """Apply exponential moving average smoothing to f_ILM and f_RPE"""
        smoothed_f_ilm, smoothed_f_rpe = session.smooth_force_values(
            new_f_ILM,
            new_f_RPE,
            alpha=f_smoothing_alpha,
        )
        with sonification_state['lock']:
            sonification_state['smoothed_f_ILM'] = smoothed_f_ilm
            sonification_state['smoothed_f_RPE'] = smoothed_f_rpe
    
    deformation_history = []  # Store recent deformation values for median calculation
    median_filter_size = 30   # Number of frames to consider for median
    
    # Start sonification timer before main loop
    session.start_sonification_timer()
    print("🎵 Sonification timer started - audio synthesis separated from video processing")
    class2_mask = session.class2_mask
    
    while index < len(frame_files):
        frame_file, frame = session.load_frame_at_index(folder, frame_files, index)
        frame_last = frame
        
        if frame is None:
            print(f"Could not load frame: {frame_file}")
            index += 1
            continue

        # Update label if this frame has one
        # if index in labels.values():
            # label = time_to_labels[index]

        # === NEEDLE POSITION TRACKING & DYNAMIC SEGMENTATION ===
        seg_img_current = None
        if remap_with_segmentation or dynamic_segs:
            seg_file = os.path.join(seg_files_path, seg_files[index]) if index < len(seg_files) else None
            seg_img_current, global_confidence = session.load_runtime_segmentation(
                seg_file,
                fallback_loader=lambda: preprocess_and_segment_bscan(frame_file),
                cls_to_use=(4 if separate_ilm else 2),
            )
        
        frame_file = os.path.join(folder, frame_files[index])
        tracked_tip, max_val = tracker.update(frame, segmentation_map=seg_img_current)
        
        needle_tip_pos = session.filter_tracked_tip(
            tracked_tip,
            max_jump=max_needle_position_jump,
        )
        if needle_tip_pos is None:
            needle_tip_pos = (None, None)

        if debug:
            if needle_tip_pos[0] is not None and needle_tip_pos[1] is not None:
                cv2.circle(frame, (int(needle_tip_pos[0]), int(needle_tip_pos[1])), 8, (0, 255, 0), 2)
                cv2.putText(frame, f"Tip: ({needle_tip_pos[0]:.1f}, {needle_tip_pos[1]:.1f})", 
                           (int(needle_tip_pos[0]) + 10, int(needle_tip_pos[1]) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        closest_node_pos, closest_node_index, dist, current_class = session.find_closest_node_and_class(
            needle_tip_pos,
            rotated_patch_centers,
            patch_centers_class,
        )
        
        if closest_node_index is not None and debug:
            cv2.circle(frame, (int(closest_node_pos[0]), int(closest_node_pos[1])), 6, (0, 255, 0), 2)
        
        class_changed = (prev_class is not None and current_class != prev_class 
                        and [prev_class, current_class] != [4,2] 
                        )
        if class_changed and current_class == 4 and prev_class == 0:
            snapped_pos, snapped_index, snapped_dist = session.snap_to_nearest_class_row(
                needle_tip_pos,
                rotated_patch_centers,
                class2_mask,
            )
            if snapped_index is not None:
                row = snapped_index[0]
                closest_node_index = snapped_index
                closest_node_pos = snapped_pos
                current_class = 2
                dist = snapped_dist
                print(f"⚠️  Class 0→4 jump detected — snapped to nearest ILM (class 2) node at row {row}")

        force_info = get_force_sum_for_frame(force_data, index)
            
        # Compute needle tip speed as force proxy if force_info is None
        needle_speed = 0.0
        velocity_x = 0.0  # Initialize for use in force proxy calculation
        velocity_y = 0.0  # Initialize for use in force proxy calculation
            
        if session.prev_needle_pos_for_speed is not None and needle_tip_pos[0] is not None:
            prev_pos = session.prev_needle_pos_for_speed
            if prev_pos[0] is not None:
                # Calculate velocity (pixels per frame)
                velocity_x = needle_tip_pos[0] - prev_pos[0]
                velocity_y = needle_tip_pos[1] - prev_pos[1]
                needle_speed = np.sqrt(velocity_x**2 + velocity_y**2)
                    
                # Display speed info if no force data available
                if force_info is None and debug:
                    cv2.putText(frame, f"Needle Speed: {needle_speed:.2f} px/frame", 
                                (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                    cv2.putText(frame, f"Velocity: ({velocity_x:.1f}, {velocity_y:.1f})", 
                                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            
        # Store current position for next frame
        session.prev_needle_pos_for_speed = needle_tip_pos
        
        # === BACKGROUND SPLINE-BASED LINE FITTING ===
        # Submit spline fitting task every 5 frames (non-blocking)
        if remap_with_segmentation and index % 3 == 0 and seg_img_current is not None:
            task_queued = session.queue_spline_task(index, seg_img_current)
            if debug and task_queued:
                print(f"📤 Frame {index}: Submitted spline fitting task to background thread")
            if debug and not task_queued:
                print(f"⚠️  Frame {index}: Spline fitting queue full, skipping this frame")
        
        # Check for completed spline fitting results (non-blocking)
        for result in session.collect_recent_spline_results(index, max_age=10):
            result_frame = result['frame_index']
            seg_img_current = result['fitted_seg']
            
            spline_prior_lines = [result['updated_lines']['ILM'].copy(), result['updated_lines']['RPE'].copy()]
            ILM_line, RPE_line = spline_prior_lines
                    
            if debug:
                print(f"📥 Frame {index}: Applied spline result from frame {result_frame}")
                print(f"   RPE confidence: {result['confidences']['RPE']:.3f}")
                print(f"   ILM confidence: {result['confidences']['ILM']:.3f}")
            
            node_update_rate = 0.99
            rotated_patch_centers, ILM_line, RPE_line, confidence = session.apply_spline_result_to_patch_centers(
                result,
                rotated_patch_centers,
                node_info,
                current_class,
                gamma=node_update_rate,
            )

            if np.isnan(confidence):
                confidence = 0.5
                print(f"⚠️  NaN confidence detected, using default 0.5")
            
            # Apply confidence-based adjustments for all classes except background (0)
            if use_confidence_weights:
                C_min = 0.5
                print(global_confidence )
                u = np.clip((C_min - global_confidence) / C_min, 0.0, 1.0)
                ex_scale = max(0.2, (1 - u)**2) 
                mult = 1 + (20 * u**2)  
                mult = min(mult, 20.0)  
                
                jitter_rate = u**2 * JITTER_MAX_HZ
                jitter_amplitude = 0.1 + u * 0.8
                jitter_cutoff = 0.2 + u * 0.3
                
                transmitter.oscTransmittProc_SetJitter(jitter_rate, jitter_amplitude, jitter_cutoff)
            # === DEFORMATION-BASED MEASUREMENT IN ROTATED FRAME ===
            rotated_seg = result['fitted_seg']
            if M_t is not None:
                rotated_seg = cv2.warpAffine(rotated_seg, M_t, (rotated_seg.shape[1], rotated_seg.shape[0]), flags=cv2.INTER_NEAREST)
            
            ILM_line_rotated = extract_line(rotated_seg, ILM_LABEL)
            RPE_line_rotated = extract_line(rotated_seg, RPE_LABEL)
            
            # Calculate thickness in rotated (horizontal-tissue) coordinate system
            current_thickness_rotated = RPE_line_rotated - ILM_line_rotated
            
            # Transform needle tip position to rotated coordinates for consistent window calculation
            if needle_tip_pos[0] is not None and M_t is not None:
                needle_point = np.array([[needle_tip_pos[0], needle_tip_pos[1]]], dtype=np.float32)
                needle_rotated = cv2.transform(needle_point.reshape(1, 1, 2), M_t).reshape(2)
                needle_x_rotated = int(needle_rotated[0])
            else:
                needle_x_rotated = int(needle_tip_pos[0]) if needle_tip_pos[0] is not None else 40
            
            # Use rotated coordinates for window calculation
            context_window = 40
            needle_window_rotated = [needle_x_rotated - context_window, needle_x_rotated + context_window]
            needle_window_rotated[0] = max(0, needle_window_rotated[0])
            needle_window_rotated[1] = min(len(ILM_line_rotated), needle_window_rotated[1])
            
            # Compute separate ILM and RPE boundary changes in rotated frame
            delta_ILM, delta_RPE = None, None
            if thickness_statistic == 'median':
                delta_ILM = np.nanmedian(np.abs(ILM_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]] - prev_ILM_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]]))
                delta_RPE = np.nanmedian(np.abs(RPE_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]] - prev_RPE_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]]))
            elif thickness_statistic == 'mean':
                delta_ILM = np.nanmean(np.abs(ILM_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]] - prev_ILM_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]]))
                delta_RPE = np.nanmean(np.abs(RPE_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]] - prev_RPE_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]]))
            elif thickness_statistic == 'max':
                delta_ILM = np.nanmax(np.abs(ILM_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]] - prev_ILM_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]]))
                delta_RPE = np.nanmax(np.abs(RPE_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]] - prev_RPE_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]]))
            elif thickness_statistic == '95th_percentile':
                delta_ILM = np.nanpercentile(np.abs(ILM_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]] - prev_ILM_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]]), 95)
                delta_RPE = np.nanpercentile(np.abs(RPE_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]] - prev_RPE_line_rotated[needle_window_rotated[0]:needle_window_rotated[1]]), 95)
            
            if np.isnan(delta_ILM):
                delta_ILM = 0.0
            if np.isnan(delta_RPE):
                delta_RPE = 0.0
            
            delta_thickness_rotated = np.nanmedian(np.abs(current_thickness_rotated[needle_window_rotated[0]:needle_window_rotated[1]] - prev_thickness_rotated[needle_window_rotated[0]:needle_window_rotated[1]]))
            if np.isnan(delta_thickness_rotated):
                delta_thickness_rotated = 0.0
            
            deformation_raw = 1.0 * delta_thickness_rotated

            if current_class == 0:
                deformation_history.append(deformation_raw)
                if len(deformation_history) > median_filter_size:
                    deformation_history = deformation_history[-median_filter_size:]
                if debug and deformation_raw > 0.1:
                    print(f"📊 Baseline deformation recorded: {deformation_raw:.3f} (history size: {len(deformation_history)})")
            
            if len(deformation_history) >= 5:
                current_median = np.median(deformation_history)
                threshold_multiplier = 1.2
                calculated_threshold = current_median * threshold_multiplier
                
                min_threshold = 0.3
                sonification_threshold = max(calculated_threshold, min_threshold)
                
                if deformation_raw > sonification_threshold and current_class != 0:
                    f_ILM = -np.clip(deformation_raw, 0, 2)
                    print(f"🎵 f_ILM TRIGGERED (rotated): deformation={deformation_raw:.3f} > threshold={sonification_threshold:.3f} (baseline_median={current_median:.3f})")
                else:
                    f_ILM = 0.0
                    if current_class != 0 and deformation_raw > 0.1:
                        print(f"🔇 f_ILM FILTERED (rotated): deformation={deformation_raw:.3f} <= threshold={sonification_threshold:.3f} (baseline_median={current_median:.3f})")
                
                if current_median < 0.1 and debug:
                    print(f"⚠️  Low baseline detected: median={current_median:.3f}, using min_threshold={min_threshold:.3f}")
            else:
                f_ILM = -np.clip(deformation_raw, 0, 2) if deformation_raw > 0.5 and current_class != 0 else 0.0
                if len(deformation_history) < 5 and current_class == 0:
                    print(f"📈 Collecting baseline data: {len(deformation_history)}/5 samples needed")
            if "injection" not in folder.lower():
                f_ILM *= 0.05
            
            prev_thickness_rotated = current_thickness_rotated.copy()
            prev_ILM_line_rotated = ILM_line_rotated.copy()
            prev_RPE_line_rotated = RPE_line_rotated.copy()
        
        magnitudes = np.ones_like([0,0,0]) * 1 * (20/sonification_rate) * scales[current_class] # TEMPORARY FIX TO CONSTANT FORCE FOR TESTING
        
        
        deflection_scaling = 1.0

        # Spread forces across nodes
        node_magnitudes = compute_node_magnitudes(num_nodes_y, magnitudes,
                                                closest_node_index, dist, deflection_scaling, sigma=3)

        # Handle class transitions
        if class_changed:
            print(f"⚡ Class change detected: {prev_class} → {current_class}")
            node_magnitudes *= 2.0
            class_changes_log.append(
                session.create_class_change_record(
                    index,
                    prev_class,
                    current_class,
                    needle_tip_pos,
                    sonification_start_time,
                )
            )
            
        prev_class = current_class
        
        # Update sonification state thread-safely (sonification happens on timer)
        with sonification_state['lock']:
            sonification_state['latest_node_magnitudes'] = node_magnitudes.copy()
            sonification_state['latest_current_class'] = current_class
            sonification_state['latest_f_ILM'] = f_ILM
            sonification_state['latest_f_RPE'] = f_RPE
            sonification_state['latest_needle_pos'] = needle_tip_pos
            sonification_state['roi_bounds'] = (x0, y0, side_x, side_y) if 'x0' in locals() else None
            sonification_state['latest_dist_to_anatomy'] = abs(dist) if dist != 0 else float('inf')
            sonification_state['sonification_ready'] = True  # Mark that we have valid data
        
        # Apply smoothing to f_ILM and f_RPE outside the lock to avoid deadlock
        smooth_f_values(f_ILM, f_RPE)
        ## CONTROLS
        cv2.imshow('Video Visualization Tool', frame)
        # Show overlay of segmentation: ILM is yellow, RPE is blue and needle is red
        # Everything else is 0 (black)
        display_frame = frame  # Default to original frame
        # if seg_img_orig is not None:
        #     overlay = np.zeros_like(frame)
        #     overlay[seg_img_orig == ILM_LABEL] = (0, 255, 255)  # Yellow for ILM
        #     overlay[seg_img_orig == RPE_LABEL] = (255, 0, 0)    # Blue for RPE
        #     if needle_tip_pos[0] is not None and needle_tip_pos[1] is not None:
        #         cv2.circle(overlay, (int(needle_tip_pos[0]), int(needle_tip_pos[1])), 8, (0, 0, 255), -1)  # Red for needle tip
        #     blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        #     display_frame = blended  # Use blended frame for both display and recording
        #     cv2.imshow('Video Visualization Tool', blended)
        
        # Record frame with timestamp for dynamic video
        if save_video:
            video_start_time, audio_started = session.buffer_video_frame(
                video_frames_buffer,
                display_frame,
                video_start_time,
                audio_started=audio_started,
                start_audio_callback=start_recording,
            )
        
        # Handle controls and get updated state
        control_result = handle_video_controls(
            paused, index, len(frame_files), show_force
        )
        
        # Unpack results
        should_exit = control_result['exit']
        paused = control_result['paused']
        index = control_result['index']
        show_force = control_result['show_force']
        
        # Break if exit was requested
        if should_exit:
            break
    
    cv2.destroyAllWindows()
    
    # Stop audio recording immediately when loop exits to match video timing
    if audio_started:
        stop_recording()

    # Clean up sonification timer
    session.stop_sonification_timer()
    
    # Create run folder first to save video there
    folder_name = os.path.basename(folder)
    config = SonificationConfig(
        script_type=folder_name,  # Use folder name instead of generic tracking type
        num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y,
        extend_roi_to_needle_tip=extend_roi_to_needle_tip,
        add_margin_to_roi=add_margin_to_roi,
        sep_f_components=sep_f_components,
        remap_with_segmentation=remap_with_segmentation,
        use_handle_forces=use_handle_forces,
        static_mapping_type=static_mapping_type,
        dynamic_detuning=dynamic_detuning,
        use_deflection_scaling=use_deflection_scaling,
        deflection_debug=deflection_debug,
        separate_ilm=separate_ilm,
        refine_with_sam=refine_with_sam,
        ranges=ranges,
        simulator_path=simulator_path,
        use_dynamic_intensity_mapping=use_dynamic_intensity_mapping,
        # Pass computed data with appropriate defaults
        has_needle=True,  # Manual tracking assumes needle presence
        is_synthetic=False,  # Manual tracking uses real data
        ROI=ROI if 'ROI' in locals() else None,
        angle=0,  # Default angle
        needle_tip_pos=needle_tip_pos if 'needle_tip_pos' in locals() else (None, None),
        RANGE_PARAMS=RANGE_PARAMS if 'RANGE_PARAMS' in locals() else None,
        masses=masses if 'masses' in locals() else None,
        stiffnesses=stiffnesses if 'stiffnesses' in locals() else None,
        damping=damping if 'damping' in locals() else None,
        seg_img_0=seg_img_0 if 'seg_img_0' in locals() else None,
        patch_centers_class=patch_centers_class if 'patch_centers_class' in locals() else None,
        parameter_evolution=parameter_evolution if 'parameter_evolution' in locals() else None
    )
    
    run_folder_path = create_run_folder_and_save_data(
        folder="inference",
        sonification_start_time=sonification_start_time,
        class_changes_log=class_changes_log,
        sound_model_config=sound_model_config if 'sound_model_config' in locals() else None,
        config=config,
        sample_name=folder.split(os.sep)[-1]
    )
    
    # Create dynamic framerate video and save to run folder
    if save_video and len(video_frames_buffer) > 1:
        print(f"🎬 Creating synced video with {len(video_frames_buffer)} frames...")
        
        # Calculate dynamic framerate based on actual viewing time
        total_time = video_frames_buffer[-1][1] - video_frames_buffer[0][1]
        if total_time > 0:
            avg_framerate = len(video_frames_buffer) / total_time
            # Clamp framerate to reasonable bounds
            avg_framerate = max(5.0, min(avg_framerate, 60.0))
        else:
            avg_framerate = 30.0
        
        # Get video dimensions from first frame
        height, width = video_frames_buffer[0][0].shape[:2]
        video_filename = os.path.join(run_folder_path, 'simulation_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, avg_framerate, (width, height))
        
        # Write all buffered frames
        for frame, timestamp in video_frames_buffer:
            video_writer.write(frame)
        
        video_writer.release()
        
        # Save timing metadata for audio synchronization
        timing_file = os.path.join(run_folder_path, 'video_timing.json')
        timing_data = {
            'total_frames': len(video_frames_buffer),
            'total_duration_seconds': total_time,
            'average_framerate': avg_framerate,
            'frame_timestamps': [timestamp for _, timestamp in video_frames_buffer],
            'video_filename': 'synced_visualization.mp4'
        }
        
        import json
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2)
        
        print(f"📹 Synced video saved: {video_filename}")
        print(f"⏱️  Video duration: {total_time:.2f}s at {avg_framerate:.1f} FPS")
        print(f"📋 Timing data saved: {timing_file}")
    elif save_video:
        print("⚠️ No video frames recorded or insufficient frames for video creation")
    
    # Stop background spline fitting thread
    print("🛑 Stopping background spline fitting thread...")
    try:
        session.stop_spline_worker(timeout=2.0)
    except Exception as e:
        print(f"⚠️  Warning during thread cleanup: {e}")

    # Wait for recording to be saved and terminate simulator
    sleep(2)
    
    # Java/Processing-specific termination logic
    kill_process_processing(simulator_process)
    
    # Move recording and generate spectrogram using the same run folder
    try:
        move_recording_and_generate_spectrogram(
            run_folder_path=run_folder_path,
            class_changes_log=class_changes_log,
            sonification_start_time=sonification_start_time
        )
    except Exception as e:
        print(f"Error during recording move/spectrogram generation: {e}")

    return run_folder_path  # Return the path for potential further use


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Visualization Tool")
    parser.add_argument("--folder_mode", default="single", help="Mode: 'single' for single folder, 'batch' for batch processing")
    parser.add_argument("root_folder", help="Path to the root folder containing video frame subfolders")
    parser.add_argument("--refine_with_sam", nargs='+', help="Masks to refine")
    parser.add_argument("--use_dynamic_intensity_mapping", action="store_true", default=False,  # Changed to False for performance
                       help="Enable dynamic intensity-based mass mapping for deformation capture")
    parser.add_argument("--ranges", default="SETUP_SPLINES", help="Parameter ranges configuration")
    parser.add_argument("--dynamic_segs", action="store_true", default=True,
                       help="Use segmentation-based needle tracking instead of template matching")
    parser.add_argument("--simulator_path", default="./scripts/simulator_script.sh")
    parser.add_argument("--save_video", action="store_true", default=True,
                       help="Save the visualization as an MP4 video file")
    parser.add_argument("--sonification_rate", type=float, default=25.0,
                       help="A-scan sonification frequency in Hz (default: 25.0)")
    parser.add_argument("--ilm_rpe_rate", type=float, default=12.0,
                       help="ILM/RPE boundary sonification frequency in Hz (default: 10.0)")
    parser.add_argument("--max_needle_jump", type=float, default=100.0,
                       help="Maximum allowed needle position jump in pixels to prevent artifacts from noise (default: 100.0)")
    args = parser.parse_args()
    
    if args.folder_mode == "single":
        parameterize_and_sonify_oct(
            args.root_folder, 
            refine_with_sam=args.refine_with_sam or [],
            use_dynamic_intensity_mapping=args.use_dynamic_intensity_mapping,
            ranges=args.ranges,
            dynamic_segs=args.dynamic_segs,
            simulator_path=args.simulator_path,
            save_video=args.save_video,
            sonification_rate=args.sonification_rate,
            ilm_rpe_rate=args.ilm_rpe_rate,
            max_needle_position_jump=args.max_needle_jump
        )
    else:
        for folder_name in os.listdir(args.root_folder):
            folder_path = os.path.join(args.root_folder, folder_name)
            if "Sample01" in folder_path:
                continue
            if "segmentation" in folder_path.lower():
                continue
            if os.path.isdir(folder_path):
                parameterize_and_sonify_oct(
                    folder_path, 
                    refine_with_sam=args.refine_with_sam or [],
                    use_dynamic_intensity_mapping=args.use_dynamic_intensity_mapping,
                    ranges=args.ranges,
                    dynamic_segs=args.dynamic_segs,
                    simulator_path=args.simulator_path,
                    save_video=args.save_video,
                    sonification_rate=args.sonification_rate,
                    ilm_rpe_rate=args.ilm_rpe_rate,
                    max_needle_position_jump=args.max_needle_jump
                )
                sleep(5)
