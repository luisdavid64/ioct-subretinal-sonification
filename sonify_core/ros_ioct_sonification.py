#!/usr/bin/env python3
"""
ROS2 Variable-Rate Sonification Node - Complete Migration from variable_rate_lite_spline_based_simulation.py

Maintains ALL functionality from the original file:
- Dual-rate sonification (A-scan + ILM/RPE)
- Background spline fitting
- Deformation tracking with median filtering
- F_ILM ramping mechanism
- Confidence-based adjustments
- Node position updates
- Class change detection

Only difference: Images fed through ROS topics instead of folder
"""

import argparse
import sys
import time
import subprocess
from time import sleep
import numpy as np
import cv2
import queue
import threading
import os

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ioct_sonification_base import BaseIOCTSonification
from segmentation.segmentation_utils import remap_segs

# Sonification imports
from sonification_main import (
    set_sonification_params, sonify_ILM_RPE, 
    sonify_ascan, send_debug_message, start_recording, stop_recording
)
from sonification_init import add_ILM_RPE_drivers, sound_model_config_init
from segmentation.segment_bscan import extrapolate_retina
from settings_sonify import RANGE_PARAMS_ALL
from needle_tracker import NeedleTracker
from utils.sim_viz import *
from mapping import map_physical_params
from utils.util import *
from forces import compute_node_magnitudes
from extrapolate import (
    extract_line,
    RPE_LABEL,
    ILM_LABEL,
)


class VariableRateROSSonificationNode(BaseIOCTSonification, Node):
    def __init__(self, 
                 num_nodes_x=5, 
                 num_nodes_y=60, 
                 extend_roi_to_needle_tip=False, 
                 add_margin_to_roi=80, 
                 sep_f_components=False,
                 static_mapping_type="dClass",
                 separate_ilm=True,
                 ranges="SETUP_SPLINES",
                 simulator_path="./scripts/simulator_script.sh",
                 include_retina_in_ilm_rpe_drivers=True,
                 thickness_statistic="median",
                 sonification_rate=25.0,
                 ilm_rpe_rate=15.0,
                 use_confidence_weights=True,
                 seg_processing="local"
                 ):
        Node.__init__(self, 'variable_rate_ros_sonification_node')
        BaseIOCTSonification.__init__(
            self,
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
        )

        self.sonification_rate = sonification_rate
        self.ilm_rpe_rate = ilm_rpe_rate
        
        # ROS components
        self.bridge = CvBridge()
        qos = QoSProfile(
            depth=1, 
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE
        )
        
        # State variables
        self.initialized = False
        self.current_frame = None
        self.current_segmentation = None
        self.frame_index = 0
        
        # Sonification state
        self.simulator_process = None
        self.audio_started = False
        self.tracker = None
        self.prev_class = None
        
        # Deformation tracking
        self.f_ILM = 0.0
        self.f_RPE = 0.0
        self.deformation_history = []
        self.median_filter_size = 30
        self.deformation_ref = 5.0
        self.frame_deformation_history = []
        
        # Sonification state variables (thread-safe)
        self.sonification_lock = threading.Lock()
        self.latest_node_magnitudes = None
        self.latest_current_class = 0
        self.latest_f_ILM = 0.0
        self.latest_f_RPE = 0.0
        self.smoothed_f_ILM = 0.0
        self.smoothed_f_RPE = 0.0
        self.sonification_ready = False
        self.latest_needle_pos = (None, None)
        self.roi_bounds = None
        self.latest_dist_to_anatomy = float('inf')
        
        # Ramping mechanism for f_ILM
        self.f_ILM_is_active = False
        self.f_ILM_ramp_counter = 0
        self.f_ILM_ramp_duration = 1.0
        
        # Smoothing parameters
        self.f_smoothing_alpha = 0.2
        self.sonification_state = {}
        
        # Timers
        self.sonification_timer = None
        self.ilm_rpe_timer = None
        
        # Class 2 mask for class jump detection
        self.class2_mask = None

        self.seg_processing = seg_processing
        
        # Create subscriptions
        self.image_subscription = self.create_subscription(
            Image,
            '/gan/bscans',
            self.image_callback,
            qos
        )
        
        self.seg_subscription = self.create_subscription(
            Image,
            '/gan/segmentation',
            self.segmentation_callback,
            qos
        )
        
        self.get_logger().info(f'🚀 Variable-Rate ROS Sonification Node initialized')
        self.get_logger().info(f'🎵 Sonification rates: A-scan={sonification_rate}Hz, ILM/RPE={ilm_rpe_rate}Hz')
        self.get_logger().info('⏳ Waiting for image and segmentation...')
    
    def image_callback(self, msg):
        """Receive and store image - processing happens in segmentation callback"""
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.current_frame = self.current_frame[:512, :512]
            if self.seg_processing in ["syntheseyes"]:
                self.current_frame = np.flip(self.current_frame, axis=0)
                
        except Exception as e:
            self.get_logger().error(f'Image callback error: {str(e)}')
    
    def segmentation_callback(self, msg):
        """Receive and store segmentation"""
        try:
            match self.seg_processing:
                case "local":
                    self.current_segmentation = self.bridge.imgmsg_to_cv2(msg, 'mono8')
                case "syntheseyes":
                    self.current_segmentation = self.bridge.imgmsg_to_cv2(msg, '32FC1')
                    self.current_segmentation = self.current_segmentation[:512, :512]  # Ensure consistent size for processing
                    self.current_segmentation = (self.current_segmentation * 255).astype(np.uint8)  # Ensure it's uint8 for remapping
                    self.current_segmentation = np.flip(self.current_segmentation, axis=0)  # Flip horizontally if needed to match image orientation
                    self.current_segmentation = remap_segs(self.current_segmentation) 
                case _:
                    current_segmentation = self.bridge.imgmsg_to_cv2(msg, 'mono8')
                    current_segmentation = current_segmentation[:512, :512]  # Ensure consistent size for processing
                    self.current_segmentation = remap_segs(current_segmentation) 
                
            
            if not self.initialized:
                self.try_initialize()
            else:
                self.process_frame()
                
        except Exception as e:
            self.get_logger().error(f'Segmentation callback error: {str(e)}')
    
    def is_segmentation_empty(self, seg):
        """Check if segmentation is empty (no meaningful labels)"""
        if seg is None:
            return True
        
        # Check for ILM (class 2) and RPE (class 3)
        has_ilm = np.any(seg == 2)
        has_rpe = np.any(seg == 3)
        
        return not (has_ilm and has_rpe)
    
    def reset_state(self):
        """Reset segmentation and spline state only (keep initialization intact)"""
        self.get_logger().warn("🔄 Clearing spline state due to empty segmentation...")
        self.clear_spline_queues()
        
        # Reset spline state to force re-fitting when valid segmentation returns
        if self.state is not None:
            self.state["prior"] = {
                "ILM": self.state["reference"]["ILM"].copy(),
                "RPE": self.state["reference"]["RPE"].copy(),
            }
        
        # Reset previous tracking to avoid using stale data
        self.prev_thickness = None
        self.prev_ILM_line = None
        self.prev_RPE_line = None
        
        # Clear recent deformation tracking (keep history for baseline)
        self.frame_deformation_history = []
        
        self.get_logger().info("✅ Spline state cleared, ready for valid segmentation...")
    
    def try_initialize(self):
        """Attempt initialization when both image and segmentation are available"""
        if self.current_frame is None or self.current_segmentation is None:
            return
        
        # Check for ILM (class 2) and RPE (class 3) instead of needle
        has_ilm = np.any(self.current_segmentation == 2)
        has_rpe = np.any(self.current_segmentation == 3)
        
        if not (has_ilm and has_rpe):
            self.get_logger().warn("⚠️  Waiting for ILM (class 2) and RPE (class 3) in segmentation...")
            return
        
        self.get_logger().info("✅ Both image and segmentation with ILM/RPE available - initializing!")
        self.initialize_sonification()
    
    def initialize_sonification(self):
        """Complete initialization matching variable_rate_lite"""
        frame_0 = self.current_frame.copy()
        seg_img_0 = self.current_segmentation.copy()
        
        self.get_logger().info('📦 Initializing advanced sonification system...')
        
        # Add padding only at bottom to extend space below RPE (no coordinate shift!)
        pad_bottom = 128
        frame_0 = cv2.copyMakeBorder(
            frame_0, 0, pad_bottom, 0, 0,  # top, bottom, left, right
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        seg_img_0 = cv2.copyMakeBorder(
            seg_img_0, 0, pad_bottom, 0, 0,
            cv2.BORDER_CONSTANT, value=0
        )
        self.get_logger().info(f'🔲 Added {pad_bottom}px bottom padding: {frame_0.shape[1]}x{frame_0.shape[0]}')
        
        self.reset_tracking_state()
        
        # Create virtual needle at top center if no real needle present
        has_real_needle = np.any(seg_img_0 == 1)
        if not has_real_needle:
            # Add virtual needle class for initialization
            img_height, img_width = seg_img_0.shape
            virtual_needle_x = img_width // 2
            virtual_needle_y = 0
            # Add small virtual needle region (just a vertical line at top center)
            seg_img_0[0:10, virtual_needle_x-2:virtual_needle_x+3] = 1
            self.get_logger().info(f"🎯 Created virtual needle at top center ({virtual_needle_x}, {virtual_needle_y})")
        
        geometry = self.prepare_initial_geometry(
            frame_0,
            seg_img_0,
            rpe_thickness_pixels=5,
            injection_mode=True,
        )
        seg_img_0 = geometry.seg_img_0
        ILM_0 = geometry.ilm_line
        RPE_0 = geometry.rpe_line
        frame_0_rotated = geometry.frame_0_rotated
        seg_0_rotated = geometry.seg_0_rotated
        needle_tip_pos = geometry.needle_tip_pos
        ROI = geometry.roi

        if needle_tip_pos[0] is None or needle_tip_pos[1] is None:
            needle_tip_pos = (frame_0.shape[1] // 2, 0)
            self.get_logger().info(f"🎯 Using virtual needle tip at ({needle_tip_pos[0]}, {needle_tip_pos[1]})")
        else:
            self.get_logger().info(f"🎯 Using real needle tip at ({needle_tip_pos[0]}, {needle_tip_pos[1]})")

        roi_has_anatomy, seg_roi_check, presence = self.roi_contains_layers(seg_0_rotated, ROI)
        if not roi_has_anatomy:
            self.get_logger().warn("⚠️  ROI missing anatomy. Trying the opposite side...")
            ROI, _ = self.compute_roi(
                seg_img_0,
                frame_0.shape,
                geometry.line_points,
                needle_tip_pos,
                to_left=False,
            )
            geometry.roi = ROI
            roi_has_anatomy, seg_roi_check, presence = self.roi_contains_layers(seg_0_rotated, ROI)

        self.get_logger().info(
            f"📍 ROI anatomy check: ILM={presence.get(2, False)}, RPE={presence.get(3, False)}"
        )
        if not roi_has_anatomy and seg_roi_check is not None:
            self.get_logger().error(f"❌ ROI does not contain anatomy! Classes in ROI: {np.unique(seg_roi_check)}")
        
        self.ROI = ROI
        if ROI is None:
            self.get_logger().error("❌ ROI is None!")
            return
        
        x0, y0, side_x, side_y = ROI
        self.get_logger().info(f"📏 ROI: ({x0}, {y0}) size {side_x}x{side_y}")
        
        model_setup = self.initialize_sound_model_setup(
            geometry,
            classification_mode="majority",
            enforce_consistency=True,
            debug=False,
        )
        frame_roi = model_setup.frame_roi
        seg_roi = model_setup.seg_roi
        patch_centers_class = model_setup.patch_centers_class
        sound_model_config = model_setup.sound_model_config
        masses = model_setup.masses
        stiffnesses = model_setup.stiffnesses
        damping = model_setup.damping
        
        # Save debug images
        cv2.imwrite("roi_debug_frame.png", frame_roi)
        cv2.imwrite("roi_debug_seg.png", seg_roi * 50)  # Scale for visibility
        self.get_logger().info(f"💾 Saved ROI debug images: roi_debug_frame.png, roi_debug_seg.png")
        
        # Start simulator with same flags as original
        try:
            self.simulator_process = self.start_simulator(suppress_output=True)
            self.get_logger().info(f'🎮 Started simulator')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to start simulator: {e}')
            return
        
        # Wait 5 seconds for simulator to fully initialize (matching original)
        self.get_logger().info('⏳ Waiting for simulator initialization...')
        sleep(5.0)
        
        # Store computed values
        self.patch_centers_class = patch_centers_class
        self.rotated_patch_centers = model_setup.rotated_patch_centers
        
        # Set sonification parameters
        set_sonification_params(
            sound_model_config, masses, stiffnesses, damping, 
            patch_centers_class.reshape((self.num_nodes_y, self.num_nodes_x))
        )
        send_debug_message("Sonification parameters set.")
        
        # Initialize needle tracker
        self.tracker = NeedleTracker(
            init_frame=frame_0,
            tip_pos=needle_tip_pos,
            alpha=0.6,
            init_segmentation=None
        )
        
        # Initialize node info for anatomical tracking
        self.node_info = model_setup.node_info
        
        self.start_spline_worker(debug=False)
        self.get_logger().info("🚀 Started background spline fitting thread")
        
        # Start sonification timers BEFORE audio recording (matching original order)
        self.start_sonification_timers()
        
        # Start audio recording AFTER timers are running (matching original)
        # In original, this happens when first frame arrives, but we do it after timer start
        start_recording()
        self.audio_started = True
        
        self.initialized = True
        needle_type = "real" if has_real_needle else "virtual"
        self.get_logger().info(f'✅ Initialization complete - ready to sonify with {needle_type} needle!')
    
    def start_sonification_timers(self):
        """Start dual-rate sonification timers"""
        # A-scan timer - runs at fixed high rate for smooth continuous excitation
        timer_period = 1.0 / self.sonification_rate
        self.sonification_timer = self.create_timer(timer_period, self.sonification_callback)
        self.get_logger().info(f'🎵 A-scan timer started at {self.sonification_rate}Hz ({timer_period*1000:.1f}ms period)')
        
        # ILM/RPE timer - separate rate for deformation-based boundary sonification
        ilm_rpe_period = 1.0 / self.ilm_rpe_rate
        self.ilm_rpe_timer = self.create_timer(ilm_rpe_period, self.ilm_rpe_sonification_callback)
        self.get_logger().info(f'🎵 ILM/RPE timer started at {self.ilm_rpe_rate}Hz ({ilm_rpe_period*1000:.1f}ms period)')
    
    def sonification_callback(self):
        """A-scan sonification callback - runs at fixed high rate for smooth continuous excitation"""
        try:
            if not self.sonification_ready:
                return
            
            with self.sonification_lock:
                node_magnitudes = self.latest_node_magnitudes.copy() if self.latest_node_magnitudes is not None else None
            
            if node_magnitudes is None:
                return
            
            # Perform continuous sonification at timer rate (smooth excitation)
            sonify_ascan(forces=node_magnitudes, separate_f_components=self.sep_f_components)
            
        except Exception as e:
            self.get_logger().error(f'❌ Sonification callback error: {str(e)}')
    
    def ilm_rpe_sonification_callback(self):
        """ILM/RPE sonification callback"""
        try:
            if not self.sonification_ready:
                return
            
            with self.sonification_lock:
                current_class = self.latest_current_class
                f_ILM = self.smoothed_f_ILM
                f_RPE = self.smoothed_f_RPE
                needle_pos = self.latest_needle_pos
                roi_bounds = self.roi_bounds
                dist_to_anatomy = self.latest_dist_to_anatomy
            
            # Check if needle is in ROI or close to anatomy
            needle_in_roi = False
            if needle_pos[0] is not None and needle_pos[1] is not None and roi_bounds is not None:
                x0, y0, side_x, side_y = roi_bounds
                needle_in_roi = (x0 <= needle_pos[0] <= x0 + side_x and
                               y0 <= needle_pos[1] <= y0 + side_y)
            
            needle_close_to_anatomy = dist_to_anatomy < 50.0

            if not (needle_in_roi or needle_close_to_anatomy):
                # Set f_ILM and f_RPE to zero if needle is not in ROI and not close to anatomy
                self.smoothed_f_ILM = 0.0
                self.smoothed_f_RPE = 0.0
            
            if current_class != 0 and (needle_in_roi or needle_close_to_anatomy):
                f_ILM_should_be_active = abs(f_ILM + f_RPE) > 1
                self.update_f_ILM_ramp_state(f_ILM_should_be_active)
                
                if f_ILM_should_be_active:
                    ramp_intensity = self.calculate_f_ILM_ramp_intensity()
                    ramped_f_ILM = f_ILM * ramp_intensity
                    sonify_ILM_RPE(f_ilm=ramped_f_ILM * 2/5, f_rpe=0)
            else:
                self.update_f_ILM_ramp_state(False)
            
        except Exception as e:
            self.get_logger().error(f'❌ ILM/RPE callback error: {str(e)}')
    
    def calculate_f_ILM_ramp_intensity(self):
        """Calculate ramping intensity for f_ILM"""
        return BaseIOCTSonification.calculate_ramp_intensity(
            is_active=self.f_ILM_is_active,
            ramp_counter=self.f_ILM_ramp_counter,
            sonification_rate=self.sonification_rate,
            ramp_duration=self.f_ILM_ramp_duration,
        )
    
    def update_f_ILM_ramp_state(self, f_ILM_active):
        """Update ramping state"""
        self.f_ILM_is_active, self.f_ILM_ramp_counter = BaseIOCTSonification.update_ramp_state(
            self.f_ILM_is_active,
            self.f_ILM_ramp_counter,
            f_ILM_active,
        )
    
    def smooth_f_values(self, new_f_ILM, new_f_RPE):
        """Apply exponential smoothing to forces"""
        return BaseIOCTSonification.smooth_force_values(
            self,
            new_f_ILM,
            new_f_RPE,
            alpha=self.f_smoothing_alpha,
        )
    
    def process_frame(self):
        """Process frame - complete implementation from variable_rate_lite"""
        if not self.initialized:
            return
        
        if self.current_frame is None or self.current_segmentation is None:
            return
        
        # Check if segmentation is empty/clear - if so, reset state
        if self.is_segmentation_empty(self.current_segmentation):
            self.reset_state()
            return
        
        frame = self.current_frame
        seg_img_current = self.current_segmentation.copy()
        seg_img_current[seg_img_current == 4] = 0
        seg_img_current = extrapolate_retina(seg_img_current, cls_to_use=(4 if self.separate_ilm else 2))
        
        # Update needle tracking
        tracked_tip, _ = self.tracker.update(frame, segmentation_map=seg_img_current)
        
        needle_tip_pos = self.filter_tracked_tip(
            tracked_tip,
            max_jump=self.max_needle_position_jump,
        )
        if needle_tip_pos is None:
            needle_tip_pos = (frame.shape[1] // 2, 0)
        
        # Find closest node
        closest_node_pos, closest_node_index, dist, current_class = self.find_closest_node_and_class(
            needle_tip_pos,
            self.rotated_patch_centers,
            self.patch_centers_class,
        )
        
        # Handle class 0->4 jump (snap to class 2)
        class_changed = (self.prev_class is not None and 
                        current_class != self.prev_class and 
                        [self.prev_class, current_class] != [4, 2])
        
        if class_changed and current_class == 4 and self.prev_class == 0:
            snapped_pos, snapped_index, snapped_dist = self.snap_to_nearest_class_row(
                needle_tip_pos,
                self.rotated_patch_centers,
                self.class2_mask,
            )
            if snapped_index is not None:
                closest_node_index = snapped_index
                closest_node_pos = snapped_pos
                current_class = 2
                dist = snapped_dist
        
        # Submit spline fitting task
        # TODO: Consider submitting only every N frames or when certain conditions are met to reduce load
        if self.frame_index % 2 == 0:
            self.queue_spline_task(self.frame_index, seg_img_current)
        
        # Process spline fitting results
        for result in self.collect_recent_spline_results(self.frame_index, max_age=10):
            seg_img_current = result['fitted_seg']
            ILM_line = result['updated_lines']['ILM']
            RPE_line = result['updated_lines']['RPE']
                    
            self.rotated_patch_centers, ILM_line, RPE_line, confidence = self.apply_spline_result_to_patch_centers(
                result,
                self.rotated_patch_centers,
                self.node_info,
                current_class,
                gamma=0.999,
            )
            
            # Deformation tracking
            current_thickness = RPE_line - ILM_line
            
            if needle_tip_pos[0] is not None:
                needle_x = int(np.clip(needle_tip_pos[0], 0, seg_img_current.shape[1] - 1))
                window_start = max(0, needle_x - 40)
                window_end = min(seg_img_current.shape[1], needle_x + 40)
            else:
                window_start = 40
                window_end = seg_img_current.shape[1] - 40
            
            delta_thickness = 0.0
            if self.prev_thickness is not None:
                delta_thickness = np.nanmedian(
                    np.abs(current_thickness[window_start:window_end] - self.prev_thickness[window_start:window_end])
                )
                if np.isnan(delta_thickness):
                    delta_thickness = 0.0
            
            deformation_raw = max(0.0, delta_thickness)
            
            if current_class == 0:
                self.deformation_history.append(deformation_raw)
                if len(self.deformation_history) > self.median_filter_size:
                    self.deformation_history = self.deformation_history[-self.median_filter_size:]
            
            if len(self.deformation_history) >= 5:
                baseline_threshold = 0.3
                soft_transition_width = 0.2
                
                if current_class != 0:
                    if deformation_raw > baseline_threshold - soft_transition_width:
                        if deformation_raw <= baseline_threshold:
                            transition_progress = (deformation_raw - (baseline_threshold - soft_transition_width)) / soft_transition_width
                            smooth_factor = transition_progress * transition_progress * (3.0 - 2.0 * transition_progress)
                            deformation_above_baseline = max(0, deformation_raw - baseline_threshold) * smooth_factor
                        else:
                            deformation_above_baseline = deformation_raw - baseline_threshold
                        
                        self.f_ILM = -deformation_above_baseline * 2.5
                    else:
                        self.f_ILM = 0.0
                else:
                    self.f_ILM = 0.0
            else:
                baseline_threshold = 0.3
                soft_transition_width = 0.2
                if current_class != 0 and deformation_raw > baseline_threshold - soft_transition_width:
                    if deformation_raw <= baseline_threshold:
                        transition_progress = (deformation_raw - (baseline_threshold - soft_transition_width)) / soft_transition_width
                        smooth_factor = transition_progress * transition_progress * (3.0 - 2.0 * transition_progress)
                        deformation_above_baseline = max(0, deformation_raw - baseline_threshold) * smooth_factor
                    else:
                        deformation_above_baseline = deformation_raw - baseline_threshold
                    
                    self.f_ILM = -deformation_above_baseline * 2.5
                else:
                    self.f_ILM = 0.0
            
            self.f_RPE = 0.0
            
            self.frame_deformation_history.append(deformation_raw)
            if len(self.frame_deformation_history) > 30:
                self.frame_deformation_history = self.frame_deformation_history[-30:]
                new_ref = np.percentile(self.frame_deformation_history, 90)
                self.deformation_ref = 0.8 * self.deformation_ref + 0.2 * max(5.0, new_ref)
            
            self.prev_thickness = current_thickness.copy()
            self.prev_ILM_line = ILM_line.copy()
            self.prev_RPE_line = RPE_line.copy()
        
        # Compute force magnitudes
        base_magnitude = 1.0 * (20/self.sonification_rate) * self.scales[current_class]
        magnitudes = np.ones(3, dtype=np.float32) * base_magnitude
        
        deflection_scaling = 1.0
        node_magnitudes = compute_node_magnitudes(
            self.num_nodes_y, magnitudes, closest_node_index, dist, 
            deflection_scaling, sigma=3
        )
        
        # Handle class transitions
        if class_changed and [self.prev_class, current_class] != [4, 2]:
            self.get_logger().info(f"⚡ Class change: {self.prev_class} → {current_class}")
            node_magnitudes *= 3.0
        
        self.prev_class = current_class
        
        # Apply smoothing
        smoothed_ILM, smoothed_RPE = self.smooth_f_values(self.f_ILM, self.f_RPE)
        
        # Update sonification state (timer will read this and sonify continuously)
        with self.sonification_lock:
            self.latest_node_magnitudes = node_magnitudes.copy()
            self.latest_current_class = current_class
            self.latest_f_ILM = self.f_ILM
            self.latest_f_RPE = self.f_RPE
            self.smoothed_f_ILM = smoothed_ILM
            self.smoothed_f_RPE = smoothed_RPE
            self.latest_needle_pos = needle_tip_pos
            if self.ROI is not None:
                x0, y0, side_x, side_y = self.ROI
                self.roi_bounds = (x0, y0, side_x, side_y)
            self.latest_dist_to_anatomy = abs(dist) if dist != 0 else float('inf')
            self.sonification_ready = True
        
        # NOTE: NO direct sonify call here! Timer reads state and sonifies continuously
        # This matches variable_rate_lite architecture for smooth excitation
        
        self.frame_index += 1
        
        if self.frame_index % 100 == 0:
            needle_x = needle_tip_pos[0] if needle_tip_pos[0] is not None else 0.0
            needle_y = needle_tip_pos[1] if needle_tip_pos[1] is not None else 0.0
            self.get_logger().info(
                f'Frame {self.frame_index}: Class {current_class}, '
                f'Needle ({needle_x:.1f}, {needle_y:.1f})'
            )
    
    def destroy_node(self):
        """Cleanup"""
        # Stop timers
        if hasattr(self, 'sonification_timer') and self.sonification_timer:
            self.sonification_timer.cancel()
        if hasattr(self, 'ilm_rpe_timer') and self.ilm_rpe_timer:
            self.ilm_rpe_timer.cancel()
        
        self.stop_spline_worker(timeout=2.0)
        
        # Stop audio
        if self.audio_started:
            stop_recording()
        
        # Stop simulator
        if self.simulator_process:
            self.simulator_process.terminate()
        
        Node.destroy_node(self)


def main():
    argsparser = argparse.ArgumentParser(description='Variable Rate ROS Sonification Node')
    # Segmentation processing
    argsparser.add_argument('--seg_processing', choices=["local", "syntheseyes", "local_s"], default="local", help='Segmentation processing method')
    args = argsparser.parse_args()

    rclpy.init()
    node = VariableRateROSSonificationNode(seg_processing=args.seg_processing)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
