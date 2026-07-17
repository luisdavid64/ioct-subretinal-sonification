#!/usr/bin/env python3
"""
Manual ROI Selection and Image Rotation Tool for Research Paper Figures

This script provides an interactive GUI for creating publication-ready figures:
1. Loading an image and its corresponding segmentation
2. Manually selecting ROI with mouse
3. Manually rotating the image
4. Generating high-quality overlays for research papers:
   - Full image + segmentation overlay
   - ROI + segmentation overlay
   - Clean, publication-ready outputs

Usage:
    python manual_roi_rotation_tool.py --image <path_to_image> --segmentation <path_to_segmentation> [--output_dir <output_directory>]

Controls:
    Mouse: Click and drag to select ROI
    'r': Rotate image clockwise by 5 degrees
    'R' (Shift+r): Rotate image counter-clockwise by 5 degrees
    'f': Fine rotation mode - rotate by 1 degree
    'F' (Shift+f): Fine rotation counter-clockwise by 1 degree
    'v': Toggle visualization mode (overlay styles)
    's': Save research figures
    'c': Clear ROI selection
    'q' or ESC: Quit without saving
    Enter: Generate and save final figures
"""

import cv2
import numpy as np
import argparse
import json
import os
from pathlib import Path
import sys

# Import spline fitting functionality
try:
    from extrapolate import extrapolate_rpe_and_ilm, update_fitted_line, extract_line, RPE_LABEL, ILM_LABEL, calculate_target_thickness
    SPLINES_AVAILABLE = True
except ImportError:
    SPLINES_AVAILABLE = False
    print("Warning: Spline fitting functionality not available (extrapolate.py not found)")

class ManualROIRotationTool:
    def __init__(self, image, segmentation, output_dir=None):
        # Handle both file paths and numpy arrays
        if isinstance(image, str):
            self.image_path = image
            self.original_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if self.original_image is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            self.image_path = "array_input"
            self.original_image = image.copy() if len(image.shape) > 2 else image
            if len(self.original_image.shape) == 3:
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        if isinstance(segmentation, str):
            self.seg_path = segmentation  
            self.original_seg = cv2.imread(segmentation, cv2.IMREAD_GRAYSCALE)
            if self.original_seg is None:
                raise ValueError(f"Could not load segmentation from {segmentation}")
        else:
            self.seg_path = "array_input"
            self.original_seg = segmentation.copy()
            
        self.output_dir = output_dir or "."
        
        # Current state
        self.current_angle = 0.0
        self.roi = None  # Will store as (center_x, center_y, width, height, angle)
        self.roi_creation_angle = 0.0  # Track angle when ROI was created
        self.roi_selecting = False
        self.roi_start = None
        self.roi_original = None  # ROI in original (unrotated) coordinates
        self.visualization_mode = 0  # 0: full overlay, 1: contours only, 2: transparent
        
        # For display
        self.display_image = None
        self.display_seg = None
        self.display_overlay = None
        
        # Mouse callback state
        self.mouse_selecting = False
        
        # Publication settings
        self.overlay_alpha = 0.3
        self.contour_thickness = 2
        self.show_splines = False  # Toggle for spline visualization
        
        # Initialize spline fitting if available
        if SPLINES_AVAILABLE:
            self.initialize_spline_state()
        
        self.update_display()
        
    def rotate_image(self, image, angle):
        """Rotate image by given angle around center"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        if len(image.shape) == 2:
            rotated = cv2.warpAffine(image, M, (w, h))
        else:
            rotated = cv2.warpAffine(image, M, (w, h))
        return rotated, M
        
    def update_display(self):
        """Update the display images with current rotation"""
        # Rotate images
        self.display_image, self.rotation_matrix = self.rotate_image(self.original_image, self.current_angle)
        self.display_seg, _ = self.rotate_image(self.original_seg, self.current_angle)
        
        # Create colored segmentation based on visualization mode
        viz_modes = ['full', 'contours', 'transparent']
        current_mode = viz_modes[self.visualization_mode]
        seg_colored = self.create_colored_segmentation(self.display_seg, current_mode)
        
        # Create overlay with appropriate alpha
        image_bgr = cv2.cvtColor(self.display_image, cv2.COLOR_GRAY2BGR)
        
        if current_mode == 'transparent':
            alpha = 0.1  # Very subtle
        elif current_mode == 'contours':
            alpha = 1.0  # Full opacity for contours
        else:
            alpha = self.overlay_alpha
            
        self.display_overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, seg_colored, alpha, 0)
        
        # Draw rotated ROI if selected
        if self.roi is not None:
            self.draw_rotated_roi(self.display_overlay, self.roi)
            
        # Draw splines if enabled
        if SPLINES_AVAILABLE and self.show_splines:
            self.display_overlay = self.draw_splines_overlay(self.display_overlay, self.display_seg)
        
        # Add information overlay
        self.add_info_overlay()
        
    def add_info_overlay(self):
        """Add information text overlay to the display"""
        # Add angle info
        angle_text = f"Angle: {self.current_angle:.1f}°"
        cv2.putText(self.display_overlay, angle_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_overlay, angle_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add visualization mode
        viz_modes = ['Full Overlay', 'Contours Only', 'Transparent']
        mode_text = f"Mode: {viz_modes[self.visualization_mode]} (v to cycle)"
        cv2.putText(self.display_overlay, mode_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(self.display_overlay, mode_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add ROI info if selected
        if self.roi is not None:
            center_x, center_y, w, h, angle = self.roi
            roi_text = f"ROI: {int(w)}×{int(h)} at ({int(center_x)},{int(center_y)}) ∠{angle:.1f}°"
            cv2.putText(self.display_overlay, roi_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(self.display_overlay, roi_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add spline info if available
        if SPLINES_AVAILABLE:
            spline_text = f"Splines: {'ON' if self.show_splines else 'OFF'} (p to toggle)"
            cv2.putText(self.display_overlay, spline_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(self.display_overlay, spline_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add instructions
        instructions = [
            "Mouse: Select ROI | r/R: Rotate ±1° | f/F: ±1° | v: Viz mode",
            f"s: Save figures | {'p: Splines | ' if SPLINES_AVAILABLE else ''}c: Clear ROI | Enter: Final save | q: Quit"
        ]
        for i, instruction in enumerate(instructions):
            y_pos = self.display_overlay.shape[0] - 40 + i * 20
            cv2.putText(self.display_overlay, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(self.display_overlay, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
    def create_colored_segmentation(self, seg, mode='full'):
        """Create a colored version of the segmentation with different visualization modes"""
        colored = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        
        # Define colors for different classes (publication-friendly colors)
        colors = {
            0: (0, 0, 0),       # Background - black
            1: (0, 255, 255),   # Cyan (needle)
            2: (255, 100, 100), # Light red (RPE)
            3: (100, 255, 100), # Light green (ILM)  
            4: (255, 255, 100), # Light yellow (other tissue)
        }
        
        if mode == 'full':
            # Full overlay mode
            for class_id, color in colors.items():
                mask = seg == class_id
                colored[mask] = color
                
        elif mode == 'contours':
            # Contours only mode
            for class_id, color in colors.items():
                if class_id == 0:
                    continue
                mask = (seg == class_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(colored, contours, -1, color, self.contour_thickness)
                
        elif mode == 'transparent':
            # Highly transparent mode for subtle overlay
            for class_id, color in colors.items():
                mask = seg == class_id
                colored[mask] = color
            
        return colored
            
    def draw_rotated_roi(self, image, roi_rect, color=(0, 0, 255), thickness=3):
        """Draw a rotated rectangle ROI"""
        if roi_rect is None:
            return image
            
        center_x, center_y, width, height, angle = roi_rect
        
        # If angle is 0 (or very close to 0), draw a normal rectangle for clarity
        if abs(angle) < 0.1:
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            cv2.rectangle(image, (x, y), (x + int(width), y + int(height)), color, thickness)
            
            # Add corner markers for better visibility
            marker_size = 8
            cv2.circle(image, (x, y), marker_size, color, -1)
            cv2.circle(image, (x, y), marker_size, (255, 255, 255), 2)
            cv2.circle(image, (x + int(width), y + int(height)), marker_size, color, -1)
            cv2.circle(image, (x + int(width), y + int(height)), marker_size, (255, 255, 255), 2)
        else:
            # Draw rotated rectangle for non-zero angles
            rect = ((center_x, center_y), (width, height), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Draw the rotated rectangle
            cv2.drawContours(image, [box], 0, color, thickness)
            
            # Add corner markers for better visibility
            marker_size = 8
            for corner in box:
                x, y = corner
                cv2.circle(image, (int(x), int(y)), marker_size, color, -1)
                cv2.circle(image, (int(x), int(y)), marker_size, (255, 255, 255), 2)
        
        return image
        
    def draw_dashed_roi_outline(self, image, roi_rect, color=(255, 255, 255), thickness=2, dash_length=10):
        """Draw a dashed outline for the ROI"""
        if roi_rect is None:
            return image
            
        center_x, center_y, width, height, angle = roi_rect
        
        # Create the rotated rectangle points
        if abs(angle) < 0.1:
            # Simple rectangle for zero angle
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            corners = [
                (x, y),
                (x + int(width), y),
                (x + int(width), y + int(height)),
                (x, y + int(height))
            ]
        else:
            # Rotated rectangle
            rect = ((center_x, center_y), (width, height), angle)
            box = cv2.boxPoints(rect)
            corners = [(int(pt[0]), int(pt[1])) for pt in box]
        
        # Draw dashed lines between corners
        for i in range(len(corners)):
            start = corners[i]
            end = corners[(i + 1) % len(corners)]
            self.draw_dashed_line(image, start, end, color, thickness, dash_length)
            
        return image
        
    def draw_dashed_line(self, image, start, end, color, thickness, dash_length):
        """Draw a dashed line between two points"""
        x1, y1 = start
        x2, y2 = end
        
        # Calculate line length and direction
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        if length == 0:
            return
            
        # Unit vector
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        
        # Draw dashes
        current_length = 0
        draw_dash = True
        
        while current_length < length:
            # Calculate dash end point
            dash_end_length = min(current_length + dash_length, length)
            dash_start_x = int(x1 + current_length * dx)
            dash_start_y = int(y1 + current_length * dy)
            dash_end_x = int(x1 + dash_end_length * dx)
            dash_end_y = int(y1 + dash_end_length * dy)
            
            if draw_dash:
                cv2.line(image, (dash_start_x, dash_start_y), (dash_end_x, dash_end_y), color, thickness)
            
            current_length = dash_end_length
            draw_dash = not draw_dash
            
    def initialize_spline_state(self):
        """Initialize spline fitting state"""
        if not SPLINES_AVAILABLE:
            return
            
        # Extract initial lines from segmentation
        initial_ilm_line = extract_line(self.original_seg, ILM_LABEL)
        initial_rpe_line = extract_line(self.original_seg, RPE_LABEL)
        
        # Initialize spline state
        self.spline_state = {
            "prior": {
                "ILM": initial_ilm_line.copy(),
                "RPE": initial_rpe_line.copy()
            },
            "reference": {
                "ILM": initial_ilm_line.copy(),
                "RPE": initial_rpe_line.copy()
            }
        }
        
    def fit_splines(self, segmentation):
        """Fit splines to segmentation data"""
        if not SPLINES_AVAILABLE or not hasattr(self, 'spline_state'):
            return None, None
            
        # Apply extrapolation like in spline_based_simulation
        seg_extrapolated = extrapolate_rpe_and_ilm(segmentation.copy(), extrapolate_distance=100)
        
        # Calculate thickness for spline fitting
        thickness = calculate_target_thickness(seg_extrapolated, self.spline_state["prior"]["ILM"])
        
        # Perform spline fitting
        try:
            fitted_seg, updated_lines, confidences = update_fitted_line(
                seg=seg_extrapolated,
                state=self.spline_state,
                extrapolate_distance=100,
                sigma_L=8.0,
                base_alpha=0.8,
                thickness=thickness
            )
            
            # Ensure splines span the full image width
            img_width = segmentation.shape[1]
            
            # Extract and extrapolate lines to full width
            ilm_line = extract_line(fitted_seg, ILM_LABEL)  # Class 2
            rpe_line = extract_line(fitted_seg, RPE_LABEL)  # Class 3
            
            # Ensure splines span the full image width with no NaN values
            img_width = segmentation.shape[1]
            
            # Fill NaN values with interpolation/extrapolation for ILM
            if len(ilm_line) < img_width:
                ilm_full = np.full(img_width, np.nan)
                ilm_full[:len(ilm_line)] = ilm_line
                ilm_line = ilm_full
            
            # Interpolate/extrapolate all NaN values in ILM line
            valid_mask = ~np.isnan(ilm_line)
            if np.sum(valid_mask) >= 2:
                valid_x = np.where(valid_mask)[0]
                valid_y = ilm_line[valid_mask]
                
                # Interpolate between valid points
                ilm_line = np.interp(np.arange(img_width), valid_x, valid_y)
            
            # Do the same for RPE
            if len(rpe_line) < img_width:
                rpe_full = np.full(img_width, np.nan)
                rpe_full[:len(rpe_line)] = rpe_line
                rpe_line = rpe_full
            
            # Interpolate/extrapolate all NaN values in RPE line
            valid_mask = ~np.isnan(rpe_line)
            if np.sum(valid_mask) >= 2:
                valid_x = np.where(valid_mask)[0]
                valid_y = rpe_line[valid_mask]
                
                # Interpolate between valid points
                rpe_line = np.interp(np.arange(img_width), valid_x, valid_y)
            
            return {"ILM": ilm_line, "RPE": rpe_line}, confidences
            
        except Exception as e:
            print(f"Spline fitting error: {e}")
            return None, None
            
    def draw_splines_overlay(self, image, segmentation):
        """Draw fitted splines on the image"""
        if not SPLINES_AVAILABLE or not self.show_splines:
            return image
            
        splines, confidences = self.fit_splines(segmentation)
        if splines is None:
            return image
            
        # Get image width to ensure full extrapolation
        img_width = image.shape[1]
        
        # Debug: Print spline info
        print(f"Image width: {img_width}")
        if "ILM" in splines:
            ilm_line = splines["ILM"]
            print(f"ILM spline length: {len(ilm_line)}, valid points: {np.sum(~np.isnan(ilm_line))}")
        if "RPE" in splines:
            rpe_line = splines["RPE"]
            print(f"RPE spline length: {len(rpe_line)}, valid points: {np.sum(~np.isnan(rpe_line))}")
        
        # Draw ILM line (darker magenta - original segmentation color for class 2)
        if "ILM" in splines:
            ilm_line = splines["ILM"]
            # Ensure we draw across the full width
            for x in range(min(len(ilm_line), img_width)):
                if not np.isnan(ilm_line[x]):
                    y = int(ilm_line[x])
                    if 0 <= y < image.shape[0]:
                        # Darker version of magenta (255,0,255) -> (150,0,150)
                        cv2.circle(image, (x, y), 2, (150, 0, 150), -1)
                        
        # Draw RPE line (darker white/gray - original segmentation color for class 3)
        if "RPE" in splines:
            rpe_line = splines["RPE"]
            # Ensure we draw across the full width
            for x in range(min(len(rpe_line), img_width)):
                if not np.isnan(rpe_line[x]):
                    y = int(rpe_line[x])
                    if 0 <= y < image.shape[0]:
                        # Darker version of white (255,255,255) -> (180,180,180)
                        cv2.circle(image, (x, y), 2, (180, 180, 180), -1)
                        
        return image
        
    def update_roi_from_selection(self, start_point, end_point):
        """Update ROI from mouse selection, accounting for current rotation"""
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Store ROI with angle 0 initially (created in current reference frame)
        # It will rotate with future image rotations
        self.roi = (center_x, center_y, width, height, 0.0)
        self.roi_creation_angle = self.current_angle  # Remember the angle when ROI was created
        
        # Also store in original coordinates (for cropping purposes)
        self.update_original_roi_coordinates()
        
    def update_original_roi_coordinates(self):
        """Convert current ROI back to original coordinate system for cropping"""
        if self.roi is None:
            self.roi_original = None
            return
            
        center_x, center_y, width, height, angle = self.roi
        
        # Get inverse rotation matrix
        h, w = self.original_image.shape
        center = (w // 2, h // 2)
        M_inv = cv2.getRotationMatrix2D(center, -self.current_angle, 1.0)
        
        # Transform ROI center back to original coordinates
        roi_center = np.array([[center_x, center_y]], dtype=np.float32)
        roi_center = roi_center.reshape(-1, 1, 2)
        original_center = cv2.transform(roi_center, M_inv)[0][0]
        
        # Store as traditional (x, y, w, h) for cropping operations
        x = int(original_center[0] - width / 2)
        y = int(original_center[1] - height / 2)
        self.roi_original = (max(0, x), max(0, y), int(width), int(height))
        
    def rotate_roi_with_image(self, angle_change):
        """Update ROI rotation when image is rotated"""
        if self.roi is not None:
            center_x, center_y, width, height, current_roi_angle = self.roi
            # Calculate the total rotation from when ROI was created
            # Negate to match image rotation direction
            total_rotation_since_roi_creation = -(self.current_angle - self.roi_creation_angle)
            # Update the ROI angle to reflect the rotation since creation
            self.roi = (center_x, center_y, width, height, total_rotation_since_roi_creation)
            self.update_original_roi_coordinates()
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_selecting = True
            self.roi_start = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_selecting:
            # Show preview rectangle
            temp_overlay = self.display_overlay.copy()
            cv2.rectangle(temp_overlay, self.roi_start, (x, y), (0, 255, 0), 2)
            cv2.imshow("ROI Selection Tool", temp_overlay)
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mouse_selecting:
                self.mouse_selecting = False
                # Update ROI using the new method that handles rotation
                self.update_roi_from_selection(self.roi_start, (x, y))
                self.update_display()
                
    def save_research_figures(self, base_name="research_figure"):
        """Save publication-ready research figures"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Full image with segmentation overlay
        full_image_bgr = cv2.cvtColor(self.display_image, cv2.COLOR_GRAY2BGR)
        seg_colored = self.create_colored_segmentation(self.display_seg, 'full')
        full_overlay = cv2.addWeighted(full_image_bgr, 0.7, seg_colored, 0.3, 0)
        
        full_path = os.path.join(self.output_dir, f"{base_name}_full_overlay.png")
        cv2.imwrite(full_path, full_overlay)
        
        # 2. ROI with segmentation overlay (if ROI selected)
        if self.roi is not None and self.roi_original is not None:
            x, y, w, h = self.roi_original
            # Ensure we don't go out of bounds
            h_max, w_max = self.display_image.shape
            x = max(0, min(x, w_max - 1))
            y = max(0, min(y, h_max - 1))
            w = min(w, w_max - x)
            h = min(h, h_max - y)
            
            roi_image = self.display_image[y:y+h, x:x+w]
            roi_seg = self.display_seg[y:y+h, x:x+w]
            
            # Set class 1 to 0 in ROI segmentation
            roi_seg_modified = roi_seg.copy()
            roi_seg_modified[roi_seg_modified == 1] = 0
            
            roi_image_bgr = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
            roi_seg_colored = self.create_colored_segmentation(roi_seg_modified, 'full')
            roi_overlay = cv2.addWeighted(roi_image_bgr, 0.5, roi_seg_colored, 0.5, 0)
            
            roi_path = os.path.join(self.output_dir, f"{base_name}_roi_overlay.png")
            cv2.imwrite(roi_path, roi_overlay)
            
            # Also save clean ROI image and segmentation
            clean_roi_path = os.path.join(self.output_dir, f"{base_name}_roi_clean.png")
            roi_seg_path = os.path.join(self.output_dir, f"{base_name}_roi_seg.png")
            cv2.imwrite(clean_roi_path, roi_image)
            cv2.imwrite(roi_seg_path, roi_seg_colored)
        
        # 3. Clean full image and segmentation
        clean_full_path = os.path.join(self.output_dir, f"{base_name}_full_clean.png")
        full_seg_path = os.path.join(self.output_dir, f"{base_name}_full_seg.png")
        cv2.imwrite(clean_full_path, full_image_bgr)
        cv2.imwrite(full_seg_path, seg_colored)
        
        # 4. Contours-only version with dashed ROI outline
        seg_contours = self.create_colored_segmentation(self.display_seg, 'contours')
        contours_overlay = cv2.addWeighted(full_image_bgr, 0.9, seg_contours, 1.0, 0)
        
        # Add fitted splines to contours version
        if SPLINES_AVAILABLE:
            contours_overlay = self.draw_splines_overlay(contours_overlay, self.display_seg)
        
        # Add dashed white ROI outline to contours version
        if self.roi is not None:
            contours_overlay = self.draw_dashed_roi_outline(contours_overlay, self.roi, color=(255, 255, 255))
            
        contours_path = os.path.join(self.output_dir, f"{base_name}_contours.png")
        cv2.imwrite(contours_path, contours_overlay)
        
        # 5. Splines-only version (if available)
        splines_only_files = []
        if SPLINES_AVAILABLE:
            splines_only = self.draw_splines_overlay(full_image_bgr.copy(), self.display_seg)
            splines_only_path = os.path.join(self.output_dir, f"{base_name}_splines.png")
            cv2.imwrite(splines_only_path, splines_only)
            splines_only_files = [splines_only_path]
        
        saved_files = [full_path, clean_full_path, full_seg_path, contours_path] + splines_only_files
        if self.roi is not None:
            saved_files.extend([roi_path, clean_roi_path, roi_seg_path])
            
        print(f"\nResearch figures saved:")
        for file_path in saved_files:
            print(f"  {file_path}")
            
        return saved_files
        
    def run(self):
        """Run the interactive tool"""
        cv2.namedWindow("ROI Selection Tool", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ROI Selection Tool", self.mouse_callback)
        
        print("Manual ROI Selection and Rotation Tool for Research Figures")
        print("=" * 60)
        print("Controls:")
        print("  Mouse: Click and drag to select ROI")
        print("  'r': Rotate clockwise by 5 degrees")
        print("  'R' (Shift+r): Rotate counter-clockwise by 5 degrees")
        print("  'f': Fine rotation clockwise by 1 degree")
        print("  'F' (Shift+f): Fine rotation counter-clockwise by 1 degree")
        print("  'v': Toggle visualization mode (full/contours/transparent)")
        print("  's': Save research figures")
        print("  'c': Clear ROI selection")
        print("  'q' or ESC: Quit without saving")
        print("  Enter: Save final figures and exit")
        print("=" * 60)
        
        while True:
            cv2.imshow("ROI Selection Tool", self.display_overlay)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # ESC
                break
                
            elif key == ord('r'):  # Rotate clockwise
                self.current_angle -= 1.0
                self.rotate_roi_with_image(-1.0)
                self.update_display()
                
            elif key == ord('R'):  # Rotate counter-clockwise
                self.current_angle += 1.0
                self.rotate_roi_with_image(1.0)
                self.update_display()
                
            elif key == ord('f'):  # Fine rotate clockwise
                self.current_angle -= 1.0
                self.rotate_roi_with_image(-1.0)
                self.update_display()
                
            elif key == ord('F'):  # Fine rotate counter-clockwise
                self.current_angle += 1.0
                self.rotate_roi_with_image(1.0)
                self.update_display()
                
            elif key == ord('v'):  # Toggle visualization mode
                self.visualization_mode = (self.visualization_mode + 1) % 3
                self.update_display()
                
            elif key == ord('p') and SPLINES_AVAILABLE:  # Toggle splines
                self.show_splines = not self.show_splines
                self.update_display()
                
            elif key == ord('c'):  # Clear ROI
                self.roi = None
                self.roi_original = None
                self.roi_creation_angle = 0.0
                self.update_display()
                
            elif key == ord('s'):  # Save research figures
                try:
                    saved_files = self.save_research_figures()
                    print(f"Saved {len(saved_files)} research figures")
                except Exception as e:
                    print(f"Error saving figures: {e}")
                    
            elif key == 13:  # Enter - final save and exit
                try:
                    saved_files = self.save_research_figures()
                    print(f"Final figures saved: {len(saved_files)} files")
                    break
                except Exception as e:
                    print(f"Error saving figures: {e}")
                    
        cv2.destroyAllWindows()


def create_research_figures(image, segmentation, output_dir="research_figures"):
    """
    Simple function to create research figures from image and segmentation
    
    Args:
        image: Path to image file or numpy array
        segmentation: Path to segmentation file or numpy array  
        output_dir: Directory to save figures (default: "research_figures")
    
    Returns:
        tool: The ManualROIRotationTool instance for further interaction
    """
    tool = ManualROIRotationTool(image, segmentation, output_dir)
    tool.run()
    return tool


def main():
    parser = argparse.ArgumentParser(description="Manual ROI Selection and Image Rotation Tool for Research Figures")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--segmentation", "-s", required=True, help="Path to segmentation image")
    parser.add_argument("--output_dir", "-o", help="Output directory (default: research_figures)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
        
    if not os.path.exists(args.segmentation):
        print(f"Error: Segmentation file not found: {args.segmentation}")
        sys.exit(1)
        
    # Set output directory
    output_dir = args.output_dir or "research_figures"
    
    try:
        create_research_figures(args.image, args.segmentation, output_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()