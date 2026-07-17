import os 
import pandas as pd
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import correlate

def load_force_data(folder):
    if os.path.exists(folder) is False:
        return None
    """Load force data from RobotLog.csv file"""
    csv_files = [f for f in os.listdir(folder) if f.endswith('RobotLog.csv')]
    if not csv_files:
        print(f"No RobotLog.csv found in {folder}")
        return None
    
    csv_file = os.path.join(folder, csv_files[0])
    try:
        # Read the first line to get the header, removing the '#' prefix
        with open(csv_file, 'r') as f:
            header_line = f.readline().strip()
            if header_line.startswith('#'):
                header_line = header_line[1:].strip()  # Remove the '#' and whitespace
        
        # Read CSV with proper header handling
        df = pd.read_csv(csv_file, skiprows=1, names=header_line.split(','))
        
        # Clean up any extra whitespace in column names
        df.columns = df.columns.str.strip()
        
        print(f"CSV columns: {list(df.columns[:5])}...")  # Debug: show first few columns
        return df
    except Exception as e:
        print(f"Error loading CSV {csv_file}: {e}")
        return None

def get_force_at_frame(force_data, frame_index, total_frames):
    """Get force data for a specific frame index accounting for different sampling rates"""
    if force_data is None or len(force_data) == 0:
        return None
    
    # Get the time span of the force data
    start_time = force_data['TimeStamp'].iloc[0]
    end_time = force_data['TimeStamp'].iloc[-1]
    total_duration = end_time - start_time
    
    # Calculate the timestamp for this video frame (assuming 20 Hz video)
    video_fps = 20.0
    frame_time_offset = frame_index / video_fps
    target_timestamp = start_time + frame_time_offset
    
    # Find the closest force measurement by timestamp
    # Use the timestamp difference to find the best match
    time_diffs = abs(force_data['TimeStamp'] - target_timestamp)
    closest_idx = time_diffs.idxmin()
    
    # Make sure we're within reasonable bounds
    if time_diffs.iloc[closest_idx] > 0.1:  # More than 100ms difference
        return None
    
    row = force_data.loc[closest_idx]
    
    # Handle potential column name variations
    ticks_col = 'Ticks' if 'Ticks' in force_data.columns else '# Ticks'
    
    return {
        'tip_force_norm': row.get('TipForceNorm_mN', 0),
        'sclera_force_norm': row.get('ScleraForceNorm_mN', 0),
        'tip_force_x': row.get('TipForce_mN_0', 0),
        'tip_force_y': row.get('TipForce_mN_1', 0),
        'tip_force_z': row.get('TipForce_mN_2', 0),
        'timestamp': row.get('TimeStamp', 0),
        'ticks': row.get(ticks_col, 0),
        'csv_index': closest_idx,
        'time_diff': time_diffs.iloc[closest_idx],
        'handle_forces': row.get('HandleForcesN1', 0),
    }

def get_force_sum_for_frame(force_data, frame_index, video_fps=20.0):
    """
    Return the sum of forces (x, y, z, and norm) for all force samples that
    fall within the time span of a given video frame.
    """
    if force_data is None or len(force_data) == 0:
        return None

    # Get overall time range
    start_time = force_data['TimeStamp'].iloc[0]
    end_time = force_data['TimeStamp'].iloc[-1]

    # Compute frame time window
    frame_duration = 1.0 / video_fps
    frame_start_time = start_time + frame_index * frame_duration
    frame_end_time = frame_start_time + frame_duration

    # Filter forces within this time window
    mask = (force_data['TimeStamp'] >= frame_start_time) & (force_data['TimeStamp'] < frame_end_time)
    frame_forces = force_data.loc[mask]

    if frame_forces.empty:
        return None

    # Compute sums
    fx_sum = frame_forces['TipForce_mN_0'].sum()
    fy_sum = frame_forces['TipForce_mN_1'].sum()
    fz_sum = frame_forces['TipForce_mN_2'].sum()

    # Compute total (vector) magnitude sum
    magnitudes = np.sqrt(
        frame_forces['TipForce_mN_0']**2 +
        frame_forces['TipForce_mN_1']**2 +
        frame_forces['TipForce_mN_2']**2
    )
    f_total_sum = magnitudes.sum()

    return {
        'frame_index': frame_index,
        'num_samples': len(frame_forces),
        'fx_sum_mN': fx_sum,
        'fy_sum_mN': fy_sum,
        'fz_sum_mN': fz_sum,
        'f_total_sum_mN': f_total_sum,
        'timestamp_start': frame_start_time,
        'timestamp_end': frame_end_time,
        'magnitudes': magnitudes.tolist(),
        'f_comps': np.array([frame_forces['TipForce_mN_0'], frame_forces['TipForce_mN_1'], frame_forces['TipForce_mN_2']]).T.tolist(),
        'handle_forces': np.array(frame_forces['HandleForcesN1'])
    }

def handle_video_controls(paused, current_index, total_frames, show_force, use_blur=False, use_attenuation_map=False):
    """
    Handle video playback controls and return updated state.
    
    Args:
        paused: Current pause state
        current_index: Current frame index
        total_frames: Total number of frames
        show_force: Current force display state
        use_blur: Current blur state (optional)
        use_attenuation_map: Current attenuation map state (optional)
    
    Returns:
        dict: Updated control states
    """
    
    # Wait for key input based on pause state
    if paused:
        key = cv2.waitKey(0)  # Wait indefinitely when paused
    else:
        key = cv2.waitKey(90)  # 10 frames per second
        #key = cv2.waitKey(50)  # 20 frames per second
    
    # Initialize result with current states
    result = {
        'exit': False,
        'paused': paused,
        'index': current_index,
        'show_force': show_force,
        'use_blur': use_blur,
        'use_attenuation_map': use_attenuation_map
    }
    
    # Handle key presses
    if key == 27:  # Escape key
        result['exit'] = True
        
    elif key == 98:  # B key - jump to beginning
        result['index'] = 0
        print("Jumping back to the beginning")
        
    elif key == 32:  # Space key - pause/resume
        result['paused'] = not paused
        if result['paused']:
            print("Paused")
        else:
            print("Resumed")
            
    elif key == 81 or key == 2:  # Left arrow key - previous frame
        result['index'] = max(0, current_index - 1)
        
    elif key == 83 or key == 3:  # Right arrow key - next frame
        result['index'] = min(total_frames - 1, current_index + 1)
        
    elif key == 103:  # G key - toggle Gaussian blur
        result['use_blur'] = not use_blur
        if result['use_blur']:
            print("Gaussian blur enabled")
        else:
            print("Gaussian blur disabled")
            
    elif key == 97:  # A key - toggle attenuation map
        result['use_attenuation_map'] = not use_attenuation_map
        if result['use_attenuation_map']:
            print("Attenuation map enabled")
        else:
            print("Attenuation map disabled")
            
    elif key == 102:  # F key - toggle force display
        result['show_force'] = not show_force
        if result['show_force']:
            print("Force display enabled")
        else:
            print("Force display disabled")
            
    else:
        # Auto-advance if not paused
        if not paused:
            result['index'] = current_index + 1
    
    return result

# def compute_roi_from_line_and_seg(line_points, M, seg_img, seg_thresh=1):
#     """
#     Compute square ROI around intersection between a fitted line and segmentation mask.

#     Args:
#         line_points: list or array of (x,y) points in original image coords.
#         M: 2x3 affine rotation matrix used to rotate the image.
#         seg_img: rotated segmentation image (2D array of ints).
#         seg_thresh: segmentation threshold (pixels > seg_thresh are considered foreground).

#     Returns:
#         ROI tuple (x0, y0, side, side) or None if no intersection,
#         and rotated_line_points as an (N,2) int array of points in rotated coords.
#     """
#     if len(line_points) == 0:
#         return None, np.empty((0, 2), dtype=int)

#     # Transform the fitted line points to rotated coordinates
#     line_points_arr = np.array(line_points)  # (N,2)
#     ones = np.ones((line_points_arr.shape[0], 1))
#     line_points_h = np.concatenate([line_points_arr, ones], axis=1)  # (N,3)
#     # Apply affine transform (M is 2x3): result (N,2)
#     rotated_line_points = (M @ line_points_h.T).T
#     rotated_line_points = np.round(rotated_line_points).astype(int)

#     # Segmentation mask (foreground where value > seg_thresh)
#     seg_mask = (seg_img > seg_thresh).astype(np.uint8)

#     # Find intersection pixels between rotated line points and segmentation
#     intersection_points = []
#     for (x, y) in rotated_line_points:
#         if 0 <= x < seg_mask.shape[1] and 0 <= y < seg_mask.shape[0]:
#             if seg_mask[y, x] > 0:
#                 intersection_points.append((x, y))

#     if not intersection_points:
#         return None, rotated_line_points

#     intersection_points = np.array(intersection_points)

#     # Bounding box around intersection and make it square
#     x_min, x_max = intersection_points[:, 0].min(), intersection_points[:, 0].max()
#     y_min, y_max = intersection_points[:, 1].min(), intersection_points[:, 1].max()

#     w = x_max - x_min
#     h = y_max - y_min
#     side = max(w, h)
#     if side <= 0:
#         side = 1

#     cx = (x_min + x_max) // 2
#     cy = (y_min + y_max) // 2
#     x0 = max(0, cx - side // 2)
#     y0 = max(0, cy - side // 2)

#     # Clamp to image bounds
#     x0 = min(x0, seg_mask.shape[1] - side)
#     y0 = min(y0, seg_mask.shape[0] - side)
#     x0 = max(0, x0)
#     y0 = max(0, y0)

#     return (int(x0), int(y0), int(side), int(side)), rotated_line_points


def compute_ilm_aligned_roi(needle_line_points, seg_img, ilm_class=2, needle_class=1, roi_height=150):
    """
    Compute ROI that is aligned with the ILM (Inner Limiting Membrane) as the top boundary.
    
    Args:
        needle_line_points: List of (x, y) points representing the needle trajectory
        seg_img: Segmentation image in rotated coordinate system
        ilm_class: Segmentation class for ILM (default 2)
        needle_class: Segmentation class for needle (default 1) 
        roi_height: Desired height of the ROI in pixels
        
    Returns:
        ROI tuple (x0, y0, width, height) or None if no valid region found
        needle_line_points as array for consistency with other functions
    """
    if len(needle_line_points) == 0:
        return None, np.array(needle_line_points)
    
    needle_line_points = np.array(needle_line_points)
    
    # Find the needle intersection with tissue (any segmentation > 0)
    tissue_mask = seg_img > 0
    intersection_points = []
    
    for x, y in needle_line_points:
        if 0 <= x < seg_img.shape[1] and 0 <= y < seg_img.shape[0]:
            if tissue_mask[y, x]:
                intersection_points.append((x, y))
    
    if not intersection_points:
        return None, needle_line_points
    
    intersection_points = np.array(intersection_points)
    
    # Find the ILM (class 2) pixels
    ilm_mask = seg_img == ilm_class
    ilm_y_coords, ilm_x_coords = np.where(ilm_mask)
    
    if len(ilm_x_coords) == 0:
        print("⚠️ No ILM found for ROI alignment")
        return None, needle_line_points
    
    # Determine ROI center based on needle intersection
    center_x = int(np.mean(intersection_points[:, 0]))
    
    # Determine ROI width based on intersection spread plus some margin
    x_min, x_max = intersection_points[:, 0].min(), intersection_points[:, 0].max()
    intersection_width = x_max - x_min
    roi_width = max(150, int(intersection_width * 1.5))  # At least 150px wide
    
    # Calculate ROI bounds
    roi_x_start = max(0, center_x - roi_width // 2)
    roi_x_end = min(seg_img.shape[1], roi_x_start + roi_width)
    roi_x_start = max(0, roi_x_end - roi_width)  # Adjust if we hit right boundary
    
    # Find ILM top boundary within our ROI x-range
    ilm_in_roi_mask = (ilm_x_coords >= roi_x_start) & (ilm_x_coords < roi_x_end)
    
    if not np.any(ilm_in_roi_mask):
        print("⚠️ No ILM found in ROI region")
        return None, needle_line_points
    
    # Get the topmost ILM points in our ROI region
    ilm_y_in_roi = ilm_y_coords[ilm_in_roi_mask]
    ilm_top_y = int(np.min(ilm_y_in_roi))
    
    # ROI starts at ILM top and extends downward
    roi_y_start = max(0, ilm_top_y)
    roi_y_end = min(seg_img.shape[0], roi_y_start + roi_height)
    actual_height = roi_y_end - roi_y_start
    
    print(f"📐 ILM-aligned ROI: x=[{roi_x_start}, {roi_x_end}], y=[{roi_y_start}, {roi_y_end}]")
    print(f"📏 ROI dimensions: {roi_width} x {actual_height}")
    
    return (roi_x_start, roi_y_start, roi_width, actual_height), needle_line_points

def compute_roi_from_line_and_seg(line_points, M, seg_img, seg_thresh=1, to_left=True):
    """
    Compute square ROI whose LEFT edge intersects the fitted line and extends to the right,
    covering the intersection region with the segmentation mask.

    Args:
        line_points: list or array of (x,y) points in original image coords.
        M: 2x3 affine rotation matrix used to rotate the image.
        seg_img: rotated segmentation image (2D array of ints).
        seg_thresh: segmentation threshold (pixels > seg_thresh are considered foreground).
        to_left: if True, skip left-anchoring and CENTER the ROI around the needle intersection.

    Returns:
        ROI tuple (x0, y0, side, side) or None if no intersection,
        and rotated_line_points as an (N,2) int array of points in rotated coords.
    """
    if len(line_points) == 0:
        return None, np.empty((0, 2), dtype=int)

    # --- Step 1: Rotate line points (if rotation matrix provided) ---
    line_points_arr = np.array(line_points)
    
    if M is not None:
        # Apply rotation transformation
        ones = np.ones((line_points_arr.shape[0], 1))
        line_points_h = np.concatenate([line_points_arr, ones], axis=1)
        rotated_line_points = (M @ line_points_h.T).T
        rotated_line_points = np.round(rotated_line_points).astype(int)
    else:
        # No rotation needed, points are already in the correct coordinate system
        rotated_line_points = line_points_arr.astype(int)

    # --- Step 2: Segmentation mask ---
    seg_mask = (seg_img > seg_thresh).astype(np.uint8)

    # --- Step 3: Find intersection pixels between line and segmentation ---
    intersection_points = []
    for (x, y) in rotated_line_points:
        if 0 <= x < seg_mask.shape[1] and 0 <= y < seg_mask.shape[0]:
            if seg_mask[y, x] > 0:
                intersection_points.append((x, y))

    # --- Step 3b: Fallback for needle shadow - search in neighborhood ---
    if not intersection_points:
        search_radius = 20  # pixels to search around each line point
        for (x, y) in rotated_line_points:
            if 0 <= x < seg_mask.shape[1] and 0 <= y < seg_mask.shape[0]:
                # Search in small neighborhood around each line point
                for dx in range(-search_radius, search_radius + 1):
                    for dy in range(-search_radius, search_radius + 1):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < seg_mask.shape[1] and 
                            0 <= ny < seg_mask.shape[0] and 
                            seg_mask[ny, nx] > 0):
                            intersection_points.append((nx, ny))
                            break
                    if intersection_points:
                        break
                if intersection_points:
                    break

    # --- Step 3c: Final fallback - use line points directly ---
    if not intersection_points:
        # When needle shadow completely obscures segmentation, use line points
        valid_line_points = []
        for (x, y) in rotated_line_points:
            if 0 <= x < seg_mask.shape[1] and 0 <= y < seg_mask.shape[0]:
                valid_line_points.append((x, y))
        
        if not valid_line_points:
            return None, rotated_line_points
            
        intersection_points = np.array(valid_line_points)

    intersection_points = np.array(intersection_points)

    # --- Step 4: Bounding box around intersection ---
    x_min, x_max = intersection_points[:, 0].min(), intersection_points[:, 0].max()
    y_min, y_max = intersection_points[:, 1].min(), intersection_points[:, 1].max()

    w = x_max - x_min
    h = y_max - y_min
    side = max(w, h)
    if side <= 0:
        side = 1

    # Compute center of intersection region
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    # --- Step 5: Choose anchoring behavior ---
    if to_left:
        # CENTER the ROI around the needle intersection (skip left-anchoring)
        x0 = int(cx - side // 2)
        y0 = int(cy - side // 2)
    else:
        # LEFT-anchored: left edge intersects the line, ROI extends to the right
        x0 = int(cx)  # start at the line x-position
        y0 = int(cy - side // 2)

    # Clamp within image bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    x0 = min(x0, seg_mask.shape[1] - side)
    y0 = min(y0, seg_mask.shape[0] - side)

    ROI = (int(x0), int(y0), int(side), int(side))
    return ROI, rotated_line_points

def get_rotated_ROI(ROI, M):
    x0, y0, side, _ = ROI
    M_inv = cv2.invertAffineTransform(M)
    # Visualize ROI on original frame
    corners = np.array([
        [x0, y0],
        [x0 + side, y0],
        [x0 + side, y0 + side],
        [x0, y0 + side]
    ])
    corners_inv = cv2.transform(np.array([corners]), M_inv)[0]
    return corners_inv

    
def extend_classes_horizontally(label_img, background_value=0, ignore_value=1):
    """
    Extends non-background classes horizontally until they fill the image width.
    Ignores 'ignore_value' pixels (does not propagate them or overwrite them).
    """
    h, w = label_img.shape
    extended = label_img.copy()

    for y in range(h):
        row = label_img[y]
        # Valid pixels = not background, not ignore_value
        valid_mask = (row != background_value) & (row != ignore_value)
        if not np.any(valid_mask):
            continue
        
        # Compute nearest valid class index for every pixel
        dist, indices = distance_transform_edt(~valid_mask, return_indices=True)
        propagated_row = row[indices[0]]  # horizontally propagate nearest valid label

        # Keep ignore_value pixels intact
        propagated_row[row == ignore_value] = ignore_value

        extended[y] = propagated_row

    return extended

def find_sonification_row(diff_roi, smooth=15, quantile_low=0.6, quantile_high=0.95):
    """
    Given an ROI of intensity difference (current - baseline),
    find the most relevant ROW (depth) for sonification.

    Args:
        diff_roi (np.ndarray): Normalized intensity difference image (0-1).
        smooth (int): Smoothing window size for row profile.
        quantile_low (float): Lower quantile to ignore weak changes.
        quantile_high (float): Upper quantile to ignore saturated extremes.

    Returns:
        sonify_y (int): Row index (depth) to sonify.
        row_profile (np.ndarray): Vertical energy per row.
    """
    # 1D profile of row energy (sum of intensity differences along width)
    row_energy = np.sum(diff_roi, axis=1)

    # Normalize
    row_energy /= np.max(row_energy) + 1e-6

    # Smooth the profile
    row_energy_smooth = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, smooth), 0).ravel()

    # Threshold by quantiles to reject noise
    low_thr = np.quantile(row_energy_smooth, quantile_low)
    high_thr = np.quantile(row_energy_smooth, quantile_high)
    valid_indices = np.where((row_energy_smooth > low_thr) & (row_energy_smooth < high_thr))[0]

    if valid_indices.size == 0:
        return diff_roi.shape[0] // 2, row_energy_smooth  # fallback: middle depth

    # Choose the deepest valid high-energy row (towards the bottom)
    sonify_y = valid_indices[-1]
    return int(sonify_y), row_energy_smooth

def compute_deflection_farneback(prev_roi, curr_roi,
                                 pyr_scale=0.5, levels=3, winsize=15,
                                 iterations=3, poly_n=5, poly_sigma=1.2):
    """
    Compute dense optical flow (Farneback) between two grayscale patches.

    Args:
        prev_roi (np.ndarray): Previous frame patch (grayscale).
        curr_roi (np.ndarray): Current frame patch (grayscale).
        Returns:
        flow (np.ndarray): 2D deflection field of shape (H, W, 2)
        mag (np.ndarray): Magnitude map of shape (H, W)
    """
    prev_roi = prev_roi.astype(np.uint8)
    curr_roi = curr_roi.astype(np.uint8)
    # Apply gaussian smoothing
    prev_roi = cv2.GaussianBlur(prev_roi, (5, 5), 0)
    curr_roi = cv2.GaussianBlur(curr_roi, (5, 5), 0)

    flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi,
                                        None, pyr_scale, levels, winsize,
                                        iterations, poly_n, poly_sigma, 0)

    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return flow, mag

def compute_deflection(prev_roi, curr_roi, max_shift=15):
    """Compute vertical deflection map between two consecutive OCT ROI frames."""
    h, w = prev_roi.shape
    shifts = np.zeros(w)

    for x in range(w):
        prev_col = prev_roi[:, x]
        curr_col = curr_roi[:, x]
        corr = correlate(curr_col, prev_col, mode='full')
        shift = np.argmax(corr) - len(prev_col) + 1
        shift = np.clip(shift, -max_shift, max_shift)
        shifts[x] = shift

    return shifts  # positive = downward deflection

def compute_deflection_force_scaling(prev_frame, curr_frame, needle_tip_pos, rotated_patch_centers, 
                                   num_nodes_x, num_nodes_y, fixed_roi_params, debug=False):
                                   
    if fixed_roi_params is None:
        # No fixed ROI established, return uniform scaling
        return 1.0
    
    # Extract ROI parameters and rotation matrix
    if len(fixed_roi_params) == 5:
        roi_x0, roi_y0, roi_width, roi_height, rotation_matrix = fixed_roi_params
    else:
        # Fallback for old format without rotation matrix
        roi_x0, roi_y0, roi_width, roi_height = fixed_roi_params
        rotation_matrix = None
    
    # Convert to grayscale if needed
    if prev_frame.ndim == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame.copy()
        
    if curr_frame.ndim == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame.copy()
    
    # Transform ROI coordinates from rotated frame space back to original frame space
    if rotation_matrix is not None:
        # Create inverse transformation matrix
        center_x, center_y = prev_gray.shape[1] // 2, prev_gray.shape[0] // 2
        inv_rotation_matrix = cv2.invertAffineTransform(rotation_matrix)
        
        # Transform ROI corners from rotated space back to original space
        roi_corners = np.array([
            [roi_x0, roi_y0, 1],
            [roi_x0 + roi_width, roi_y0, 1],
            [roi_x0, roi_y0 + roi_height, 1],
            [roi_x0 + roi_width, roi_y0 + roi_height, 1]
        ]).T
        
        transformed_corners = inv_rotation_matrix @ roi_corners
        
        # Get bounding box of transformed ROI in original frame space
        min_x = int(np.floor(np.min(transformed_corners[0])))
        max_x = int(np.ceil(np.max(transformed_corners[0])))
        min_y = int(np.floor(np.min(transformed_corners[1])))
        max_y = int(np.ceil(np.max(transformed_corners[1])))
        
        # Ensure bounds are within frame
        roi_x0 = max(0, min_x)
        roi_y0 = max(0, min_y) 
        roi_x1 = min(prev_gray.shape[1], max_x)
        roi_y1 = min(prev_gray.shape[0], max_y)
    else:
        # No rotation, use ROI coordinates directly
        roi_x1 = roi_x0 + roi_width
        roi_y1 = roi_y0 + roi_height
    
    # Use the FIXED ROI parameters (transformed back to original frame space if needed)
    
    # Extract ROI patches
    prev_roi = prev_gray[roi_y0:roi_y1, roi_x0:roi_x1]
    curr_roi = curr_gray[roi_y0:roi_y1, roi_x0:roi_x1]
    # mask upper right triangle so (0,0) , n,n) and (n,0) are in the mask
    # mask = np.zeros_like(prev_roi, dtype=bool)
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if i >= j :
    #             mask[i, j] = True

    # prev_roi[mask] = 0
    # curr_roi[mask] = 0
    
    if prev_roi.size == 0 or curr_roi.size == 0:
        return 1.0

    prev_roi = cv2.GaussianBlur(prev_roi,(7,7), 1 ) 
    curr_roi = cv2.GaussianBlur(curr_roi, (7,7), 1)

    # Alternative: use intensity difference as deflection proxy
    intensity_diff = np.abs(curr_roi.astype(np.float32) - prev_roi.astype(np.float32))
        
    # Combine deflection magnitude and intensity change
    combined_deflection = intensity_diff 
    
    # Calculate single deflection scaling value for entire ROI
    # Get average deflection across the entire ROI
    avg_deflection = np.mean(combined_deflection)
    
    deflection_scaling = avg_deflection

    if debug:
        # Visualize deflection map
        deflection_vis = (combined_deflection * 255).astype(np.uint8)
        deflection_colored = cv2.applyColorMap(deflection_vis, cv2.COLORMAP_JET)
        
        # Draw fixed ROI in original frame coordinates
        if curr_frame.ndim == 3:
            vis_frame = curr_frame.copy()
        else:
            vis_frame = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
            
        cv2.rectangle(vis_frame, (roi_x0, roi_y0), (roi_x1, roi_y1), (0, 255, 0), 2)
        
        # Draw needle position if available (both are now in original frame coordinates)
        if needle_tip_pos[0] is not None and needle_tip_pos[1] is not None:
            cv2.circle(vis_frame, (int(needle_tip_pos[0]), int(needle_tip_pos[1])), 5, (0, 0, 255), -1)
        
        # Add text showing single scaling value
        cv2.putText(vis_frame, f"Deflection Scaling: {deflection_scaling:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Deflection Analysis", vis_frame)
        cv2.imshow("Deflection Map", deflection_colored)
        cv2.waitKey(1)
    
    deflection_scaling = deflection_scaling / 10
    return deflection_scaling

def compute_needle_force_proxy(needle_speed, velocity_x, velocity_y):
    
    base_force = needle_speed  
    
    force_x = abs(velocity_x) 
    force_y = abs(velocity_y)
    force_z = base_force * 0.1 # Small comp


    static = np.exp(-needle_speed * 2.5)

    dynamic = needle_speed ** 1.3

    force_x += static + dynamic * 0.15
    force_y += static + dynamic * 0.7
    force_z += static + dynamic * 0.15

    # Create force components similar to actual force data structure
    f_comps = np.array([[force_x, force_y, force_z]])
    
    # Compute magnitude
    magnitude = np.sqrt(force_x**2 + force_y**2 + force_z**2)
    
    # Create synthetic force_info structure
    force_proxy = {
        "f_comps": f_comps,
        "magnitudes": [magnitude],
        "tip_force_norm": magnitude,
        "tip_force_x": force_x,
        "tip_force_y": force_y, 
        "tip_force_z": force_z,
        "sclera_force_norm": 0, 
        "handle_forces": [0.0, 0.0, 0.0],  
        "timestamp": 0.0, 
        "csv_index": 0,    
        "time_diff": 0.0 
    }
    
    return force_proxy

def get_range_list(range_params_all, index):
    """
    Extracts a list of [min, max] ranges for a specific property index across classes.
    Example: index=0 -> mass ranges, index=1 -> stiffness ranges.
    """
    return [rp[index] for rp in range_params_all]

def postprocess_patch_center_classes(patch_centers_class, num_nodes_x, num_nodes_y):
    """
    Propagate class labels into nearby background (0) regions
    using morphological closing + dilation, but never overwrite
    existing non-background points.
    """
    processed_classes = patch_centers_class.copy()

    # If a class 3 appears, make column below class 3 to class 3
    class_3_indices = np.where(patch_centers_class == 3)[0]
    for idx in class_3_indices:
        row = idx // num_nodes_x
        col = idx % num_nodes_x
        for r in range(row + 1, num_nodes_y):
            proc_idx = r * num_nodes_x + col
            if processed_classes[proc_idx] == 0:
                processed_classes[proc_idx] = 3
            else:
                break  # Stop if we hit a non-background class
    
    # Also replace all ones with 0
    processed_classes[processed_classes == 1] = 0

    return processed_classes

def propagate_patch_center_classes_horizontally(patch_centers_class, num_nodes_x, num_nodes_y):
    """
    Propagate class labels horizontally into background (0) regions.
    For each background pixel, find the nearest non-background class in the same row.
    """
    processed_classes = patch_centers_class.copy().reshape(num_nodes_y, num_nodes_x)
    
    for row in range(num_nodes_y):
        # Find all background positions in this row
        background_cols = np.where(processed_classes[row] == 0)[0]
        
        if len(background_cols) == 0:
            continue  # No background pixels in this row
        
        # Find all non-background positions in this row
        non_background_cols = np.where(processed_classes[row] != 0)[0]
        
        if len(non_background_cols) == 0:
            continue  # No non-background pixels to propagate from
        
        # For each background pixel, find the nearest non-background pixel
        for bg_col in background_cols:
            # Find the nearest non-background column
            distances = np.abs(non_background_cols - bg_col)
            nearest_idx = np.argmin(distances)
            nearest_col = non_background_cols[nearest_idx]
            
            # Propagate the class from the nearest non-background pixel
            processed_classes[row, bg_col] = processed_classes[row, nearest_col]
    
    return processed_classes.flatten()
    
    


def find_closest_node_to_tip(needle_tip_pos, rotated_patch_centers, num_nodes_x=16, num_nodes_y=16):
    """
    Given the needle tip position (x,y) and the rotated patch centers,
    find the index of the patch center closest to the needle tip.
    """
    if needle_tip_pos[0] is None or needle_tip_pos[1] is None:
        return None  # No valid needle tip position

    # Reshape to (num_nodes_y, num_nodes_x, 2)
    rotated_patch_centers = rotated_patch_centers.reshape(num_nodes_y, num_nodes_x, 2)

    tip_x, tip_y = needle_tip_pos

    # Compute distances for every (y, x)
    distances = np.linalg.norm(rotated_patch_centers - np.array([tip_x, tip_y]), axis=2)

    # Find the (y, x) index of the minimum
    y_idx, x_idx = np.unravel_index(np.argmin(distances), distances.shape)

    return rotated_patch_centers[y_idx, x_idx], (y_idx, x_idx), distances[y_idx, x_idx]


def compute_and_rotate_patch_centers(ROI, num_nodes_x, num_nodes_y, frame_shape, angle):
    """
    Return (patch_centers, rotated_patch_centers).
    patch_centers are in the rotated ROI coordinate system (x,y).
    rotated_patch_centers are mapped back into the original frame using -angle.
    """
    if ROI is None:
        return np.zeros((0, 2)), np.zeros((0, 2))

    x0, y0, side_x, side_y = ROI
    patch_centers = []
    patch_size_x = side_x // num_nodes_x
    patch_size_y = side_y // num_nodes_y
    for i in range(num_nodes_y):
        for j in range(num_nodes_x):
            center_x = x0 + j * patch_size_x + patch_size_x // 2
            center_y = y0 + i * patch_size_y + patch_size_y // 2
            patch_centers.append((center_x, center_y))
    patch_centers = np.array(patch_centers, dtype=float)

    # Rotate patch centers back using inverse rotation (apply -angle)
    center = (frame_shape[1] // 2, frame_shape[0] // 2)
    M_inv = cv2.getRotationMatrix2D(center, angle, 1)
    ones = np.ones((patch_centers.shape[0], 1))
    patch_centers_homogeneous = np.hstack([patch_centers, ones])
    rotated_patch_centers = M_inv.dot(patch_centers_homogeneous.T).T

    return patch_centers, rotated_patch_centers

def rotate_patch_centers(patch_centers, frame_shape, angle):
    """
    Rotate patch centers back into original frame coordinates using -angle.
    """
    if patch_centers.shape[0] == 0:
        return patch_centers

    center = (frame_shape[1] // 2, frame_shape[0] // 2)
    M_inv = cv2.getRotationMatrix2D(center, angle, 1)
    ones = np.ones((patch_centers.shape[0], 1))
    patch_centers_homogeneous = np.hstack([patch_centers, ones])
    rotated_patch_centers = M_inv.dot(patch_centers_homogeneous.T).T

    return rotated_patch_centers

def get_needle_tip_pos_from_seg(needle_seg, M=None):
    needle_tip_pos = (None, None)
    if np.any(needle_seg):
        ys, xs = np.where(needle_seg)
        bottom_idx = np.argmax(ys)  # index of the lowest point (max y)
        bottom_x = xs[bottom_idx]
        bottom_y = ys[bottom_idx]
        if bottom_x and bottom_y and M is not None:
            needle_tip_pos_rotated = M.dot(np.array([bottom_x, bottom_y, 1]))
            bottom_x, bottom_y = int(needle_tip_pos_rotated[0]), int(needle_tip_pos_rotated[1])
        needle_tip_pos = (bottom_x, bottom_y)
        # Visualize needle tip
    else:
        needle_tip_pos = (None, None)
    return needle_tip_pos

def smoothen_and_assign_prev_position(prev_tip_pos, needle_tip_pos, alpha):
    if  prev_tip_pos is None:
        return prev_tip_pos, needle_tip_pos
    else:
        if needle_tip_pos[0] is not None and prev_tip_pos[0] is not None:
            smoothed_x = alpha * prev_tip_pos[0] + (1 - alpha) * needle_tip_pos[0]
            smoothed_y = alpha * prev_tip_pos[1] + (1 - alpha) * needle_tip_pos[1]
            needle_tip_pos = (smoothed_x, smoothed_y)
        # Make pos doesnt move backwards in x and y
        if needle_tip_pos[0] is not None and prev_tip_pos[0] is not None:
            if needle_tip_pos[0] < prev_tip_pos[0]:
                needle_tip_pos = (prev_tip_pos[0], needle_tip_pos[1])
            if needle_tip_pos[1] < prev_tip_pos[1]:
                needle_tip_pos = (needle_tip_pos[0], prev_tip_pos[1])
        prev_tip_pos = needle_tip_pos
    return prev_tip_pos, needle_tip_pos
    
def adjust_roi(
    ROI, img_shape, num_nodes_x, num_nodes_y,
    needle_tip=None, extend_tip=True, margin=20
):
    """
    Adjust ROI size and position:
      - Round to multiples of grid size
      - Optionally extend to include needle tip
      - Optionally add uniform margin
    """
    if ROI is None:
        return (0, 0, 1, 1)

    h, w = img_shape[:2]
    x0, y0, side_x, side_y = ROI

    # Extend ROI upward if tip is above
    if extend_tip and needle_tip is not None:
        tip_x, tip_y = needle_tip
        if tip_y < y0:
            extra = y0 - tip_y
            y0 = max(0, y0 - extra)
            side_y = min(side_y + extra, h - y0)

    # Add uniform margin
    if margin > 0:
        orig_x0, orig_y0 = x0, y0
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        # Calculate actual margin added to left/top
        left_margin = orig_x0 - x0
        top_margin = orig_y0 - y0
        # Add symmetric margin to right/bottom
        side_x = min(w - x0, side_x + left_margin + margin)
        side_y = min(h - y0, side_y + top_margin + margin)

    # Snap again to grid multiples
    side_x = ((side_x + num_nodes_x - 1) // num_nodes_x) * num_nodes_x
    side_y = ((side_y + num_nodes_y - 1) // num_nodes_y) * num_nodes_y

    # Move ROI if it exceeds image bounds
    if x0 + side_x > w:
        x0 = w - side_x
    if y0 + side_y > h:
        y0 = h - side_y

    return (x0, y0, side_x, side_y)


def normalize_oct(img):
    gray = img.astype(np.float32)

    # A) Remove background bias using top + bottom rows
    top = gray[:10, :].reshape(-1)
    bottom = gray[-10:, :].reshape(-1)
    bg = np.concatenate([top, bottom])
    bg_sorted = np.sort(bg)
    bg_val = np.mean(bg_sorted[int(len(bg_sorted)*0.66):])  # top 1/3
    gray = gray - bg_val
    gray[gray < 0] = 0

    # B) Clip extreme bright outliers (0.05%)
    sorted_all = np.sort(gray.reshape(-1))
    clip_val = sorted_all[int(len(sorted_all)*0.9995)]
    gray[gray > clip_val] = clip_val

    # C) Normalize to 0–1
    gray = gray / np.max(gray)
    return gray

def normalize_oct_fast(img):
    gray = img.astype(np.float32)
    H, W, _ = gray.shape

    # --- A) Fast background bias estimation using percentiles ---
    bg_rows = np.concatenate([gray[:10, :], gray[-10:, :]], axis=0).reshape(-1)

    # Approximate "top third" of background using 70th percentile
    bg_val = np.percentile(bg_rows, 70)   
    gray = gray - bg_val
    gray[gray < 0] = 0

    # --- B) Fast clipping of extreme outliers using 99.95th percentile ---
    clip_val = np.percentile(gray, 99.95)
    gray = np.clip(gray, 0, clip_val)

    # --- C) Normalize to [0, 1] ---
    max_val = gray.max()
    if max_val > 0:
        gray /= max_val

    return gray

def generate_attenuation_map_fast(img, window_size=5):
    img = img.astype(float)

    # Compute rolling sums using cumsum trick
    cumsum = np.cumsum(img, axis=0)
    rolling_sum = cumsum[window_size:, :] - cumsum[:-window_size, :]

    curr = img[:-window_size, :]
    denom = 2 * 0.0097 * (rolling_sum + 1e-6)
    attenuation = curr / denom

    # Normalize attenuation
    attenuation -= attenuation.min()
    attenuation /= attenuation.max()

    multiplied = attenuation * (img[:-window_size, :] / 255.0)

    multiplied -= multiplied.min()
    multiplied /= multiplied.max()

    # Resize back to original height by padding last few rows with zeros
    out = np.zeros_like(img)
    out[:multiplied.shape[0]] = multiplied

    return (out * 255).astype(np.uint8)