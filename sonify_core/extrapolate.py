import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt  # For debugging visualizations

RPE_LABEL = 3          # RPE class
ILM_LABEL = 2          # ILM class
EXTRAP_DIST = 100

DROP_PROB = 0.4        # probability of dropping segmentation columns
JITTER_STD = 3.0       # pixel jitter


def extrapolate_curve(binary_mask, extrapolate_distance=20, spline_degree=3, smoothing=0):
    """
    Extrapolate a partial curve in a binary mask using spline interpolation and derivative-based extrapolation.
    
    Args:
        binary_mask (np.ndarray): 2D binary array containing curve points (1s) on background (0s)
        extrapolate_distance (int): Maximum distance to extrapolate on each side
        spline_degree (int): Degree of spline interpolation (1-5, typically 3)
        smoothing (float): Smoothing factor for spline (0 for exact interpolation)
    
    Returns:
        np.ndarray: Binary mask with extrapolated curve
    """
    if binary_mask.ndim != 2:
        raise ValueError("Input must be a 2D binary mask")
    
    H, W = binary_mask.shape
    
    # Extract curve points
    ys, xs = np.nonzero(binary_mask)
    
    if len(xs) < spline_degree + 1:
        # Not enough points for spline, return original
        return binary_mask.copy()
    
    # Sort by x coordinate for proper spline fitting
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    
    # Remove duplicate x values by using median and outlier filtering
    unique_xs, inverse_indices = np.unique(xs_sorted, return_inverse=True)
    if len(unique_xs) < len(xs_sorted):
        unique_ys = []
        for i in range(len(unique_xs)):
            ys_at_x = ys_sorted[inverse_indices == i]
            if len(ys_at_x) == 1:
                unique_ys.append(ys_at_x[0])
            else:
                # Use median and filter outliers (beyond 2 standard deviations)
                median_y = np.median(ys_at_x)
                if len(ys_at_x) > 2:
                    std_y = np.std(ys_at_x)
                    filtered_ys = ys_at_x[np.abs(ys_at_x - median_y) <= 2 * std_y]
                    if len(filtered_ys) > 0:
                        unique_ys.append(np.mean(filtered_ys))
                    else:
                        unique_ys.append(median_y)
                else:
                    unique_ys.append(median_y)
        unique_ys = np.array(unique_ys)
        xs_sorted, ys_sorted = unique_xs, unique_ys
    
    # Require more points for stable spline fitting
    min_points = max(spline_degree + 3, 8)  # At least 8 points for stability
    if len(xs_sorted) < min_points:
        return binary_mask.copy()
    
    try:
        # Fit spline with appropriate smoothing for noisy data
        adaptive_smoothing = max(smoothing, len(xs_sorted) * 0.1) if smoothing == 0 else smoothing
        spl = UnivariateSpline(xs_sorted, ys_sorted, k=min(2, len(xs_sorted)-1), s=adaptive_smoothing)
        
        # Interpolate existing range
        x_interp = np.arange(xs_sorted.min(), xs_sorted.max() + 1)
        y_interp = spl(x_interp)
        
        # Left extrapolation using spline derivative (more robust)
        x_start = x_interp[0]
        y_start = y_interp[0]
        # Use spline derivative instead of linear fit
        dy_dx_start = spl.derivative()(x_start)
        # Limit extreme slopes to prevent unrealistic extrapolation
        dy_dx_start = np.clip(dy_dx_start, -2.0, 2.0)
        
        x_left_start = max(0, x_start - extrapolate_distance)
        x_left = np.arange(x_left_start, x_start)
        y_left = y_start + dy_dx_start * (x_left - x_start)
        
        # Right extrapolation with slope limiting
        x_end = x_interp[-1]
        y_end = y_interp[-1]
        dy_dx_end = spl.derivative()(x_end)
        # Limit extreme slopes to prevent unrealistic extrapolation
        dy_dx_end = np.clip(dy_dx_end, -2.0, 2.0)
        
        x_right_end = min(x_end + extrapolate_distance + 1, W)
        x_right = np.arange(x_end + 1, x_right_end)
        y_right = y_end + dy_dx_end * (x_right - x_end)
        
        # Combine all segments
        x_full = np.concatenate([x_left, x_interp, x_right])
        y_full = np.concatenate([y_left, y_interp, y_right])
        
        # Apply smoothing to reduce sudden jumps
        y_full = smooth_curve_transitions(x_full, y_full, max_jump_threshold=15)
        
        # Convert to integers and filter valid coordinates
        x_int = x_full.astype(int)
        y_int = y_full.astype(int)
        
        # Vectorized bounds checking
        valid_mask = (x_int >= 0) & (x_int < W) & (y_int >= 0) & (y_int < H)
        x_valid = x_int[valid_mask]
        y_valid = y_int[valid_mask]
        
        # Create output mask
        out = np.zeros_like(binary_mask)
        out[y_valid, x_valid] = 1
        
        return out
        
    except Exception:
        # Fallback: return original mask if spline fitting fails
        return binary_mask.copy()

def smooth_curve_transitions(x_coords, y_coords, max_jump_threshold=10):
    """
    Smooth sudden jumps in curve that exceed a threshold.
    
    Args:
        x_coords, y_coords: Arrays of curve coordinates
        max_jump_threshold: Maximum allowed jump between adjacent points
    
    Returns:
        Smoothed y_coords
    """
    if len(y_coords) < 3:
        return y_coords.copy()
        
    y_smooth = y_coords.copy()
    
    # Forward pass - detect and smooth large jumps
    for i in range(1, len(y_coords)):
        jump = abs(y_coords[i] - y_coords[i-1])
        if jump > max_jump_threshold:
            # Use linear interpolation between surrounding points
            if i < len(y_coords) - 1:
                # Interpolate between previous and next point
                y_smooth[i] = (y_coords[i-1] + y_coords[i+1]) / 2
            else:
                # At end, use previous point with small continuation
                y_smooth[i] = y_coords[i-1] + np.sign(y_coords[i] - y_coords[i-1]) * max_jump_threshold/2
    
    return y_smooth


def debug_plot_extrapolation(binary_mask, result_mask, title="Curve Extrapolation"):
    """
    Debug visualization of curve extrapolation process.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original points
    ys_orig, xs_orig = np.nonzero(binary_mask)
    ax1.scatter(xs_orig, ys_orig, c='red', s=20, alpha=0.7, label='Original points')
    ax1.set_ylim(binary_mask.shape[0], 0)  # Flip y-axis for image coordinates
    ax1.set_title('Original Curve Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Extrapolated result
    ys_result, xs_result = np.nonzero(result_mask)
    ax2.scatter(xs_result, ys_result, c='blue', s=10, alpha=0.5, label='Extrapolated curve')
    ax2.scatter(xs_orig, ys_orig, c='red', s=20, alpha=0.8, label='Original points')
    ax2.set_ylim(result_mask.shape[0], 0)  # Flip y-axis for image coordinates
    ax2.set_title('Extrapolated Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def extrapolate_rpe_and_ilm(seg_mask, extrapolate_distance=20):
    """
    Extrapolate RPE and ILM layers in a segmentation mask.
    
    Args:
        seg_mask (np.ndarray): 2D array with segmentation labels (1 for RPE, 2 for ILM)
        extrapolate_distance (int): Distance to extrapolate on each side
    
    Returns:
        np.ndarray: Segmentation mask with extrapolated RPE and ILM layers
    """
    rpe_mask = (seg_mask == 3).astype(np.uint8)
    ilm_mask = (seg_mask == 2).astype(np.uint8)
    for x in range(seg_mask.shape[1]):
        ys = np.where(ilm_mask[:, x] == 1)[0]
        if len(ys) > 0:
            highest_y = ys.min()
            ilm_mask[:, x] = 0
            ilm_mask[highest_y, x] = 1
    
    rpe_extrapolated = extrapolate_curve(rpe_mask, extrapolate_distance=extrapolate_distance)
    ilm_extrapolated = extrapolate_curve(ilm_mask, extrapolate_distance=extrapolate_distance)
    
    out_seg = np.zeros_like(seg_mask)
    out_seg[rpe_extrapolated == 1] = 3 
    out_seg[ilm_extrapolated == 1] = 2
    
    return out_seg

def segmentation_line_confidence(seg, fitted_line, reference_line, label, sigma_L=8.0):
    valid_mask = (seg == label).any(axis=0)

    # coverage = valid_mask.sum() / valid_mask.shape[0]
    expected_mask = np.isfinite(reference_line)
    coverage = valid_mask[expected_mask].sum() / expected_mask.sum()
    coverage = np.clip(coverage, 0.2, 1.0)  # floor is important

    if valid_mask.sum() == 0:
        return 0.0

    delta = np.abs(fitted_line - reference_line)
    
    # Filter out NaN values more carefully
    valid_delta = delta[valid_mask]
    finite_delta = valid_delta[np.isfinite(valid_delta)]
    
    if len(finite_delta) == 0:
        # No valid deltas to compute, return low confidence
        mean_delta = sigma_L  # This will give C_motion ≈ 1/e ≈ 0.37
    else:
        mean_delta = np.mean(finite_delta)
    
    # Ensure mean_delta is finite
    if not np.isfinite(mean_delta):
        mean_delta = sigma_L  # Fallback value

    C_motion = np.exp(-mean_delta / sigma_L)
    return float(np.clip(coverage * C_motion, 0.0, 1.0))

def extract_line(seg, label):
    """Extract 1D curve (mean y per column)"""
    H, W = seg.shape
    line = np.full(W, np.nan)
    for x in range(W):
        ys = np.where(seg[:, x] == label)[0]
        if len(ys) > 0:
            line[x] = ys.mean()
    return line


def perturb_segmentation(seg, labels, drop_prob=0.4, jitter_std=3.0):
    """Randomly drop or jitter segmentation points for multiple labels"""
    seg = seg.copy()
    H, W = seg.shape

    for label in labels:
        for x in range(W):
            ys = np.where(seg[:, x] == label)[0]
            if len(ys) == 0:
                continue

            if np.random.rand() < drop_prob:
                seg[ys, x] = 0
            else:
                seg[ys, x] = 0
                y_new = int(np.clip(ys.mean() + np.random.randn() * jitter_std, 0, H - 1))
                seg[y_new, x] = label

    return seg


def create_segmentation_image(rpe_line, ilm_line, height, width, thickness=10):
    """Create segmentation image from fitted lines with specified thickness extending downward"""
    seg_img = np.zeros((height, width), dtype=np.uint8)
    
    for x in range(width):
        # RPE line with thickness extending downward
        if not np.isnan(rpe_line[x]):
            y_center = int(np.clip(rpe_line[x], 0, height - 1))
            y_start = y_center
            y_end = min(height, y_center + thickness)
            seg_img[y_start:y_end, x] = RPE_LABEL
            
        # ILM line with thickness extending downward
        if not np.isnan(ilm_line[x]):
            y_center = int(np.clip(ilm_line[x], 0, height - 1))
            y_start = y_center
            y_end = min(height, y_center + thickness)
            seg_img[y_start:y_end, x] = ILM_LABEL
    
    return seg_img

def update_fitted_line(
    seg,
    state,
    extrapolate_distance=20,
    sigma_L=8.0,
    base_alpha=0.5,
    ref_alpha=0.05,
    thickness=10,
):
    # --- Extrapolate ---
    seg_ext = extrapolate_rpe_and_ilm(seg, extrapolate_distance)

    obs_ilm = extract_line(seg_ext, ILM_LABEL)
    obs_rpe = extract_line(seg_ext, RPE_LABEL)

    prior_ilm = state["prior"]["ILM"]
    prior_rpe = state["prior"]["RPE"]

    ref_ilm = state["reference"]["ILM"]
    ref_rpe = state["reference"]["RPE"]

    # --- Confidence (vs reference, not prior!) ---
    ilm_conf = segmentation_line_confidence(
        seg, obs_ilm, ref_ilm, ILM_LABEL, sigma_L
    )
    rpe_conf = segmentation_line_confidence(
        seg, obs_rpe, ref_rpe, RPE_LABEL, sigma_L
    )

    # --- Update reference slowly ---
    state["reference"]["ILM"] = (
        (1 - ref_alpha) * ref_ilm + ref_alpha * obs_ilm
    )
    state["reference"]["RPE"] = (
        (1 - ref_alpha) * ref_rpe + ref_alpha * obs_rpe
    )

    # --- Confidence-gated update ---
    ilm_alpha = base_alpha * (0.3 + 0.7 * ilm_conf)
    rpe_alpha = base_alpha * (0.3 + 0.7 * rpe_conf)

    ilm_alpha = np.clip(ilm_alpha, 0.05, base_alpha)
    rpe_alpha = np.clip(rpe_alpha, 0.05, base_alpha)

    fitted_ilm = (1 - ilm_alpha) * prior_ilm + ilm_alpha * obs_ilm
    fitted_rpe = (1 - rpe_alpha) * prior_rpe + rpe_alpha * obs_rpe

    # --- Store back ---
    state["prior"]["ILM"] = fitted_ilm.copy()
    state["prior"]["RPE"] = fitted_rpe.copy()

    # --- Build segmentation ---
    fitted_seg = create_segmentation_image(
        fitted_rpe, fitted_ilm,
        seg.shape[0], seg.shape[1],
        thickness=thickness
    )

    return (
        fitted_seg,
        {"ILM": fitted_ilm, "RPE": fitted_rpe},
        {"ILM": ilm_conf, "RPE": rpe_conf}
    )

def project_monotonic(y_targets, eps=1.0):
    """
    Enforce strict top-to-bottom ordering per column.
    
    Args:
        y_targets : (num_nodes_y, num_nodes_x) array
        eps       : minimum spacing between nodes (pixels)

    Returns:
        y_proj : monotonic version of y_targets
    """
    y_proj = y_targets.copy()
    num_nodes_y, num_nodes_x = y_proj.shape

    for j in range(num_nodes_x):
        for i in range(1, num_nodes_y):
            y_proj[i, j] = max(y_proj[i, j], y_proj[i - 1, j] + eps)

    return y_proj

def calculate_target_thickness(seg_img, ILM):
    thickness = 0
    cols = 0
    for x in range (seg_img.shape[1]):
        # Find avg thickness in y direction of ILM
        if not np.isnan(ILM[x]):
            thickness += (seg_img[:, x] == ILM_LABEL).sum()
            cols += 1
    thickness = 2*int(thickness / cols) if cols > 0 else 5
    return thickness
    
def ensure_anatomical_representation(patch_centers_class, seg_img, rotated_patch_centers, num_nodes_x, num_nodes_y, side_x, side_y):
    """
    Ensure each column has at least one RPE and one ILM node if these classes exist in the segmentation.
    """
    # Reshape to 2D grid
    class_grid = patch_centers_class.reshape((num_nodes_y, num_nodes_x))
    
    # Check if RPE and ILM classes exist in the segmentation
    unique_seg_classes = np.unique(seg_img)
    has_rpe = 2 in unique_seg_classes
    has_ilm = 3 in unique_seg_classes or 4 in unique_seg_classes
    
    for col in range(num_nodes_x):
        column_classes = class_grid[:, col]
        
        # Ensure RPE representation if it exists in segmentation
        if has_rpe and 2 not in column_classes:
            # Find the node most likely to be RPE based on position and surrounding pixels
            best_rpe_candidate = None
            best_rpe_score = 0
            
            for row in range(num_nodes_y):
                node_idx = row * num_nodes_x + col
                center = rotated_patch_centers[node_idx]
                cx, cy = int(center[0]), int(center[1])
                
                # Get larger patch to check for RPE pixels
                patch_size_x = side_x // num_nodes_x
                patch_size_y = side_y // num_nodes_y
                x0, x1 = max(0, cx - patch_size_x), min(seg_img.shape[1], cx + patch_size_x)
                y0, y1 = max(0, cy - patch_size_y), min(seg_img.shape[0], cy + patch_size_y)
                
                if x0 < x1 and y0 < y1:
                    extended_patch = seg_img[y0:y1, x0:x1]
                    rpe_pixel_count = np.sum(extended_patch == 2)
                    
                    if rpe_pixel_count > best_rpe_score:
                        best_rpe_score = rpe_pixel_count
                        best_rpe_candidate = row
            
            # Assign the best candidate to RPE class
            if best_rpe_candidate is not None and best_rpe_score > 0:
                class_grid[best_rpe_candidate, col] = 2
        
        # Ensure ILM representation if it exists in segmentation  
        if has_ilm and 3 not in column_classes and 4 not in column_classes:
            # Find the node most likely to be ILM
            best_ilm_candidate = None
            best_ilm_score = 0
            best_ilm_class = 3
            
            for row in range(num_nodes_y):
                node_idx = row * num_nodes_x + col
                center = rotated_patch_centers[node_idx]
                cx, cy = int(center[0]), int(center[1])
                
                # Get larger patch to check for ILM pixels
                patch_size_x = side_x // num_nodes_x
                patch_size_y = side_y // num_nodes_y
                x0, x1 = max(0, cx - patch_size_x), min(seg_img.shape[1], cx + patch_size_x)
                y0, y1 = max(0, cy - patch_size_y), min(seg_img.shape[0], cy + patch_size_y)
                
                if x0 < x1 and y0 < y1:
                    extended_patch = seg_img[y0:y1, x0:x1]
                    ilm_3_count = np.sum(extended_patch == 3)
                    ilm_4_count = np.sum(extended_patch == 4)
                    
                    if ilm_3_count > best_ilm_score:
                        best_ilm_score = ilm_3_count
                        best_ilm_candidate = row
                        best_ilm_class = 3
                    elif ilm_4_count > best_ilm_score:
                        best_ilm_score = ilm_4_count
                        best_ilm_candidate = row  
                        best_ilm_class = 4
            
            # Assign the best candidate to ILM class
            if best_ilm_candidate is not None and best_ilm_score > 0:
                class_grid[best_ilm_candidate, col] = best_ilm_class
    
    return class_grid.flatten()


def enforce_anatomical_consistency(patch_centers_class, seg_img, rotated_patch_centers, num_nodes_x, num_nodes_y, separate_ilm):
    """
    Enforce anatomical consistency: nodes between RPE and ILM should be retina tissue.
    """
    # Extract boundary lines from the segmentation
    ILM_line = extract_line(seg_img, ILM_LABEL)
    RPE_line = extract_line(seg_img, RPE_LABEL)
    
    # Reshape to 2D grid for easier manipulation
    class_grid = patch_centers_class.reshape((num_nodes_y, num_nodes_x))
    
    # Determine retina class to use
    retina_class = 4 if separate_ilm else 2
    
    for i, center in enumerate(rotated_patch_centers):
        cx, cy = int(center[0]), int(center[1])
        row = i // num_nodes_x
        col = i % num_nodes_x
        
        # Skip if coordinates are out of bounds
        if cx < 0 or cx >= len(ILM_line) or cx >= len(RPE_line):
            continue
            
        # Get boundary positions at this x-coordinate
        ilm_y = ILM_line[cx]
        rpe_y = RPE_line[cx]
        
        # Skip if boundaries are not defined (NaN)
        if np.isnan(ilm_y) or np.isnan(rpe_y):
            continue
            
        # Check if node is anatomically between RPE and ILM
        tolerance = 5.0  # pixels tolerance for boundary detection
        
        if ilm_y + tolerance < cy < rpe_y - tolerance:
            # Node is clearly in retina region
            if class_grid[row, col] == 0:  # Currently background
                print(f"🔧 Correcting background node at ({row},{col}) to retina class {retina_class}")
                class_grid[row, col] = retina_class
        elif abs(cy - ilm_y) <= tolerance:
            # Node is at ILM boundary
            if class_grid[row, col] == 0:
                class_grid[row, col] = 3  # ILM class
        elif abs(cy - rpe_y) <= tolerance:
            # Node is at RPE boundary  
            if class_grid[row, col] == 0:
                class_grid[row, col] = 2  # RPE class
    
    return class_grid.flatten()