import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Global variable to track smoothed maximum change for temporal mapping
_global_smoothed_max_change = None
deformation_baseline = None

def save_debug_params_plot(masses, stiffnesses, damping, num_nodes_y, num_nodes_x, static_mapping_type="", filename="inference/debug_params.png"):
    """Save a clear spatial grid visualization of physical parameters matching node layout"""
    # Reshape arrays to 2D grid
    masses_vis = np.array(masses).reshape((num_nodes_y, num_nodes_x))
    stiffnesses_vis = np.array(stiffnesses).reshape((num_nodes_y, num_nodes_x))
    damping_vis = np.array(damping).reshape((num_nodes_y, num_nodes_x))
    
    # Create figure with proper aspect ratio for the grid
    fig_width = max(12, num_nodes_x * 1.5)
    fig_height = max(8, num_nodes_y * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Set up grid layout matching actual node positions
    ax.set_xlim(-0.5, num_nodes_x - 0.5)
    ax.set_ylim(-0.5, num_nodes_y - 0.5)
    ax.set_aspect('equal')
    
    # Create grid lines for clarity
    for i in range(num_nodes_x + 1):
        ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)
    for i in range(num_nodes_y + 1):
        ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
    
    # Add parameter values at each node position
    for i in range(num_nodes_y):
        for j in range(num_nodes_x):
            # Node center position
            # x, y = j, num_nodes_y - 1 - i  # Flip Y to match image coordinates
            x,y = j, i
            
            # Get parameter values
            mass_val = masses_vis[i, j]
            stiff_val = stiffnesses_vis[i, j]
            damp_val = damping_vis[i, j]
            
            # Create text box for each node with colored background
            box_props = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='black')
            
            # Multi-line text showing all three parameters
            param_text = f"M: {mass_val:.3f}\nK: {stiff_val:.3f}\nD: {damp_val:.3f}"
            
            ax.text(x, y, param_text, 
                   ha='center', va='center', 
                   fontsize=8, fontweight='bold',
                   bbox=box_props)
            
            # Add small colored indicators for each parameter
            # Mass indicator (red circle)
            circle_m = patches.Circle((x - 0.35, y + 0.25), 0.05, color='red', alpha=0.7)
            ax.add_patch(circle_m)
            
            # Stiffness indicator (green square)
            square_k = patches.Rectangle((x - 0.05, y + 0.2), 0.1, 0.1, color='green', alpha=0.7)
            ax.add_patch(square_k)
            
            # Damping indicator (blue triangle)
            triangle_d = patches.Polygon([(x + 0.25, y + 0.2), (x + 0.35, y + 0.2), (x + 0.3, y + 0.3)], 
                                       color='blue', alpha=0.7)
            ax.add_patch(triangle_d)
    
    # Set labels and title
    ax.set_xlabel('Node X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Node Y Position', fontsize=12, fontweight='bold')
    ax.set_title(f'Physical Parameters Spatial Grid ({static_mapping_type})\n' +
                 '🔴 Mass | 🟢 Stiffness | 🔵 Damping', fontsize=14, fontweight='bold')
    
    # Set integer ticks to match node positions
    ax.set_xticks(range(num_nodes_x))
    ax.set_yticks(range(num_nodes_y))
    
    # Invert Y-axis to match image coordinates (0,0 at top-left)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✅ Saved spatial parameters grid: {filename}")
    plt.close()  # Close to free memory

def map_physical_params(
    static_mapping_type,
    seg_img_0,
    rotated_patch_centers,
    patch_centers_class,
    num_nodes_x,
    num_nodes_y,
    RANGE_PARAMS_ALL,
    means=None,
    std=None,
    ROI=None,
    frame_0_rotated=None,
    debug=False,
):
    if static_mapping_type == "intensity":
        if means is None or std is None:
            raise ValueError("For 'intensity' mapping, 'means' and 'std' must be provided.")
        masses = remap_classwise(means, patch_centers_class, RANGE_PARAMS_ALL, 0, scale="linear")
        stiffnesses = remap_classwise(std, patch_centers_class, RANGE_PARAMS_ALL, 1, scale="linear")
        damping = remap_classwise(means, patch_centers_class, RANGE_PARAMS_ALL, 2, scale="linear")

    elif static_mapping_type == "dRPE":
        print("Remapping based on distance from RPE")
        seg_RPE = seg_img_0 == 3
        y_first = np.argmax(seg_RPE, axis=0)
        valid_cols = seg_RPE.any(axis=0)
        top1 = np.zeros_like(seg_RPE, dtype=bool)
        top1[y_first[valid_cols], np.where(valid_cols)[0]] = True
        rpe_ys, rpe_xs = np.where(top1)
        rpe_points = np.array(list(zip(rpe_xs, rpe_ys)))

        distances = []
        for cx, cy in rotated_patch_centers:
            if rpe_points.size == 0:
                distances.append(max(seg_img_0.shape))
                continue
            dists = np.linalg.norm(rpe_points - np.array([cx, cy]), axis=1)
            distances.append(dists.min())

        distances = np.array(distances).reshape((num_nodes_y, num_nodes_x))

        if debug and ROI and frame_0_rotated is not None:
            x0, y0, side_x, side_y = ROI
            vis = frame_0_rotated.copy()
            for i in range(num_nodes_y):
                for j in range(num_nodes_x):
                    cx = x0 + j * (side_x // num_nodes_x) + (side_x // num_nodes_x) // 2
                    cy = y0 + i * (side_y // num_nodes_y) + (side_y // num_nodes_y) // 2
                    cv2.putText(vis, f"{distances[i,j]:.1f}", (cx-15, cy+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            cv2.imshow("Distances to RPE", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        masses = remap_classwise(distances, patch_centers_class, RANGE_PARAMS_ALL, 0)
        stiffnesses = remap_classwise(distances, patch_centers_class, RANGE_PARAMS_ALL, 1)
        damping = remap_classwise(distances, patch_centers_class, RANGE_PARAMS_ALL, 2)

    elif static_mapping_type == "dClass":
        print("Remapping based on distance to next semantic class (dClass)")
        
        def find_next_class_spatially(cx, cy, current_class, seg_img):
            """Find the next different class encountered when moving down from (cx, cy)"""
            height, width = seg_img.shape
            
            # Start from current position and move downward
            for y in range(int(cy) + 1, height):
                if 0 <= int(cx) < width:
                    next_cls = seg_img[y, int(cx)]
                    if next_cls != current_class and next_cls != 1:  # Skip background class 1
                        return next_cls
            
            # If no different class found moving straight down, expand search in nearby columns
            for dy in range(1, height - int(cy)):
                y = int(cy) + dy
                if y >= height:
                    break
                    
                # Search in expanding radius around the column
                for dx in range(-min(5, int(cx)), min(6, width - int(cx))):
                    x = int(cx) + dx
                    if 0 <= x < width:
                        next_cls = seg_img[y, x]
                        if next_cls != current_class and next_cls != 1:
                            return next_cls
            
            return None  # No next class found

        # Precompute coordinates for each class
        points_by_class = {}
        for c in np.unique(seg_img_0):
            ys, xs = np.where(seg_img_0 == c)
            points_by_class[c] = np.vstack([xs, ys]).T if xs.size > 0 else np.empty((0, 2), dtype=float)

        distances = []
        max_fallback = max(seg_img_0.shape)
        
        for idx, (cx, cy) in enumerate(rotated_patch_centers):
            cls = int(patch_centers_class[idx]) if idx < len(patch_centers_class) else None
            
            if cls == 3:  # RPE - use distance to bottom of image
                distances.append(abs(seg_img_0.shape[0] - cy))
            else:
                # Find the next class spatially (moving down in Y direction)
                next_cls = find_next_class_spatially(cx, cy, cls, seg_img_0)
                
                if next_cls is not None:
                    # Calculate distance to nearest point of the next class
                    pts = points_by_class.get(next_cls, np.empty((0, 2), dtype=float))
                    if pts.size > 0:
                        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
                        distances.append(dists.min())
                    else:
                        distances.append(max_fallback)
                else:
                    # No next class found - use distance to bottom of image
                    distances.append(abs(seg_img_0.shape[0] - cy))

        distances = np.array(distances).reshape((num_nodes_y, num_nodes_x))

        # Normalize distances for better scaling PER CLASS, so it is [0,1]
        for c in np.unique(patch_centers_class):
            class_mask = (patch_centers_class.reshape((num_nodes_y, num_nodes_x)) == c)
            class_distances = distances[class_mask]
            if class_distances.size > 0:
                min_d = class_distances.min()
                max_d = class_distances.max()
                if max_d > min_d:
                    distances[class_mask] = (class_distances - min_d) / (max_d - min_d)
                else:
                    # If all distances are the same, use a meaningful default (0.5) instead of 0.0
                    # This prevents class 4 (highest class) from getting all zeros
                    distances[class_mask] = 0.5

        if debug and ROI and frame_0_rotated is not None:
            x0, y0, side_x, side_y = ROI
            vis = frame_0_rotated.copy()
            for i in range(num_nodes_y):
                for j in range(num_nodes_x):
                    cx = x0 + j * (side_x // num_nodes_x) + (side_x // num_nodes_x) // 2
                    cy = y0 + i * (side_y // num_nodes_y) + (side_y // num_nodes_y) // 2
                    cv2.putText(vis, f"{distances[i,j]:.1f}", (cx-15, cy+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,128,255), 1)
            cv2.imshow("Distances to Next Class (dClass)", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        masses = remap_classwise(distances, patch_centers_class, RANGE_PARAMS_ALL, 0)
        stiffnesses = remap_classwise(distances, patch_centers_class, RANGE_PARAMS_ALL, 1)
        damping = remap_classwise(distances, patch_centers_class, RANGE_PARAMS_ALL, 2)

        if debug and ROI and frame_0_rotated is not None:
            # Save a clear spatial visualization plot instead of unreadable text overlay
            save_debug_params_plot(masses, stiffnesses, damping, num_nodes_y, num_nodes_x, 
                                 static_mapping_type="dClass", filename="debug_params.png")
        print("Mean mass per class:", [np.mean(masses[patch_centers_class == c]) for c in np.unique(patch_centers_class)])
        print("Mean stiffness per class:", [np.mean(stiffnesses[patch_centers_class == c]) for c in np.unique(patch_centers_class)])
        print("Mean damping per class:", [np.mean(damping[patch_centers_class == c]) for c in np.unique(patch_centers_class)])
    else:
        raise ValueError(f"Invalid static_mapping_type: {static_mapping_type}")

    return (
        masses.reshape((num_nodes_y, num_nodes_x)),
        stiffnesses.reshape((num_nodes_y, num_nodes_x)),
        damping.reshape((num_nodes_y, num_nodes_x))
    )


def compute_dynamic_intensity_mapping(
    current_frame,
    rotated_patch_centers,
    patch_centers_class,
    num_nodes_x,
    num_nodes_y,
    ROI,
    RANGE_PARAMS_ALL,
    base_masses,
    base_stiffnesses,
    base_damping,
    rotation_matrix=None,
    debug=False,
    baseline_frame=None,  # NEW: Reference frame for temporal comparison
    intensity_history=None,  # NEW: Track intensity changes over time,
    needle_tip_pos=None
):
    if ROI is None:
        if debug:
            print("⚠️ No ROI available for dynamic intensity mapping")
        return base_masses, base_stiffnesses, base_damping
        
    x0, y0, side_x, side_y = ROI

    
    # OPTIMIZATION 1: Fast frame preprocessing (avoid transformations when possible)
    if rotation_matrix is not None:
        frame_rotated = cv2.warpAffine(current_frame, rotation_matrix, 
                                     (current_frame.shape[1], current_frame.shape[0]))
    else:
        frame_rotated = current_frame
    
    # OPTIMIZATION 2: Fast grayscale conversion
    if len(frame_rotated.shape) == 3:
        frame_gray = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame_rotated
    
    # Extract ROI
    frame_roi = frame_gray[y0:y0 + side_y, x0:x0 + side_x]
    
    needle_tip_pos_within_ROI = None
    if needle_tip_pos is not None:
        # Calculate relative position within ROI
        rel_x = needle_tip_pos[0] - x0
        rel_y = needle_tip_pos[1] - y0
        
        # Check if needle tip is within ROI bounds
        if 0 <= rel_x < side_x and 0 <= rel_y < side_y:
            needle_tip_pos_within_ROI = (int(rel_x), int(rel_y))
        else:
            # Needle tip is outside ROI - clamp to ROI boundaries
            clamped_x = max(0, min(side_x - 1, int(rel_x)))
            clamped_y = max(0, min(side_y - 1, int(rel_y)))
            needle_tip_pos_within_ROI = (clamped_x, clamped_y)
            print(f"⚠️ Needle tip outside ROI, clamped from ({rel_x:.1f}, {rel_y:.1f}) to {needle_tip_pos_within_ROI}")

    mask_below_needle = None
    if needle_tip_pos_within_ROI is not None:
        # Create mask for region below and to the left of needle tip
        mask_below_needle = np.zeros_like(frame_roi, dtype=bool)
        tip_x, tip_y = needle_tip_pos_within_ROI
        
        # Ensure coordinates are valid for array indexing
        if tip_y < frame_roi.shape[0] and tip_x < frame_roi.shape[1]:
            mask_below_needle[tip_y:, :tip_x + 1] = True
    
    # OPTIMIZATION 3: Vectorized block-based intensity calculation
    patch_size_x = side_x // num_nodes_x
    patch_size_y = side_y // num_nodes_y
    
    # Ensure frame dimensions are multiples of patch grid
    target_height = num_nodes_y * patch_size_y
    target_width = num_nodes_x * patch_size_x
    
    # Method 1: Fast numpy reshape and averaging (fastest approach)
    try:
        if frame_roi.shape[0] != target_height or frame_roi.shape[1] != target_width:
            frame_resized = cv2.resize(frame_roi, (target_width, target_height))
        else:
            frame_resized = frame_roi
        
        # Vectorized block averaging using reshape - extremely fast!
        patch_intensities = frame_resized.reshape(num_nodes_y, patch_size_y, 
                                                 num_nodes_x, patch_size_x).mean(axis=(1, 3))
        
    except (ValueError, MemoryError):
        # Fallback: Manual but still optimized approach
        patch_intensities = np.zeros((num_nodes_y, num_nodes_x))
        for i in range(num_nodes_y):
            y_start, y_end = i * patch_size_y, min((i + 1) * patch_size_y, side_y)
            for j in range(num_nodes_x):
                x_start, x_end = j * patch_size_x, min((j + 1) * patch_size_x, side_x)
                patch = frame_roi[y_start:y_end, x_start:x_end]
                patch_intensities[i, j] = np.mean(patch) if patch.size > 0 else 0
    
    # NEW: Temporal evolution tracking
    baseline_intensities = patch_intensities
    if baseline_frame is not None and intensity_history is not None:
        # Compute baseline intensities for comparison
        if len(baseline_frame.shape) == 3:
            baseline_gray = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY)
        else:
            baseline_gray = baseline_frame
        baseline_roi = baseline_gray[y0:y0 + side_y, x0:x0 + side_x]
        
        # Get baseline patch intensities
        if baseline_roi.shape[0] != target_height or baseline_roi.shape[1] != target_width:
            baseline_resized = cv2.resize(baseline_roi, (target_width, target_height))
        else:
            baseline_resized = baseline_roi
        
        baseline_intensities = baseline_resized.reshape(num_nodes_y, patch_size_y, 
                                                       num_nodes_x, patch_size_x).mean(axis=(1, 3))
        
    if mask_below_needle is not None:
        mask_below_needle = cv2.resize(mask_below_needle.astype(np.uint8), (num_nodes_x, num_nodes_y)).astype(bool).T
        patch_intensities[mask_below_needle.reshape(num_nodes_y, num_nodes_x)] = baseline_intensities[mask_below_needle.reshape(num_nodes_y, num_nodes_x)]
        # Compute TEMPORAL changes (current vs baseline)
    updated_masses, updated_stiffnesses, updated_damping, deformation = \
        per_patch_percent_change_mapping(
            patch_intensities,
            baseline_intensities,
            base_masses,
            base_stiffnesses,
            base_damping,
        )
    global deformation_baseline
    return updated_masses, updated_stiffnesses, updated_damping, (deformation - deformation_baseline if deformation_baseline is not None else 0)

def remap_classwise(values, classes, range_params_all, param_index, scale="linear", invert=False):
    values = np.asarray(values, dtype=float).flatten()
    classes = np.asarray(classes).astype(int)
    remapped_values = np.zeros_like(values, dtype=float)

    for c in np.unique(classes):
        mask = (classes == c)
        vals_c = values[mask]
        if vals_c.size == 0 or c >= len(range_params_all):
            continue

        vmin_target, vmax_target = range_params_all[c][param_index]
        vmin, vmax = vals_c.min(), vals_c.max()

        # Handle degenerate case
        if np.isclose(vmax, vmin):
            remapped_values[mask] = 0.5 * (vmin_target + vmax_target)
            continue

        # Normalize to [0, 1]
        norm_vals = (vals_c - vmin) / (vmax - vmin)

        # Exponential scaling in log domain (for perceptual growth)
        if scale == "exp":
            vals_c_safe = np.clip(vals_c, a_min=1e-6, a_max=None)
            vmin_safe, vmax_safe = max(vmin, 1e-6), max(vmax, 1e-6)
            log_vals = np.log(vals_c_safe)
            log_vmin, log_vmax = np.log(vmin_safe), np.log(vmax_safe)
            norm_vals = (log_vals - log_vmin) / (log_vmax - log_vmin)

        # Invert for distance-based mapping (closer → stronger)
        if invert:
            norm_vals = 1.0 - norm_vals
        

        # Map to target range (handles increasing or decreasing)
        remapped_c = vmin_target + norm_vals * (vmax_target - vmin_target)

        remapped_values[mask] = remapped_c

    return remapped_values

def per_patch_percent_change_mapping(
    patch_intensities,        
    baseline_intensities,     
    base_masses,
    base_stiffnesses,
    base_damping,
    alpha=2,   # ↑ stronger stiffness modulation
    beta=2,    # ↑ stronger damping modulation
    gamma=0.1,   # mass mostly unchanged, remain small
    K_MIN=0.05,  # prevents zero stiffness
    C_MIN=0.00005, # prevents zero damping
    C_MAX=0.5,  # avoids overdamping (silent system)
):
    """
    Nonlinear (cubic-enhanced) deformation mapping:
    - darker → more deformation
    - deformation is amplified using x^3 for expressiveness
    - stiffness decreases, damping increases
    - all outputs remain in safe ranges
    """
    global deformation_baseline

    # Avoid divide-by-zero
    baseline_safe = np.clip(baseline_intensities, 5, None)

    # Base deformation: darker → more deformation
    deformation_linear = (baseline_safe - patch_intensities) / baseline_safe
    deformation_linear = np.clip(deformation_linear, 0.0, 1.0)

    # ⭐ Nonlinear expressive deformation curve
    # Keeps small changes small, amplifies moderate ones, saturates at high deformation
    deformation = deformation_linear**3
    def_mean = deformation.mean()
    if deformation_baseline is None:
        deformation_baseline = def_mean.copy()

    # -------------------------------
    # PHYSICAL PARAMETER UPDATES
    # -------------------------------

    # Stiffness decreases
    stiffness_scale = 1.0 - alpha * deformation
    updated_stiffnesses = base_stiffnesses * stiffness_scale
    updated_stiffnesses = np.clip(updated_stiffnesses, K_MIN, None)

    # Damping increases
    damping_scale   = 1.0 + beta * deformation
    updated_damping = base_damping * damping_scale
    updated_damping = np.clip(updated_damping, C_MIN, C_MAX)

    return base_masses, updated_stiffnesses, updated_damping, def_mean
