"""
Segmentation utilities for OCT image analysis and needle detection.

This module contains functions for:
- Needle detection and tracking
- Shadow detection below needle peaks
- Layer parameter extraction from segmentation maps
- Region-based analysis for tissue characterization
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter


def find_needle_peaks(frame):
    """
    Find needle peaks by detecting the brightest points in each column.
    
    Parameters
    ----------
    frame : np.ndarray
        Input OCT frame (BGR or grayscale)
    
    Returns
    -------
    max_indices : np.ndarray
        Array of y-coordinates of brightest points for each column
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    max_indices = np.argmax(gray, axis=0)
    return max_indices


def find_shadow_start_below_peaks(image, brightest_y,
                                  dark_thresh=0.4, grad_thresh=0.05,
                                  min_dark_len=20, offset=5):
    """
    Detects the shadow onset starting from the known bright reflection (needle surface)
    in each A-scan.
    
    Parameters
    ----------
    image : np.ndarray
        2D OCT B-scan (float or uint8).
    brightest_y : np.ndarray
        Array (length = image width) with y positions of brightest points per A-scan.
    dark_thresh : float
        Intensity threshold (normalized [0,1]) below which region is considered dark.
    grad_thresh : float
        Minimum negative gradient to be considered a shadow onset.
    min_dark_len : int
        Minimum number of consecutive pixels below threshold for a valid shadow.
    offset : int
        Number of pixels below bright point to start checking (avoid reflection glare).
    
    Returns
    -------
    shadow_start : np.ndarray
        Array of y-coordinates where shadow starts for each column
    """
    h, w = image.shape
    shadow_start = np.full(w, np.nan)

    # Normalize image
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    for x in range(w):
        y_peak = int(brightest_y[x])
        if np.isnan(y_peak) or y_peak >= h - min_dark_len:
            continue

        # Column intensity profile
        col = gaussian_filter1d(img_norm[:, x], sigma=2)

        # Region below the peak
        start_y = min(y_peak + offset, h - min_dark_len)
        sub_col = col[start_y:]

        # Gradient in that region
        grad = np.gradient(sub_col)

        # Find first strong negative gradient
        candidates = np.where(grad < grad_thresh)[0]
        for c in candidates:
            y = start_y + c
            # Check persistence of darkness
            if np.all(col[y:y+min_dark_len] < dark_thresh):
                shadow_start[x] = y
                break

    # Smooth laterally for continuity
    if np.any(~np.isnan(shadow_start)):
        mean_val = float(np.nanmean(shadow_start))
        shadow_start = gaussian_filter1d(
            np.nan_to_num(shadow_start, nan=mean_val), sigma=5
        )
    return shadow_start


def find_shadow_start_below_peaks_masked(
    image, brightest_y,
    dark_thresh=0.4, grad_thresh=0.05,
    min_dark_len=20, offset=5
):
    """
    Optimized version of shadow detection using vectorized operations.
    
    Parameters
    ----------
    image : np.ndarray
        2D OCT B-scan (float or uint8).
    brightest_y : np.ndarray
        Array with y positions of brightest points per A-scan.
    dark_thresh : float
        Intensity threshold below which region is considered dark.
    grad_thresh : float
        Minimum negative gradient for shadow onset.
    min_dark_len : int
        Minimum consecutive pixels below threshold for valid shadow.
    offset : int
        Pixels below bright point to start checking.
    
    Returns
    -------
    shadow_start : np.ndarray
        Array of shadow start y-coordinates for each column
    """
    h, w = image.shape

    # Normalize and smooth image once
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    img_smooth = gaussian_filter(img_norm, sigma=(2, 0))

    # --- Step 1: Mask everything above each bright peak ---
    Y = np.arange(h)[:, None]
    mask_above = Y < (brightest_y[None, :] + offset)

    # Peak intensity per column
    peak_vals = img_smooth[np.clip(brightest_y.astype(int), 0, h-1), np.arange(w)]
    peak_map = np.tile(peak_vals, (h, 1))  # broadcast across height
    img_masked = img_smooth.copy()
    img_masked[mask_above] = peak_map[mask_above]

    # --- Step 2: Compute gradient once ---
    grad = np.gradient(img_masked, axis=0)

    # --- Step 3: Detect shadow onset ---
    dark_mask = img_masked < dark_thresh
    grad_mask = grad < -grad_thresh
    candidate_mask = dark_mask & grad_mask

    # Persistent darkness criterion
    dark_int = dark_mask.astype(np.int16)
    dark_run = np.cumsum(dark_int, axis=0)
    dark_below = dark_run[-1, None, :] - dark_run
    persistent_dark = dark_below >= min_dark_len

    combined = candidate_mask & persistent_dark

    # --- Step 4: First valid y per column ---
    shadow_start = np.argmax(combined, axis=0).astype(float)
    shadow_start[~np.any(combined, axis=0)] = np.nan

    # --- Step 5: Smooth laterally ---
    if np.any(~np.isnan(shadow_start)):
        mean_val = float(np.nanmean(shadow_start))
        shadow_start = gaussian_filter1d(
            np.nan_to_num(shadow_start, nan=mean_val), sigma=5
        )

    return shadow_start


def detect_needle_points(max_indices, shadow_start):
    """
    Detect needle points by comparing distance between peaks and shadow starts.
    
    Parameters
    ----------
    max_indices : np.ndarray
        Y-coordinates of brightest points for each column
    shadow_start : np.ndarray
        Y-coordinates of shadow start for each column
    
    Returns
    -------
    needle_mask : np.ndarray
        Boolean array indicating which columns contain needle points
    needle_coords : list of tuples
        List of (x, y) coordinates of detected needle points
    """
    # Distance between peaks and shadow starts
    dist = shadow_start - max_indices
    
    # Needle points are where this distance is less than the mean
    needle_mask = dist < dist.mean()
    
    # Get coordinates of needle points
    needle_coords = []
    for col, is_needle in enumerate(needle_mask):
        if is_needle and 0 <= max_indices[col] < len(max_indices):
            needle_coords.append((col, int(max_indices[col])))
    
    return needle_mask, needle_coords


def get_layer_params(image, seg, x_tip, y_tip, valid_labels=None, smooth=True):
    """
    Extracts per-layer parameters (m, c, k, w) for the column under the needle tip,
    using segmentation map boundaries.

    Parameters
    ----------
    image : np.ndarray
        Grayscale OCT image (2D float array normalized to [0,1]).
    seg : np.ndarray
        Segmentation map (same size as image), where each pixel has an integer label for the layer.
    x_tip : int
        x-position of the needle tip in pixels.
    y_tip : int
        y-position of the needle tip in pixels.
    valid_labels : list[int] or None
        If given, only these labels will be used as anatomical layers.
        Otherwise, all unique labels > 0 are used.
    smooth : bool
        If True, fills holes and removes isolated pixels in the segmentation column.

    Returns
    -------
    layers : list[tuple[int,int]]
        Pixel index ranges (y0,y1) for each detected layer along this A-scan.
    params : list[tuple[float,float,float,float]]
        Corresponding (m, c, k, w) tuples: mass, damping, stiffness, weight (distance-to-tip factor).
    """
    h, w = seg.shape
    seg_col = seg[:, x_tip].copy()

    if valid_labels is None:
        valid_labels = [l for l in np.unique(seg_col) if l > 0]

    layers, params = [], []
    for idx, lid in enumerate(valid_labels):
        ys = np.where(seg_col == lid)[0]
        if len(ys) < 3:
            continue  # skip tiny regions or noise
        y0, y1 = ys[0], ys[-1]
        if idx == 0:
            y0 = y_tip
        region = image[y0:y1, x_tip]

        # Compute local features
        m = 0.1 + 0.9 * (y1 - y0) / h        # normalized thickness as mass
        k = np.mean(region)                   # stiffness ~ reflectivity
        c = np.var(region)                    # damping ~ variance
        dist = abs((y0 + y1)/2 - y_tip)
        w = np.exp(-dist / 20)                # proximity weighting

        layers.append((y0, y1))
        params.append((m, c, k, w))

    return layers, params


def get_uniform_regions(image, seg, x_tip, layer_bounds, subdivisions=(6, 2)):
    """
    Extract subregions (nodes) for each tissue layer around the needle.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale OCT B-scan normalized to [0,1].
    seg : np.ndarray
        Segmentation map (same size as image) with layer IDs.
    x_tip : int
        x coordinate of needle tip.
    layer_bounds : list[tuple[int,int]]
        [(y0,y1), (y2,y3)] for first two tissues of interest.
    subdivisions : tuple[int,int]
        Number of subdivisions for each layer.
    
    Returns
    -------
    regions : list[tuple[int,int,int]]
        [(layer_id, y_start, y_end), ...]
    region_intensities : list[float]
        Mean intensity for each region.
    region_centers : list[int]
        Center y coordinate of each region.
    """
    regions, region_intensities, region_centers = [], [], []

    for lid, (y0, y1) in enumerate(layer_bounds):
        n_sub = subdivisions[lid]
        height = y1 - y0
        step = height / n_sub

        for i in range(n_sub):
            ys = int(y0 + i*step)
            ye = int(y0 + (i+1)*step)
            patch = image[ys:ye, x_tip]
            mean_intensity = np.mean(patch)
            center = (ys + ye) // 2

            regions.append((lid, ys, ye))
            region_intensities.append(mean_intensity)
            region_centers.append(center)

    return regions, region_intensities, region_centers


def compute_proximity_weights(region_centers, y_tip, falloff=20):
    """
    Compute proximity weights (1.0 near tip, 0.0 far).
    
    Parameters
    ----------
    region_centers : array-like
        Y-coordinates of region centers
    y_tip : int
        Y-coordinate of needle tip
    falloff : float
        Distance falloff parameter
    
    Returns
    -------
    weights : np.ndarray
        Proximity weights for each region
    """
    region_centers = np.array(region_centers)
    dists = np.abs(region_centers - y_tip)
    weights = np.exp(-dists / falloff)
    weights /= weights.max() + 1e-8  # normalize
    return weights


def rotate_to_align_segmentation(frame, seg_img, target_label=1):
    """
    Rotate frame and segmentation to align a specific segmentation label horizontally.
    
    Parameters
    ----------
    frame : np.ndarray
        Input OCT frame
    seg_img : np.ndarray
        Segmentation image
    target_label : int
        Segmentation label to align (default: 1)
    
    Returns
    -------
    rotated_frame : np.ndarray
        Rotated frame
    rotated_seg : np.ndarray
        Rotated segmentation
    angle : float
        Rotation angle in degrees
    """
    # Find pixels with target label
    ys, xs = np.where(seg_img == target_label)
    
    if len(xs) > 1:
        # Fit line to the segmentation
        m, b = np.polyfit(xs, ys, 1)
        angle = -np.degrees(np.arctan(m))
        
        # Create rotation matrix
        center = (frame.shape[1]//2, frame.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate both images
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        rotated_seg = cv2.warpAffine(seg_img, M, (seg_img.shape[1], seg_img.shape[0]), 
                                    flags=cv2.INTER_NEAREST)
        
        return rotated_frame, rotated_seg, angle
    else:
        return frame, seg_img, 0.0


def find_rightmost_segmentation_point(seg_img, target_label=1):
    """
    Find the rightmost point of a specific segmentation label.
    
    Parameters
    ----------
    seg_img : np.ndarray
        Segmentation image
    target_label : int
        Label to find rightmost point for
    
    Returns
    -------
    rightmost_x : int or None
        X-coordinate of rightmost point
    rightmost_y : int or None
        Y-coordinate of rightmost point
    """
    mask = seg_img == target_label
    
    if np.any(mask):
        rightmost_x = np.where(mask)[1].max()
        # Get the corresponding y coordinate
        rightmost_y = np.where(mask)[0][np.where(mask)[1] == rightmost_x][0]
        return rightmost_x, rightmost_y
    else:
        return None, None


def visualize_needle_detection(frame, max_indices, shadow_start, needle_coords):
    """
    Visualize needle detection results on a frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Input frame to draw on
    max_indices : np.ndarray
        Y-coordinates of brightness peaks
    shadow_start : np.ndarray
        Y-coordinates of shadow starts
    needle_coords : list of tuples
        List of (x, y) needle coordinates
    
    Returns
    -------
    vis_frame : np.ndarray
        Frame with visualization overlays
    """
    vis_frame = frame.copy()
    
    # Draw brightness peaks (red dots)
    for col, row in enumerate(max_indices):
        if 0 <= row < frame.shape[0]:
            cv2.circle(vis_frame, (col, row), 1, (0, 0, 255), -1)
    
    # Draw shadow starts (blue dots)
    for col, row in enumerate(shadow_start.astype(int)):
        if 0 <= row < frame.shape[0]:
            cv2.circle(vis_frame, (col, row), 1, (255, 0, 0), -1)
    
    # Draw needle points (green dots)
    for col, row in needle_coords:
        cv2.circle(vis_frame, (col, row), 1, (0, 255, 0), -1)
    
    return vis_frame


def visualize_segmentation_overlay(frame, seg_img, colors=None):
    """
    Overlay segmentation contours on a frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Input frame
    seg_img : np.ndarray
        Segmentation image
    colors : list of tuples, optional
        BGR colors for each label. Default: [(255,255,0), (0,255,255), (255,0,255)]
    
    Returns
    -------
    overlay_frame : np.ndarray
        Frame with segmentation overlay
    """
    if colors is None:
        colors = [(255,255,0), (0,255,255), (255,0,255)]
    
    overlay_frame = frame.copy()
    
    for i, label in enumerate(np.unique(seg_img)):
        if label == 0:
            continue  # skip background
        
        mask = (seg_img == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[i % len(colors)]
        cv2.drawContours(overlay_frame, contours, -1, color, 1)
    
    return overlay_frame


def extract_needle_tip_analysis(frame, seg_img, target_label=1):
    """
    Complete pipeline for needle tip analysis from OCT frame and segmentation.
    
    Parameters
    ----------
    frame : np.ndarray
        Input OCT frame
    seg_img : np.ndarray
        Segmentation image
    target_label : int
        Segmentation label for needle surface
    
    Returns
    -------
    analysis_results : dict
        Dictionary containing:
        - 'needle_tip_x': X-coordinate of needle tip
        - 'needle_tip_y': Y-coordinate of needle tip  
        - 'max_indices': Brightness peaks array
        - 'shadow_start': Shadow start positions
        - 'needle_coords': List of needle point coordinates
        - 'rotated_frame': Frame rotated to align segmentation
        - 'rotated_seg': Rotated segmentation
        - 'rotation_angle': Rotation angle applied
    """
    # Rotate to align segmentation
    rotated_frame, rotated_seg, angle = rotate_to_align_segmentation(frame, seg_img, target_label)
    
    # Find needle peaks and shadows
    gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY) if len(rotated_frame.shape) == 3 else rotated_frame
    max_indices = find_needle_peaks(gray)
    shadow_start = find_shadow_start_below_peaks(gray, max_indices)
    
    # Detect needle points
    needle_mask, needle_coords = detect_needle_points(max_indices, shadow_start)
    
    # Find rightmost segmentation point as needle tip
    needle_tip_x, needle_tip_y = find_rightmost_segmentation_point(rotated_seg, target_label)
    
    return {
        'needle_tip_x': needle_tip_x,
        'needle_tip_y': needle_tip_y,
        'max_indices': max_indices,
        'shadow_start': shadow_start,
        'needle_coords': needle_coords,
        'rotated_frame': rotated_frame,
        'rotated_seg': rotated_seg,
        'rotation_angle': angle
    }

def extract_ilm(ascan, rpe_y):
    # 1. Find all local maxima
    peaks, props = find_peaks(ascan, height=threshold, distance=3)

    # 2. Filter peaks to those above RPE
    peaks = [p for p in peaks if p < rpe_y - 20]

    if len(peaks) == 0:
        return None  # no ILM in this A-scan

    # 3. Choose the brightest peak in a plausible ILM band
    ilm_candidates = [p for p in peaks if rpe_y - p < 150]

    if len(ilm_candidates) == 0:
        return None

    # the ILM is usually the lowest (closest to RPE) candidate
    ilm = max(ilm_candidates)  

    return ilm

if __name__ == "__main__":
    path = "/Users/luisreyes/Sonify/SonifyOCT/data/data_sample_with_tip/07_25_23/b_i17/b_i17001.png"
    path = "/Users/luisreyes/Sonify/SonifyOCT/data/data_sample_with_tip/07_18_23/b_i12/Images 1133.png"
    path = "/Users/luisreyes/Sonify/SonifyOCT/data/data_sample_injection/Bscans-dt/A/Image/00067.png"
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = gaussian_filter(img, sigma=5)
    peaks = find_needle_peaks(img)
    # # Visualize
    # import matplotlib.pyplot as plt
    # plt.imshow(img, cmap='gray')
    # plt.scatter(range(len(peaks)), peaks, c='r', s=1)
    # plt.show()

mapping_segs = {
    0: 0,
    4: 0,
    1: 2,
    6: 2,
    3: 3,
    2: 1,
    5: 0
}

def remap_segs(seg):
    seg_copy = np.zeros_like(seg)
    for k, v in mapping_segs.items():
        seg_copy[seg == k] = v
    return seg_copy
    