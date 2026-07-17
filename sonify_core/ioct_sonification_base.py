import copy
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from extrapolate import (
    ILM_LABEL,
    RPE_LABEL,
    calculate_target_thickness,
    enforce_anatomical_consistency,
    ensure_anatomical_representation,
    extract_line,
    extrapolate_rpe_and_ilm,
    update_fitted_line,
)
from mapping import map_physical_params
from settings_sonify import RANGE_PARAMS_ALL
from segmentation.segment_bscan import extrapolate_retina, thicken_rpe_down
from sonification_init import add_ILM_RPE_drivers, sound_model_config_init
from utils.util import (
    adjust_roi,
    compute_and_rotate_patch_centers,
    compute_roi_from_line_and_seg,
    find_closest_node_to_tip,
    get_needle_tip_pos_from_seg,
    postprocess_patch_center_classes,
)

try:
    from sklearn.linear_model import HuberRegressor
except ImportError:  # pragma: no cover - optional dependency in some environments
    HuberRegressor = None


@dataclass
class OfflineInitialData:
    frame_files: list
    seg_files_path: str
    seg_files: list
    frame_0: np.ndarray
    seg_img_0: np.ndarray
    original_shape: tuple
    scale_x: float
    scale_y: float
    add_margin_to_roi: int | None
    initial_confidence: float


@dataclass
class InitialGeometry:
    frame_0: np.ndarray
    seg_img_0: np.ndarray
    ilm_line: np.ndarray
    rpe_line: np.ndarray
    baseline_thickness: np.ndarray
    frame_0_rotated: np.ndarray
    seg_0_rotated: np.ndarray
    needle_tip_pos: tuple
    line_points: list
    roi: tuple | None
    rotated_line_points: np.ndarray | None


@dataclass
class SoundModelSetup:
    roi: tuple | None
    frame_roi: np.ndarray | None
    seg_roi: np.ndarray | None
    rotated_patch_centers: np.ndarray
    patch_centers_class: np.ndarray
    ilm_bound: int | None
    rpe_bound: int | None
    sound_model_config: object
    means: object
    std: object
    range_params: dict
    masses: np.ndarray
    stiffnesses: np.ndarray
    damping: np.ndarray
    node_info: list


def spline_fitting_worker(input_queue, output_queue, stop_event=None, debug=False):
    """
    Background worker for spline-based line fitting.
    """
    while stop_event is None or not stop_event.is_set():
        task = None
        try:
            task = input_queue.get(timeout=1.0)
            if task is None:
                input_queue.task_done()
                break

            frame_index, seg_img_current, state, thickness = task

            fitted_seg, updated_lines, confidences = update_fitted_line(
                seg=seg_img_current,
                state=state,
                extrapolate_distance=100,
                sigma_L=8.0,
                base_alpha=0.8,
                thickness=thickness,
            )
            fitted_seg = extrapolate_retina(fitted_seg, cls_to_use=4)

            result = {
                "frame_index": frame_index,
                "fitted_seg": fitted_seg,
                "updated_lines": updated_lines,
                "confidences": confidences,
                "timestamp": time.time(),
            }

            try:
                output_queue.put(result, block=False)
            except queue.Full:
                if debug:
                    print(f"Background: output queue full, dropping frame {frame_index}")

            input_queue.task_done()
        except queue.Empty:
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            if debug:
                print(f"Spline fitting error: {exc}")
            if task is not None:
                try:
                    input_queue.task_done()
                except ValueError:
                    pass


class BaseIOCTSonification:
    def __init__(
        self,
        *,
        num_nodes_x,
        num_nodes_y,
        extend_roi_to_needle_tip,
        add_margin_to_roi,
        sep_f_components,
        static_mapping_type,
        separate_ilm,
        ranges,
        simulator_path,
        include_retina_in_ilm_rpe_drivers,
        thickness_statistic,
        use_confidence_weights=True,
        target_resolution=None,
        max_needle_position_jump=100.0,
    ):
        self.num_nodes_x = num_nodes_x
        self.num_nodes_y = num_nodes_y
        self.extend_roi_to_needle_tip = extend_roi_to_needle_tip
        self.add_margin_to_roi = add_margin_to_roi
        self.sep_f_components = sep_f_components
        self.static_mapping_type = static_mapping_type
        self.separate_ilm = separate_ilm
        self.ranges = ranges
        self.simulator_path = simulator_path
        self.include_retina_in_ilm_rpe_drivers = include_retina_in_ilm_rpe_drivers
        self.thickness_statistic = thickness_statistic
        self.use_confidence_weights = use_confidence_weights
        self.target_resolution = target_resolution
        self.max_needle_position_jump = max_needle_position_jump

        self.scale_x = 1.0
        self.scale_y = 1.0
        self.state = None
        self.target_thickness = None
        self.prev_thickness = None
        self.prev_ILM_line = None
        self.prev_RPE_line = None
        self.prev_tip_pos = None
        self.prev_needle_pos_for_speed = None
        self.prev_def_amount = 0
        self.M = None
        self.angle = 0.0
        self.ROI = None
        self.rotated_patch_centers = None
        self.patch_centers_class = None
        self.node_info = None
        self.class2_mask = None
        self.scales = [1.0, 1.0, 1.2, 1.2, 1.1]
        self._prev_f_smoothed = (0.0, 0.0)

        self.spline_input_queue = None
        self.spline_output_queue = None
        self.spline_stop_event = None
        self.spline_thread = None

    def log_info(self, message):
        logger = self._get_logger()
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    def log_warn(self, message):
        logger = self._get_logger()
        if logger is not None:
            logger.warn(message)
        else:
            print(message)

    def log_error(self, message):
        logger = self._get_logger()
        if logger is not None:
            logger.error(message)
        else:
            print(message)

    def _get_logger(self):
        get_logger = getattr(self, "get_logger", None)
        if callable(get_logger):
            return get_logger()
        return None

    def reset_tracking_state(self):
        self.prev_tip_pos = None
        self.prev_needle_pos_for_speed = None
        self.prev_def_amount = 0

    def prepare_offline_initial_data(
        self,
        folder,
        add_margin_to_roi,
        *,
        seg_folder_suffix="_segmentation",
        seg_predicate=None,
        fallback_loader=None,
    ):
        frame_files = self.list_frame_files(folder)
        if not frame_files:
            return None

        seg_files_path, seg_files = self.list_segmentation_files(
            folder,
            seg_folder_suffix=seg_folder_suffix,
            predicate=seg_predicate,
        )

        frame_0_path = os.path.join(folder, frame_files[0])
        frame_0 = cv2.imread(frame_0_path)
        if frame_0 is None:
            return None

        original_shape = frame_0.shape[:2]
        frame_0, scale_x, scale_y, adjusted_margin = self.normalize_frame(frame_0, add_margin_to_roi)

        seg_0_path = os.path.join(seg_files_path, seg_files[0]) if seg_files else None
        seg_img_0, initial_confidence = self.load_segmentation_from_source(
            seg_0_path,
            fallback_loader=(
                None if fallback_loader is None else lambda: fallback_loader(frame_0_path, frame_0)
            ),
        )

        return OfflineInitialData(
            frame_files=frame_files,
            seg_files_path=seg_files_path,
            seg_files=seg_files,
            frame_0=frame_0,
            seg_img_0=seg_img_0,
            original_shape=original_shape,
            scale_x=scale_x,
            scale_y=scale_y,
            add_margin_to_roi=adjusted_margin,
            initial_confidence=initial_confidence,
        )

    @staticmethod
    def list_frame_files(folder, suffixes=(".jpg", ".png")):
        return sorted([f for f in os.listdir(folder) if f.endswith(suffixes)])

    @staticmethod
    def list_segmentation_files(folder, seg_folder_suffix="_segmentation", predicate=None):
        seg_files_path = os.path.join(os.path.dirname(folder), os.path.basename(folder) + seg_folder_suffix)
        seg_files = []
        if os.path.exists(seg_files_path):
            candidates = os.listdir(seg_files_path)
            if predicate is not None:
                seg_files = sorted([f for f in candidates if predicate(f)])
            else:
                seg_files = sorted([f for f in candidates if f.endswith((".png", ".jpg"))])
        return seg_files_path, seg_files

    def load_segmentation_from_source(self, seg_file=None, *, fallback_loader=None, default_confidence=1.0):
        seg_img = None
        confidence = default_confidence

        if seg_file and os.path.exists(seg_file):
            seg_img = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
        elif fallback_loader is not None:
            result = fallback_loader()
            if isinstance(result, tuple):
                seg_img, confidence = result
            else:
                seg_img = result

        if isinstance(seg_img, np.ndarray):
            seg_img = self.normalize_segmentation(seg_img)

        return seg_img, confidence

    def load_frame_at_index(self, folder, frame_files, index):
        frame_file = os.path.join(folder, frame_files[index])
        frame = cv2.imread(frame_file)
        if frame is None:
            return frame_file, None

        if self.target_resolution is not None and (abs(self.scale_x - 1.0) > 0.05 or abs(self.scale_y - 1.0) > 0.05):
            frame = cv2.resize(frame, self.target_resolution)

        return frame_file, frame

    def load_runtime_segmentation(self, seg_file=None, *, fallback_loader=None, cls_to_use=None):
        seg_img, confidence = self.load_segmentation_from_source(
            seg_file,
            fallback_loader=fallback_loader,
        )
        if seg_img is not None and cls_to_use is not None:
            seg_img = extrapolate_retina(seg_img, cls_to_use=cls_to_use)
        return seg_img, confidence

    def normalize_frame(self, frame, add_margin_to_roi=None):
        self.scale_x = 1.0
        self.scale_y = 1.0
        resized_frame = frame
        adjusted_margin = add_margin_to_roi

        if self.target_resolution is None:
            return resized_frame, self.scale_x, self.scale_y, adjusted_margin

        target_width, target_height = self.target_resolution
        self.scale_x = target_width / frame.shape[1]
        self.scale_y = target_height / frame.shape[0]

        if abs(self.scale_x - 1.0) > 0.05 or abs(self.scale_y - 1.0) > 0.05:
            resized_frame = cv2.resize(frame, (target_width, target_height))
            if adjusted_margin is not None:
                adjusted_margin = int(adjusted_margin * min(self.scale_x, self.scale_y))

        return resized_frame, self.scale_x, self.scale_y, adjusted_margin

    def normalize_segmentation(self, seg_img):
        if seg_img is None:
            return None

        seg_img = seg_img.copy()
        if self.target_resolution is None:
            return seg_img

        if abs(self.scale_x - 1.0) > 0.05 or abs(self.scale_y - 1.0) > 0.05:
            target_width, target_height = self.target_resolution
            return cv2.resize(seg_img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        return seg_img

    def prepare_initial_segmentation(self, seg_img, *, rpe_thickness_pixels):
        seg_img = seg_img.copy()
        seg_img[seg_img == 4] = 0
        needle_mask = seg_img == 1

        seg_img = extrapolate_rpe_and_ilm(seg_img, extrapolate_distance=100)
        ilm_line = extract_line(seg_img, ILM_LABEL)
        rpe_line = extract_line(seg_img, RPE_LABEL)

        self.state = {
            "prior": {
                "ILM": ilm_line.copy(),
                "RPE": rpe_line.copy(),
            },
            "reference": {
                "ILM": ilm_line.copy(),
                "RPE": rpe_line.copy(),
            },
        }
        self.target_thickness = calculate_target_thickness(seg_img, ilm_line)

        baseline_thickness = rpe_line - ilm_line
        self.prev_thickness = baseline_thickness.copy()
        self.prev_ILM_line = ilm_line.copy()
        self.prev_RPE_line = rpe_line.copy()

        seg_img = update_fitted_line(
            seg=seg_img,
            state=self.state,
            extrapolate_distance=100,
            sigma_L=8.0,
            base_alpha=0.8,
            thickness=self.target_thickness,
        )[0]
        seg_img = extrapolate_retina(seg_img, cls_to_use=(4 if self.separate_ilm else 2))
        seg_img[needle_mask] = 1
        seg_img = thicken_rpe_down(seg_img, no_pixels=rpe_thickness_pixels, cls=2)
        seg_img = thicken_rpe_down(seg_img, no_pixels=rpe_thickness_pixels, cls=3)

        return seg_img, ilm_line, rpe_line, baseline_thickness

    def prepare_initial_geometry(
        self,
        frame_0,
        seg_img_0,
        *,
        rpe_thickness_pixels,
        injection_mode,
        use_huber_regressor=False,
        needle_tip_pos=None,
        to_left=True,
        margin=None,
        extend_tip=None,
    ):
        seg_img_0, ilm_line, rpe_line, baseline_thickness = self.prepare_initial_segmentation(
            seg_img_0,
            rpe_thickness_pixels=rpe_thickness_pixels,
        )
        frame_0_rotated, seg_0_rotated, _, _ = self.compute_rotation(
            frame_0,
            seg_img_0,
            use_huber_regressor=use_huber_regressor,
        )

        if needle_tip_pos is None:
            needle_tip_pos = get_needle_tip_pos_from_seg(seg_img_0 == 1)

        line_points = self.build_line_points(
            seg_img_0,
            frame_0.shape,
            needle_tip_pos,
            injection_mode=injection_mode,
        )
        roi, rotated_line_points = self.compute_roi(
            seg_img_0,
            frame_0.shape,
            line_points,
            needle_tip_pos,
            to_left=to_left,
            margin=margin,
            extend_tip=extend_tip,
        )

        return InitialGeometry(
            frame_0=frame_0,
            seg_img_0=seg_img_0,
            ilm_line=ilm_line,
            rpe_line=rpe_line,
            baseline_thickness=baseline_thickness,
            frame_0_rotated=frame_0_rotated,
            seg_0_rotated=seg_0_rotated,
            needle_tip_pos=needle_tip_pos,
            line_points=line_points,
            roi=roi,
            rotated_line_points=rotated_line_points,
        )

    def compute_rotation(self, frame, seg_img, *, use_huber_regressor=False):
        ys_ilm, xs_ilm = np.where(seg_img == 3)
        if len(xs_ilm) < 2:
            self.M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            self.angle = 0.0
            return (
                frame.copy(),
                seg_img.copy(),
                self.M,
                self.angle,
            )

        if use_huber_regressor and HuberRegressor is not None and len(xs_ilm) > 2:
            huber = HuberRegressor(epsilon=1.35, max_iter=100)
            huber.fit(xs_ilm.reshape(-1, 1), ys_ilm)
            slope = huber.coef_[0]
        else:
            slope = np.polyfit(xs_ilm, ys_ilm, 1)[0]

        tilt = np.degrees(np.arctan(slope))
        self.M = cv2.getRotationMatrix2D(
            (frame.shape[1] // 2, frame.shape[0] // 2),
            tilt,
            1.0,
        )
        self.angle = -tilt

        frame_rotated = cv2.warpAffine(frame, self.M, (frame.shape[1], frame.shape[0]))
        seg_rotated = cv2.warpAffine(
            seg_img,
            self.M,
            (seg_img.shape[1], seg_img.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        return frame_rotated, seg_rotated, self.M, self.angle

    def build_line_points(self, seg_img, frame_shape, needle_tip_pos, *, injection_mode):
        ys, xs = np.where(seg_img == 1)
        if ys.size == 0 or xs.size == 0:
            return []

        if injection_mode:
            if needle_tip_pos[0] is None:
                return []
            return [(needle_tip_pos[0], y) for y in range(frame_shape[0])]

        if len(xs) < 2:
            return []

        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        line_points = []
        for x in range(frame_shape[1]):
            y = int(p(x))
            if 0 <= y < frame_shape[0]:
                line_points.append((x, y))
        return line_points

    def compute_roi(
        self,
        seg_img,
        frame_shape,
        line_points,
        needle_tip_pos,
        *,
        to_left=True,
        margin=None,
        extend_tip=None,
    ):
        if not line_points:
            return None, None

        clinical_seg = seg_img.copy()
        clinical_seg[seg_img == 1] = 0

        roi, rotated_line_points = compute_roi_from_line_and_seg(
            line_points,
            self.M,
            clinical_seg,
            seg_thresh=0,
            to_left=to_left,
        )
        roi = adjust_roi(
            roi,
            frame_shape,
            self.num_nodes_x,
            self.num_nodes_y,
            needle_tip=needle_tip_pos,
            extend_tip=self.extend_roi_to_needle_tip if extend_tip is None else extend_tip,
            margin=self.add_margin_to_roi if margin is None else margin,
        )
        self.ROI = roi
        return roi, rotated_line_points

    @staticmethod
    def roi_contains_layers(seg_rotated, roi, *, layers=(2, 3)):
        if roi is None:
            return False, None, {}

        x0, y0, side_x, side_y = roi
        seg_roi = seg_rotated[y0:y0 + side_y, x0:x0 + side_x]
        presence = {layer: np.any(seg_roi == layer) for layer in layers}
        return all(presence.values()), seg_roi, presence

    @staticmethod
    def weighted_patch_classification(patch):
        if patch.size == 0:
            return 0

        unique_classes, counts = np.unique(patch, return_counts=True)
        class_weights = {
            0: 1.0,
            1: 1.0,
            2: 4.0,
            3: 4.0,
            4: 1.0,
        }
        weighted_scores = {
            cls: count * class_weights.get(cls, 1.0)
            for cls, count in zip(unique_classes, counts)
        }

        total_pixels = patch.size
        bg_pixels = np.sum(patch == 0)
        non_bg_ratio = (total_pixels - bg_pixels) / total_pixels
        if non_bg_ratio > 0.25:
            anatomical_scores = {cls: score for cls, score in weighted_scores.items() if cls != 0}
            if anatomical_scores:
                return max(anatomical_scores, key=anatomical_scores.get)

        return max(weighted_scores, key=weighted_scores.get) if weighted_scores else 0

    def compute_patch_center_classes(
        self,
        seg_img,
        rotated_patch_centers,
        side_x,
        side_y,
        *,
        mode="weighted",
        enforce_consistency=False,
    ):
        patch_centers_class = np.zeros((rotated_patch_centers.shape[0],), dtype=np.uint8)
        patch_size_x = max(1, side_x // self.num_nodes_x)
        patch_size_y = max(1, side_y // self.num_nodes_y)

        for i, center in enumerate(rotated_patch_centers):
            cx, cy = int(center[0]), int(center[1])
            x0 = max(0, cx - patch_size_x // 2)
            x1 = min(seg_img.shape[1], cx + patch_size_x // 2)
            y0 = max(0, cy - patch_size_y // 2)
            y1 = min(seg_img.shape[0], cy + patch_size_y // 2)
            patch = seg_img[y0:y1, x0:x1]

            if mode == "median":
                patch_centers_class[i] = np.median(patch) if patch.size else 0
            elif mode == "majority":
                if patch.size == 0:
                    patch_centers_class[i] = 0
                else:
                    unique_classes, counts = np.unique(patch, return_counts=True)
                    patch_centers_class[i] = unique_classes[np.argmax(counts)]
            else:
                patch_centers_class[i] = self.weighted_patch_classification(patch)

        patch_centers_class = ensure_anatomical_representation(
            patch_centers_class,
            seg_img,
            rotated_patch_centers,
            self.num_nodes_x,
            self.num_nodes_y,
            side_x,
            side_y,
        )
        if enforce_consistency:
            patch_centers_class = enforce_anatomical_consistency(
                patch_centers_class,
                seg_img,
                rotated_patch_centers,
                self.num_nodes_x,
                self.num_nodes_y,
                self.separate_ilm,
            )
        patch_centers_class = postprocess_patch_center_classes(
            patch_centers_class,
            self.num_nodes_x,
            self.num_nodes_y,
        )

        self.patch_centers_class = patch_centers_class
        self.class2_mask = patch_centers_class == 2
        return patch_centers_class

    def find_layer_bounds(self, patch_centers_class):
        ilm_bound = None
        rpe_bound = None
        patch_centers_reshaped = patch_centers_class.reshape((self.num_nodes_y, self.num_nodes_x))
        middle_x = self.num_nodes_x // 2
        for i in range(self.num_nodes_y):
            cls = patch_centers_reshaped[i, middle_x]
            if cls == 2 and ilm_bound is None:
                ilm_bound = i
            if cls == 3 and rpe_bound is None:
                rpe_bound = i
        return ilm_bound, rpe_bound

    def build_node_info(self, rotated_patch_centers, ilm_line, rpe_line, *, eps=2.0):
        node_info = [[None for _ in range(self.num_nodes_x)] for _ in range(self.num_nodes_y)]
        pos = rotated_patch_centers.reshape((self.num_nodes_y, self.num_nodes_x, 2))

        for i in range(self.num_nodes_y):
            for j in range(self.num_nodes_x):
                x = int(pos[i, j, 0])
                y = pos[i, j, 1]

                if x < 0 or x >= len(ilm_line):
                    node_info[i][j] = {"type": "inactive"}
                    continue
                if np.isnan(ilm_line[x]) or np.isnan(rpe_line[x]):
                    node_info[i][j] = {"type": "inactive"}
                    continue

                ilm_y = ilm_line[x]
                rpe_y = rpe_line[x]
                thickness = rpe_y - ilm_y
                if thickness < 5:
                    node_info[i][j] = {"type": "inactive"}
                    continue

                if y < ilm_y - eps:
                    node_info[i][j] = {"type": "vitreous", "delta": y - ilm_y}
                elif abs(y - ilm_y) <= eps:
                    node_info[i][j] = {"type": "ilm"}
                elif ilm_y + eps < y < rpe_y - eps:
                    rho = (y - ilm_y) / thickness
                    node_info[i][j] = {"type": "retina", "rho": rho}
                elif abs(y - rpe_y) <= eps:
                    node_info[i][j] = {"type": "rpe"}
                else:
                    node_info[i][j] = {"type": "subrpe", "delta": y - rpe_y}

        self.node_info = node_info
        return node_info

    def initialize_sound_model_setup(
        self,
        geometry,
        *,
        classification_mode,
        enforce_consistency,
        debug=False,
        range_params_all=RANGE_PARAMS_ALL,
    ):
        roi = geometry.roi
        if roi is None:
            raise ValueError("ROI must be defined before sound model initialization")

        x0, y0, side_x, side_y = roi
        frame_roi = geometry.frame_0_rotated[y0:y0 + side_y, x0:x0 + side_x]
        seg_roi = geometry.seg_0_rotated[y0:y0 + side_y, x0:x0 + side_x]

        _, rotated_patch_centers = compute_and_rotate_patch_centers(
            roi,
            self.num_nodes_x,
            self.num_nodes_y,
            geometry.frame_0.shape,
            self.angle,
        )
        patch_centers_class = self.compute_patch_center_classes(
            geometry.seg_img_0,
            rotated_patch_centers,
            side_x,
            side_y,
            mode=classification_mode,
            enforce_consistency=enforce_consistency,
        )
        ilm_bound, rpe_bound = self.find_layer_bounds(patch_centers_class)

        sound_model_config, means, std = sound_model_config_init(
            roi,
            frame_roi,
            seg_roi,
            self.num_nodes_x,
            self.num_nodes_y,
            classes=patch_centers_class,
        )
        add_ILM_RPE_drivers(
            ilm_bound,
            rpe_bound,
            self.num_nodes_x,
            self.num_nodes_y,
            self.include_retina_in_ilm_rpe_drivers,
        )
        range_params, _ = self.swap_range_layers_and_compute_scales(range_params_all)
        masses, stiffnesses, damping = map_physical_params(
            static_mapping_type=self.static_mapping_type,
            seg_img_0=geometry.seg_img_0,
            rotated_patch_centers=rotated_patch_centers,
            patch_centers_class=patch_centers_class,
            num_nodes_x=self.num_nodes_x,
            num_nodes_y=self.num_nodes_y,
            RANGE_PARAMS_ALL=range_params,
            means=means,
            std=std,
            ROI=roi,
            frame_0_rotated=geometry.frame_0_rotated,
            debug=debug,
        )
        node_info = self.build_node_info(
            rotated_patch_centers,
            geometry.ilm_line,
            geometry.rpe_line,
        )

        self.rotated_patch_centers = rotated_patch_centers
        self.patch_centers_class = patch_centers_class

        return SoundModelSetup(
            roi=roi,
            frame_roi=frame_roi,
            seg_roi=seg_roi,
            rotated_patch_centers=rotated_patch_centers,
            patch_centers_class=patch_centers_class,
            ilm_bound=ilm_bound,
            rpe_bound=rpe_bound,
            sound_model_config=sound_model_config,
            means=means,
            std=std,
            range_params=range_params,
            masses=masses,
            stiffnesses=stiffnesses,
            damping=damping,
            node_info=node_info,
        )

    def swap_range_layers_and_compute_scales(self, range_params_all):
        range_params = copy.deepcopy(range_params_all[self.ranges])
        range_params[3], range_params[4] = range_params[4], range_params[3]
        self.scales = np.clip(
            [np.array(range_params[el][1]).mean() / 4 for el in range(5)],
            0.5,
            1.5,
        )
        return range_params, self.scales

    def start_spline_worker(self, debug=False):
        self.spline_input_queue = queue.Queue(maxsize=3)
        self.spline_output_queue = queue.Queue(maxsize=10)
        self.spline_stop_event = threading.Event()
        self.spline_thread = threading.Thread(
            target=spline_fitting_worker,
            args=(self.spline_input_queue, self.spline_output_queue, self.spline_stop_event, debug),
            daemon=True,
        )
        self.spline_thread.start()
        return self.spline_thread

    def queue_spline_task(self, frame_index, seg_img_current):
        if self.spline_input_queue is None:
            return False
        try:
            task = (frame_index, seg_img_current.copy(), self.state, self.target_thickness)
            self.spline_input_queue.put(task, block=False)
            return True
        except queue.Full:
            return False

    def collect_recent_spline_results(self, frame_index, *, max_age=10):
        results = []
        if self.spline_output_queue is None:
            return results

        while True:
            try:
                result = self.spline_output_queue.get(block=False)
            except queue.Empty:
                break

            if frame_index - result["frame_index"] <= max_age:
                results.append(result)
            self.spline_output_queue.task_done()

        return results

    def clear_spline_queues(self):
        for work_queue in (self.spline_input_queue, self.spline_output_queue):
            if work_queue is None:
                continue
            while not work_queue.empty():
                try:
                    work_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    work_queue.task_done()
                except ValueError:
                    pass

    def stop_spline_worker(self, timeout=2.0):
        if self.spline_stop_event is not None:
            self.spline_stop_event.set()
        if self.spline_input_queue is not None:
            try:
                self.spline_input_queue.put_nowait(None)
            except queue.Full:
                pass
        if self.spline_thread is not None and self.spline_thread.is_alive():
            self.spline_thread.join(timeout=timeout)

    def start_simulator(self, *, suppress_output=False, new_process_group=False):
        kwargs = {}
        if suppress_output:
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
        if new_process_group:
            kwargs["preexec_fn"] = os.setsid
        return subprocess.Popen([self.simulator_path], **kwargs)

    @staticmethod
    def show_startup_preview(frame, *, wait_ms=0, window_name="Press any key to start sonification"):
        cv2.imshow(window_name, frame)
        cv2.waitKey(wait_ms)
        cv2.destroyAllWindows()

    def filter_tracked_tip(self, tracked_tip, *, max_jump=None):
        if tracked_tip is None:
            return None

        new_tip_pos = (tracked_tip[0], tracked_tip[1])
        if self.prev_tip_pos is not None and max_jump is not None:
            dx = new_tip_pos[0] - self.prev_tip_pos[0]
            dy = new_tip_pos[1] - self.prev_tip_pos[1]
            distance_change = np.sqrt(dx * dx + dy * dy)
            if distance_change > max_jump:
                return self.prev_tip_pos

        self.prev_tip_pos = new_tip_pos
        return new_tip_pos

    def find_closest_node_and_class(self, needle_tip_pos, rotated_patch_centers, patch_centers_class):
        closest_node_pos = None
        closest_node_index = None
        dist = 0
        current_class = 0

        if needle_tip_pos[0] is None or needle_tip_pos[1] is None:
            return closest_node_pos, closest_node_index, dist, current_class

        closest_node_result = find_closest_node_to_tip(
            needle_tip_pos,
            rotated_patch_centers,
            self.num_nodes_x,
            self.num_nodes_y,
        )
        if closest_node_result is None:
            return closest_node_pos, closest_node_index, dist, current_class

        closest_node_pos, closest_node_index, dist = closest_node_result
        if needle_tip_pos[1] > closest_node_pos[1]:
            dist = -dist

        middle_x = self.num_nodes_x // 2
        closest_node_index = (closest_node_index[0], middle_x)
        current_class = patch_centers_class[closest_node_index[0] * self.num_nodes_x + closest_node_index[1]]
        return closest_node_pos, closest_node_index, dist, current_class

    def snap_to_nearest_class_row(self, needle_tip_pos, rotated_patch_centers, class_mask):
        if needle_tip_pos[0] is None or needle_tip_pos[1] is None or not np.any(class_mask):
            return None, None, None

        class_indices = np.where(class_mask)[0]
        class_positions = rotated_patch_centers[class_indices]
        tip_arr = np.array([needle_tip_pos[0], needle_tip_pos[1]])
        dists = np.linalg.norm(class_positions - tip_arr, axis=1)
        best = np.argmin(dists)
        flat_idx = class_indices[best]
        row = flat_idx // self.num_nodes_x
        middle_x = self.num_nodes_x // 2
        closest_node_index = (row, middle_x)
        closest_node_pos = rotated_patch_centers[row * self.num_nodes_x + middle_x]
        dist = dists[best]
        if needle_tip_pos[1] > closest_node_pos[1]:
            dist = -dist
        return closest_node_pos, closest_node_index, dist

    def apply_spline_result_to_patch_centers(self, result, rotated_patch_centers, node_info, current_class, *, gamma):
        ilm_line = result["updated_lines"]["ILM"].copy()
        rpe_line = result["updated_lines"]["RPE"].copy()

        confidence = np.mean(list(result["confidences"].values()))
        if current_class == 3:
            confidence = result["confidences"]["RPE"]
        else:
            confidence = result["confidences"]["ILM"]
        if np.isnan(confidence):
            confidence = 0.5

        pos = rotated_patch_centers.reshape((self.num_nodes_y, self.num_nodes_x, 2))
        for i in range(self.num_nodes_y):
            for j in range(self.num_nodes_x):
                x = int(pos[i, j, 0])
                if x < 0 or x >= len(ilm_line):
                    continue
                if np.isnan(ilm_line[x]) or np.isnan(rpe_line[x]):
                    continue

                node = node_info[i][j]
                if node is None or node.get("type") == "inactive":
                    continue

                y_current = pos[i, j, 1]
                node_type = node["type"]
                if node_type == "retina":
                    y_target = ilm_line[x] + node["rho"] * (rpe_line[x] - ilm_line[x])
                    y_min = ilm_line[x]
                    y_max = rpe_line[x]
                elif node_type == "vitreous":
                    y_target = ilm_line[x] + node["delta"]
                    y_min = -np.inf
                    y_max = ilm_line[x]
                elif node_type == "subrpe":
                    y_target = rpe_line[x] + node["delta"]
                    y_min = rpe_line[x]
                    y_max = np.inf
                elif node_type == "ilm":
                    y_target = ilm_line[x]
                    y_min = ilm_line[x]
                    y_max = ilm_line[x]
                elif node_type == "rpe":
                    y_target = rpe_line[x]
                    y_min = rpe_line[x]
                    y_max = rpe_line[x]
                else:
                    continue

                y_target = np.clip(y_target, y_min, y_max)
                pos[i, j, 1] = (1 - gamma) * y_current + gamma * y_target

        return pos.reshape((-1, 2)), ilm_line, rpe_line, confidence

    def smooth_force_values(self, new_f_ilm, new_f_rpe, *, alpha=0.2, decay_rate=0.02):
        current_smooth_ilm, current_smooth_rpe = self._prev_f_smoothed
        smoothed_ilm = alpha * new_f_ilm + (1 - alpha) * current_smooth_ilm
        smoothed_rpe = alpha * new_f_rpe + (1 - alpha) * current_smooth_rpe

        if abs(new_f_ilm) < 0.01:
            smoothed_ilm *= 1 - decay_rate
        if abs(new_f_rpe) < 0.01:
            smoothed_rpe *= 1 - decay_rate

        self._prev_f_smoothed = (smoothed_ilm, smoothed_rpe)
        return smoothed_ilm, smoothed_rpe

    @staticmethod
    def calculate_ramp_intensity(*, is_active, ramp_counter, sonification_rate, ramp_duration, min_intensity=0.15, max_intensity=1.0):
        if not is_active:
            return 0.0

        time_elapsed = ramp_counter / sonification_rate
        if time_elapsed >= ramp_duration:
            return max_intensity

        progress = time_elapsed / ramp_duration
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        return min_intensity + smooth_progress * (max_intensity - min_intensity)

    @staticmethod
    def update_ramp_state(current_active, ramp_counter, requested_active):
        if requested_active and not current_active:
            return True, 0
        if requested_active and current_active:
            return True, ramp_counter + 1
        if not requested_active and current_active:
            return False, 0
        return False, ramp_counter

    @staticmethod
    def create_class_change_record(frame_index, prev_class, current_class, needle_tip_pos, start_time):
        elapsed_seconds = time.time() - start_time
        return {
            "frame": frame_index,
            "time_seconds": elapsed_seconds,
            "from_class": int(prev_class) if prev_class is not None else None,
            "to_class": int(current_class),
            "needle_position": needle_tip_pos,
        }

    @staticmethod
    def buffer_video_frame(video_frames_buffer, frame, video_start_time, *, audio_started, start_audio_callback):
        current_time = time.time()
        if video_start_time is None:
            video_start_time = current_time
            if not audio_started:
                start_audio_callback()
                audio_started = True
        frame_timestamp = current_time - video_start_time
        video_frames_buffer.append((frame.copy(), frame_timestamp))
        return video_start_time, audio_started
