import cv2
import numpy as np
import os

def to_cv_points(pts):
    return np.ascontiguousarray(pts.reshape(-1, 1, 2), dtype=np.float32)

def from_cv_points(pts):
    return pts.reshape(-1, 2)


def extract_needle_keypoints(segmentation, tip_pos, n_shaft_points=20):
    mask = segmentation == 1
    ys, xs = np.where(mask)

    pts = np.column_stack((xs, ys)).astype(np.float32)

    # PCA to get shaft direction
    center = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - center, full_matrices=False)
    direction = Vt[0]

    # project points
    proj = (pts - center) @ direction
    order = np.argsort(proj)

    # sample shaft points uniformly
    shaft_pts = pts[order][::max(1, len(order)//n_shaft_points)]

    # include tip explicitly
    tip_pt = np.array(tip_pos, dtype=np.float32)[None]

    keypoints = np.vstack([tip_pt, shaft_pts])
    return keypoints

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

class NeedleTracker:
    def __init__(
        self,
        init_frame=None,
        tip_pos=(0, 0),
        init_segmentation=None,
        alpha=0.6
    ):
        self.alpha = alpha
        self.tracked_tip = np.array(tip_pos, dtype=float)
        self.prev_tip_pos = self.tracked_tip.copy()

        self.use_lk = init_segmentation is not None

        if self.use_lk:
            # ----- LK tracking initialization -----
            self.prev_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)

            pts = extract_needle_keypoints(init_segmentation, tip_pos)
            self.points = to_cv_points(pts)

            self.lk_params = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS |
                    cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01
                )
            )
        else:
            # ----- segmentation-based tracking -----
            self.points = None
            self.prev_gray = None

    def update(self, frame, segmentation_map=None):
        if self.use_lk:
            return self._update_with_lk(frame)
        else:
            return self._update_with_segmentation(segmentation_map)

    # ------------------------------------------------------------------
    # Segmentation-based tracking (unchanged, simple & deterministic)
    # ------------------------------------------------------------------
    def _update_with_segmentation(self, segmentation_map):
        if segmentation_map is None:
            return self.tracked_tip, 0.0

        needle_seg = segmentation_map == 1
        if not np.any(needle_seg):
            return self.tracked_tip, 0.0

        ys, xs = np.where(needle_seg)
        idx = np.argmax(ys)
        tip = np.array([xs[idx], ys[idx]], dtype=float)

        self.tracked_tip = tip
        self.prev_tip_pos = tip.copy()

        return tip, 1.0

    # ------------------------------------------------------------------
    # Lucas–Kanade keypoint tracking (replaces template matching)
    # ------------------------------------------------------------------
    def _update_with_lk(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.points,
            None,
            **self.lk_params
        )

        if new_pts is None or status is None:
            return self.tracked_tip, 0.0

        status = status.squeeze().astype(bool)
        old_pts = from_cv_points(self.points)
        new_pts = from_cv_points(new_pts)

        good_old = old_pts[status]
        good_new = new_pts[status]

        if len(good_new) < 5:
            # tracking failure
            self.prev_gray = gray
            return self.tracked_tip, 0.0

        # rigid motion estimation for stability
        H, _ = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC
        )

        if H is not None:
            ones = np.ones((old_pts.shape[0], 1))
            pts_h = np.hstack([old_pts, ones])
            updated = (H @ pts_h.T).T
        else:
            updated = new_pts

        self.points = to_cv_points(updated)
        self.prev_gray = gray

        tip = updated[0]
        tip = (
            self.alpha * tip +
            (1 - self.alpha) * self.prev_tip_pos
        )

        self.tracked_tip = tip
        self.prev_tip_pos = tip.copy()

        confidence = len(good_new) / len(old_pts)
        return tip, confidence
