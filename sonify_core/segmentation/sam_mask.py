import torch
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
# from segment_anything import sam_model_registry, SamPredictor
from mobile_sam import sam_model_registry, SamPredictor

import matplotlib.pyplot as plt

def refine_mask_with_sam(
    image,
    initial_mask=None,
    sam_checkpoint="/Users/luisreyes/Sonify/SonifyOCT/checkpoints/sam_vit_l_0b3195.pth",
    model_type="vit_l",
    remove_radius=10,
):
    """
    Iterative SAM refinement with persistent point registry and toggle behavior.
      🟢 Left-click = add/remove foreground point
      🔴 Right-click = add/remove background point
      ↩️ ENTER = refine using SAM
      🔁 R = clear all points
      ✅ Q = finish
    """
    print("🧠 SAM interactive refinement mode (persistent clicks)")
    print("   🟢 Left-click = toggle needle FG   🔴 Right-click = toggle background")
    print("   ↩️ ENTER = refine mask   🔁 R = reset points   ✅ Q = accept")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    refined_mask = initial_mask
    done = False

    # Persistent registry
    points, labels = [], []

    def find_existing_point(x, y):
        """Return index of a nearby point (for toggling), or None if none found."""
        if not points:
            return None
        pts = np.array(points)
        dists = np.sqrt(((pts[:, 0] - x) ** 2) + ((pts[:, 1] - y) ** 2))
        close_idx = np.where(dists < remove_radius)[0]
        return close_idx[0] if len(close_idx) > 0 else None

    while not done:
        fig, ax = plt.subplots()
        ax.imshow(image_rgb)
        if refined_mask is not None:
            ax.imshow(refined_mask, alpha=0.4, cmap="jet")

        # Draw existing points
        for (x, y), lbl in zip(points, labels):
            color = "g" if lbl == 1 else "r"
            ax.plot(x, y, f"{color}o", markersize=6)
        plt.title("Click points — ENTER=refine, R=reset, Q=accept")

        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return
            x, y = event.xdata, event.ydata
            existing = find_existing_point(x, y)
            if existing is not None:
                # Toggle/remove existing point
                removed_label = "FG" if labels[existing] == 1 else "BG"
                print(f"❌ Removed {removed_label} point at ({int(x)}, {int(y)})")
                points.pop(existing)
                labels.pop(existing)
            else:
                if event.button == 1:  # foreground
                    points.append([x, y])
                    labels.append(1)
                    print(f"🟢 Added FG point at ({int(x)}, {int(y)})")
                elif event.button == 3:  # background
                    points.append([x, y])
                    labels.append(0)
                    print(f"🔴 Added BG point at ({int(x)}, {int(y)})")
            ax.clear()
            ax.imshow(image_rgb)
            if refined_mask is not None:
                ax.imshow(refined_mask, alpha=0.4, cmap="jet")
            for (px, py), lbl in zip(points, labels):
                color = "g" if lbl == 1 else "r"
                ax.plot(px, py, f"{color}o", markersize=6)
            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal done, refined_mask, points, labels
            if event.key == "enter":
                plt.close()
                if len(points) == 0:
                    print("⚠️ No points yet.")
                    return
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.array(labels),
                    multimask_output=True,
                )
                best_mask = masks[scores.argmax()]
                refined_mask = (best_mask * 255).astype("uint8")
                print("✅ Mask refined.")
            elif event.key.lower() == "r":
                print("🔁 Cleared all points.")
                points.clear()
                labels.clear()
                plt.close()
            elif event.key.lower() == "q":
                print("✅ Refinement finished.")
                done = True
                plt.close()

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

    return refined_mask


class RetinalLayerSegmenter:
    """
    Automated retinal layer segmentation using SAM with intelligent point sampling.
    
    This class provides automated segmentation by sampling anatomically stable points
    from both the center and edges of retinal structures for robust SAM predictions.
    """
    
    def __init__(self, sam_checkpoint=None, model_type="vit_t", device=None):
        """
        Initialize the retinal layer segmenter.
        
        Args:
            sam_checkpoint: Path to SAM checkpoint. If None, uses default mobile SAM.
            model_type: SAM model type ("vit_t" for mobile, "vit_l" for full SAM)
            device: Device to run SAM on. If None, auto-detects.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if sam_checkpoint is None:
            # Use mobile SAM by default for faster inference
            sam_checkpoint = "/Users/luisreyes/Sonify/SonifyOCT/checkpoints/mobile_sam.pt"
            
        print(f"🧠 Initializing SAM on {self.device}")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        self.current_image = None
        
    def set_image(self, image):
        """Set the input image for segmentation."""
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Grayscale to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        self.current_image = image
        self.predictor.set_image(image_rgb)
        print(f"📸 Image set: {image.shape}")
    
    def sample_anatomical_points(self, binary_mask, n_center=5, n_edge=5, min_distance=10):
        """
        Sample anatomically stable points from a binary mask.
        
        Args:
            binary_mask: Binary mask where 1/255 represents the target region
            n_center: Number of points to sample from the center regions
            n_edge: Number of points to sample from edge regions
            min_distance: Minimum distance between sampled points
            
        Returns:
            points: List of (x, y) coordinates
            labels: List of labels (all 1 for foreground)
        """
        # Normalize mask to 0-1
        mask = (binary_mask > 0).astype(np.uint8)
        
        if mask.sum() == 0:
            print("⚠️ Empty mask provided")
            return [], []
        
        # Find center points using distance transform
        distance_transform = ndimage.distance_transform_edt(mask)
        
        # Ensure we have a valid mask
        mask_pixels = mask > 0
        if not np.any(mask_pixels):
            print("⚠️ No valid mask pixels found")
            return [], []
        
        # Get center points (high distance values)
        mask_values = distance_transform[mask_pixels]
        center_threshold = np.percentile(mask_values, 70)
        center_candidates = np.where((distance_transform >= center_threshold) & mask_pixels)
        
        # Get edge points (low distance values but still inside mask)
        edge_threshold = np.percentile(mask_values, 30)
        edge_candidates = np.where((distance_transform <= edge_threshold) & mask_pixels)
        
        points = []
        labels = []
        
        # Sample center points
        if len(center_candidates[0]) > 0:
            center_points = list(zip(center_candidates[1], center_candidates[0]))  # (x, y)
            sampled_center = self._sample_with_distance_constraint(
                center_points, n_center, min_distance
            )
            points.extend(sampled_center)
            labels.extend([1] * len(sampled_center))
            print(f"🎯 Sampled {len(sampled_center)} center points")
        
        # Sample edge points
        if len(edge_candidates[0]) > 0:
            edge_points = list(zip(edge_candidates[1], edge_candidates[0]))  # (x, y)
            # Exclude points too close to already sampled center points
            if points:
                edge_points = [p for p in edge_points 
                             if min([np.linalg.norm(np.array(p) - np.array(cp)) 
                                   for cp in points]) >= min_distance]
            
            sampled_edge = self._sample_with_distance_constraint(
                edge_points, n_edge, min_distance
            )
            points.extend(sampled_edge)
            labels.extend([1] * len(sampled_edge))
            print(f"🎯 Sampled {len(sampled_edge)} edge points")
        
        print(f"📍 Total sampled points: {len(points)}")
        return points, labels
    
    def _sample_with_distance_constraint(self, candidates, n_points, min_distance):
        """Sample points ensuring minimum distance between them."""
        if len(candidates) == 0:
            return []
        
        if len(candidates) <= n_points:
            return candidates
        
        # Greedy sampling with distance constraint
        sampled = []
        candidates = np.array(candidates)
        
        # Start with random point
        first_idx = np.random.randint(len(candidates))
        sampled.append(candidates[first_idx])
        remaining_indices = list(range(len(candidates)))
        remaining_indices.remove(first_idx)
        
        while len(sampled) < n_points and remaining_indices:
            # Find candidates far enough from all sampled points
            valid_candidates = []
            for idx in remaining_indices:
                candidate = candidates[idx]
                distances = [np.linalg.norm(candidate - sampled_point) 
                           for sampled_point in sampled]
                if min(distances) >= min_distance:
                    valid_candidates.append(idx)
            
            if not valid_candidates:
                # If no candidates meet distance constraint, pick the one with max min distance
                best_idx = max(remaining_indices, 
                             key=lambda idx: min([np.linalg.norm(candidates[idx] - sp) 
                                                 for sp in sampled]))
                valid_candidates = [best_idx]
            
            # Randomly select from valid candidates
            chosen_idx = np.random.choice(valid_candidates)
            sampled.append(candidates[chosen_idx])
            remaining_indices.remove(chosen_idx)
        
        return [tuple(point) for point in sampled]
    
    def segment_layer(self, binary_mask, n_center=5, n_edge=5, min_distance=10, 
                     multimask_output=True, visualize=False):
        """
        Automatically segment a retinal layer using the provided binary mask as guidance.
        
        Args:
            binary_mask: Binary mask indicating the approximate layer region
            n_center: Number of center points to sample
            n_edge: Number of edge points to sample  
            min_distance: Minimum distance between sampled points
            multimask_output: Whether to use multiple mask outputs (recommended)
            visualize: Whether to show the segmentation process
            
        Returns:
            Binary mask (0-255 uint8) of the segmented layer
        """
        if self.current_image is None:
            raise ValueError("No image set. Call set_image() first.")
        
        print(f"🔬 Segmenting retinal layer...")
        
        # Sample anatomically stable points
        points, labels = self.sample_anatomical_points(
            binary_mask, n_center, n_edge, min_distance
        )
        
        if not points:
            print("❌ No valid points sampled")
            return binary_mask
        
        # Predict with SAM
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=multimask_output,
        )
        
        # Select best mask
        if multimask_output:
            best_mask = masks[scores.argmax()]
            print(f"🏆 Best mask score: {scores.max():.3f}")
        else:
            best_mask = masks[0]
        
        # Convert to uint8
        output_mask = (best_mask * 255).astype(np.uint8)
        
        if visualize:
            self._visualize_segmentation(binary_mask, points, labels, output_mask)
        
        print("✅ Layer segmentation complete")
        return output_mask
    
    def _visualize_segmentation(self, input_mask, points, labels, output_mask):
        """Visualize the segmentation process."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with points
        axes[0].imshow(self.current_image, cmap='gray')
        for (x, y), label in zip(points, labels):
            color = 'lime' if label == 1 else 'red'
            axes[0].plot(x, y, 'o', color=color, markersize=8)
        axes[0].set_title('Image + Sampled Points')
        axes[0].axis('off')
        
        # Input mask
        axes[1].imshow(input_mask, cmap='gray')
        axes[1].set_title('Input Binary Mask')
        axes[1].axis('off')
        
        # Output mask
        axes[2].imshow(output_mask, cmap='gray')
        axes[2].set_title('SAM Output Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

