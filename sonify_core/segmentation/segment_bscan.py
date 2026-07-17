import argparse
import onnxruntime as ort
import numpy as np
import os
import cv2


ONNX_PATH = "../model/segmentation.onnx"
IMG_PATH = "image_path.png"

session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name

color_map = np.array([
    [0, 0, 0],        
    [255, 99, 71],    
    [60, 179, 113],   
    [30, 144, 255],    
    [255, 215, 0],
], dtype=np.uint8)


def read_bscan_image_as_tensor(path):
    
    img = cv2.imread(path)
    orig_shape = img.shape
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_tensor = img[np.newaxis, np.newaxis, :, :]
    
    return input_tensor, orig_shape

def get_probabilities(bscan_tensor, onnx_session, input_name):
    
    inputs = {input_name: bscan_tensor}
    outputs = onnx_session.run(None, inputs)
    probs = outputs[0].squeeze()
    
    return probs


def extrapolate_retina_old(segmentation, cls_to_use=2):
    # make segmentation == 2 go down rows until it hits segmentation == 3, without overwriting 3
    # If there is no class 3 below class 2 in the same column, do nothing
    height, width = segmentation.shape
    for col in range(width):
        seg_col = segmentation[:, col]
        for row in range(height):
            if seg_col[row] == 2:
                if not 3 in list(np.unique(seg_col[row:])):
                    # No class 3 below, do nothing
                    break
                for k in range(row, height):
                    if seg_col[k] == 3:
                        break
                    if seg_col[k] != 2:
                        seg_col[k] = cls_to_use
                break
        segmentation[:, col] = seg_col
    return segmentation

def extrapolate_retina(segmentation, cls_to_use=4):
    seg = segmentation.copy()
    H, W = seg.shape
    
    rows = np.arange(H)[:, None]  # shape (H,1)

    # First 2 per column
    is_two = seg == 2
    has_two = is_two.any(axis=0)
    first_two = np.argmax(is_two, axis=0)
    first_two[~has_two] = -1

    # First 3 below the first 2
    first_three = np.full(W, -1)
    for col in np.where(has_two)[0]:
        r2 = first_two[col]
        below = np.where(seg[r2:, col] == 3)[0]
        if len(below) > 0:
            first_three[col] = r2 + below[0]

    valid = first_three > 0

    # Create fill mask (H,W)
    # rows between r2 and r3 AND NOT class 2 or 3
    r2 = first_two[valid]
    r3 = first_three[valid]

    R2 = r2[None, :]  # broadcast to (H, valid_cols)
    R3 = r3[None, :]

    fill_mask = ((rows >= R2) & (rows < R3))

    # Place fill_mask into full image mask
    full_mask = np.zeros_like(seg, dtype=bool)
    valid_cols = np.where(valid)[0]
    full_mask[:, valid_cols] = fill_mask

    # Do NOT overwrite class 2 or 3
    safe_mask = full_mask & (seg != 2) & (seg != 3)

    seg[safe_mask] = cls_to_use
    return seg

def thicken_rpe(segmentation, percentage=0.02):
    # make segmentation == 3 thicker by expanding it up and down by a percentage of its own height
    height, width = segmentation.shape
    new_segmentation = segmentation.copy()
    thickness = int(height * percentage)
    for col in range(width):
        seg_col = segmentation[:, col]
        rpe_indices = np.where(seg_col == 3)[0]
        for idx in rpe_indices:
            start = max(0, idx - thickness)
            end = min(height, idx + thickness + 1)
            new_segmentation[start:end, col] = 3
    return new_segmentation

def thicken_rpe_down(segmentation, no_pixels=20, cls=3):
    # make segmentation == 3 thicker by expanding it up and down by a percentage of its own height
    height, width = segmentation.shape
    new_segmentation = segmentation.copy()
    for col in range(width):
        seg_col = segmentation[:, col]
        rpe_indices = np.where(seg_col == cls)[0]
        for idx in rpe_indices:
            start =  idx 
            end = min(height, idx + no_pixels + 1)
            new_segmentation[start:end, col] = cls
    return new_segmentation


def thicken_retina(segmentation, thickness=10, cls=3):
    # make segmentation == 3 thicker by expanding it up and down by a percentage of its own height
    height, width = segmentation.shape
    new_segmentation = segmentation.copy()
    for col in range(width):
        seg_col = segmentation[:, col]
        rpe_indices = np.where(seg_col == cls)[0]
        for idx in rpe_indices:
            start =  idx 
            end = min(height, idx + thickness + 1)
            new_segmentation[start:end, col] = cls
    return new_segmentation
    
    

def remap_segments(segmentation):
    remapped = segmentation.copy()
    remapped[segmentation == 0] = 0
    remapped[segmentation == 3] = 1
    remapped[segmentation == 1] = 2
    remapped[segmentation == 2] = 3
    return remapped

def segment_bscan(img, threshold=0.5):
    orig_size = img.shape
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_tensor = img[np.newaxis, np.newaxis, :, :]
    probs = get_probabilities(input_tensor, session, input_name)
    segmentation = np.argmax(probs, 0)
    segmentation = cv2.resize(segmentation.astype(np.uint8), (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    return segmentation

def segment_folder(folder_path, output_folder):
    """
    Segment all images in a single folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(folder_path):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            bscan_tensor, orig_shape = read_bscan_image_as_tensor(img_path)
            segmentation = segment_bscan(bscan_tensor)
            segmentation = remap_segments(segmentation)
            segmentation = extrapolate_retina(segmentation)

            segmentation = cv2.resize(segmentation.astype(np.uint8), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            # View the segmentation
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, segmentation)
            print(f"Saved segmentation to: {output_path}")


def segment_dataset_recursive(root_folder, output_root=None, skip_existing=True):
    """
    Recursively segment all subfolders in the OCT Force Dataset
    
    Args:
        root_folder: Path to the root dataset folder (e.g., "/Users/luisreyes/Sonify/SonifyOCT/OCT Force Dataset")
        output_root: Root path for output segmentations. If None, will create alongside input folders with "_seg" suffix
        skip_existing: If True, skip folders that already have segmentations
    """
    
    if output_root is None:
        output_root = root_folder
    
    processed_count = 0
    skipped_count = 0
    
    print(f"🔍 Scanning dataset: {root_folder}")
    print("=" * 60)
    
    # Walk through all directories recursively
    for root, dirs, files in os.walk(root_folder):
        # Check if this directory contains image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if image_files:
            # This is a leaf directory with images - process it
            relative_path = os.path.relpath(root, root_folder)
            
            # Create output path
            if output_root == root_folder:
                # Create segmentation folder alongside the input folder
                output_folder = root + "_seg"
            else:
                # Create mirrored structure in output_root
                output_folder = os.path.join(output_root, relative_path + "_seg")
            
            # Check if already processed
            if skip_existing and os.path.exists(output_folder) and os.listdir(output_folder):
                print(f"⏭️  Skipping (already exists): {relative_path}")
                skipped_count += 1
                continue
            
            print(f"🔄 Processing: {relative_path}")
            print(f"   📁 Input:  {root}")
            print(f"   📁 Output: {output_folder}")
            print(f"   📸 Images: {len(image_files)}")
            
            try:
                segment_folder(root, output_folder)
                processed_count += 1
                print(f"   ✅ Completed successfully!")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                continue
            
            print()
    
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   ✅ Processed: {processed_count} folders")
    print(f"   ⏭️  Skipped: {skipped_count} folders")
    print(f"   🎯 Total: {processed_count + skipped_count} folders found")


def segment_specific_experiments(root_folder, date_patterns=None, experiment_patterns=None, output_root=None):
    """
    Segment specific experiments based on date and experiment patterns
    
    Args:
        root_folder: Path to the root dataset folder
        date_patterns: List of date patterns to include (e.g., ['08_08_23', '08_15_23'])
        experiment_patterns: List of experiment patterns to include (e.g., ['f_i1', 'f_i2'])
        output_root: Root path for output segmentations
    """
    
    if output_root is None:
        output_root = root_folder
    
    processed_count = 0
    
    print(f"🎯 Processing specific experiments...")
    if date_patterns:
        print(f"   📅 Date patterns: {date_patterns}")
    if experiment_patterns:
        print(f"   🧪 Experiment patterns: {experiment_patterns}")
    print("=" * 60)
    
    # Iterate through date folders
    for date_folder in os.listdir(root_folder):
        date_path = os.path.join(root_folder, date_folder)
        
        # Skip if not a directory or doesn't match date pattern
        if not os.path.isdir(date_path):
            continue
        if date_patterns and not any(pattern in date_folder for pattern in date_patterns):
            continue
        
        # Iterate through experiment folders
        for exp_folder in os.listdir(date_path):
            exp_path = os.path.join(date_path, exp_folder)
            
            # Skip if not a directory or doesn't match experiment pattern
            if not os.path.isdir(exp_path):
                continue
            if experiment_patterns and not any(pattern in exp_folder for pattern in experiment_patterns):
                continue
            
            # Check if this folder contains images
            files = os.listdir(exp_path)
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if image_files:
                relative_path = os.path.join(date_folder, exp_folder)
                
                # Create output path
                if output_root == root_folder:
                    output_folder = exp_path + "_seg"
                else:
                    output_folder = os.path.join(output_root, relative_path + "_seg")
                
                print(f"🔄 Processing: {relative_path}")
                print(f"   📁 Input:  {exp_path}")
                print(f"   📁 Output: {output_folder}")
                print(f"   📸 Images: {len(image_files)}")
                
                try:
                    segment_folder(exp_path, output_folder)
                    processed_count += 1
                    print(f"   ✅ Completed successfully!")
                except Exception as e:
                    print(f"   ❌ Error: {str(e)}")
                    continue
                
                print()
    
    print("=" * 60)
    print(f"📊 Summary: Processed {processed_count} experiments")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment B-scan images using ONNX model')
    parser.add_argument('--input_folder', type=str, help='Path to a single folder containing B-scan images')
    parser.add_argument('--dataset_root', type=str, help='Path to root dataset folder for recursive processing')
    parser.add_argument('--output_root', type=str, help='Root path for output segmentations (optional)')
    parser.add_argument('--recursive', action='store_true', help='Process all subfolders recursively')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Skip folders that already have segmentations')
    parser.add_argument('--dates', nargs='*', help='Specific date patterns to process (e.g., 08_08_23 08_15_23)')
    parser.add_argument('--experiments', nargs='*', help='Specific experiment patterns to process (e.g., f_i1 f_i2)')
    
    args = parser.parse_args()
    
    if args.dataset_root and args.recursive:
        # Recursive processing of entire dataset
        print("🚀 Starting recursive segmentation of OCT Force Dataset")
        segment_dataset_recursive(
            root_folder=args.dataset_root,
            output_root=args.output_root,
            skip_existing=args.skip_existing
        )
    elif args.dataset_root and (args.dates or args.experiments):
        # Process specific experiments
        print("🎯 Processing specific experiments")
        segment_specific_experiments(
            root_folder=args.dataset_root,
            date_patterns=args.dates,
            experiment_patterns=args.experiments,
            output_root=args.output_root
        )
    elif args.input_folder:
        # Single folder processing (original functionality)
        print("📁 Processing single folder")
        output_folder = args.input_folder + "_seg"
        segment_folder(args.input_folder, output_folder)
    else:
        print("❌ Error: Please provide either --input_folder for single folder or --dataset_root for recursive processing")
        parser.print_help()
        
    print("🎉 All done!")