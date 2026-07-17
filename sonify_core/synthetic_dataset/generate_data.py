import cv2
import os
import numpy
EXAMPLE_IMAGE_PATH = "/Users/luisreyes/Sonify/SonifyOCT/data/data_sample/06_28_23/b_i3/b_scans000000_1687979370.0099738.jpg"

image = cv2.imread(EXAMPLE_IMAGE_PATH)

target_shape = image.shape
OUT_ROOT="/Users/luisreyes/Sonify/SonifyOCT/data/synthetic"
OUT_PATH = os.path.join(OUT_ROOT, "base_case_data_static")

if os.path.exists(OUT_PATH) is False:
    os.makedirs(OUT_PATH)
    
cv2.imwrite(os.path.join(OUT_PATH, "reference_image.png"), image)

"""Sample 1: Three layers with distinct intensities, white, gray and black"""
three_layers = numpy.zeros_like(image)
segmentation_map = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)

three_layers[0:target_shape[0]//3, :, :] = 255
three_layers[target_shape[0]//3:2*target_shape[0]//3, :, :] = 127
three_layers[2*target_shape[0]//3:, :, :] = 0

segmentation_map[target_shape[0]//3:2*target_shape[0]//3, :] = 2
segmentation_map[2*target_shape[0]//3:, :] = 3 

"""Sample 2: Four layers with distinct intensities, white, light gray, dark gray and black"""
four_layers = numpy.zeros_like(image)
segmentation_map_4 = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)
four_layers[0:target_shape[0]//4, :, :] = 255
four_layers[target_shape[0]//4:2*target_shape[0]//4, :, :] = 191
four_layers[2*target_shape[0]//4:3*target_shape[0]//4, :, :] = 63
four_layers[3*target_shape[0]//4:, :, :] = 0
segmentation_map_4[target_shape[0]//4:2*target_shape[0]//4, :] = 2 
segmentation_map_4[2*target_shape[0]//4:3*target_shape[0]//4, :] = 3
segmentation_map_4[3*target_shape[0]//4:, :] = 4

"""Sample 3: Gradient image from black to white"""
gradient_image = numpy.zeros_like(image)
for i in range(target_shape[0]):
    gradient_image[i, :, :] = i * 255 // target_shape[0]
segmentation_map_gradient = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)
segmentation_map_gradient[:, :] = 0  # Single class for gradient image

"""Sample 4: 3 classes but gradient within each class"""
gradient_classes = numpy.zeros_like(image)
segmentation_map_gradient_classes = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)
for i in range(target_shape[0]):
    if i < target_shape[0] // 3:
        intensity = (i * 255 // (target_shape[0] // 3))
        gradient_classes[i, :, :] = intensity
        segmentation_map_gradient_classes[i, :] = 0
    elif i < 2 * target_shape[0] // 3:
        intensity = ((i - target_shape[0] // 3) * 255 // (target_shape[0] // 3))
        gradient_classes[i, :, :] = intensity
        segmentation_map_gradient_classes[i, :] = 2
    else:
        intensity = ((i - 2 * target_shape[0] // 3) * 255 // (target_shape[0] // 3))
        gradient_classes[i, :, :] = intensity
        segmentation_map_gradient_classes[i, :] = 3

"""Sample 5: Gradient going left to right"""
gradient_lr_image = numpy.zeros_like(image)
segmentation_map_gradient_lr = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)
for j in range(target_shape[1]):
    intensity = j * 255 // target_shape[1]
    gradient_lr_image[:, j, :] = intensity
    segmentation_map_gradient_lr[:, j] = 0  # Single class for gradient image
# Save all samples as follows: folder/sample_image.png and folder/sample_segmentation.png

"""Sample 6: Gradient going left to right with 3 classes"""
gradient_lr_classes = numpy.zeros_like(image)
segmentation_map_gradient_lr_classes = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)
for j in range(target_shape[1]):
    intensity = j * 255 // target_shape[1]
    gradient_lr_classes[:, j, :] = intensity
    if j < target_shape[1] // 3:
        segmentation_map_gradient_lr_classes[:, j] = 0
    elif j < 2 * target_shape[1] // 3:
        segmentation_map_gradient_lr_classes[:, j] = 2
    else:
        segmentation_map_gradient_lr_classes[:, j] = 3

"""Sample 7: 4 layers with distinct intensities, white, light gray, dark gray and black, but single segmentation class"""
four_layers_single_class = numpy.zeros_like(image)
segmentation_map_4_single_class = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)
four_layers_single_class[0:target_shape[0]//4, :, :] = 255
four_layers_single_class[target_shape[0]//4:2*target_shape[0]//4, :, :] = 191
four_layers_single_class[2*target_shape[0]//4:3*target_shape[0]//4, :, :] = 63
four_layers_single_class[3*target_shape[0]//4:, :, :] = 0

""" Sample 8: Sample 7 but flipped horizontally """
four_layers_single_class_flipped = numpy.flip(four_layers_single_class, axis=0)
segmentation_map_4_single_class_flipped = numpy.flip(segmentation_map_4_single_class, axis=0)

""" Sample 9: Four layers, black, white, light gray, dark gray, single segmentation class """
four_layers_bwgg_single_class = numpy.zeros_like(image)
segmentation_map_4_bwgg_single_class = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)
segmentation_map_4_bwgg_diff_classes = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)
four_layers_bwgg_single_class[0:target_shape[0]//4, :, :] = 0
four_layers_bwgg_single_class[target_shape[0]//4:2*target_shape[0]//4, :, :] = 255
four_layers_bwgg_single_class[2*target_shape[0]//4:3*target_shape[0]//4, :, :] = 63
four_layers_bwgg_single_class[3*target_shape[0]//4:, :, :] = 191
segmentation_map_4_bwgg_diff_classes[0:target_shape[0]//4, :] = 0
segmentation_map_4_bwgg_diff_classes[target_shape[0]//4:2*target_shape[0]//4, :] = 2
segmentation_map_4_bwgg_diff_classes[2*target_shape[0]//4:3*target_shape[0]//4, :] = 4
segmentation_map_4_bwgg_diff_classes[3*target_shape[0]//4:, :] = 3

""" Sample 10: Plain black image with single segmentation class """
plain_black_single_class = numpy.zeros_like(image)
segmentation_map_plain_black_single_class = numpy.zeros((target_shape[0], target_shape[1]), dtype=numpy.uint8)

# Example: three_layers/sample_image.png and three_layers/sample_segmentation.png
# Add four_layers_single_class flipped horizontally
samples = [
    (three_layers, segmentation_map, "three_layers"),
    (four_layers, segmentation_map_4, "four_layers"),
    (gradient_image, segmentation_map_gradient, "gradient_image"),
    (gradient_classes, segmentation_map_gradient_classes, "gradient_classes"),
    (gradient_lr_image, segmentation_map_gradient_lr, "gradient_left_right"),
    (gradient_lr_classes, segmentation_map_gradient_lr_classes, "gradient_left_right_classes"),
    (four_layers_single_class, segmentation_map_4_single_class, "four_layers_w_to_b_single_class"),
    (four_layers_single_class_flipped, segmentation_map_4_single_class_flipped, "four_layers_b_to_w_single_class"),
    (four_layers_bwgg_single_class, segmentation_map_4_bwgg_single_class, "four_layers_bwgg_single_class"),
    (plain_black_single_class, segmentation_map_plain_black_single_class, "plain_black_single_class"),
]
for sample_image, sample_segmentation, sample_name in samples:
    sample_dir = os.path.join(OUT_PATH, sample_name)
    seg_dir = sample_dir + "_segmentation"
    if os.path.exists(sample_dir) is False:
        os.makedirs(sample_dir)
    if os.path.exists(seg_dir) is False:
        os.makedirs(seg_dir)
    cv2.imwrite(os.path.join(sample_dir, "sample_image.png"), sample_image)
    cv2.imwrite(os.path.join(seg_dir, "sample_segmentation.png"), sample_segmentation)

""" Now let's make some dynamic sequences by modulating the intensity over time """

num_frames = 100
OUT_PATH_DYNAMIC = os.path.join(OUT_ROOT, "base_case_data_dynamic")
if os.path.exists(OUT_PATH_DYNAMIC) is False:
    os.makedirs(OUT_PATH_DYNAMIC)

"""Dynamic Sample 1: One Layer going from gray to white and back"""
dynamic_one_layer = numpy.zeros((num_frames, target_shape[0], target_shape[1], target_shape[2]), dtype=numpy.uint8)
segmentation_map_dynamic_one_layer = numpy.zeros((num_frames,target_shape[0], target_shape[1]), dtype=numpy.uint8)
for f in range(num_frames):
    intensity = int(127 + 128 * numpy.sin(2 * numpy.pi * f / num_frames))
    dynamic_one_layer[f, :, :, :] = intensity
    segmentation_map_dynamic_one_layer[f, :, :] = 0  # Single class for dynamic image

"""Dynamic Sample 2: Three tissue layers each modulating intensity independently (like tissue deformation)"""
dynamic_three_layers = numpy.zeros((num_frames, target_shape[0], target_shape[1], target_shape[2]), dtype=numpy.uint8)
segmentation_map_dynamic_three_layers = numpy.zeros((num_frames, target_shape[0], target_shape[1]), dtype=numpy.uint8)

# Define layer boundaries (similar to tissue structure)
layer1_end = target_shape[0] // 3      # Top layer (e.g., surface tissue)
layer2_end = 2 * target_shape[0] // 3  # Middle layer (e.g., intermediate tissue)
# Bottom layer goes from layer2_end to end (e.g., deep tissue)

for f in range(num_frames):
    # Each layer modulates with different frequency and phase (simulating independent tissue deformation)
    intensity1 = int(127 + 100 * numpy.sin(2 * numpy.pi * f / num_frames))           # Fast oscillation
    intensity2 = int(127 + 80 * numpy.sin(2 * numpy.pi * f / (num_frames // 2) + numpy.pi/3))  # Medium oscillation, phase shifted
    intensity3 = int(127 + 60 * numpy.sin(2 * numpy.pi * f / (num_frames // 3) + numpy.pi))    # Slow oscillation, phase shifted
    
    # Create three distinct tissue layers with different base intensities and modulation
    dynamic_three_layers[f, :layer1_end, :, :] = intensity1           # Top layer
    dynamic_three_layers[f, layer1_end:layer2_end, :, :] = intensity2  # Middle layer  
    dynamic_three_layers[f, layer2_end:, :, :] = intensity3            # Bottom layer
    
    # Set segmentation classes for each layer
    segmentation_map_dynamic_three_layers[f, :layer1_end, :] = 0        # Top layer class
    segmentation_map_dynamic_three_layers[f, layer1_end:layer2_end, :] = 2  # Middle layer class
    segmentation_map_dynamic_three_layers[f, layer2_end:, :] = 3        # Bottom layer class

# Save dynamic samples as sequences of images
dynamic_samples = [
    (dynamic_one_layer, segmentation_map_dynamic_one_layer, "dynamic_one_layer"),
    (dynamic_three_layers, segmentation_map_dynamic_three_layers, "dynamic_three_layers"),
]

for dynamic_image_seq, dynamic_segmentation_seq, sample_name in dynamic_samples:
    sample_dir = os.path.join(OUT_PATH_DYNAMIC, sample_name)
    seg_dir = sample_dir + "_segmentation"
    if os.path.exists(sample_dir) is False:
        os.makedirs(sample_dir)
    if os.path.exists(seg_dir) is False:
        os.makedirs(seg_dir)
    for f in range(num_frames):
        cv2.imwrite(os.path.join(sample_dir, f"frame_{f:03d}_image.png"), dynamic_image_seq[f, :, :, :])
        cv2.imwrite(os.path.join(seg_dir, f"frame_{f:03d}_segmentation.png"), dynamic_segmentation_seq[f, :, :])
