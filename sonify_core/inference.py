import onnxruntime as ort
import numpy as np
import cv2
import time

ONNX_PATH = "../model/segmentation.onnx"
session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]

                               )
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


def segment_bscan(img, threshold=0.5):
    probs = get_probabilities(img, session, input_name)
    # perturb probabilities to test confidence-based adjustments
    global_confidence = np.mean(np.max(probs, axis=0))
    # randomly drop it below threshold 10% of the time to simulate low-confidence cases
    if np.random.rand() < 0.1:
        global_confidence = 0.4
    thresh_mask = np.zeros_like(probs)
    thresh_mask[probs > threshold] = 1
    probs = probs * thresh_mask
    segmentation = np.argmax(probs, 0)
    return segmentation, global_confidence

def preprocess_and_segment_bscan(path, threshold=0.9):
    input_tensor, orig_shape = read_bscan_image_as_tensor(path)
    segmentation, global_confidence = segment_bscan(input_tensor, threshold)
    segmentation = cv2.resize(segmentation.astype(np.uint8), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    segmentation = remap_segments(segmentation)
    return segmentation, global_confidence

def extrapolate_retina(segmentation):
    # make segmentation == 2 go down rows until it hits segmentation == 3, without overwriting 3
    # If there is no class 3 below class 2 in the same column, do nothing
    height, width = segmentation.shape
    for col in range(width):
        seg_col = segmentation[:, col]
        for row in range(height):
            if seg_col[row] == 2:
                if not seg_col[row:].unique().contains(3):
                    # No class 3 below, do nothing
                    break
                for k in range(row, height):
                    if seg_col[k] == 3:
                        break
                    seg_col[k] = 2
                break
        segmentation[:, col] = seg_col

    return segmentation

def remap_segments(segmentation):
    remapped = segmentation.copy()
    remapped[segmentation == 0] = 0
    remapped[segmentation == 3] = 1
    remapped[segmentation == 1] = 2
    remapped[segmentation == 2] = 3
    return remapped
