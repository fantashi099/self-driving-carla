import os
import argparse
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from openvino.runtime import Core

import torch
import segmentation_models_pytorch as smp
import albumentations as albu

def openvino_infer(compiled_model_ir, output_layer_ir, image: np.ndarray) -> np.ndarray:
    """
    OpenVINO Inference
    """

    # Convert the resized images to network input shape
    input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)

    # Run inference on the input image
    model_output = compiled_model_ir([input_image])[output_layer_ir]
    result_mask_ir = np.squeeze(np.argmax(model_output, axis=1)).astype(np.uint8)
    return result_mask_ir


def torch_norm(fn):
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
    transform = [
        albu.Lambda(image=fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(transform)

def torch_infer(model, to_tensor_func, image: np.ndarray) -> np.ndarray:
    """
    Pytorch Inference
    """

    img_tensor = to_tensor_func(image=image)['image']
    x_tensor = torch.from_numpy(img_tensor).to("cpu").unsqueeze(0)

    # Run inference on the input image
    model_output = model.predict(x_tensor).cpu().numpy()
    result_mask_torch = np.squeeze(np.argmax(model_output, axis=1)).astype(np.uint8)
    return result_mask_torch

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred)
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark between OpenVINO model and PyTorch Model')
    parser.add_argument("--path", default = "./images/test.png", help="Choose an input image path")
    parser.add_argument("--groundtruth", default = "./images/groundtruth.png", help="Choose an input groundtruth path")
    parser.add_argument("--folder", default = "", help="Choose folder data path, file format: <image_filename>, see also folder_gt (without display)")
    parser.add_argument("--folder_gt", default = "", help="Choose folder groundtruth path, groundtruth format: <image_filename>_label (without display)")
    args = parser.parse_args()

    image_filename = args.path
    groundtruth_filename = args.groundtruth
    folder_path = args.folder
    folder_gt = args.folder_gt

    # Load the network in Inference Engine
    ie = Core()
    model_ir = ie.read_model(model='./converted_model/lane_model.xml')
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    # Get input and output layers
    input_layer_ir = next(iter(compiled_model_ir.inputs))
    output_layer_ir = next(iter(compiled_model_ir.outputs))

    torch_model = torch.load("./lane_detection/Deeplabv3+(MobilenetV2).pth").to("cpu")
    preprocessing_fn = smp.encoders.get_preprocessing_fn("efficientnet-b0", "imagenet")
    to_tensor_func = torch_norm(preprocessing_fn)


    if folder_path == "":
        image = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)

        groundtruth = cv2.imread(groundtruth_filename, 0)

        start_time = time.time()
        openvino_output = openvino_infer(compiled_model_ir, output_layer_ir, image)
        end_time = time.time()
        print("FPS OpenVINO:", 1/(end_time - start_time))
        print("IoU OpenVINO:", compute_iou(openvino_output, groundtruth))
        print('-'*50)

        start_time = time.time()
        torch_output = torch_infer(torch_model, to_tensor_func, image)
        end_time = time.time()
        print("FPS Pytorch:", 1/(end_time - start_time))
        print("IoU Pytorch:", compute_iou(torch_output, groundtruth))
        print('-'*50)

        fig = plt.figure(figsize=(8,8))
        fig.add_subplot(2, 2, 1)
        plt.imshow(openvino_output)
        plt.title(f"OpenVINO Inference")

        fig.add_subplot(2, 2, 2)
        plt.imshow(torch_output)
        plt.title("PyTorch Inference")

        fig.add_subplot(2, 2, 3)
        plt.imshow(groundtruth)
        plt.title("Groundtruth")
        plt.show()

    else:
        openVINO_iou = 0
        torch_iou = 0

        openVINO_exc_time = 0
        torch_exc_time = 0
        dir_path = os.listdir(folder_path)[:100]

        for fpath in dir_path:
            gt_path = fpath.replace(".png", "_label.png")
            groundtruth_path = os.path.join(folder_gt, gt_path)
            groundtruth = cv2.imread(groundtruth_path, 0)

            file_path = os.path.join(folder_path, fpath)
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            start = time.time()
            openvino_output = openvino_infer(compiled_model_ir, output_layer_ir, image)
            openVINO_exc_time += time.time() - start
            openVINO_iou += compute_iou(openvino_output, groundtruth)

            start = time.time()
            torch_output = torch_infer(torch_model, to_tensor_func, image)
            torch_exc_time += time.time() - start
            torch_iou += compute_iou(torch_output, groundtruth)
        
        openVINO_iou = openVINO_iou / len(dir_path)
        torch_iou = torch_iou / len(dir_path)

        openVINO_exc_time = len(dir_path) / openVINO_exc_time
        torch_exc_time = len(dir_path) / torch_exc_time

        print(f"OpenVINO mean_IoU = {openVINO_iou} --- OpenVINO FPS = {openVINO_exc_time}")
        print(f"Pytorch mean_IoU = {torch_iou} --- Pytorch FPS = {torch_exc_time}")


