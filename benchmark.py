import argparse
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from openvino.runtime import Core

import torch
import segmentation_models_pytorch as smp
import albumentations as albu

# def normalize(image: np.ndarray) -> np.ndarray:
#     """
#     Normalize the image to the given mean and standard deviation
#     for CityScapes models.
#     """
#     image = image.astype(np.float32)
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     image /= 255.0
#     image -= mean
#     image /= std
#     return image

def openvino_infer(image: np.ndarray) -> np.ndarray:
    """
    OpenVINO Inference
    """
    print('OpenVINO Inference')

    # Convert the resized images to network input shape
    input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)

    # Load the network in Inference Engine
    ie = Core()
    model_ir = ie.read_model(model='./converted_model/lane_model.xml')
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    # Get input and output layers
    input_layer_ir = next(iter(compiled_model_ir.inputs))
    output_layer_ir = next(iter(compiled_model_ir.outputs))

    # Run inference on the input image
    start_time = time.time()
    model_output = compiled_model_ir([input_image])[output_layer_ir]
    background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
    # result_mask_ir = np.squeeze(np.argmax(model_output, axis=1)).astype(np.uint8)
    end_time = time.time()
    out_img = left + right
    print("FPS:", 1/(end_time - start_time))
    print('-'*50)
    return out_img


def torch_norm(fn):
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
    transform = [
        albu.Lambda(image=fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(transform)

def torch_infer(image: np.ndarray) -> np.ndarray:
    """
    Pytorch Inference
    """
    print('PyTorch Inference')

    model = torch.load("./lane_detection/Deeplabv3+(MobilenetV2).pth").to("cpu")
    preprocessing_fn = smp.encoders.get_preprocessing_fn("efficientnet-b0", "imagenet")

    to_tensor_func = torch_norm(preprocessing_fn)
    start_time = time.time()
    img_tensor = to_tensor_func(image=resized_image)['image']
    x_tensor = torch.from_numpy(img_tensor).to("cpu").unsqueeze(0)
    model_output = model.predict(x_tensor).cpu().numpy()
    background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
    end_time = time.time()

    out_img = left + right
    print("FPS:", 1/(end_time - start_time))
    print('-'*50)
    return out_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark between OpenVINO model and PyTorch Model')
    parser.add_argument("--path", default = "./images/test.png", help="Choose input image path")
    args = parser.parse_args()

    image_filename = args.path
    image = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (512, 512))

    openvino_output = openvino_infer(resized_image)
    torch_output = torch_infer(resized_image)

    fig = plt.figure(figsize=(8,8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(openvino_output)
    plt.title("OpenVINO Inference")

    fig.add_subplot(1, 2, 2)
    plt.imshow(torch_output)
    plt.title("PyTorch Inference")
    plt.show()
