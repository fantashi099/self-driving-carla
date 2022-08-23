import torch
import argparse

def onxx_converted(model_path):
    model = torch.load(model_path).to("cpu")
    model.eval()

    dummy_input = torch.randn(1,3,512,1024)
    torch.onnx.export(
        model, 
        (dummy_input, ), 
        "lane_model.onnx", 
        opset_version=11,
        do_constant_folding=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Model Pytorch to ONNX')
    parser.add_argument("--path", default = "./lane_detection/Deeplabv3+(MobilenetV2).pth", help="Choose input model path")
    args = parser.parse_args()

    onxx_converted(args.path)

# OpenVINO Converted ------------------------------------------
# mo --input_model lane_model.onnx --input_shape [1,3,512,1024] \
#  --data_type FP16 --scale_values [58.395,57.12,57.375]  \ 
# --mean_values [123.675,116.28,103.53] --output_dir ./converted_model