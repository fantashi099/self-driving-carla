from .camera_geometry import CameraGeometry
from openvino.runtime import Core
import numpy as np
import cv2

class OpenVINOLaneDetector():
    def __init__(self, cam_geom=CameraGeometry(), model_path='./converted_model/lane_model.xml', device="CPU") -> None:
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        
        ie = Core()
        model_ir = ie.read_model(model_path)
        self.compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    def read_imagefile_to_array(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect_from_file(self, filename):
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)

    def detect(self, img_array):
        img_array = np.expand_dims(np.transpose(img_array, (2, 0, 1)), 0)
        output_layer_ir = next(iter(self.compiled_model_ir.outputs))
        model_output = self.compiled_model_ir([img_array])[output_layer_ir]
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
        return background, left, right

    def detect_and_fit(self, img_array):
        _, left, right = self.detect(img_array)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right

    def fit_poly(self, probs):
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
        return np.poly1d(coeffs)

    def __call__(self, img):
        if isinstance(img, str):
            img = self.read_imagefile_to_array(img)
        return self.detect_and_fit(img)
