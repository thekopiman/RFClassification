import os
from pathlib import Path

import sys
import platform
import torch
import numpy as np
import pandas as pd
from models.common import DetectMultiBackend
import glob
import cv2
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)  # Might have to use these functions in the future
from utils.plots import Annotator, colors, save_one_box
import math
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Detection:
    def __init__(
        self,
        yaml_file=None,
        to_slice=True,
        dataset_path=None,  # Directory, not the individual png file
        model_path=None,
    ):
        """
        data_utils and detection.py have to be inside yolov5 folder

        ** Future work: Analyse annotate to obtain various detected information.

        Input: yaml_file, to_slice, dataset_path, model_path
        Output: None
        """
        self.yaml_file = yaml_file
        self.to_slice = to_slice
        self.dataset_path = dataset_path
        self.model_path = model_path

        if yaml_file == None or dataset_path == None or model_path == None:
            raise FileNotFoundError(
                "Don't be a troller and not put the required paths inside"
            )

    def obtain_np_array(
        self,
        path: str = None,
    ) -> np.array:
        """
        Obtain the np.array file

        Input: path
        output: img : np.array
        """
        self.img = cv2.imread(path)  # this is im0

        return self.img

    def split(self, img: np.array = None) -> list:
        """
        Spltis the img (in np.array) into the respective
        mini images.

        Input: img
        output: list of divided imgs
        """
        self.lst = []

        try:
            if img == None:
                img = self.img
        except ValueError:
            pass

        self.dimensions = img.shape

        # (Number of blocks, excess)
        self.slice_info = (
            math.floor(self.dimensions[1] / self.dimensions[0]),
            self.dimensions[1] % self.dimensions[0],
        )

        self.block_count = self.slice_info[0] + int(self.slice_info[1] != 0)

        h = self.dimensions[0]
        l = self.dimensions[1]

        # Slice for the first n blocks
        for block in range(self.slice_info[0]):
            self.lst.append(img[:, block * h : (block + 1) * h, :])

        # Slice for the last block (excess)
        if self.slice_info[1]:
            self.lst.append(
                img[
                    :,
                    self.slice_info[0] * h : self.slice_info[0] * h
                    + self.slice_info[1]
                    - 1,
                    :,
                ]
            )

        return self.lst

    def wrapper_dir(self, write_des: str) -> None:
        files = glob.glob(
            self.dataset_path + "\*.png"
            if platform.system() == "Windows"
            else self.dataset_path + "/*.png"
        )
        counter = 0
        for file in files:
            if self.to_slice == False:
                self.lst = [self.obtain_np_array(file)]
            else:
                self.lst = self.split(self.obtain_np_array(file))

            stride, names, pt, batch_size, model, device, imgsz = self.model_insert(
                self.lst, self.model_path
            )
            res = []
            for img in self.lst:
                pred, im = self.prediction(
                    img, stride, names, pt, batch_size, model, device, imgsz
                )
                res.append(self.annotate(pred, img, names, im))

            res = np.concatenate(res, axis=1)

            final_write_des = (
                f"{write_des}\{counter}.png"
                if platform.system() == "Windows"
                else f"{write_des}/{counter}.png"
            )
            cv2.imwrite(final_write_des, res)
            counter += 1

    def wrapper_one(self, img_des: str, show=True) -> np.array:
        if self.to_slice == False:
            self.lst = [self.obtain_np_array(img_des)]
        else:
            self.lst = self.split(self.obtain_np_array(img_des))

        stride, names, pt, batch_size, model, device, imgsz = self.model_insert(
            self.lst, self.model_path
        )
        res = []
        for img in self.lst:
            pred, im = self.prediction(
                img, stride, names, pt, batch_size, model, device, imgsz
            )
            res.append(self.annotate(pred, img, names, im))

        res = np.concatenate(res, axis=1)

        if show:
            cv2.imshow("image", res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return res

    def annotate(self, pred, im0, names, im):
        det = pred[0]
        annotator = Annotator(
            np.ascontiguousarray(im0), line_width=1, example=str(names)
        )

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class

            # ---------------- Extraction of various detected information ----------------------------

            # xyxy contains the coordinate of the classification box (not sure how it's classified tho)
            # conf is the confidence
            # cls is the class value corresponding to the yaml file
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                # print(xyxy, conf, cls)
                label = (
                    None if False else (names[c] if False else f"{names[c]} {conf:.2f}")
                )
                annotator.box_label(xyxy, label, color=colors(c, True))

        return annotator.result()

    def prediction(
        self, im0: np.array, stride, names, pt, batch_size, model, device, imgsz
    ) -> torch.tensor:
        im = letterbox(im0, imgsz[0], stride=32, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        dt = (Profile(), Profile(), Profile())

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        with dt[1]:
            pred = model(im)
        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred,
                conf_thres=0.25,
                iou_thres=0.45,
                classes=None,
                agnostic=False,
                max_det=1000,
            )

        return pred, im

    def model_insert(self, imgs: list, model_path):
        """
        This is the initialisation of the model
        Insert the imgs as a list of np.arrays
        Even if there is only 1 img, put the np.array inside a list first

        Input: imgs, model_path
        output: stride, names, pt, batch_size, model, device
        """
        # Initialization of model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DetectMultiBackend(
            model_path, device=device, dnn=False, data=self.yaml_file, fp16=False
        )
        stride, names, pt = model.stride, model.names, model.pt

        # Might want to explore this part of the code further next time

        imgsz = check_img_size((640, 640), s=stride)  # check image size

        batch_size = len(imgs)

        model.warmup(imgsz=(1 if pt or model.triton else batch_size, 3, *imgsz))

        return stride, names, pt, batch_size, model, device, imgsz


if __name__ == "__main__":
    demo = Detection(
        yaml_file=r"C:\Users\chiny\OneDrive - Nanyang Technological University\Internships\AY23 DSO Summer\Classification of RF Project\yolov5\data\rfclasses.yaml",
        dataset_path=r"C:\Users\chiny\OneDrive - Nanyang Technological University\Internships\AY23 DSO Summer\Classification of RF Project\toexport\toexport\testing_env\images",
        model_path=r"C:\Users\chiny\OneDrive - Nanyang Technological University\Internships\AY23 DSO Summer\Classification of RF Project\exp6_train\exp6\weights\best.pt",
    )

    # Directory version

    # demo.wrapper_dir(
    #     write_des=r"C:\Users\chiny\OneDrive - Nanyang Technological University\Internships\AY23 DSO Summer\Classification of RF Project\toexport\toexport\testing_env\results"
    # )

    # One individual img version

    img = demo.wrapper_one(
        img_des=r"C:\Users\chiny\OneDrive - Nanyang Technological University\Internships\AY23 DSO Summer\Classification of RF Project\toexport\toexport\testing_env\images\result_frame_138847877310877880_bw_25E+6.png"
    )
