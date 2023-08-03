import sys
import os
import glob
import numpy as np
import torch
from torch.nn import functional as F


sys.path.append(os.path.join("yolov5"))
sys.path.append(os.path.join("AFC"))

from yolov5.detection import Detection
from AFC.inclearn import models, parser, train

import numpy as np
from yolov5.utils.general import (
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
)


class HybridDetection(Detection):
    def __init__(
        self,
        yaml_file: str = None,
        to_slice: bool = True,
        dataset_path: str = None,
        yolo_model_path: str = None,
    ):
        """

        Updated Detection module for Hybrid model

        Args:
            yaml_file (str, optional): _description_. Defaults to None.
            to_slice (bool, optional): _description_. Defaults to True.
            dataset_path (str, optional): _description_. Defaults to None.
            yolo_model_path (str, optional): _description_. Defaults to None.
        """
        super().__init__(yaml_file, to_slice, dataset_path, yolo_model_path)

    def predict(self, pred, im0, names, im) -> any:
        det = pred[0]

        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            res = []
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = [int(i) for i in xyxy]
                res.append(im0[x1:x2, y1:y2])
            return res
        else:
            return None

    def wrapper_one(self, img_des: str, show=True) -> list:
        """
        Sliced doesn't work rn

        Args:
            img_des (str): img path
            show (bool, optional): NIL. Defaults to True.

        Returns:
            np.array: _description_
        """
        self.lst = [self.obtain_np_array(img_des)]

        # Can add slice subsequently

        stride, names, pt, batch_size, model, device, imgsz = self.model_insert(
            self.lst, self.model_path
        )
        res = []
        # Can add slice subsequently
        for img in self.lst:
            pred, im = self.prediction(
                img, stride, names, pt, batch_size, model, device, imgsz
            )
            res.append(self.predict(pred, img, names, im))

        return res

    def model_load(self, model_path: str = None, args=None) -> any:
        """
        Load AFC model

        Args:
            model_path (str): Path of the AFC model
        """

        model = models.AFC(args=args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        checkpoints = torch.load(model_path)
        to_del = []
        for k, v in checkpoints.items():
            if "classifier._weights" in k:
                to_del.append(k)

        for i in to_del:
            checkpoints.pop(i)
        try:
            model.network.load_state_dict(checkpoints)
        except Exception as e:
            # print(e)
            model._network.load_state_dict(checkpoints)

        return model


def resize_with_pad(
    image: np.array,
    new_shape: tuple[int, int],
    padding_color: tuple[int] = (255, 255, 255),
) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple(
        [int(x * ratio) if int(x * ratio) >= 1 else 1 for x in original_shape]
    )
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color
    )
    return image


if __name__ == "__main__":
    print("Starting")
    demo = HybridDetection(
        yaml_file=r"yolov5\transfer_learning\spectrogram_v3.yaml",
        dataset_path=r"images_final",
        yolo_model_path=r"yolov5\weights\unsliced.pt",
        to_slice=False,
    )

    # res = demo.wrapper_one(
    #     r"images_final\result_frame_138847940791413510_bw_45E+6 (1).png"
    # )
    # print(len(res[0]))

    args = parser.get_parser()
    args = args.parse_args()
    args = vars(args)

    train._set_up_options(args)

    # print("args", args)

    # Error here
    model = demo.model_load(
        r"20230714_AFC_cnn_custom_10stepsv2\net_0_task_4.pth", args=args
    )

    from torchsummary import summary

    print(summary(model._network, (3, 224, 224)))
    # for input in res[0]:
    #     # Need to change the dimensions of the thingy

    #     input = resize_with_pad(input, (224, 224))

    #     input = torch.tensor(input)
    #     inputs = input.to(demo.device)
    #     logits = model._network(inputs)
    #     print(logits)
    #     preds = F.softmax(logits, dim=-1)
    #     print(preds.cpu().numpy())
