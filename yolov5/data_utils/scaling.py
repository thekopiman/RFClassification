import numpy as np
import pandas as pd
import cv2
import glob
import math
import argparse
import os


class Scaling:
    def __init__(self, img_dir: str = None, labels_dir: str = None) -> None:
        """
        Immediately obtains the first file by default.
        Use obtain_info() to change the file to be used

        Args:
            img_dir (str, optional): _description_. Defaults to None.
            labels_dir (str, optional): _description_. Defaults to None.
        """
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.img_path = None  # Stores the current full image
        self.img = None  # Stores the respective img in (red, green, blue, alpha)
        self.frame_no = None  # Respective frame no in use
        self.slice_info = None  # (Number of blocks, excess)
        self.labels_path = None  # Stores the current label wrt to the img
        self.labels_data = None  # List of dataframes corresponding to new labels data
        self.block_count = None  # Correspond to slice_info

        self.obtain_info()

    def obtain_info(self, img_path: str = None) -> None:
        """
        Specifies the targetted img_file to be used
        Will use the first file in the img folder by default

        Args:
            img_path (str, optional): _description_. Defaults to None.
        """

        if img_path == None:
            self.img_path = glob.glob(f"{self.img_dir}\*.png")[0]
        else:
            self.img_path = img_path

        self.img = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
        self.frame_no = self.img_path.split("result_frame_")[1][:-4]
        self.dimensions = self.img.shape

        # (Number of blocks, excess)
        self.slice_info = (
            math.floor(self.dimensions[1] / self.dimensions[0]),
            self.dimensions[1] % self.dimensions[0],
        )

        self.block_count = self.slice_info[0] + int(self.slice_info[1] != 0)

        self.labels_path = glob.glob(f"{self.labels_dir}\*{self.frame_no}.txt")[0]

    def slice_img(self, img: np.array = None) -> list:
        """
        Slice the image wrt to height


        Args:
            img (np.array, optional): Slice the images into a list of imgs. Defaults to None.

        Returns:
            list: list of imgs
        """

        if img == None:
            img = self.img

        self.new_images = []
        h = self.dimensions[0]
        l = self.dimensions[1]

        # Slice for the first n blocks
        for block in range(self.slice_info[0]):
            self.new_images.append(img[:, block * h : (block + 1) * h, :])

        # Slice for the last block (excess)
        if self.slice_info[1]:
            self.new_images.append(
                img[
                    :,
                    self.slice_info[0] * h : self.slice_info[0] * h
                    + self.slice_info[1]
                    - 1,
                    :,
                ]
            )

        return self.new_images

    def save_imgs(self, images: list[np.array] = None, dir: str = None) -> None:
        """
        aves the new images into the respective dir specified

        Args:
            images (list[np.array], Optional): List of imgs(np.array). Defaults to None.
            dir (str, Mandatory): Directory Path. Defaults to None.

        Raises:
            IsADirectoryError: _description_
        """

        if images == None:
            images = self.new_images

        if dir == None:
            raise IsADirectoryError("Put a imges_new folder dir path here")

        for i in range(len(images)):
            cv2.imwrite(f"{dir}\{self.frame_no}_{i}.png", images[i])

    def write_files(
        self, inputs: list[dict], path_dir: str, name: str = None, auto: bool = True
    ):
        """
        Save the files

        Args:
            inputs (list[dict]): List of dict data
            path_dir (str): Path directory
            name (str, optional): Not completed. Defaults to None.
            auto (bool, optional): Not completed. Defaults to True.
        """
        # Auto = True
        if auto:
            for i in range(len(inputs)):
                name = f"{self.frame_no}_{i}.txt"
                np.savetxt(f"{path_dir}\{name}", inputs[i].values, fmt="%s")

        # Auto = False | Code later

    def label_wrapper(self, path_dir: str) -> None:
        """
        Wrapper function to automatically create new labels files

        Args:
            path_dir (str): Path Directory
        """

        self.write_files(self.slice_labels(), path_dir)

    def slice_labels(self, path: str = None, tolerance: int = 0) -> list:
        """
        Slice the labels wrt to dimensions used in the img slicing


        Args:
            path (str, optional): Labels path. Defaults to None.
            tolerance (int, optional): Tolernce to delete signals that are too small. Defaults to 0.

        Returns:
            list: _description_
        """

        if path == None:
            path = self.labels_path

        # Initialization
        data = []
        for i in range(self.block_count):
            data.append({"label": [], "x": [], "y": [], "w": [], "h": []})

        with open(path, "r") as r:
            for line in r.readlines():
                label, x, y, w, h = line.split(" ")
                w = float(w)
                x_start = float(x) - w / 2
                y = float(y)
                x_end = float(x) + w / 2
                h = float(h)

                number_of_blocks = self.block_count
                for index in range(number_of_blocks):
                    x_start_new, x_end_new = self.check_inside(
                        x_start=x_start,
                        x_end=x_end,
                        block_index=index,
                        block_w=self.dimensions[0],
                        map_w=self.dimensions[1],
                        slice_info=self.slice_info,
                        tolerance=tolerance,
                    )

                    # Write into the dictionary
                    if x_start_new != None and x_end_new != None:
                        data[index]["label"].append(label)
                        data[index]["x"].append((x_start_new + x_end_new) / 2)
                        data[index]["y"].append(y)
                        data[index]["w"].append(x_end_new - x_start_new)
                        data[index]["h"].append(h)

        self.labels_data = [pd.DataFrame(data=data[i]) for i in range(len(data))]
        return self.labels_data

    def check_inside(
        self,
        x_start: float,
        x_end: float,
        block_index: int,
        block_w: int,
        map_w: int,
        slice_info: tuple,
        tolerance: int = 0,  # 0 means tolerate everything. In px
    ):
        """
        Check if the sliced signal is inside the square

        Args:
            x_start (float): Can't rmb
            x_end (float): Can't rmb
            block_index (int): Can't rmb
            block_w (int): Can't rmb
            map_w (int): Can't rmb
            slice_info (tuple): Can't rmb
            tolerance (int, optional): Tolerance. Defaults to 0.

        Returns:
            _type_: _description_
        """
        x_start_new = None
        x_end_new = None

        division_ratio = float(block_w / map_w)

        starting_index_block = math.floor(x_start / division_ratio)
        ending_index_block = math.floor(x_end / division_ratio)

        current_block_width = (
            slice_info[1]
            if (block_index == slice_info[0] and slice_info[1] != 0)
            else block_w
        )

        tolerance_percentage = tolerance / current_block_width

        # It's inside
        if block_index <= ending_index_block and block_index >= starting_index_block:
            # Settle Starting
            if block_index == starting_index_block:
                x_start_new = (
                    (x_start - block_index * division_ratio)
                    * map_w
                    / current_block_width
                )
            else:
                x_start_new = 0

            # Settle Ending
            if block_index == ending_index_block:
                x_end_new = (
                    (x_end - block_index * division_ratio) * map_w / current_block_width
                )
            else:
                x_end_new = 1

        # Tolerance
        if x_start_new != None and (x_end_new - x_start_new) >= tolerance_percentage:
            return (x_start_new, x_end_new)
        else:
            return (None, None)


def get_parser() -> any:
    """
    Self explanatory

    Returns:
        argparse.ArgumentParser: Haven't .parse_arg()
    """

    parser = argparse.ArgumentParser(
        "Scaling", description="Used in slicing/scaling various signals"
    )
    parser.add_argument(
        "-src",
        "--src",
        type=str,
        help="Source dataset. Ensure that there are 2 subdirectories called images and labels.",
    )
    parser.add_argument(
        "-dest",
        "--dest",
        type=str,
        help="Dest dataset. Will create new directories called images and labels.",
    )
    parser.add_argument(
        "-tol",
        "--tolerance",
        type=int,
        defaults=5,
        help="Px tolerance level to ignore the sliced signal if it's too small. ",
    )

    return parser


def main(args: dict) -> None:
    """
    Main function

    Args:
        args (dict): Put vars(args) here
    """
    img = Scaling(
        img_dir=os.path.join(args["src"], "images"),
        labels_dir=os.path.join(args["src"], "labels"),
    )

    dest_img = os.path.join(args["dest"], "images")
    dest_label = os.path.join(args["dest"], "labels")

    if not os.path.exists(dest_img):
        print(f"----- {dest_img} Not Found, creating new directory ------")
        os.makedirs(dest_img)

    if not os.path.exists(dest_label):
        print(f"----- {dest_label} Not Found, creating new directory ------")
        os.makedirs(dest_label)

    print("----- Scaling/Slicing in progress -----")

    length = len(glob.glob(os.path.join(img.img_dir, "*.png")))
    percent10 = int(length / 10)
    for idx, l in enumerate(glob.glob(os.path.join(img.img_dir, "*.png"))):
        if idx % percent10 == 0:
            print(f"-- Progress {idx/percent10}% --")

        img.obtain_info(img_path=l)
        img.slice_img()
        img.save_imgs(dir=dest_img)
        data = img.slice_labels(tolerance=args["tolerance"])
        img.write_files(inputs=data, path_dir=dest_label)


if __name__ == "__main__":
    args = get_parser().parse_args()
    args = vars(args)
    main(args)