import numpy as np
import pandas as pd
import cv2
import glob
import math


class Scaling:
    def __init__(self, img_dir=None, labels_dir=None) -> None:
        """
        Immediately obtains the first file by default.
        Use obtain_info() to change the file to be used

        ** Future work: Automatically create all the labels in the whole folder

        Input: img_dir, labels_dir
        """
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.img_path = None  # Stores the current full image
        self.img = None  # Stores the respective img in (red, green, blue, alpha)
        self.frame_no = None  # Respective frame no in use
        self.slice_info = None  # (Number of blocks, excess)
        self.labels_path = None  # Stores the current label wrt to the img
        self.labels_data = None  # List of dataframes corresponding to new labels data

        self.obtain_info()

    def obtain_info(self, img_path=None) -> None:
        """
        Specifies the targetted img_file to be used
        Will use the first file in the img folder by default

        input: img_path = None
        output: None
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

        self.labels_path = glob.glob(f"{self.labels_dir}\*{self.frame_no}.txt")[0]

    def slice_img(self, img=None) -> list:
        """
        Slice the image wrt to height

        Input: img = None (defaults to the original img)
        Output: new_images <List>
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

    def save_imgs(self, images: list = None, dir=None) -> None:
        """
        Saves the new images into the respective dir specified

        Input: images <list> (defaults to new_images), dir <str>
        Output: None
        """
        if images == None:
            images = self.new_images
        if dir == None:
            raise IsADirectoryError("Put a imges_new folder dir path here")
        for i in range(len(images)):
            cv2.imwrite(f"{dir}\{self.frame_no}_{i}.png", images[i])

    def write_files(self, inputs, path_dir, name=None, auto=True):
        # Auto = True
        if auto:
            for i in range(len(inputs)):
                name = f"{self.frame_no}_{i}.txt"
                np.savetxt(f"{path_dir}\{name}", inputs[i].values, fmt="%s")

        # Auto = False | Code later

    def label_wrapper(self, path_dir) -> None:
        """
        Wrapper function to automatically create new labels files

        Input: path_dir (of the new label folder)
        Output: None
        """
        self.write_files(self.slice_labels(), path_dir)

    def slice_labels(self, path=None, tolerance: int = 0) -> list:
        """
        Slice the labels wrt to dimensions used in the img slicing

        Input: path = None (defaults to the original label)
        Output: data <List>
        """

        if path == None:
            path = self.labels_path

        # Initialization
        data = []
        for i in range(len(self.new_images)):
            data.append({"label": [], "x": [], "y": [], "w": [], "h": []})

        with open(path, "r") as r:
            for line in r.readlines():
                label, x, y, w, h = line.split(" ")
                w = float(w)
                x_start = float(x) - w / 2
                y = float(y)
                x_end = float(x) + w / 2
                h = float(h)

                number_of_blocks = len(self.new_images)
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


if __name__ == "__main__":
    pass
