import os
import csv
import numpy as np
import pandas as pd
import glob
import argparse
import yaml


class Extraction:
    def __init__(self, path: str = None) -> None:
        """
        This Extraction Module is used for Extracting more information classes from the
        3 orginal classes (WLAN, Collision, BT).

        Args:
            path (str, optional): Contains the yaml file path. Defaults to None. If path is None,
            then the classes will be generated automatically (albeit wrong).
        """
        self.classes = None
        self.paths = None
        self.obtain_classes(path=path)  # Add the new function next time

    def obtain_files(self, path: str, file_type: str = "csv") -> any:
        """
        Obtain all the specific files from the folder ending with file_type

        Args:
            path (str): _description_
            file_type (str, optional): _description_. Defaults to 'csv'.

        Returns:
            list[str]: List of paths
        """
        self.paths = glob.glob(os.path.join(path, f"*.{file_type}"))
        return self.paths

    def wlan_create(self) -> dict:
        """
        WLAN Generation.
        Will be used if yaml path is not included during initialization of this Class

        Ask Jia Ze for more information
        Returns:
            dict: ??
        """
        rf_std = ["WAC", "WN", "WBG"]
        mcs = ["0", "1", "2", "3", "7"]
        lan_dict = {}
        counter = 0
        for i in range(len(rf_std)):
            for j in range(len(mcs)):
                label = "WLAN_" + rf_std[i] + "_" + mcs[j]
                if label == "WLAN_WAC_2" or label == "WLAN_WAC_0":
                    continue
                else:
                    lan_dict.update({label: counter})
                    counter += 1

        return lan_dict

    def bt_create(self) -> dict:
        """
        BT Generation.
        Will be used if yaml path is not included during initialization of this Class

        Ask Jia Ze for more information
        Returns:
            dict: ??
        """
        classtype = ["BT_classic"]
        packetype = ["ADH5", "AEDH5", "DH5"]
        bt_dict = {}
        counter = 0

        for i in range(len(classtype)):
            for j in range(len(packetype)):
                label = classtype[i] + "_" + packetype[j]
                bt_dict.update({label: counter})
                counter += 1

        bt_dict.update({"BLE_2MHz_DATA": counter})
        counter += 1
        bt_dict.update({"BLE_1MHz_DATA": counter})
        counter += 1
        bt_dict.update({"BLE_1MHz_AIND": counter})
        return bt_dict

    def obtain_classes(self, path: str = None) -> any:
        """
        Reads from yaml file if possible
        or else will create the classes on the fly


        Args:
            path (str, optional): eg path /home/jovyan/yolov5/transfer_learning/spectrogram_v3.yaml. Defaults to None.

        Returns:
            dict: {class name: idx}
        """

        if self.classes != None:
            return self.classes
        else:
            try:
                if path != None:
                    with open(path, "r") as stream:
                        try:
                            classes = yaml.safe_load(stream)["names"]
                        except yaml.YAMLError as exc:
                            print(exc)

                    inv_map = {v: k for k, v in classes.items()}
                    self.classes = inv_map
                    return self.classes
            except Exception:
                lan_dict = self.wlan_create()
                bt_dict = self.bt_create()
                for key in bt_dict:
                    bt_dict[key] += len(lan_dict)

                lan_dict.update(bt_dict)
                self.classes = lan_dict
                return self.classes

    def extract(
        self,
        filepath: str,
        dest_path: str,
        txt_file_path: str,
        add_new=True,
        add_result_frame=False,
    ) -> None:
        """
        Main function for Extraction

        Args:
            filepath (str): filepath of the csv file that contains all the extra information
            dest_path (str): output folder for the new files
            txt_file_path (str): folder name of the txt file
            add_new (bool, optional): boolean for addeding the word "new" to the end of the label name. Defaults to True.
            add_result_frame (bool, optional): boolean for adding the word "result_frame" to the start of the label name. Defaults to False.
        """
        df = pd.read_csv(filepath)
        try:
            frameNo = df["identifier"][0]
            identifier = "result_frame_" + str(frameNo) + ".txt"
            resultfile = os.path.join(txt_file_path, identifier)

            # remove collision
            index = []
            index = df[df["class"] == "collision"].index
            for i in range(len(index)):
                df.drop(index[i], inplace=True)
            df = df.reset_index()

            with open(resultfile, "r") as file:
                lines = file.readlines()

            for i in range(len(index)):
                del lines[index[i] - i]

            if len(lines) != 0:
                for index, row in df.iterrows():
                    label = lines[index][1:]

                    if row["class"] == "WLAN":
                        c = (
                            "WLAN_"
                            + str(row["rf_std"])
                            + "_"
                            + str(int(row["WLAN_mcs"]))
                        )
                    else:
                        c = str(row["class"]) + "_" + str(row["BT_packet_type"])

                    key = self.classes[c]

                    label = str(key) + label
                    lines[index] = label

            header = "result_frame_" if add_result_frame else ""
            tail = "_new" if add_new else ""

            with open(
                os.path.join(dest_path, f"{header}{frameNo}{tail}.txt"), "w"
            ) as file:
                file.writelines(lines)

        except (KeyError, IndexError) as error:
            frame_no = filepath.split("labels_")[-1]
            frame_no = frame_no.replace(".csv", "")
            # print(frame_no, filepath)

            header = "result_frame_" if add_result_frame else ""
            tail = "_new" if add_new else ""

            with open(
                os.path.join(dest_path, f"{header}{frame_no}{tail}.txt"), "w"
            ) as file:
                file.write("")
            # print(error)

    def label_changer(self, filepath):
        """
        Consider this as deprecated
        """
        import re

        with open(filepath, "r") as file:
            lines = file.readlines()

        for i in range(len(lines)):
            match = re.search(r"\d+", lines[i])
            num = int(match.group())
            print(num)
            if num == 1:
                num -= 1
            elif num > 1 and num < 15:
                num = num - 2
            elif num == 26:
                num = num - 13
            elif num == 27 or num == 28 or num == 30:
                num = num - 14
            elif num == 35:
                num = num - 18
            else:
                num = num - 19
            print("new", num)
            num = str(num)
            pos = len(num)
            label = lines[i][pos:]
            label = num + label
            lines[i] = label

        with open(filepath, "w") as file:
            file.writelines(lines)


def version_update(file_name: str, dir: str) -> str:
    """Obtain the Correct file name so that the there won't be duplicates

    eg.
    file_name = hello_v0.txt

    If there exist up to hello_v10.txt inside the dir, then
    this function will return hello_v11.txt


    Args:
        file_name (str): name of the file.
        dir (str): directory path

    Returns:
        str: final name of the file with version numbers
    """
    ver_int = 0
    prefix = file_name.split("_v")[0] if "_v" in file_name else file_name.split(".")[0]
    suffix = file_name.split(".")[-1]

    while f"{prefix}_v{ver_int}.{suffix}" in glob.glob(os.path.join(dir, "*")):
        ver_int += 1

    return f"{prefix}_v{ver_int}.{suffix}"


class Parser():
    def __init__(self):
        """
        Parser class for Extract since it's getting too big
        
        Args: None
        
        Raises:
            Exception: Missing paths
        """
        self.args = None
        self.get_parser()
    
    # @classmethod
    def main(self, args : dict = None):
        if not isinstance(args, dict):
            args = self.args
        
        # DUPLICATE IMGS
        if args["duplicate_imgs"] != None:
            assert args["duplicate_imgs"] != None, "Missing duplicate imgs path"
            assert args["data"] != None, "Missing data path"
            self.duplicate_imgs(args)
        
        # GENERATE .txt FILES
        if args["generate"] != None or args["duplicate_imgs"] != None:
            gen_path = (
            args["generate"] if args["generate"] != None else args["duplicate_imgs"]
        )
        
            save_path = args['data'] if args['data'] != None else os.getcwd()
            self.generate(gen_path = gen_path, save_path = save_path)
        
        # Running of the various modes
        
        # classes 18
        
        if args["mode"] == "classes18":
            assert args["yaml"] != None, "Missing yaml path"
            assert args["data"] != None, "Missing data path"
            
            try:
                assert args["dest"] != None, "Missing dest path"
            
            except AssertionError:
                counter = 0
                while os.path.exists(os.path.join(args["data"], f"dataset{counter}")):
                    counter += 1

                args["dest"] =  os.path.join(args["data"], f"dataset{counter}")
                os.makedirs(args["dest"])

                args["dest"] = os.path.join(args["dest"], "labels")
                os.makedirs(args["dest"])

                print(f"--- Dest Directory Not Found, creating new directory: {args['dest']} ----")
            
            self.args = args
            self.classes18(args)
            
        # getsignals
        
        elif args["mode"] == "getsignals":
            assert args["yaml"] != None, "Missing yaml path"
            assert args["dest"] != None, "Missing dest path"
            assert args["labels_path"] != None, "Missing labels path"
            assert args["imgs_path"] != None, "Missing imgs path"
            self.getsignals(args)
        
        
        
        # Automatically run classes18 and getsignals and also duplicate files 
        # Only yaml and data path are required here
        elif args["mode"] == "auto":
            assert args["yaml"] != None, "Missing yaml path"
            assert args["data"] != None, "Missing data path"
            
            counter = 0
            while os.path.exists(os.path.join(args["data"], f"dataset{counter}")):
                counter += 1

            args["dest"] =  os.path.join(args["data"], f"dataset{counter}")
            os.makedirs(args["dest"])
            
            print(f"--- classes18 directory is {args['dest']} ---")
            
            args["duplicate_imgs"] = os.path.join(args["dest"], "images")
            
            self.duplicate_imgs(args)
            
            # Got bug here
            self.generate(gen_path = args["duplicate_imgs"], save_path = args["data"])
            
            args["dest"] = os.path.join(args["dest"], "labels")
            os.makedirs(args["dest"])
            
            # Start classes18
            self.classes18(args)
            
            
            # Start getsignals
            args["labels_path"] = args["dest"]
            args["imgs_path"] = args["duplicate_imgs"]
            
            while os.path.exists(os.path.join(args["data"], f"dataset{counter}")):
                counter += 1

            args["dest"] =  os.path.join(args["data"], f"dataset{counter}")
            os.makedirs(args["dest"])
            print(f"--- getsignals directory is {args['dest']} ---")
            
            self.args = args
            self.getsignals(args)
            
        elif args["mode"] == "false": 
            pass
        else:
            raise Exception("Mode name is wrong") 
        
    def classes18(self, args : dict) -> None:
        print("------- Signal Classification has began -------")

        labels_more_classification = Extraction(path=args["yaml"])
        for idx, j in enumerate(["25", "45", "60", "125"]):
            print(f"-- Progress {idx*25}% --")
            dest_path = args["dest"]

            if not os.path.exists(dest_path):
                print("----- Directory Not Found, creating new directory ------")
                os.makedirs(dest_path)

            labels_more_classification.obtain_files(
                os.path.join(args["data"], "merged_packets", f"bw_{j}e6")
            )

            # print(os.path.join(args['data'], "merged_packets", f"bw_{j}e6"), labels_more_classification.paths)

            # For debugging purposes
            for i in labels_more_classification.paths:
                filepath = i
                txt_file_path = os.path.join(args["data"], "results")
                add_new = args["addnew"]
                add_result_frame = args["addresultframe"]
                labels_more_classification.extract(
                    filepath, dest_path, txt_file_path, add_new, add_result_frame
                )
        
    def getsignals(self, args : dict) -> None:
        print("------- Signals will now be extracted -------")

        import cv2

        with open(args["yaml"], "r") as stream:
            try:
                classes = yaml.safe_load(stream)["names"]
            except yaml.YAMLError as exc:
                print(exc)

        delete = False  # Hard code this part

        dest_path = args["dest"]
        if not os.path.exists(dest_path):
            print("----- Directory Not Found, creating new directory ------")
            os.makedirs(dest_path)

        for i in classes:
            try:
                # os.mkdir(classes[i])
                os.mkdir(os.path.join(dest_path, classes[i]))
            except FileExistsError as e:
                # Delete everything, ie restart the whole thing
                if delete:
                    for j in glob.glob(os.path.join(dest_path, classes[i], "*")):
                        os.remove(j)

        # For progress purposes
        length = len(glob.glob(os.path.join(args["imgs_path"], "*.png"))[: args["cap"]])
        percent10 = int(length / 10)

        for idx, image in enumerate(
            glob.glob(os.path.join(args["imgs_path"], "*.png"))[: args["cap"]]
        ):  # Keep in mind that there are 20k images
            if idx % percent10 == 0:
                print(f"-- Progress {idx/percent10 * 10}% --")

            index = image.replace(args["imgs_path"], "")
            index = index.replace(".png", "")
            
            # print("labels_path", args["labels_path"])
            
            
            # Cannot use os.path.join for some reason
            label_path = f"{args['labels_path']}{os.path.sep}{index}.txt"
            # print(label_path)

            # print("OS path Joined", os.path.join(args['labels_path'], f"{index}.txt"))
            # print("OS path Not Joined",args['labels_path'], f"{index}.txt")

            img = cv2.imread(image)
            # print(img.shape)
            with open(label_path, "r") as r:
                lines = r.readlines()
                for index, line in enumerate(lines):
                    label, x, y, w, h = line.split(" ")
                    label = int(label)
                    x = float(x)
                    y = float(y)
                    w = float(w)
                    h = float(h)
                    # print(label, x, y, w, h)
                    y1 = (y - h / 2) * img.shape[0]
                    y2 = (y + h / 2) * img.shape[0]
                    x1 = (x - w / 2) * img.shape[1]
                    x2 = (x + w / 2) * img.shape[1]
                    counter_files = len(
                        glob.glob(os.path.join(args["dest"], classes[label], "*"))
                    )
                    final_img = img[int(y1) : int(y2), int(x1) : int(x2), :]
                    try:
                        cv2.imwrite(
                            os.path.join(
                                args["dest"], classes[label], f"{counter_files}.png"
                            ),
                            final_img,
                        )
                    except:
                        print(classes[label], final_img.shape)
    
    def duplicate_imgs(self, args : dict = None):
        import shutil

        # print(args['duplicate_imgs'], os.path.exists(args['duplicate_imgs']))
        if not os.path.exists(args["duplicate_imgs"]):
            print("----- Directory Not Found, creating new directory ------")
            os.makedirs(args["duplicate_imgs"])

        print("------- Currently Duplicating images ---------")
        folder_size = len(glob.glob(os.path.join(args["data"], "results", "*+6.png")))
        division_size = folder_size / 4
        for idx, file in enumerate(
            glob.glob(os.path.join(args["data"], "results", "*+6.png"))
        ):  # Debugging
            if idx % division_size == 0:
                print(f"---- Progress {idx/division_size *25}% ----")
            shutil.copy(file, args["duplicate_imgs"])
    
    def generate(self, gen_path : str, save_path : str):
        files = glob.glob(os.path.join(gen_path, "*.png"))
        length = len(files)

        train_int = int(0.7 * length)
        val_int = int(0.2 * length) + train_int
        # test_int = int(0.1*length) + val_int

        with open(os.path.join(save_path,"train_list.txt"), "w+") as w:
            for l in files[:train_int]:
                w.writelines(l)
                w.writelines("\n")

        with open(os.path.join(save_path,"val_list.txt"), "w+") as w:
            for l in files[train_int:val_int]:
                w.writelines(l)
                w.writelines("\n")

        with open(os.path.join(save_path,"test_list.txt"), "w+") as w:
            for l in files[val_int:]:
                w.writelines(l)
                w.writelines("\n")

        print(
            f"train/val/test_list.txt files have been saved at {save_path}."
        )  # Implement new feature here
    
    def __repr__(self) -> dict:
        return self.args
    
    
    def get_parser(self) -> any:
        """Parser

        Returns:
            parser: Haven't .parse_arg()
        """
        parser = argparse.ArgumentParser(
            "Extraction", description="Used in extracting various signals"
        )

        # Mode
        parser.add_argument(
            "-m",
            "--mode",
            default="false",
            type=str,
            help="Extraction modes: classes18, getsignals, false, auto",
        )

        # Classes 18 - Splitting the 3 classes into 18 different classes while removing collision
        parser.add_argument("-y", "--yaml", type=str, help="yaml path")
        parser.add_argument(
            "-data", "--data", type=str, help="path of spectrogram, the 80gb folder"
        )
        parser.add_argument(
            "-dest", "--dest", type=str, help="destination path of the labels files"
        )
        parser.add_argument(
            "-dupli",
            "--duplicate-imgs",
            type=str,
            default=None,
            help="Duplicate the *.png files (not marked) files into a path",
        )
        parser.add_argument(
            "-addnew",
            "--addnew",
            default=False,
            type=lambda x: (str(x).lower() == "true"),
            help="add *_new to the end of each label file",
        )
        parser.add_argument(
            "-addresultframe",
            "--addresultframe",
            default=True,
            type=lambda x: (str(x).lower() == "true"),
            help="boolean for adding the word 'result_frame' to the start of the label name",
        )

        parser.add_argument(
            "-cap",
            "--cap",
            type=int,
            default=-1,
            help="Caps/Limits the number of images being extract during 'getsignals'. Defaults to -1, basically everything.",
        )

        # Generate train/val/test.txt files
        parser.add_argument(
            "-generate",
            "--generate",
            type=str,
            default=None,
            help="Generate the train/val/test files from the images folder. Fixed at 70:20:10. Ensure that the folder do not include _marked.png pics as I'm searching for *.png. Note: If you set this arg as None but duplicate has a path, then it will automatically use the duplicate path.",
        )
        # Get signals - Extract out every signal as a png file

        parser.add_argument(
            "-imgs",
            "--imgs-path",
            type=str,
            help="Src Path of the images. Ensure that the name of the label and imgs are identical (excluding the .png/.txt).",
        )
        parser.add_argument(
            "-labels",
            "--labels-path",
            type=str,
            help="Src Path of the labels. Ensure that the name of the label and imgs are identical (excluding the .png/.txt).",
        )
        
        self.args = vars(parser.parse_args())
        return parser
        


if __name__ == "__main__":
    args = Parser()
    args.main()