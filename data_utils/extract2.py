import os
import csv 
import numpy as np
import pandas as pd
import glob
import re

class Extraction():
    def __init__(self, path = None) -> None:
        self.classes = None
        self.paths = None
        self.obtain_classes(path = path) # Add the new function next time

    def obtain_files(self, path, file_type = 'csv') -> None:
        self.paths = glob.glob(f'{path}/*.{file_type}')
        return self.paths

    def wlan_create(self) -> dict:
        rf_std = ["WAC", "WN", "WBG"]
        mcs = ["0", "1" , "2", "3", "7"]
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
        '''
        Creates a dictionary containing all possible classifications for Bluetooth 

        returns a dictionary
        '''
        classtype = ['BT_classic']
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
    
    def obtain_classes(self, path = None) -> dict:
        '''
        Reads from yaml file if possible
        or else will create the classes on the fly
        
        example path = "/home/jovyan/yolov5/transfer_learning/spectrogram_v2.yaml"

        returns a dictionary
        '''
        if self.classes != None:
            return self.classes
        else:
            try:
                if path != None:
                    import yaml
                    with open(path, "r") as stream:
                        try:
                            classes = yaml.safe_load(stream)['names']
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
        
    def extract(self, filepath, dest_path, txt_file_path , add_new = True, add_result_frame = False) -> None:
        '''
        The main function for the whole extraction
        
        filepath = filepath of the csv file that contains all the extra information
        dest_path = output folder for the new files
        txt_file_path = folder name of the txt file
        add_new = boolean for addeding the word "new" to the end of the label name
        add_result_frame = boolean for addeding the word "result_frame" to the start of the label name

        
        output: None
        '''
        df = pd.read_csv(filepath)
        try:
            frameNo = df["identifier"][0]
            identifier = "result_frame_" + str(frameNo) + ".txt"
            resultfile = os.path.join(txt_file_path, identifier)

            #remove collision 
            index = []
            index = df[df['class'] == 'collision'].index
            for i in range(len(index)):
                df.drop(index[i], inplace = True)
            df = df.reset_index()
            
            with open(resultfile, "r") as file: 
                lines = file.readlines()

            
            for i in range(len(index)):
                del lines[index[i] - i]

            if len(lines) != 0:
                for index, row in df.iterrows():
                    label = lines[index][1:]
                    
                    if row["class"] == "WLAN":
                        c = "WLAN_" + str(row["rf_std"]) + "_" + str(int(row["WLAN_mcs"]))
                    else:
                        c = str(row["class"]) + "_" + str(row["BT_packet_type"])
                    
                    key = self.classes[c]

                    label = str(key) + label
                    lines[index] = label
            
            header = "result_frame_" if add_result_frame else ""
            tail = "_new" if add_new else ""
            
            with open(f"{dest_path}/{header}{frameNo}{tail}.txt", "w") as file:     
                file.writelines(lines)


        except (KeyError, IndexError) as error:
            frame_no = filepath.split("labels_")[-1]
            frame_no = frame_no.replace(".csv", "_new.txt")
            # print(frame_no, filepath)
            
            
            header = "result_frame_" if add_result_frame else ""
            tail = "_new" if add_new else ""
            
            with open(f"{dest_path}/{header}{frame_no}{tail}.txt", "w") as file:     
                file.write("")
            # print(error)
    
    def label_changer(self, filepath):
        '''
        Consider this as deprecated
        '''
        with open(filepath, "r") as file: 
            lines = file.readlines()
        
        for i in range(len(lines)):
            match = re.search(r'\d+', lines[i])
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






if __name__ == "__main__":
    testing = Extraction()
    # for j in ["25", "45", "60", "125"]:
    #     testing.obtain_files(r"C:\Users\DSO\Downloads\Spectrogram\spectrogram_training_data_20221006\merged_packets\bw_" + j +"e6")
    #     for i in testing.paths:
    #         testing.extract(i, r"C:\Users\DSO\Downloads\Spectrogram\test")

    # with open(r"C:\Users\DSO\Downloads\Spectrogram\label_key.txt", "w") as file:
    #     dict = testing.obtain_classes()
    #     for key, value in dict.items():
    #         line = str(value) + ": " + str(key) + "\n"
    #         file.write(line)

    # testing.extract(r"C:\Users\DSO\Downloads\Spectrogram\spectrogram_training_data_20221006\merged_packets\bw_25e6\labels_138847877310877880_bw_25E+6.csv", r"C:\Users\DSO\Downloads\Spectrogram\test")

    # dict = testing.obtain_classes()
    # for key, value in dict.items():
    #     print(key, ":", value)

    testing.label_changer(r"C:\JZ\deepsig\testfile.txt")
    

