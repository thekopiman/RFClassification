import os
import csv 
import numpy as np
import pandas as pd
import glob

class Extraction():
    def __init__(self) -> None:
        self.classes = None
        self.paths = None
        self.obtain_classes(path = None) # Add the new function next time

    def obtain_files(self, path, type = 'csv') -> None:
        self.paths = glob.glob(f'{path}\*.{type}')
        return self.paths

    def wlan_create(self) -> dict:
        rf_std = ["WAC", "WN", "WBG", "WPJ", "WAG"]
        mcs = ["0", "1" , "2", "3", "7"]
        lan_dict = {}
        counter = 0
        for i in range(len(rf_std)):
            for j in range(len(mcs)):
                label = "WLAN_" + rf_std[i] + "_" + mcs[j]
                lan_dict.update({label: counter})
                counter += 1
        
        return lan_dict

    def bt_create(self) -> dict:
        '''
        Creates a dictionary containing all possible classifications for Bluetooth 

        returns a dictionary
        '''
        classtype = ['BT_classic', 'BLE_2MHz', 'BLE_1MHz']
        packetype = ["DATA", "ADH5", "AIND", "AEDH5", "DH5"]
        bt_dict = {}
        counter = 0

        for i in range(len(classtype)):
            for j in range(len(packetype)):
                label = classtype[i] + "_" + packetype[j]
                bt_dict.update({label: counter})
                counter += 1
        
        return bt_dict
    
    def obtain_classes(self, path = None) -> dict:
        '''
        Reads from yaml file if possible
        or else will create the classes on the fly

        returns a dictionary
        '''
        if self.classes != None:
            return self.classes
        else:
            lan_dict = self.wlan_create()
            bt_dict = self.bt_create()
            for key in bt_dict:
                bt_dict[key] += len(lan_dict)
            

            lan_dict.update(bt_dict)
            self.classes = lan_dict
            return self.classes
        
    def extract(self, filepath, dest_path):
        df = pd.read_csv(filepath)
        try:
            frameNo = df["identifier"][0]
            identifier = "result_frame_" + str(frameNo) + ".txt"
            resultfile = os.path.join(r"C:\Users\DSO\Downloads\Spectrogram\spectrogram_training_data_20221006\results results", identifier)

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

            with open(f"{dest_path}/{frameNo}_new.txt", "w") as file:     
                file.writelines(lines)

        except KeyError:
            pass




if __name__ == "__main__":
    testing = Extraction()
    testing.obtain_files(r"C:\Users\DSO\Downloads\Spectrogram\spectrogram_training_data_20221006\merged_packets\bw_25e6")
    for i in testing.paths:
        testing.extract(i, r"C:\Users\DSO\Downloads\Spectrogram\test")

    # testing.extract(r"C:\Users\DSO\Downloads\Spectrogram\spectrogram_training_data_20221006\merged_packets\bw_25e6\labels_138847877310877880_bw_25E+6.csv", r"C:\Users\DSO\Downloads\Spectrogram\test")
