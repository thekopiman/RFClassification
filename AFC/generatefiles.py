import os
import yaml
import glob
import argparse

def get_parser() -> any:
    """
    Parser

    Returns:
        parser: Haven't .parse_arg()
    """
    parser = argparse.ArgumentParser(
        "Generate Files for AFC", description="Used in creating train/val/test.txt files for AFC"
    )
    
    parser.add_argument(
            "-data",
            "--data",
            type=str,
            help="Path of the signals. This signals directory should contain the 18 different subdirectories. If you follow the steps aforementioned, the path should be called dataset1",
        )
    
    parser.add_argument(
            "-dest",
            "--dest",
            default = os.getcwd()
            type=str,
            help="Path of dest directory",
        )
    
    parser.add_argument(
            "-yaml",
            "--yaml",
            type=str,
            help="Path of the yaml file. This yaml file contains the 18 classes (the one used in Yolov5)",
        )
    

    return parser

def main(args : dict) -> None:
    """
    Main function for this module
    
    """
    
    assert args['data'] != None, "data argument is empty"
    assert args['dest'] != None, "dest argument is empty"
    assert args['yaml'] != None, "yaml argument is empty"

    
    
    data_path = args['data']
    DEST = args['dest'] 
    yaml_file = args['yaml']

    with open(yaml_file) as file:
        _ = yaml.safe_load(file)

    num_to_name = _['names']
    name_to_num = {v: k for k, v in num_to_name.items()}

    # Assuming that we have a 70:20:10 split

    # string - /* or \*
    suffix = "\*" if os.name == 'nt' else "/*"
    complete_data = {i: [] for i in range(18)}
    
    for folder in glob.glob(data_path + suffix):
        index_name = folder.replace(data_path, "")[1:]
        index = name_to_num[index_name]
        for file_name in glob.glob(folder + suffix):
            complete_data[index].append(file_name.replace(data_path, "")[1:])


    with open(os.path.join(DEST,'train_18.txt'), "w") as w:
        for key in complete_data:
            size = len(complete_data[key])
            w.writelines(f"{line} {key}" + '\n' for line in complete_data[key][:int(0.7*size)])

    with open(os.path.join(DEST,'val_18.txt'), "w") as w:
        for key in complete_data:
            size = len(complete_data[key])
            w.writelines(f"{line} {key}" + '\n' for line in complete_data[key][int(0.7*size): int(0.9*size)])

    with open(os.path.join(DEST,'test_18.txt'), "w") as w:
        for key in complete_data:
            size = len(complete_data[key])
            w.writelines(f"{line} {key}" + '\n' for line in complete_data[key][int(0.9*size):])
            
    print(f"The files have been save at {DEST}")
if __name__ == "__main__":
    args = get_parser().parse_arg()
    args = vars(args)
    main(args)