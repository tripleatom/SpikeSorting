import re
import spikeinterface.extractors as se
from pathlib import Path
import numpy as np
import spikeinterface as si
from process_func.readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
import matplotlib.pyplot as plt


def get_dio_folders(rec_folder):
    # get the folder under rec_folder that ends with ".DIO"
    rec_folder = Path(rec_folder)
    dio_folders = [f for f in rec_folder.iterdir() if f.is_dir() and f.name.endswith(".DIO")]
    # sort by part number: no part suffix -> 1, .part2 -> 2, .part3 -> 3, etc.
    def _part_num(f):
        m = re.search(r'\.part(\d+)\.DIO$', f.name)
        return int(m.group(1)) if m else 1
    return sorted(dio_folders, key=_part_num)


def extract_DIN(DIO_folder, channel_id):
    # get the file name
    DIO_folder = Path(DIO_folder)

    # file name is end with Din1.dat
    din_files = [f for f in DIO_folder.iterdir() if f.is_file() and f.name.endswith(f"Din{channel_id}.dat")]
    if len(din_files) == 0:
        # return error if no file found
        return None
    
    din_file = din_files[0]
    # read the file
    time = readTrodesExtractedDataFile(din_file)['data']['time']
    state = readTrodesExtractedDataFile(din_file)['data']['state']
    return time, state


def concatenate_din_data(dio_folders, channel_id: int):
    # initialize the din_data
    time, state = extract_DIN(dio_folders[0], channel_id)

    if len(dio_folders) == 1:
        return time, state
    
    
    for i in range(1, len(dio_folders)):
        time_, state_ = extract_DIN(dio_folders[i], channel_id)
        # if the end of the last state is the same as the start of the current state, remove the first element of the current state and time
        if state[-1] == state_[0]:
            state_ = state_[1:]
            time_ = time_[1:]

        time = np.concatenate((time, time_))
        state = np.concatenate((state, state_))
    return time, state