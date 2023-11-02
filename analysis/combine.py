import numpy as np
import os



# load all files from directory "./processed_data" np.load

def load_data(directory):
    files = os.listdir(directory)
    data = []
    for file in files:
        file_path = os.path.join(directory, file)
        data.append(np.load(file_path, allow_pickle=True))
    return data


res = load_data("./processed_data")



