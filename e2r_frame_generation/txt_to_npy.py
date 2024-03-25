import numpy as np
import os
import argparse

def main(filepath:str): #Script to convert txt files to npy files for easy loading
    with open(filepath) as f:
        filename = filepath.split("/")[-1].split(".")[0]
        assert(filename != "", "Filename is empty")
        file_data = np.loadtxt(f, dtype=np.string_)
        print(file_data.shape)
        print(file_data[202])
        print(file_data[-1])
        n = 1
        folder_path = "/home/eshan/Downloads/e_data/"
        assert(file_data.shape[0]%n == 0,"File is not evenly divisible by n")
        chunks = np.split(file_data, n)
        for i in range(len(chunks)):
            np.save(os.path.join(folder_path, filename + "_" + str(i) + ".npy"), chunks[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process filepath.')
    parser.add_argument('filepath', type=str, help='Path to the file')
    args = parser.parse_args()

    main(args.filepath)
    print("Done!")