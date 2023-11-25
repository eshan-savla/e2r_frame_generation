import numpy as np
import cv2
import os

class Events:
    # attributes
    events_vector = np.ndarray()

    def __init__(self, all_events:np.array) -> None:
        self.events_vector = all_events

    def loadFromFiles(self,path_to_folder:str, file_count:int = None) -> None:
        i = 0
        for filename in os.listdir(path_to_folder):
            file = os.path.join(path_to_folder,filename)
            if (not os.path.isfile(file)) and (filename[-3:] != ".npy"):
                continue
            if(not file_count is None) and (i > file_count):
                break
            events = np.load(file)
            self.events_vector = np.append(self.events_vector, events)
            i += 1

    def getGreyscaleImages(self, time_resolution:float) -> np.ndarray:
        baseline_image = np.full((1280,720),128,dtype=np.uint8)
        
