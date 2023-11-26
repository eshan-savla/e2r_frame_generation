import numpy as np
import cv2
import os

class Events:
    # attributes
    events_vector = np.empty((1,4),dtype=np.uint32)

    def __init__(self, all_events:np.array = None) -> None:
        if(all_events is not None):
            self.events_vector = all_events

    def loadFromFiles(self,path_to_folder:str, file_count:int = None) -> None:
        i = 0
        for filename in os.listdir(path_to_folder):
            file = os.path.join(path_to_folder,filename)
            if (not os.path.isfile(file)) and (filename[-3:] != ".npy") and (filename[:5] != "event"):
                continue
            if(not file_count is None) and (i > file_count):
                break
            events = np.load(file)
            self.events_vector = np.concatenate((self.events_vector, events),axis=0)
            i += 1
        self.events_vector = np.delete(self.events_vector, (0), axis=0)
        print(self.events_vector[0])

    def getGreyscaleImage(self,img_resolution:tuple, num_of_frames:int, amplification: float) -> np.ndarray:
        assert(self.events_vector.shape[0]%num_of_frames == 0, "no. of events not divisible by time resolution")
        self.events_vector[self.events_vector == 0] = -1
        for count in range(0,self.events_vector.shape[0], num_of_frames):
            baseline_image = np.full(img_resolution,128,dtype=np.uint8)
            t0 = self.events_vector[0][0]
            events_slice = self.events_vector[count:count + num_of_frames]
            for event in events_slice:
                pixel_val = baseline_image[event[2]][event[1]]
                if pixel_val + event[3]*amplification < 255:
                    baseline_image[event[2]][event[1]] += event[3]*int(amplification)
                else:
                    baseline_image[event[2]][event[1]] = 255
                b = 0
            yield baseline_image
        b = 0



if __name__ == "__main__":
    events_obj = Events()
    events_obj.loadFromFiles("./data/",1)
    img_iterator = events_obj.getGreyscaleImage((720,1280),100000,50)
    img = next(img_iterator)
    img_2 = next(img_iterator)
    cv2.imshow("greyscale event", img)
    cv2.imshow("next image", img_2)
    cv2.imwrite("./data/cheetah_grey.png",img)
    cv2.waitKey()