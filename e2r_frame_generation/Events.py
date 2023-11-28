import numpy as np
import cv2
import os
import pandas as pd

class Events:
    # attributes
    _events_vector = np.empty((1,4),dtype=np.uint32)
    _c_threshhold:float = .0
    _delta_epsilon:float = .0
    _exposure_time:float = .0
    
    _last_timestamp_event:int = 0
    _last_timestamp_img:int = 0
    
    _img_meta_data:pd.DataFrame = pd.DataFrame()
    _img_resolution:tuple = (0,0)
    _rgb_framerate:int = 0
    _img_folder_path:str = ""

    _intermediate_img:np.ndarray = []
    _frame_generated = False
    _img_idx = 0

    def __init__(self, c_threshhold:float, delta_eps:float, exposure_time:float) -> None:
        self._c_threshhold = c_threshhold
        self._delta_epsilon = delta_eps
        self._exposure_time = exposure_time

    def loadEventsFromFiles(self,path_to_folder:str, file_count:int = None) -> None:
        i = 0
        for filename in sorted(os.listdir(path_to_folder)):
            file = os.path.join(path_to_folder,filename)
            dtype = filename[-3:]
            if (not os.path.isfile(file)) or (filename[-3:] != "npy") or (filename[:5] != "event"):
                continue
            if(not file_count is None) and (i > file_count):
                break
            events = np.load(file,allow_pickle=True)
            self._events_vector = np.concatenate((self._events_vector, events),axis=0)
            i += 1
        self._events_vector = np.delete(self._events_vector, (0), axis=0)
        self._events_vector[self._events_vector == 0] = -1
        print(self._events_vector[0])

    def loadImgMetaData(self, path_to_file:str, img_resolution:tuple, video_framerate:int, folder_path:str) -> None:
        self._img_meta_data = pd.read_csv(path_to_file, header=None)
        self._img_resolution = img_resolution
        self._rgb_framerate = video_framerate
        self._img_folder_path = folder_path
        self._intermediate_img = np.full(img_resolution,128,dtype=np.uint8)

    def _cumulateEvents(self, start_ind:int, num_of_events:int):
        events_slice = self._events_vector[start_ind:start_ind+num_of_events]
        event_array = np.empty(self._img_resolution,dtype=float)
        for event in events_slice:
            event_array[event[2]][event[1]] += event[3]
        return event_array
    
    def _getNearestImage(self, timestamp:int, temporal_thresh:int = None) -> tuple:
        times = np.asarray(self._img_meta_data.iloc[:,0])
        min_diff = timestamp - (times[times <= timestamp]).max()
        min_diff_idx = np.argmax(times[times <= timestamp])
        if (temporal_thresh is not None) and (min_diff >= temporal_thresh):
            return 0, np.zeros(1)
        if(self._frame_generated) and (min_diff_idx <= self._img_idx):
            return timestamp - self._last_timestamp_event, np.log(np.add(np.divide(self._intermediate_img,255),self._delta_epsilon))
        filename = os.path.join(self._img_folder_path,self._img_meta_data.iloc[min_diff_idx,1])
        color_frame = cv2.imread(os.path.join(self._img_folder_path,self._img_meta_data.iloc[min_diff_idx,1]))
        greyscale = cv2.cvtColor(color_frame,cv2.COLOR_BGR2GRAY)
        return timestamp-self._img_meta_data.iloc[min_diff_idx,0],np.log(np.add(np.divide(greyscale, 255), self._delta_epsilon))

        
    def generateFrames(self, num_of_events:int, break_count:int = None) -> np.ndarray:
        count = 0
        end = 0
        if(break_count is not None):
            end = break_count
        else:
            end = self._img_meta_data.shape[0]
        for i in range(count, end*num_of_events,num_of_events):
            timestamp, base_image = self._getNearestImage(self._events_vector[i][0])
            event_stream = self._cumulateEvents(i,num_of_events)
            new_frame_ln = np.add(base_image, np.multiply(event_stream, self._c_threshhold))
            new_frame = np.asarray(np.multiply(np.subtract(np.exp(new_frame_ln),self._delta_epsilon),255),dtype=np.uint8)
            self._frame_generated = True
            self._intermediate_img = new_frame
            self._last_timestamp_event = self._events_vector[i][0]
            yield new_frame



if __name__ == "__main__":
    events_obj = Events(0.15,0.1,10)
    events_obj.loadEventsFromFiles("./data/",1)
    events_obj.loadImgMetaData("./data/images.csv",(720,1280),1200,"./data/")
    img_iterator = events_obj.generateFrames(200,10)
    img = next(img_iterator)
    img_2 = next(img_iterator)
    cv2.imshow("greyscale event", img)
    cv2.imshow("next image", img_2)
    cv2.imwrite("./data/cheetah_grey.png",img)
    cv2.waitKey()