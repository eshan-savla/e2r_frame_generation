import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm


class Events:
    # attributes
    # _events_vector = np.empty((1, 4), dtype=np.uint32)
    # _c_threshhold: float = .0
    # _delta_epsilon: float = .0
    # _exposure_time: float = .0
    # _avg_frequency_events: float = .0

    # _last_timestamp_event: int = 0
    # _last_timestamp_img: int = 0

    # _img_meta_data: pd.DataFrame = pd.DataFrame()
    # _img_resolution: tuple = (0, 0)
    # _rgb_framerate: int = 0
    # _img_folder_path: str = ""
    # _avg_frequency_imgs: float = .0

    # _intermediate_img: np.ndarray = np.empty((720, 1280), dtype=np.uint8)
    # _frame_generated = False
    # _img_idx = 0

    def __init__(self, c_threshhold: float, delta_eps: float, exposure_time: float) -> None:
        self._events_vector = np.empty((1, 4), dtype=np.uint32)
        self._c_threshhold = c_threshhold
        self._delta_epsilon = delta_eps
        self._exposure_time = exposure_time
        self._initAttributes()

    def _initAttributes(self) -> None:
        self._avg_frequency_events = .0
        self._last_timestamp_event = 0
        self._last_timestamp_img = 0

        self._img_meta_data = pd.DataFrame()
        self._img_resolution = (0, 0)
        self._rgb_framerate = 0
        self._img_folder_path = ""
        self._avg_frequency_imgs = .0

        self._intermediate_img = np.empty((720, 1280), dtype=np.uint8)
        self._frame_generated = False
        self._img_idx = 0

    def loadEventsFromFiles(self, path_to_folder: str, file_count: int = None) -> None:
        i = 0
        for filename in sorted(os.listdir(path_to_folder)):
            file = os.path.join(path_to_folder, filename)
            dtype = filename[-3:]
            if (not os.path.isfile(file)) or (filename[-3:] != "npy") or (filename[:5] != "event"):
                continue
            if (not file_count is None) and (i > file_count):
                break
            events = np.load(file, allow_pickle=True)
            self._events_vector = np.concatenate((self._events_vector, events), axis=0)
            i += 1
        self._events_vector = np.delete(self._events_vector, 0, axis=0)
        self._events_vector[self._events_vector == 0] = -1
        self._avg_frequency_events = np.average(np.diff(self._events_vector[:,0]))

    def loadImgMetaData(self, path_to_file: str, img_resolution: tuple, video_framerate: int, folder_path: str,
                        max_images=None) -> None:
        self._img_meta_data = pd.read_csv(path_to_file, header=None)
        if max_images is not None:
            self._img_meta_data = self._img_meta_data.iloc[:max_images, :]
        self._img_resolution = img_resolution
        self._rgb_framerate = video_framerate
        self._img_folder_path = folder_path
        diff = self._img_meta_data.iloc[1,0] - self._img_meta_data.iloc[0,0]
        self._avg_frequency_imgs = np.average(np.diff(self._img_meta_data.iloc[:,0]))
        self._intermediate_img = np.full(img_resolution, 128, dtype=np.uint8)

    def _cumulateEvents(self, start_ind: int, end_ind: int = None, num_of_events: int = 0):
        events_slice = self._events_vector[start_ind:start_ind + num_of_events]
        if end_ind is not None:
            if end_ind <= start_ind:
                events_slice = self._events_vector[end_ind:start_ind]
                events_slice[:, 3] = -events_slice[:, 3]
            else:
                events_slice = self._events_vector[start_ind:end_ind]
        event_array = np.full(self._img_resolution, 0, dtype=np.double)
        for event in events_slice:
            event_array[event[2], event[1]] += event[3]
        return event_array

    def deblurImage(self, blurred_img: np.ndarray, timestamp: int) -> np.ndarray:
        average_intensities = self._computeAverageIntensities(timestamp)
        blurred_img = np.log(np.add(np.divide(blurred_img, 255), self._delta_epsilon))
        sharp_img = np.subtract(blurred_img, average_intensities)
        sharp_img = np.multiply(np.subtract(np.exp(sharp_img), self._delta_epsilon), 255)
        return sharp_img.astype(np.uint8)
        

    def _computeAverageIntensities(self, timestamp: int) -> np.ndarray:
        average_intensities = np.zeros((self._img_resolution[0], self._img_resolution[1]), dtype=np.double)
        events_slice_0 = self._getEventsInTimeFrame(timestamp, int(timestamp + self._exposure_time/2))
        events_slice_1 = self._getEventsInTimeFrame(timestamp, int(timestamp - self._exposure_time/2))
        intensities_0 = np.exp(np.multiply(events_slice_0, self._c_threshhold))
        intensities_1 = np.exp(np.multiply(events_slice_1, self._c_threshhold))
        average_intensities = np.subtract(intensities_0, intensities_1)
        max = np.max(average_intensities)
        min = np.min(average_intensities)
        average_intensities = np.divide(average_intensities, self._exposure_time)
        # for i in range(int(timestamp - self._exposure_time/2), int(timestamp + self._exposure_time/2)):
        #     events_slice = self._getEventsInTimeFrame(timestamp, i)
        #     average_intensities = np.add(average_intensities, events_slice)
        # average_intensities = np.divide(average_intensities, self._exposure_time)
        return average_intensities
    
    def _getEventsInTimeFrame(self, start_time: int, end_time: int) -> np.ndarray:
        start_idx = np.argwhere(self._events_vector[:, 0] >= start_time)[0][0]
        end_idx = np.argwhere(self._events_vector[:, 0] <= end_time)[-1][0]
        return self._cumulateEvents(start_idx, end_idx)

        
    def _getNearestImage(self, timestamp: int, temporal_thresh: int = None) -> tuple:
        times = np.asarray(self._img_meta_data.iloc[:, 0])
        min_diff = timestamp - (times[times <= timestamp]).max()
        min_diff_idx = np.argmax(times[times <= timestamp])
        if (temporal_thresh is not None) and (min_diff > temporal_thresh):
            return 0, np.zeros(1)
        if self._frame_generated and (min_diff_idx <= self._img_idx):
            return timestamp - self._last_timestamp_event, np.log(np.add(
                np.divide(self._intermediate_img, 255), self._delta_epsilon))
        color_frame = cv2.imread(os.path.join(self._img_folder_path, self._img_meta_data.iloc[min_diff_idx, 1]))
        greyscale = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        return timestamp - self._img_meta_data.iloc[min_diff_idx, 0], np.log(
            np.add(np.divide(greyscale, 255), self._delta_epsilon))

    def generateFrames(self, num_of_events: int, break_count: int = None) -> np.ndarray:
        count = 0
        end = 0
        if break_count is not None:
            end = break_count
        else:
            end = self._img_meta_data.shape[0]*int(self._avg_frequency_imgs/(self._avg_frequency_events*num_of_events))
        for i in tqdm(range(count, end * num_of_events, num_of_events)):
            timestamp, base_image = self._getNearestImage(self._events_vector[i][0])
            if timestamp >= self._exposure_time/2:
                base_image = self.deblurImage(base_image, timestamp)
            event_stream = self._cumulateEvents(i, num_of_events)
            new_frame_ln = np.add(base_image, np.multiply(event_stream, self._c_threshhold))
            new_frame = np.multiply(np.subtract(np.exp(new_frame_ln), self._delta_epsilon), 255)
            new_frame[new_frame > 255.0] = 255.0
            new_frame[new_frame < 0.0] = 0.0
            new_frame = np.asarray(new_frame, dtype=np.uint8)
            self._frame_generated = True
            self._intermediate_img = new_frame
            # cv2.imshow("intermediate image", self._intermediate_img)
            self._last_timestamp_event = self._events_vector[i][0]
            yield new_frame


if __name__ == "__main__":
    events_obj = Events(0.15, 0.1, 833333)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    events_obj.loadEventsFromFiles("../data/", 1)
    events_obj.loadImgMetaData("../data/images.csv", (720, 1280), 1200, "../data/", max_images=3)
    blurred_img = cv2.imread("../data/frames_0000000003.png", cv2.IMREAD_GRAYSCALE)
    blurred_img = cv2.GaussianBlur(blurred_img, (5, 5), 0)
    timestamp = 1666667
    sharp_img = events_obj.deblurImage(blurred_img, timestamp)
    cv2.imshow("blurred image", blurred_img)
    cv2.imshow("sharp image", sharp_img)
    cv2.waitKey(0)
    a = 0
    
    # img_iterator = events_obj.generateFrames(300)
    # i = 0
    # for img in img_iterator:
    #     # cv2.imwrite("../generated_imgs/gen_frame_" + str(i) + ".png", img)
    #     i += 1
