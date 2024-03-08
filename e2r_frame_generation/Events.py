import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
from scipy.signal import convolve2d
import h5py

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

    def __init__(self, c_threshhold: float, exposure_time: float = None, delta_eps: float = 0) -> None:
        self._events_vector = np.empty((1, 4), dtype=np.uint32)
        self._c_threshhold = c_threshhold
        self._delta_epsilon = delta_eps
        self._exposure_time = exposure_time
        self._initAttributes()

    def _initAttributes(self) -> None:
        self._mean_events_diff = .0
        self._last_timestamp_event = 0
        self._last_timestamp_img = 0

        self._img_meta_data = pd.DataFrame()
        self._img_resolution = None
        self._rgb_framerate = 0
        self._img_folder_path = ""
        self._avg_frequency_imgs = .0

        self._intermediate_img = np.empty((720, 1280), dtype=np.uint8)
        self._frame_generated = False
        self._img_idx = 0
        self._deblurred_img = np.empty((720, 1280), dtype=np.uint8)
        self._timestamp_col = None
        self._img_data = None

    def getImage(self, img_idx: any) -> np.ndarray:
        if isinstance(img_idx, int):
            return self._img_data[img_idx]
        elif isinstance(img_idx, str):
            return cv2.imread(os.path.join(self._img_folder_path, img_idx), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Invalid image index")

    def loadEventsFromFiles(self, path_to_folder: str, file_count: int = None, timestamp_col: int = 0) -> None:
        i = 0
        for filename in sorted(os.listdir(path_to_folder)):
            file = os.path.join(path_to_folder, filename)
            dtype = filename.split(".")[-1]
            if (not os.path.isfile(file)): # or (filename[-3:] != "npy") or (filename[:5] != "event")
                continue
            if (not file_count is None) and (i > file_count):
                break
            if dtype == "npy":
                events = np.load(file, allow_pickle=True)
            elif dtype == "hdf5":
                data = h5py.File(file, 'r')
                events = np.asarray(data['davis']['left']['events'])
            else:
                continue
            self._events_vector = np.concatenate((self._events_vector, events), axis=0)
            i += 1
        self._events_vector = np.delete(self._events_vector, 0, axis=0)
        self._events_vector[self._events_vector == 0] = -1
        if timestamp_col != 0:
            self._events_vector[:, [0, timestamp_col]] = self._events_vector[:, [timestamp_col, 0]]
        self._mean_events_diff = np.average(np.diff(self._events_vector[:,0]))
        self._avg_events_freq = 1/self._mean_events_diff

    def loadImgMetaData(self, path_to_file: str, img_resolution: tuple = None, video_framerate: int = None, folder_path: str =  None,
                        max_images=None) -> None:
        if path_to_file.split(".")[-1] == "csv":
            self._img_meta_data = pd.read_csv(path_to_file, header=None)
            if video_framerate is None:
                self._rgb_framerate = int(1 / np.mean(np.diff(self._img_meta_data.iloc[:,0])))
            if img_resolution is None:
                raise ValueError("Image resolution must be provided. Cannot be inferred.")
        elif path_to_file.split(".")[-1] == "txt":
            self._img_meta_data = pd.read_csv(path_to_file, header=None, sep=" ")
            if video_framerate is None:
                self._rgb_framerate = int(1 / np.mean(np.diff(self._img_meta_data.iloc[:,0])))
            if img_resolution is None:
                raise ValueError("Image resolution must be provided. Cannot be inferred.")
        elif path_to_file.split(".")[-1] == "hdf5":
            self._imgs_in_data = True
            data = h5py.File(path_to_file, 'r')
            timestamps = np.asarray(data['davis']['left']['image_raw_ts'])
            if max_images is None:
                max_images = len(timestamps)
            self._img_data = np.asarray(data['davis']['left']['image_raw'])[:max_images]
            self._img_meta_data = pd.DataFrame({'timestamp': timestamps, 'img_position': np.arange(len(timestamps))})
            self._img_resolution = (self._img_data[0].shape[0], self._img_data[0].shape[1])
            self._rgb_framerate = int(1 / np.mean(np.diff(timestamps)))
            a = 0
        else:
            raise ValueError("Invalid file format")
        if max_images is not None:
            self._img_meta_data = self._img_meta_data.iloc[:max_images, :]
        if self._img_resolution is None:
            self._img_resolution = img_resolution
        if self._rgb_framerate is None:
            self._rgb_framerate = video_framerate
        self._img_folder_path = folder_path
        self._exposure_time = np.mean(np.diff(self._img_meta_data.iloc[:,0]))
        self._avg_frequency_imgs = 1/self._exposure_time
        self._intermediate_img = np.full(img_resolution, 128, dtype=np.uint8)


    def convertIntensitiesToFrame(self, intensities: np.ndarray) -> np.ndarray:
        return_val = np.multiply(np.subtract(np.exp(intensities), self._delta_epsilon), 255)
        return_val[return_val > 255.0] = 255.0
        return_val[return_val < 0.0] = 0.0
        return return_val
    
    def convertFrameToIntensities(self, frame: np.ndarray) -> np.ndarray:
        return np.log(np.add(np.divide(frame, 255), self._delta_epsilon))

    def _cumulateEvents(self, start_ind: int, end_ind: int = None, num_of_events: int = 0):
        if end_ind is not None:
            if end_ind <= start_ind:
                events_slice = self._events_vector[end_ind:start_ind]
            else:
                events_slice = self._events_vector[start_ind:end_ind]
        else:
            events_slice = self._events_vector[start_ind:start_ind + num_of_events]

        event_array = np.zeros(self._img_resolution, dtype=np.double)
        indices = (events_slice[:, 1].astype(np.int32), events_slice[:, 2].astype(np.int32))
        np.add.at(event_array, indices, events_slice[:, 3])

        if end_ind is not None and end_ind <= start_ind:
            event_array = np.multiply(event_array, -1)

        return event_array

    def deblurImage(self, blurred_img: np.ndarray, timestamp: int) -> np.ndarray:
        average_intensities = np.log(self._computeAverageIntensities(timestamp))
        blurred_img = self.convertFrameToIntensities(blurred_img)
        return np.subtract(blurred_img, average_intensities)

    def _computeAverageIntensities(self, timestamp: int) -> np.ndarray:
        events_slice_0 = self._getEventsInTimeFrame(timestamp, (timestamp + self._exposure_time/2))
        events_slice_1 = self._getEventsInTimeFrame(timestamp, (timestamp - self._exposure_time/2))

        integrand_0 = np.divide(np.exp(np.multiply(events_slice_0, self._c_threshhold)),self._c_threshhold)
        integrand_1 = np.divide(np.exp(np.multiply(events_slice_1, self._c_threshhold)),self._c_threshhold)

        average_intensities = np.divide(np.add(integrand_0, integrand_1), 2*self._mean_events_diff*10e6)

        return average_intensities
    
    def _getEventsInTimeFrame(self, start_time: float, end_time: float) -> np.ndarray:
        start_idx = np.searchsorted(self._events_vector[:, 0], start_time, side='left')
        end_idx = np.searchsorted(self._events_vector[:, 0], end_time, side='right') - 1
        return self._cumulateEvents(start_idx, end_idx)

    def compute_edge_map(self, event_sequence:np.ndarray, alpha=1.0):
        # Create an exponentially decaying window
        t = np.arange(max(event_sequence.shape))
        window = np.exp(-alpha * np.abs(t - t[:, None]))

        # Convolve the event sequence with the window
        M = convolve2d(event_sequence, window, mode='same')

        # Use the Sobel filter to get a sharper binary edge map
        edge_map = cv2.Sobel(M, cv2.CV_64F, 1, 1)

        return edge_map
        
    def _getNearestImage(self, timestamp: int, temporal_thresh: int = None, denoise:bool = False) -> tuple:
        times = np.asarray(self._img_meta_data.iloc[:, 0])
        min_diff = timestamp - (times[times <= timestamp]).max()
        min_diff_idx = np.argmax(times[times <= timestamp])
        if (temporal_thresh is not None) and (min_diff > temporal_thresh):
            return 0, np.zeros(1)
        if self._frame_generated and (min_diff_idx <= self._img_idx):
            return timestamp - self._last_timestamp_event, self.convertFrameToIntensities(self._intermediate_img)
        if self._img_data is None:
            intensity_img = cv2.imread(os.path.join(self._img_folder_path, self._img_meta_data.iloc[min_diff_idx, 1]))
        else:
            intensity_img = self._img_data[self._img_meta_data.iloc[min_diff_idx, 1]]
        if len(intensity_img.shape) > 2:
            intensity_img = cv2.cvtColor(intensity_img, cv2.COLOR_BGR2GRAY)
        deblurred_greyscale = self.deblurImage(intensity_img, timestamp)
        if denoise:
            deblurred_greyscale = cv2.fastNlMeansDenoising(deblurred_greyscale, None, 10, 7, 21)
        return timestamp - self._img_meta_data.iloc[min_diff_idx, 0], deblurred_greyscale

    def _getEventsFromTime(self, start_time: int, num_events:int) -> tuple:
        start_idx = np.searchsorted(self._events_vector[:, 0], start_time, side='left')
        end_idx = start_idx + num_events if start_idx + num_events < self._events_vector.shape[0] else self._events_vector.shape[0] - 1
        return self._cumulateEvents(start_idx, end_idx), self._events_vector[end_idx][0]

    def generateFrames(self, num_of_events: int, break_count: int = None) -> np.ndarray:
        count = 0
        end = 0
        offset = sum(self._events_vector[:, 0] < self._img_meta_data.iloc[0, 0])
        offset = offset if offset%num_of_events == 0 else offset + num_of_events - (offset%num_of_events)
        if break_count is not None:
            end = break_count + offset
        else:
            end = self._img_meta_data.shape[0]*int(self._exposure_time/(self._mean_events_diff*num_of_events))
        for i in tqdm(range(count, end * num_of_events, num_of_events)):
            if self._events_vector[i][0] < self._img_meta_data.iloc[0, 0]:
                continue
            timestamp, base_image = self._getNearestImage(self._events_vector[i][0])
            event_stream = self._cumulateEvents(i, num_of_events=num_of_events)
            new_frame_ln = np.add(base_image, np.multiply(event_stream, self._c_threshhold))
            new_frame = self.convertIntensitiesToFrame(new_frame_ln).astype(np.uint8)
            self._frame_generated = True
            self._intermediate_img = new_frame
            # cv2.imshow("intermediate image", self._intermediate_img)
            self._last_timestamp_event = self._events_vector[i][0]
            yield new_frame

    def generateFramesByImg(self, num_events:int, break_count = None) -> np.ndarray:
        end = 0
        if break_count is not None:
            end = break_count
        else:
            end = self._img_meta_data.shape[0]
        with tqdm(total=end) as pbar:
            for i in range(0, end):
                frames = []
                timestamp = self._img_meta_data.iloc[i, 0]
                next_timestamp = self._img_meta_data.iloc[i+1, 0] if i+1 < end else timestamp + 1
                _, base_image = self._getNearestImage(timestamp)
                while(timestamp < next_timestamp):
                    event_stream, end_timestamp = self._getEventsFromTime(timestamp, num_events)
                    event_stream = np.multiply(event_stream, self._c_threshhold)
                    base_image = np.add(base_image, np.multiply(event_stream, self._c_threshhold))
                    new_frame = self.convertIntensitiesToFrame(base_image).astype(np.uint8)
                    timestamp = end_timestamp
                    frames.append(new_frame)
                yield frames
                pbar.update(1)

if __name__ == "__main__":
    events_obj = Events(0.02, delta_eps=0.1)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    events_obj.loadEventsFromFiles("/home/eshan/Downloads/e_data/flying/", 1, timestamp_col=2) # /home/eshan/Downloads/e_data/flying/
    events_obj.loadImgMetaData("/home/eshan/Downloads/e_data/flying/indoor_flying1_data.hdf5", (720, 1280), 1200, max_images=1000) # ../data/images.csv
    psnrs = []
    psnr_gt = []
    vid = []
    count = 0
    for timestamp, img in events_obj._img_meta_data.values:
        original_img = events_obj.getImage(int(img))
        blurred_img = cv2.blur(original_img, (5, 5))
        vid.append(original_img)
        deblurred_img = events_obj.convertIntensitiesToFrame(events_obj.deblurImage(blurred_img, timestamp)).astype(np.uint8)
        if count % 500 == 0:
            cv2.imwrite(f"original_drone_{count}.png", original_img)
            cv2.imwrite(f"blurred_drone_{count}.png", blurred_img)
            cv2.imwrite(f"deblurred_drone_{count}.png", deblurred_img)
        psnr = cv2.PSNR(blurred_img, deblurred_img)
        psnrs.append(psnr)
        psnr_gt.append(cv2.PSNR(blurred_img, original_img))
        count += 1
    print(f"Mean psnr through deblurring: {np.mean(psnrs)}")
    print(f"Std psnr through deblurring: {np.std(psnrs)}")
    print(f"Mean psnr gt: {np.mean(psnr_gt)}")
    print(f"Std psnr gt: {np.std(psnr_gt)}")
    frames = events_obj.generateFramesByImg(50, 1000)
    video = []
    for frame in frames:
        video.extend(frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = np.array(video)
    out = cv2.VideoWriter('output_drone.mp4', fourcc, float(events_obj._rgb_framerate * int(len(video)/950)), (events_obj._img_resolution[1], events_obj._img_resolution[0]), isColor=False)
    [out.write(frame) for frame in video]
    orig_out = cv2.VideoWriter('original_drone.mp4', fourcc, float(events_obj._rgb_framerate), (events_obj._img_resolution[1], events_obj._img_resolution[0]), isColor=False)
    [orig_out.write(frame) for frame in vid]
    out.release()
    orig_out.release()
    
