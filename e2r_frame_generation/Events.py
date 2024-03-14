import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
from scipy.signal import convolve2d
import h5py
import nexusformat.nexus as nx

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
        self._events_vector = np.empty((0, 4), dtype=np.uint32)
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
        self._img_data = []
        self._gt_img_data = []

        self._prev_start_idx = 0

    def getImage(self, img_idx: any) -> np.ndarray:
        if isinstance(img_idx, int):
            return self._img_data[img_idx]
        elif isinstance(img_idx, str):
            return cv2.imread(os.path.join(self._img_folder_path, img_idx), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Invalid image index")

    def loadEventsFromFile(self, path_to_file: str, timestamp_col: int = 0, append: bool = False) -> None:
        file = path_to_file
        dtype = file.split(".")[-1]
        if (not os.path.isfile(file)): # or (filename[-3:] != "npy") or (filename[:5] != "event")
            raise ValueError("Invalid file path")
        if dtype == "npy":
            events = np.load(file, allow_pickle=True)
        elif dtype == "hdf5" or dtype == "h5":
            data = h5py.File(file, 'r')
            ts = np.asarray(data['events']['ts'])
            xs = np.asarray(data['events']['xs'])
            ys = np.asarray(data['events']['ys'])
            ps = np.asarray(data['events']['ps'])
            # if len(np.unique(ts)) != len(ts):
            #     print(f"Duplicate timestamps in {filename}")
            #     duplicate_indices = np.where(np.diff(ts) == 0)[0]
            #     for idx in duplicate_indices:
            #         ts[idx+1:] += 1
            events = np.column_stack((ts, xs, ys, ps))
        else:
            raise ValueError("Invalid file format")
        if append:
            self._events_vector = np.concatenate((self._events_vector, events), axis=0)
        else:
            self._events_vector = events
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
        elif path_to_file.split(".")[-1] == "hdf5" or path_to_file.split(".")[-1] == "h5":
            self._imgs_in_data = True
            data = nx.nxload(path_to_file)
            print(data.tree)
            self._img_resolution = data.attrs['sensor_resolution']
            images = data['images']
            sharp_images = data['sharp_images']
            timestamps = []
            self._exposure_time = 0.0
            for image in images:
                # im = cv2.cvtColor(np.asarray(images[image]), cv2.COLOR_BGR2GRAY)
                # sharp_im = cv2.cvtColor(np.asarray(sharp_images[image]), cv2.COLOR_BGR2GRAY)
                # cv2.imshow("image", im)
                # cv2.imshow("sharp image", sharp_im)
                # cv2.waitKey(0)
                timestamps.append(images[image].attrs['timestamp'])
                self._exposure_time += images[image].attrs['exposure_time']
                self._img_data.append(cv2.cvtColor(np.asarray(images[image]), cv2.COLOR_BGR2GRAY))
                self._gt_img_data.append(cv2.cvtColor(np.asarray(sharp_images[image]), cv2.COLOR_BGR2GRAY))
            self._exposure_time /= len(images)
            if max_images is None:
                max_images = len(self._img_data)
            timestamps = np.asarray(timestamps)[:max_images]
            self._img_data = self._img_data[:max_images]
            self._img_meta_data = pd.DataFrame({'timestamp': timestamps, 'img_position': np.arange(len(timestamps))})
            self._avg_frequency_imgs = 1/(self._exposure_time * 1e-6)
        else:
            raise ValueError("Invalid file format")
        
        self._img_data = np.asarray(self._img_data)
        if max_images is not None:
            self._img_meta_data = self._img_meta_data.iloc[:max_images, :]
        if self._img_resolution is None:
            self._img_resolution = img_resolution
        if self._avg_frequency_imgs is None:
            self._avg_frequency_imgs = video_framerate
        self._img_folder_path = folder_path
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
        indices = (events_slice[:, 2].astype(np.int32), events_slice[:, 1].astype(np.int32))
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

        average_intensities = np.divide(np.add(integrand_0, integrand_1), 2*self._mean_events_diff)

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
        if self._img_data.size == 0:
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
        while start_idx <= self._prev_start_idx:
            start_idx += num_events
        end_idx = start_idx + num_events if start_idx + num_events < self._events_vector.shape[0] else self._events_vector.shape[0] - 1
        events = self._cumulateEvents(start_idx, end_idx), self._events_vector[end_idx][0]
        self._prev_start_idx = start_idx
        return events

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
                frames.append(self.convertIntensitiesToFrame(base_image).astype(np.uint8))
                while(timestamp < next_timestamp):
                    if timestamp == 21964189.0:
                        a = 0
                    event_stream, end_timestamp = self._getEventsFromTime(timestamp, num_events)
                    base_image = np.add(base_image, np.multiply(event_stream, self._c_threshhold))
                    new_frame = self.convertIntensitiesToFrame(base_image).astype(np.uint8)
                    timestamp = end_timestamp
                    frames.append(new_frame)
                yield frames
                pbar.update(1)

if __name__ == "__main__":
    events_obj = Events(0.17, delta_eps=0.1)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    events_obj.loadEventsFromFile("../data/1-3-circle-50-zju.h5", timestamp_col=0) # /home/eshan/Downloads/e_data/flying/
    events_obj.loadImgMetaData("../data/1-3-circle-50-zju.h5") # ../data/images.csv
    psnrs = []
    psnr_gt = []
    vid = []
    vid_gt = []
    count = 0
    for timestamp, img in events_obj._img_meta_data.values:
        blurred_img = events_obj.getImage(int(img))
        original_img = events_obj._gt_img_data[int(img)]
        vid.append(blurred_img)
        vid_gt.append(original_img)
        deblurred_img = events_obj.convertIntensitiesToFrame(events_obj.deblurImage(blurred_img, timestamp)).astype(np.uint8)
        if count % 20 == 0:
            cv2.imwrite(f"original_circle_{count}.png", original_img)
            cv2.imwrite(f"blurred_circle_{count}.png", blurred_img)
            cv2.imwrite(f"deblurred_circle_{count}.png", deblurred_img)
        psnr = cv2.PSNR(blurred_img, deblurred_img)
        psnrs.append(psnr)
        psnr_gt.append(cv2.PSNR(blurred_img, original_img))
        count += 1
    print(f"Mean psnr through deblurring: {np.mean(psnrs)}")
    print(f"Std psnr through deblurring: {np.std(psnrs)}")
    print(f"Mean psnr gt: {np.mean(psnr_gt)}")
    print(f"Std psnr gt: {np.std(psnr_gt)}")
    frames_collection = events_obj.generateFramesByImg(100)
    video = []
    for frames in frames_collection:
        video.extend(frames)
    # for frame in video:
    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = np.array(video)
    out = cv2.VideoWriter('./output_circle.mp4', fourcc, float(events_obj._avg_frequency_imgs*16.5), (events_obj._img_resolution[1], events_obj._img_resolution[0]), isColor=False)
    [out.write(frame) for frame in video]
    orig_out = cv2.VideoWriter('./original_circle.mp4', fourcc, float(events_obj._avg_frequency_imgs/12), (events_obj._img_resolution[1], events_obj._img_resolution[0]), isColor=False)
    [orig_out.write(frame) for frame in vid]
    gt_out = cv2.VideoWriter('./gt_circle.mp4', fourcc, float(events_obj._avg_frequency_imgs/12), (events_obj._img_resolution[1], events_obj._img_resolution[0]), isColor=False)
    [gt_out.write(frame) for frame in vid_gt]
    out.release()
    orig_out.release()
    gt_out.release()
    
