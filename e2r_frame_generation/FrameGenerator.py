import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import h5py
import nexusformat.nexus as nx
import skimage
import time

class FrameGenerator:
    """
    Class to deblur and generate additional frames from input video and corresponding events.

    Attributes:
        _events_vector (numpy.ndarray): Array to store event data.
        _c_threshhold (float): Threshold value for event accumulation.
        _delta_epsilon (float): Value to add to intensities during conversion.
        _exposure_time (float): Exposure time for images.
        _mean_events_diff (float): Mean time difference between events.
        _last_timestamp_event (float): Timestamp of the last event.
        _last_timestamp_img (float): Timestamp of the last image.
        _img_meta_data (pandas.DataFrame): DataFrame to store image metadata.
        _img_resolution (tuple): Resolution of the images.
        _rgb_framerate (int): Framerate of the RGB video.
        _img_folder_path (str): Path to the folder containing images.
        _avg_frequency_imgs (float): Average frequency of images.
        _intermediate_img (numpy.ndarray): Intermediate image array.
        _frame_generated (bool): Flag indicating if a frame has been generated.
        _img_idx (int): Index of the current image.
        _deblurred_img (numpy.ndarray): Deblurred image array.
        _timestamp_col (int): Column index of the timestamp in the event data.
        _img_data (list): List to store image data.
        _gt_img_data (list): List to store ground truth image data.
        _prev_start_idx (int): Index of the previous start time.
    """

    def __init__(self, c_threshold: float, exposure_time: float = None, delta_eps: float = 0) -> None:
        """
        Initializes the Events object.

        Args:
            c_threshhold (float): Intenstiy threshold value for event accumulation.
            exposure_time (float, optional): Exposure time for images. Defaults to None.
            delta_eps (float, optional): Delta value to add to intensities during conversion. Defaults to 0.
        """
        self._events_vector = np.empty((0, 4), dtype=np.uint32)
        self._c_threshold = c_threshold
        self._delta_epsilon = delta_eps
        self._exposure_time = exposure_time
        self._initAttributes()

    def _initAttributes(self) -> None:
        """
        Initializes the private attributes of the Events object.
        """
        # Initialize attributes
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
        """
        Retrieves an image from the image data based on the given index.

        Parameters:
            img_idx (int or str): The index of the image to retrieve. If an integer is provided, it returns the image from the
                                  `_img_data` list. If a string is provided, it reads the image from the `_img_folder_path`
                                  directory using OpenCV and returns it as a grayscale image.

        Returns:
            np.ndarray: The retrieved image as a NumPy array.

        Raises:
            ValueError: If the provided image index is invalid.

        """
        if isinstance(img_idx, int):
            return self._img_data[img_idx]
        elif isinstance(img_idx, str):
            return cv2.imread(os.path.join(self._img_folder_path, img_idx), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Invalid image index")

    def loadEventsFromFile(self, path_to_file: str, timestamp_col: int = 0, append: bool = False) -> None:
        """
        Load events from a file and store them in the events vector.

        Args:
            path_to_file (str): The path to the hdf5 or numpy file containing the events.
            timestamp_col (int, optional): The column index of the timestamp in the file. Defaults to 0.
            append (bool, optional): Whether to append the events to the existing events vector. Defaults to False.

        Raises:
            ValueError: If the file path is invalid or the file format is not supported.

        Returns:
            None
        """
        file = path_to_file
        dtype = file.split(".")[-1]
        if (not os.path.isfile(file)):
            raise ValueError("Invalid file path")
        if dtype == "npy":
            events = np.load(file, allow_pickle=True)
        elif dtype == "hdf5" or dtype == "h5":
            data = h5py.File(file, 'r')
            ts = np.asarray(data['events']['ts'])
            xs = np.asarray(data['events']['xs'])
            ys = np.asarray(data['events']['ys'])
            ps = np.asarray(data['events']['ps'])
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
        """
        Loads image metadata from a file.

        Args:
            path_to_file (str): The path to the file containing the image metadata. 
            img_resolution (tuple, optional): The resolution of the images. Defaults to None. Can be infered from HDF5 file.
            video_framerate (int, optional): The framerate of the video. Defaults to None. If none, computed from image timestamps
            folder_path (str, optional): The path to the folder containing the images. Defaults to None. Necessary if images are stored externally and need to be loaded in.
            max_images (int, optional): The maximum number of images to load. Defaults to None.

        Raises:
            ValueError: If the file format is invalid or if image resolution is not provided for certain file formats.

        Returns:
            None
        """
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
            self._img_resolution = data.attrs['sensor_resolution']
            images = data['images']
            sharp_images = data['sharp_images']
            timestamps = []
            self._exposure_time = 0.0
            for image in images:
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
        """
        Converts image intensities to a frame of pixel values. Events should be multiplied by the intensity threshold before applying this method

        Args:
            intensities (np.ndarray): Array of intensity values.

        Returns:
            np.ndarray: Array of pixel values representing the frame.
        """
        return_val = np.multiply(np.subtract(np.exp(intensities), self._delta_epsilon), 255)
        return_val[return_val > 255.0] = 255.0
        return_val[return_val < 0.0] = 0.0
        return return_val
    
    def convertFrameToIntensities(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts a frame of pixel values to intensities using a logarithmic transformation. Intensity threshold can be used to convert to events

        Args:
            frame (np.ndarray): The input frame of pixel values.

        Returns:
            np.ndarray: The converted frame of intensities.
        """
        return np.log(np.add(np.divide(frame, 255), self._delta_epsilon))

    def _cumulateEvents(self, start_ind: int, end_ind: int = None, num_of_events: int = 0):
        """
        Cumulates events within a specified range and returns the resulting event array.

        Args:
            start_ind (int): The starting index of the events to be cumulated.
            end_ind (int, optional): The ending index of the events to be cumulated. If not provided, `num_of_events` will be used instead. Defaults to None.
            num_of_events (int, optional): The number of events to be cumulated starting from `start_ind`. This parameter is used only when `end_ind` is not provided. Defaults to 0.

        Returns:
            numpy.ndarray: Cumulated events as numpy matrix of the same resolution as images.

        """
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

    def deblurImage(self, blurred_img: np.ndarray, timestamp: int, compute_integral: bool = False) -> np.ndarray:
        """
        Deblurs the given image using the average intensities computed from events around the specified timestamp.

        Args:
            blurred_img (np.ndarray): The blurred image to be deblurred.
            timestamp (int): The timestamp at which the average intensities are computed.
            compute_integral (bool, optional): Whether to compute the integral of intensities. Defaults to False.

        Returns:
            np.ndarray: The deblurred image.

        """
        average_intensities = np.log(self._computeAverageIntensities(timestamp, compute_integral))
        blurred_img = self.convertFrameToIntensities(blurred_img)
        return np.subtract(blurred_img, average_intensities)

    def _computeAverageIntensities(self, timestamp: int, compute_integral:bool) -> np.ndarray:
        """
        Compute the average intensity changes using event within a given time frame.

        Args:
            timestamp (int): The timestamp of the events.
            compute_integral (bool): Flag indicating whether to compute the integral of events. If False, average intensities are approximated. Approximation is considerably faster.

        Returns:
            np.ndarray: The array of average intensities.

        """
        events_slice_0 = self._getEventsInTimeFrame(timestamp, (timestamp + self._exposure_time/2))
        events_slice_1 = self._getEventsInTimeFrame(timestamp, (timestamp - self._exposure_time/2))

        if compute_integral:
            average_intensities = np.zeros(self._img_resolution, dtype=np.double)
            for i in tqdm(range(int(timestamp - self._exposure_time/2), int(timestamp + self._exposure_time/2))):
                average_intensities += np.exp(np.multiply(self._getEventsInTimeFrame(timestamp, i), self._c_threshold))
            average_intensities = np.divide(average_intensities, self._exposure_time)
                
        else:
            integrand_0 = np.divide(np.exp(np.multiply(events_slice_0, self._c_threshold)),self._c_threshold)
            integrand_1 = np.divide(np.exp(np.multiply(events_slice_1, self._c_threshold)),self._c_threshold)

            average_intensities = np.divide(np.add(integrand_0, integrand_1), 2*self._mean_events_diff)

        return average_intensities
    
    def _getEventsInTimeFrame(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Returns cumulated events within the specified time frame.

        Args:
            start_time (float): The start time of the time frame.
            end_time (float): The end time of the time frame.

        Returns:
            np.ndarray: An array containing the events within the specified time frame.
        """
        start_idx = np.searchsorted(self._events_vector[:, 0], start_time, side='left')
        end_idx = np.searchsorted(self._events_vector[:, 0], end_time, side='right') - 1
        return self._cumulateEvents(start_idx, end_idx)
        
    def _getNearestImage(self, timestamp: int, temporal_thresh: int = None, denoise:bool = False) -> tuple:
        """
        Returns the nearest image and its corresponding timestamp based on the given timestamp.

        Args:
            timestamp (int): The timestamp to find the nearest image for.
            temporal_thresh (int, optional): The temporal threshold to consider. If the difference between the nearest image's timestamp and the given timestamp is greater than this threshold, a default value is returned. Defaults to None.
            denoise (bool, optional): Flag to indicate whether to apply denoising to the image. Defaults to False.

        Returns:
            tuple: A tuple containing the time difference between the given timestamp and the nearest image's timestamp, and the deblurred grayscale image.

        """
        times = np.asarray(self._img_meta_data.iloc[:, 0])
        min_diff = timestamp - (times[times <= timestamp]).max()
        min_diff_idx = np.argmax(times[times <= timestamp])
        if (temporal_thresh is not None) and (min_diff > temporal_thresh):
            return 0, np.zeros(1)
        # if self._frame_generated and (min_diff_idx <= self._img_idx):
        #     return timestamp - self._last_timestamp_event, self.convertFrameToIntensities(self._intermediate_img)
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
        """
        Retrieves a tuple of events and the end time from the events vector based on the given start time and number of events.

        Args:
            start_time (int): The start time to search for events.
            num_events (int): The number of events to retrieve.

        Returns:
            tuple: A tuple containing the accumulated events and the end time of the retrieved events.
        """
        start_idx = np.searchsorted(self._events_vector[:, 0], start_time, side='left')
        while start_idx <= self._prev_start_idx:
            start_idx += num_events
        end_idx = start_idx + num_events if start_idx + num_events < self._events_vector.shape[0] else self._events_vector.shape[0] - 1
        events = self._cumulateEvents(start_idx, end_idx), self._events_vector[end_idx][0]
        self._prev_start_idx = start_idx
        return events

    def generateFrames(self, num_of_events: int, break_count: int = None) -> np.ndarray:
        """
        Generates frames based on the given number of events. Events are iterated over to generate intermediate frames.

        Args:
            num_of_events (int): The number of events per frame.
            break_count (int, optional): The number of frames to generate. If not provided, frames will be generated until the end of the image metadata.

        Yields:
            np.ndarray: The generated frames one by one.
        """
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
            new_frame_ln = np.add(base_image, np.multiply(event_stream, self._c_threshold))
            new_frame = self.convertIntensitiesToFrame(new_frame_ln).astype(np.uint8)
            self._frame_generated = True
            self._intermediate_img = new_frame
            # cv2.imshow("intermediate image", self._intermediate_img)
            self._last_timestamp_event = self._events_vector[i][0]
            yield new_frame

    def generateFramesByImg(self, num_events:int, break_count = None) -> np.ndarray:
        """
        Generates intermediate frames from events using existing frames as refernece. Recommended method to reconstruct high frame rate video.

        Args:
            num_events (int): The number of events to cumulate to generate a frame.
            break_count (int, optional): The maximum number of frames to generate. Defaults to None.

        Yields:
            np.ndarray: Individual frames of higher frame rate video choronologically
        """
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
                    event_stream, end_timestamp = self._getEventsFromTime(timestamp, num_events)
                    base_image = np.add(base_image, np.multiply(event_stream, self._c_threshold))
                    new_frame = self.convertIntensitiesToFrame(base_image).astype(np.uint8)
                    timestamp = end_timestamp
                    frames.append(new_frame)
                yield frames
                pbar.update(1)
        self._prev_start_idx = 0

    def evaluateLatentFrames(self, num_events:int):
        """
        Evaluates the latent frames by reconstructing the frames using event data and calculating the structural similarity index (SSIM) between the original frames and the reconstructed frames.

        Args:
            num_events (int): The number of events to consider for each reconstructed frame.

        Returns:
            tuple: A tuple containing the following:
                - ssim_vals (list): A list of SSIM values calculated for each reconstructed frame.
                - reconstructed_frames (list): A list of reconstructed frames.
                - orig_frames (list): A list of original frames.
        """
        ssim_vals = []
        orig_frames = []
        reconstructed_frames = []
        for i in tqdm(range(1, self._img_meta_data.shape[0])):
            next_timestamp = self._img_meta_data.iloc[i, 0]
            orig_unblurred_img = self._gt_img_data[i]
            timestamp = self._img_meta_data.iloc[i-1, 0]
            orig_frames.append(orig_unblurred_img)
            _, base_img_eval = self._getNearestImage(timestamp)
            while (timestamp < next_timestamp):
                event_stream, end_timestamp = self._getEventsFromTime(timestamp, num_events)
                base_img_eval = np.add(base_img_eval, np.multiply(event_stream, self._c_threshold))
                timestamp = end_timestamp
            reconstructed_img = self.convertIntensitiesToFrame(base_img_eval).astype(np.uint8)
            reconstructed_frames.append(reconstructed_img)
            ssim = skimage.metrics.structural_similarity(orig_unblurred_img, reconstructed_img, full=True)[0]
            ssim_vals.append(ssim)
        
        self._prev_start_idx = 0
        return ssim_vals, reconstructed_frames, orig_frames
            

if __name__ == "__main__":
    events_obj = FrameGenerator(0.17, delta_eps=0.0) # Events object to perform operations.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    events_obj.loadEventsFromFile("../data/1-3-circle-50-zju.h5", timestamp_col=0) # dataset contains events as well as blurred and ground truth image frames
    events_obj.loadImgMetaData("../data/1-3-circle-50-zju.h5")
# ------------------- Latent image reconstruction evaluation -------------------
    ssims, reconstructeds, origs = events_obj.evaluateLatentFrames(100)

    print(f"Average ssim of latent-orignal images: {np.mean(ssims)}")
    print(f"Standard deviation ssim of latent-orignal frames: {np.std(ssims)}")

    # save exemplars of original and reconstructed frames
    for i in range(0, len(origs), 20):
        cv2.imwrite(f"original_frame_circ_{i}.png", origs[i])
        cv2.imwrite(f"reconstructed_frame_circ_{i}.png", reconstructeds[i])

# --------------------- Image deblurring evaluation -----------------------------
    psnrs = [] # list to store psnr values of deblurred images using approximation
    ssims = [] # list to store ssim values of deblurred images using approximation
    psnrs_int = [] # list to store psnr values of deblurred images using integral
    ssims_int = [] # list to store ssim values of deblurred images using integral
    psnr_gt = [] # list to store psnr values of ground truth images
    ssims_gt = [] # list to store ssim values of ground truth images
    vid = [] # list to store deblurred images
    vid_gt = [] # list to store ground truth images
    times_int = [] # list to store time taken for deblurring using integral
    times = [] # list to store time taken for deblurring using approximation
    count = 0
    for timestamp, img in events_obj._img_meta_data.values:
        blurred_img = events_obj.getImage(int(img))
        original_img = events_obj._gt_img_data[int(img)] # ground truth sharp image
        vid.append(blurred_img)
        vid_gt.append(original_img)
        start_time = time.time()
        deblurred_img = events_obj.convertIntensitiesToFrame(events_obj.deblurImage(blurred_img, timestamp)).astype(np.uint8)
        end_time = time.time()
        times.append(end_time - start_time)
        start_time = time.time()
        deblurred_int_img = events_obj.convertIntensitiesToFrame(events_obj.deblurImage(blurred_img, timestamp, True)).astype(np.uint8)
        end_time = time.time()
        times_int.append(end_time - start_time)
        # saving exemplary deblurred, blurred and ground truth images.
        if count % 20 == 0:
            cv2.imwrite(f"original_circ_{count}.png", original_img)
            cv2.imwrite(f"blurred_circ_{count}.png", blurred_img)
            cv2.imwrite(f"deblurred_circ_{count}.png", deblurred_img)
            cv2.imwrite(f"deblurred_int_circ_{count}.png", deblurred_int_img)
        psnr_int = cv2.PSNR(original_img, deblurred_int_img)
        psnrs_int.append(psnr_int)
        psnr = cv2.PSNR(original_img, deblurred_img)
        psnrs.append(psnr)
        psnr_gt.append(cv2.PSNR(original_img, blurred_img))
        ssim = skimage.metrics.structural_similarity(original_img, deblurred_img, full=True)[0]
        ssims.append(ssim)
        ssim_int = skimage.metrics.structural_similarity(original_img, deblurred_int_img, full=True)[0]
        ssims_int.append(ssim_int)
        ssim_gt = skimage.metrics.structural_similarity(blurred_img, original_img, full=True)[0]
        ssims_gt.append(ssim_gt)
        count += 1
    print(f"Mean time for deblurring: {np.mean(times)}")
    print(f"Std time for deblurring: {np.std(times)}")
    print(f"Mean time for deblurring with integral: {np.mean(times_int)}")
    print(f"Std time for deblurring with integral: {np.std(times_int)}")
    print(f"Mean psnr through deblurring: {np.mean(psnrs)}")
    print(f"Std psnr through deblurring: {np.std(psnrs)}")
    print(f"Mean psnr gt: {np.mean(psnr_gt)}")
    print(f"Std psnr gt: {np.std(psnr_gt)}")
    print(f"Mean psnr through deblurring with integral: {np.mean(psnrs_int)}")
    print(f"Std psnr through deblurring with integral: {np.std(psnrs_int)}")
    print(f"Mean ssim through deblurring: {np.mean(ssims)}")
    print(f"Std ssim through deblurring: {np.std(ssims)}")
    print(f"Mean ssim through deblurring with integral: {np.mean(ssims_int)}")
    print(f"Std ssim through deblurring with integral: {np.std(ssims_int)}")
    print(f"Mean ssim gt: {np.mean(ssims_gt)}")
    print(f"Std ssim gt: {np.std(ssims_gt)}")

# --------------------- High frame rate video generation ----------------------------------------
    
    frames_collection = events_obj.generateFramesByImg(100) # generator object containing new frames
    video = []
    for frames in frames_collection:
        video.extend(frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = np.array(video)
    
# --------------------- Saving the generated videos ----------------------------------------------
    slow_down_factor = 10 # Determines speed of original videos. Change if timebase error for MPEG 4 Standard recieved
    output_fps = (len(video)/len(vid)) * (events_obj._avg_frequency_imgs / slow_down_factor) # to assure length of all videos is same
    out = cv2.VideoWriter('./output_circ.mp4', fourcc, float(output_fps), (events_obj._img_resolution[1], events_obj._img_resolution[0]), isColor=False)
    [out.write(frame) for frame in video]
    orig_out = cv2.VideoWriter('./original_circ.mp4', fourcc, float(events_obj._avg_frequency_imgs/slow_down_factor), (events_obj._img_resolution[1], events_obj._img_resolution[0]), isColor=False)
    [orig_out.write(frame) for frame in vid]
    gt_out = cv2.VideoWriter('./gt_circ.mp4', fourcc, float(events_obj._avg_frequency_imgs/slow_down_factor), (events_obj._img_resolution[1], events_obj._img_resolution[0]), isColor=False)
    [gt_out.write(frame) for frame in vid_gt]
    out.release()
    orig_out.release()
    gt_out.release()
    
