import numpy as np


def get_slice(video_duration, target_number_segment=32, one_segment_size=16):
    
    end_point = video_duration - one_segment_size
    idx_float_list = np.arange(
        0, end_point, 
        step=end_point/(target_number_segment-1),
        dtype=float)
    idx_floor = np.floor(idx_float_list)
    idx_int = idx_floor.astype(np.int32)
    slice_start = np.append(idx_int, [end_point])
    return slice_start
