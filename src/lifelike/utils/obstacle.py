import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R


def obstacles_in_frame(frames, frame_rate):
    base_poss = []
    base_orns = []
    assert len(frames[0]) == 19
    for frame in frames:
        base_poss.append(frame[0:3])
        base_orns.append(frame[3:7])
    base_poss = np.array(base_poss)
    base_orns = np.array(base_orns)
    base_heights = base_poss[:, 2]
    peak_ids, _ = find_peaks(base_heights, height=0.5, distance=120)
    if len(peak_ids) == 0:
        return None
    else:
        ob_poss = base_poss[peak_ids]
        ob_orn_oth = base_orns[peak_ids]
        peak_time = peak_ids / frame_rate
        ob_dict = {'pos': ob_poss, 'orn_otho': ob_orn_oth, 'time': peak_time}
        return ob_dict


def get_obstacle_pose(pos, orn):
    pos = [pos[0], pos[1], 0]
    # orn = obstacle_orn_otho[0, :]
    mat = R.from_quat(orn).as_matrix()
    yaw = np.arctan2(mat[1, 0], mat[0, 0])
    orn = R.from_euler('xyz', [0, 0, yaw]).as_quat()
    return pos, orn
