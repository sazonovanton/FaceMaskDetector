import numpy as np

def dec_to_bin(x):
    return str(bin(x)[2:])

def reduce_output(det_by_frame, fps):
    det_by_frame = np.array(det_by_frame)
    length = len(det_by_frame)

    # target value - 200 bins on x axis
    # closest power of 2 is
    if length > 200:
        n = 2**len(dec_to_bin(int(length / 200)))
        # reduce size in n times
        y = det_by_frame.reshape(-1, n).sum(axis=1)
    else:
        y = det_by_frame
        n = 1
    x = (np.arange(0,len(y))/(fps/n)).tolist()

    return y.tolist(), x


