import numpy as np

def read_datafile(path):
    '''
    :params
        :path - path to data file to read
    :returns
        :numpy array of lines of file
    '''
    lines = []
    with open(path, mode="rb") as f:
        for line in f:
            tokens = [int(token) for token in line.strip().split()]
            lines.append(tokens)
    return np.array(lines)


def pad_sequences(data, length, pad_token):
    """
        Pads sequences in data so that they are all of length
        params:
            -data: list of lists containing data
            -length: the length each list in that list of list should be after padding
            -pad_token: the token to use for padding
    """
    ret = []
    for seq in data:
        len_diff = max(length - len(seq), 0)
        ret.append(seq + [pad_token]*len_diff)
    return ret

def clip_sequences(data, length):
    """
        Clips sequences in data so that they are all of length
        params:
            -data: list of lists containing data
            -length: the length each list in that list of in data should be after clipping
    """
    return [seq[:length] for seq in data]

def read_clip_and_pad(path, length, pad_token):
    data = read_datafile(path)
    data = pad_sequences(data, length, pad_token)
    return np.array(clip_sequences(data, length), dtype=np.int32)

def read_and_pad(path, length, pad_token):
    data = read_datafile(path)
    return np.array(pad_sequences(data, length, pad_token),  dtype=np.int32)
