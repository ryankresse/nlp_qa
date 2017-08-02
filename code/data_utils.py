import numpy as np
import pdb

def read_token_data_file(path):
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


def read_text_data_file(path):
    gathered = []
    with open(path, 'r') as f:
        for i, l in enumerate(f):
                gathered.append(l.strip())
    return np.array(gathered)


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
    data = read_token_data_file(path)
    data = pad_sequences(data, length, pad_token)
    return np.array(clip_sequences(data, length), dtype=np.int32)

def read_and_pad(path, length, pad_token):
    data = read_token_data_file(path)
    return np.array(pad_sequences(data, length, pad_token),  dtype=np.int32)

def invert_map(answer_map):
    return {v: k for k, v in answer_map.iteritems()}


def make_dense_answers(ans_span, cont_length):
    ans = np.zeros((ans_span.shape[0], cont_length), dtype=np.float64)

    ans_span = np.clip(ans_span, None, cont_length-1) # need the minus one because of zero indexing

    for i in np.arange(ans_span.shape[0]):
        low = ans_span[i, 0]
        high = ans_span[i, 1]
        #single word answer
        if  low == high:
            ans[i, low] = 1.0
        else:
            # need the plus one because in np, the ending slice is not included,
            #whereas in our data set the ending slice indicates the last word included in the answer
            ans[i, low:high+1] = 1


    return ans
