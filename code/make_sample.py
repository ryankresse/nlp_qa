import os
import pdb

def readNumLines(path, numLines):
    lines = []
    with open(path, 'r') as f:
        for i, l in enumerate(f):
            if i == numLines - 1: break
            tokens = l.strip().split()
            lines.append(tokens)
    return lines

def writeLines(path, lines):
    with open(path, 'w') as f:
        for l in lines:
            f.write(' '.join(l) + '\n')

def readWriteLines(readPath, writePath, numLines):
    toWrite = readNumLines(readPath, numLines)
    writeLines(writePath, toWrite)

if __name__ == '__main__':
    TRAIN_NUM = 1000
    VAL_NUM =  300
    data_path = os.path.join(os.getcwd(), 'data', 'squad')

    readWriteLines(os.path.join(data_path, 'train.ids.context'), os.path.join(data_path, 'samp.train.ids.context'), TRAIN_NUM)
    readWriteLines(os.path.join(data_path, 'train.ids.question'), os.path.join(data_path, 'samp.train.ids.question'), TRAIN_NUM)
    readWriteLines(os.path.join(data_path, 'train.span'), os.path.join(data_path, 'samp.train.span'), TRAIN_NUM)

    readWriteLines(os.path.join(data_path, 'val.ids.context'), os.path.join(data_path, 'samp.val.ids.context'), VAL_NUM)
    readWriteLines(os.path.join(data_path, 'val.ids.question'), os.path.join(data_path, 'samp.val.ids.question'), VAL_NUM)
    readWriteLines(os.path.join(data_path, 'val.span'), os.path.join(data_path, 'samp.vals.span'), VAL_NUM)
