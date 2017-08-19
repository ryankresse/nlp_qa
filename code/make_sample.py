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
    VAL_NUM =  400
    PREPEND = ''
    data_path = os.path.join(os.getcwd(), 'data', 'squad')

    readWriteLines(os.path.join(data_path, 'train.ids.context'), os.path.join(data_path, PREPEND+'samp.train.ids.context'), TRAIN_NUM+1)
    readWriteLines(os.path.join(data_path, 'train.ids.question'), os.path.join(data_path, PREPEND+'samp.train.ids.question'), TRAIN_NUM+1)
    readWriteLines(os.path.join(data_path, 'train.span'), os.path.join(data_path, PREPEND+'samp.train.span'), TRAIN_NUM+1)
    readWriteLines(os.path.join(data_path, 'train.context'), os.path.join(data_path, PREPEND+'samp.train.context'), TRAIN_NUM+1)
    readWriteLines(os.path.join(data_path, 'train.answer'), os.path.join(data_path, PREPEND+'samp.train.answer'), TRAIN_NUM+1)
    readWriteLines(os.path.join(data_path, 'train.question'), os.path.join(data_path, PREPEND+'samp.train.question'), TRAIN_NUM+1)

    readWriteLines(os.path.join(data_path, 'val.ids.context'), os.path.join(data_path, PREPEND+'samp.val.ids.context'), VAL_NUM+1)
    readWriteLines(os.path.join(data_path, 'val.ids.question'), os.path.join(data_path, PREPEND+'samp.val.ids.question'), VAL_NUM+1)
    readWriteLines(os.path.join(data_path, 'val.span'), os.path.join(data_path, PREPEND+'samp.val.span'), VAL_NUM+1)
    readWriteLines(os.path.join(data_path, 'val.context'), os.path.join(data_path, PREPEND+'samp.val.context'), VAL_NUM+1)
    readWriteLines(os.path.join(data_path, 'val.answer'), os.path.join(data_path, PREPEND+'samp.val.answer'), VAL_NUM+1)
    readWriteLines(os.path.join(data_path, 'val.question'), os.path.join(data_path, PREPEND+'samp.val.question'), VAL_NUM+1)
