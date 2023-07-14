import mxnet as mx
import os
import numpy as np
import torch
import numbers
import cv2

root_dir = "/home/sonnt373/Desktop/SoNg/Face_quality/dev/data_train/faces_webface_112x112"
path_imgrec = os.path.join(root_dir, 'train.rec')
path_imgidx = os.path.join(root_dir, 'train.idx')
imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)
if header.flag > 0:
    header0 = (int(header.label[0]), int(header.label[1]))
    imgidx = np.array(range(1, int(header.label[0])))
else:
    imgidx = np.array(list(imgrec.keys))
index = 10000
idx = imgidx[index]
s = imgrec.read_idx(idx)
header, img = mx.recordio.unpack(s)
label = header.label
if not isinstance(label, numbers.Number):
    label = label[0]
label = torch.tensor(label, dtype=torch.long)
sample = mx.image.imdecode(img).asnumpy()

print(label, sample.shape)
cv2.imwrite("test.jpg",cv2.cvtColor(sample, cv2.COLOR_BGR2RGB) )