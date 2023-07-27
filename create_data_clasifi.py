import time
import json
import cv2
import numpy
import os
from tqdm import tqdm
import random
import shutil
import json
import torch
import numpy as np
from backbones.iresnet import iresnet100, iresnet50


def save_dict_box(dict_save,  out_path):
    # save dict box 
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(out_path , "w") as outfile:
        json.dump(dict_save, outfile)
    print("=======================================save dict done=====================================================")
    return "save dict done"
if __name__ == '__main__':

    # Define algorithms

    backbone = iresnet100(dropout=0.4,num_features=512).cuda()

    backbone_pth = "/media/thainq97/DATA/GHTK/sonnt373/CR-FIQA_Tuning/cp/181952backbone.pth"
    backbone.load_state_dict(torch.load(backbone_pth, map_location='cuda'))
    backbone.cuda()
    model = torch.nn.DataParallel(backbone)
    model.eval()


    data_folder = "/media/thainq97/DATA/GHTK/sonnt373/data/data_train_100k_140723"
    data_folder_out = "/media/thainq97/DATA/GHTK/sonnt373/data/data_train_classifi_2class"

    dict_img_qualified = {}
    dict_img_notqualified = {}
    for clas in os.listdir(data_folder):
        folder_clas = os.path.join(data_folder, clas)

        folder_clas_out = os.path.join(data_folder_out, clas)
        os.makedirs(folder_clas_out, exist_ok=True)

        for path in os.listdir(folder_clas):
            img_path = os.path.join(folder_clas, path)
            t0 = time.time()
            img_face = cv2.imread(img_path)

            image = cv2.resize(img_face, (112, 112))
            image = np.transpose(image, (2, 0, 1))
            input_blob = np.expand_dims(image, axis=0)
            imgs = torch.Tensor(input_blob)
            imgs.div_(255).sub_(0.5).div_(0.5)
            feat, qs = model(imgs)

            score_quality = qs.cpu().detach().numpy()[0][0]
            if score_quality > 2.32:
                # shutil.copy(img_path, folder_clas_out)
                dict_img_qualified[os.path.join(clas, path)] = "qualified"
            if score_quality < 1.8:
                dict_img_notqualified[os.path.join(clas, path)] = "not qualified"

    print(f'len dict_img_notqualified {len(dict_img_notqualified.keys())}, dict_img_qualified { len(dict_img_qualified.keys())}')

    save_dict_box(dict_img_qualified, "save_dict_box.json")
    save_dict_box(dict_img_notqualified, "dict_img_notqualified.json")



