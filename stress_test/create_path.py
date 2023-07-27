import json
import os
from sklearn.model_selection import train_test_split
import random

def save_dict_box(dict_save,  out_path):
    # save dict box 
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(out_path , "w") as outfile:
        json.dump(dict_save, outfile)
    print("=======================================save dict done=====================================================")
    return "save dict done"

folder_input = "../../data/data_train_100k_140723_300k/data_train_100k_140723"
count = 0
dict_data_train = {}
dict_data_test = {}

class_id_all = os.listdir(folder_input)
ratio = 1
list_class_id_train = random.sample(class_id_all, k=int(ratio * len(class_id_all)))
list_class_id_test = list(set(class_id_all) - set(list_class_id_train))
print("list_class_id_train",len(list_class_id_train))
for idx, class_id in enumerate(list_class_id_train):
    folder_class_id = os.path.join(folder_input, class_id)
    print(idx)
    for path in os.listdir(folder_class_id):
        dicct_img = {}
        img_path = os.path.join(class_id, path)
        dicct_img["path"] = img_path
        dicct_img["labels"] = str(idx)
        dict_data_train[str(count)]  = dicct_img
        count += 1
print("coubnt=", count)
count = 0

for idx, class_id in enumerate(list_class_id_test):
    folder_class_id = os.path.join(folder_input, class_id)
    for path in os.listdir(folder_class_id):
        dicct_img = {}
        img_path = os.path.join(class_id, path)
        dicct_img["path"] = img_path
        dicct_img["labels"] = str(idx)
        dict_data_test[str(count)]  = dicct_img
        count += 1

print(f'len all data train: {len(dict_data_train.keys())}')
print(f'len all data test: {len(dict_data_test.keys())}')

save_dict_box(dict_data_train, f'{len(list_class_id_train)}_data_train.json')
save_dict_box(dict_data_test, f'{len(list_class_id_test)}_data_test.json')
