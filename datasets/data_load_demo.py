import numpy as np
from scipy import sparse
import ast
import os
import json

"""
You should download the whole dataset first at: https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg
Then, use this loader tool to load dataset
"""
uniseg_path = '/PATH/M3D_Seg'  # PATH : your path
dataset_code = '0001'
json_path = os.path.join('../', dataset_code, dataset_code + '.json')
with open(json_path, 'r') as f:
    dataset_dict = json.load(f)

ct_file_path = os.path.join(uniseg_path, dataset_dict['train'][0]['image'])
gt_file_path = os.path.join(uniseg_path, dataset_dict['train'][0]['label'])

img_array = np.load(ct_file_path)[0]
# print('img_array.shape ', img_array.shape)

allmatrix_sp= sparse.load_npz(gt_file_path)
gt_shape = ast.literal_eval(gt_file_path.split('.')[-2].split('_')[-1])
gt_array=allmatrix_sp.toarray().reshape(gt_shape)
# print('gt_array.shape ', gt_array.shape)
