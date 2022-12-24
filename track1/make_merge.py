import os
import sys
import numpy as np
from tqdm import tqdm

from gdal_lib import read_img, write_img

def make_merge(rgb_path, sar_path, merge_path):
    ds_sar, im_proj_sar, im_geotrans_sar, height_sar, width_sar, data_sar = read_img(sar_path, True)
    ds_opt, im_proj_opt, im_geotrans_opt, height_opt, width_opt, data_opt = read_img(rgb_path, True)
    data_sar_cat = data_sar[np.newaxis, :, :]
    data_merge = np.concatenate([data_opt, data_sar_cat], axis=0)
    write_img(merge_path, im_proj_opt, im_geotrans_opt, data_merge)

if __name__ == '__main__':
    root = sys.argv[1]
    sets = ['train', 'val', 'test']
    for set in sets:
        set_dir = os.path.join(root, set)
        rgb_dir = os.path.join(set_dir, 'rgb')
        sar_dir = os.path.join(set_dir, 'sar')
        merge_dir = os.path.join(set_dir, 'merge')
        if not os.path.exists(merge_dir):
            os.mkdir(merge_dir)
        for f in tqdm(os.listdir(rgb_dir)):
            if f.endswith('.tif'):
                rgb_path = os.path.join(rgb_dir, f)
                sar_path = os.path.join(sar_dir, f)
                merge_path = os.path.join(merge_dir, f)
                make_merge(rgb_path, sar_path, merge_path)