import os
import sys
import numpy as np
from tqdm import tqdm

from osgeo import gdal
from osgeo import osr

def read_img(filename, with_data=True):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    if not with_data:
        return dataset, im_proj, im_geotrans, im_height, im_width
    else:
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
        return dataset, im_proj, im_geotrans, im_height, im_width, im_data

def write_img(filename, im_proj, im_geotrans, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64
    
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

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