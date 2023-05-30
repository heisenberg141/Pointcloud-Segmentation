import numpy as np
import struct
from open3d import *
import os



def main():
    bin_dir = "/media/heisenberg/Data/kitti360/test_1/2013_05_28_drive_0008_sync/velodyne_points/data"
    pcd_dir = "velodyne_points/data_pcd"
    if not os.path.isdir(pcd_dir):
        os.mkdir(pcd_dir)

    for file in os.listdir(bin_dir):
        file_path = os.path.join(bin_dir,file)
        current_point_cloud = convert_kitti_bin_to_pcd(file_path)
        # open3d.visualization.draw_geometries([current_point_cloud])
        open3d.io.write_point_cloud(os.path.join(pcd_dir, f'{os.path.splitext(file)[0]}.pcd'), current_point_cloud, write_ascii=True, compressed=False, print_progress=False)



def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pcd)
    return pcd



if __name__== '__main__':
    main()