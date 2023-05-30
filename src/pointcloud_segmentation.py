from gen_color_pcd import *
import numpy as np
import open3d as o3d
import copy
from load_calibration import *
import matplotlib.pyplot as plt

def main():
    # RUN FROM OUTSIDE SRC
    pointcloud_files, image00_files = LoadData()
    cam_id = 0
    GenSegAndRegisterPCDs(pointcloud_files,image00_files,cam_id, register = True)


#=================================================================================================================


def GenSegAndRegisterPCDs(pointcloud_files,image00_files,cam_id, register = True):
    TVeloToCam = GetVeloToCam(cam_id = cam_id)
    intrinsics = GetIntrinsics(cam_id)
    TVeloToRect = np.matmul(intrinsics["R_rect"],TVeloToCam)
    T_current_target = np.identity(4)
    big_pcd = o3d.geometry.PointCloud()
    prev_pcd = o3d.geometry.PointCloud()
    n_files = len(pointcloud_files)
    vis = o3d.visualization.Visualizer()
    
#
    for i in range(n_files):
        pcd_file = pointcloud_files[i]
        image_path = image00_files[i]
        
        seg_pcd = ColorPCD(image_path, pcd_file, TVeloToRect, intrinsics, segmentation = True)
        
        T_seg_prev = get_transformation(prev_pcd,seg_pcd)
        T_current_target = np.matmul(T_current_target,T_seg_prev)
        prev_pcd = copy.deepcopy(seg_pcd)
        seg_pcd.transform(T_current_target)
        
        big_pcd.points.extend(seg_pcd.points)
        big_pcd.colors.extend(seg_pcd.colors)
        if i==0:
            if not register:
                break
            vis.create_window()
            vis.add_geometry(big_pcd)
            
        else: 
            vis.update_geometry(big_pcd)
        
        keep_running = vis.poll_events()
        vis.update_renderer()
    o3d.visualization.draw_geometries([big_pcd])
    return big_pcd


#=================================================================================================================


def get_transformation(target, source):
    if(len(target.points)==0):
        return np.identity(4)

    current_transformation = np.identity(4)
    voxel_radius = [0.4, 0.2, 0.1]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    T = result_icp.transformation
    # draw_registration_result_original_color(source, target,
                                            # result_icp.transformation)
    return T
        






if __name__== '__main__':
    main()
