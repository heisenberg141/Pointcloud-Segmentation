import numpy as np
import struct
import open3d as o3d
import os
import copy
import time
from load_calibration import *
import cv2
import matplotlib.pyplot as plt
from segment_image import *

def main():
    # RUN FROM OUTSIDE SRC
    pointcloud_files, image00_files = LoadData()
    cam_id = 0
    TVeloToCam = GetVeloToCam(cam_id = cam_id)
    # DisplayPointCloudOnImage(pointcloud_files,image00_files,TVeloToCam,cam_id)
    GenerateColoredPCDs(pointcloud_files,image00_files,TVeloToCam,cam_id)
    # ColorAndDisplayPCD(pointcloud_files,image00_files,TVeloToCam,cam_id)
    return


#========================================================================================================================


def LoadData():
    data_dir = "data"
    image00_dir = os.path.join(data_dir,"image_00/good_data")
    pointcloud_dir = os.path.join(data_dir,"velodyne_points/good_data")
    pointcloud_files,image00_files = [], []
    for file_name in os.listdir(image00_dir):
        image00_files.append(os.path.join(image00_dir,file_name))
    for file_name in os.listdir(pointcloud_dir):
        pointcloud_files.append(os.path.join(pointcloud_dir,file_name))
    pointcloud_files.sort()
    image00_files.sort()
    # print(pointcloud_files[0], "\n",image00_files[0])
    return pointcloud_files, image00_files

def GetVeloToCam(cam_id):
    '''
        DESCRIPTION:
        
        Under the calibration dir, we will use cam_to_pose.txt, cam_to_velo.txt and perspective.txt files.
        cam_to_velo.txt has transformation matrix from cam_0 to velodyne. T_C0_V
        cam_to pose.txt has transformations from IMU's frame to cam_i's frame.
        perspective.txt has intrinsics of perspective cameras (Cam_0 and Cam_1)
        
        From world to camera:
        [x y 1].T = P * [X Y Z 1].T
        where P is projection matrix. It Projects world points to image plane.
        The P matrix made of two parts. on is [R|T] from world frame to camera's frame
        and intrinsic matrix K which contains information such as camera center, focal length and skew.
        Thus to go from world point [X Y Z 1] to image plane [x y 1] we use [K] * [R|T] * [X Y Z 1].T 
        Now, this all was pinhole camera projections. IE without any lens or distortion, or any angle between image plane and lenses.
        All these parameters are also given to us in the perspective.txt. And we can rectify the camera images to actually transform them to 
        [x y 1] plane.   
    '''
    
    # load all transformations
    TCam0ToVelo, TCam_kToIMU = LoadTransformations()
    TVeloToCam = TransformVeloToCam(cam_id,TCam0ToVelo, TCam_kToIMU)
    
    return TVeloToCam

def DisplayPointCloudOnImage(pointcloud_files,image00_files,TVeloToCam,cam_id):
    '''
        File Structure of perspective.txt
            S_xx: 1x2 size of image xx before rectification
            K_xx: 3x3 calibration matrix of camera xx before rectification
            D_xx: 1x5 distortion vector of camera xx before rectification
            R_xx: 3x3 rotation matrix of camera xx (extrinsic)
            T_xx: 3x1 translation vector of camera xx (extrinsic)
            S_rect_xx: 1x2 size of image xx after rectification
            R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
            P_rect_xx: 3x4 projection matrix after rectification

    '''
    intrinsics = GetIntrinsics(cam_id)
    TVeloToRect = np.matmul(intrinsics["R_rect"],TVeloToCam)
    data_dir = "data"
    image00_dir = os.path.join(data_dir,"image_00/good_data")
    
    for pcd_file in pointcloud_files:
        file_name,_ = os.path.splitext(os.path.basename(pcd_file))
        image_path = os.path.join(image00_dir,file_name + ".png")
        points = loadVelodyneData(pcd_file)
        u,v,depth = ProjectToImage(points,intrinsics,TVeloToRect)
        u = u.astype(np.int64)
        v = v.astype(np.int64)
        depthMap = np.zeros((intrinsics["height"], intrinsics["width"]))
        depthImage = np.zeros((intrinsics["height"], intrinsics["width"], 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<intrinsics["width"]), v>=0), v<intrinsics["height"])
        mask = np.logical_and(np.logical_and(mask, depth>0), depth<100)
        depthMap[v[mask],u[mask]] = depth[mask]
        print(depthMap.max())
        
        depth_image = ((depthMap/depthMap.max())*255).astype(np.uint8)
        image = cv2.imread(image_path)
        image[depth_image>0,:] = 255
        
        print(depth_image)
        cv2.imshow("DepthMap",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # return 
        # break
        
def GenerateColoredPCDs(pointcloud_files,image00_files,TVeloToCam,cam_id,vis = True,segmentation = False):
    
    intrinsics = GetIntrinsics(cam_id)
    TVeloToRect = np.matmul(intrinsics["R_rect"],TVeloToCam)
    data_dir = "data"
    image00_dir = os.path.join(data_dir,"image_00/good_data")
    
    for pcd_file in pointcloud_files:
        print(f"NOTE: Coloring {pcd_file}")
        file_name,_ = os.path.splitext(os.path.basename(pcd_file))
        image_path = os.path.join(image00_dir,file_name + ".png")
        pcd = ColorPCD(image_path, pcd_file, TVeloToRect, intrinsics, segmentation = segmentation)
        if vis:
            o3d.visualization.draw_geometries([pcd])
        # return pcd


#=================================================================================================================


def LoadTransformations():
    kitti360Path = "data"
    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    TCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)
    TCam_kToIMU = loadCalibrationCameraToPose(fileCameraToPose)
    
    return TCam0ToVelo, TCam_kToIMU

def TransformVeloToCam(cam_id,TCam0ToVelo, TCam_kToIMU):
    if cam_id == 0:
        return np.linalg.inv(TCam0ToVelo)
    
    TCam0ToIMU = TCam_kToIMU['image_00']
    TCam_idToIMU = TCam_kToIMU['image_%02d' % cam_id]
    return TCam0ToVelo @ np.linalg.inv(TCam0ToIMU) @ TCam_idToIMU 

def loadVelodyneData(pcdFile):
    if not os.path.isfile(pcdFile):
        raise RuntimeError('%s does not exist!' % pcdFile)
    pcd = np.fromfile(pcdFile, dtype=np.float32)
    pcd = np.reshape(pcd,[-1,4])
    return pcd 

def ProjectToImage(points,intrinsics,TVeloToRect):
    # this is a list of 3 points. N x 3
    # homogenous coordinates
    # print("T_velo_rect: ", TVeloToRect.shape)
    # TVeloToRect = np.eye(4).astype(np.float64)
    points[:,3] = 1
    # N x 4
    # print("WorldCoordinates: ", points.shape)
    pointsCam = np.matmul(TVeloToRect, points.T).T
    # pointsCam = points
    pointsCam = pointsCam[:,:3]
    # print("From Camera: ", pointsCam.shape)
    
    return cam2image(pointsCam.T,intrinsics)

def ColorPCD(image_path, pcd_file,TVeloToRect,intrinsics,segmentation = False):
    points = loadVelodyneData(pcd_file)
    print("Total Points in pointcloud: ", points.shape)
    # display_points = points[:,:3]
    # big_pcd = o3d.geometry.PointCloud()
    # big_pcd.points = o3d.utility.Vector3dVector(display_points)
    # big_pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(display_points))
    # o3d.visualization.draw_geometries([big_pcd])
    u,v,depth = ProjectToImage(points,intrinsics,TVeloToRect)
    u = u.astype(np.int64)
    v = v.astype(np.int64)
    PointToColorMap = {}
    width = intrinsics["width"]
    height = intrinsics["height"]
    print(f"Image (width,height): ({width}, {height})")
    image = cv2.imread(image_path)
    for i in range(len(points)):
        if (0<=u[i]<width):
            if (0<=v[i]<height):
                if(0<depth[i]<100):
                    if tuple([v[i],u[i]]) not in PointToColorMap:
                        PointToColorMap[tuple([v[i],u[i]])] = [points[i], image[v[i],u[i],:],np.linalg.norm(points[i],2)]
                    else:
                        if PointToColorMap[tuple([v[i],u[i]])][2]> np.linalg.norm(points[i],2):
                            PointToColorMap[tuple([v[i],u[i]])] = [points[i], image[v[i],u[i],:],np.linalg.norm(points[i],2)]
    colored_pcd = list()
    colors = list()

    if segmentation:
        seg_img = generate_segmentated_img(image_path)
        for im_point in PointToColorMap:
            colored_pcd.append(PointToColorMap[im_point][0])
            colors.append(seg_img[im_point[0],im_point[1],:])
    else:
        for im_point in PointToColorMap:
            colored_pcd.append(PointToColorMap[im_point][0])
            colors.append(PointToColorMap[im_point][1])
    print("DONE")
    colored_pcd = np.array(colored_pcd)
    print("Smaller by",(len(points)-len(colored_pcd))/len(points)*100,"%")
    colored_pcd = colored_pcd[:,0:3]
    print(colored_pcd[0])
    colors = np.array(colors).astype(np.float32)
    colors[:,[2,0]]  = colors[:,[0,2]]
    colors/=255
    points = points[:,:3]
    print(points[0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(colored_pcd)
    pcd.colors = o3d.utility.Vector3dVector(colors)
   
    return pcd

    

#=============================================================================================================================

def cam2image(points, intrinsics):
        print("But now, stacked column wise: ", points.shape)
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        print("With extra dimension at 0th axis: ",points.shape)
        # return  0, 0, 0
        points_proj = np.matmul(intrinsics["K"][:3,:3].reshape([1,3,3]), points)
        print("Finally project onto the image plane: ", points_proj.shape)
        # After multiplication from K matrix, the 3rd coordinate from each point is not 1. 
        # For each pixel coordinate, it has a depth value in the 3rd coordinate.
        depth = points_proj[:,2,:]
        # print(depth)
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int64)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int64)
        # print(v)
        # return  0, 0, 0
        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth


# ===========================================================================================================================
if __name__== '__main__':
    main()