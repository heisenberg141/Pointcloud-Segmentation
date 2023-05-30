import numpy as np
import struct
import open3d as o3d
import os
import copy
import time
from load_calibration import *
import matplotlib.pyplot as plt
from PIL import Image

def main():
    projectVeloToImage()

#====================================================================================================

def projectVeloToImage(cam_id=0, kitti360Path = "data"):

    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    filePersIntrinsic = os.path.join(kitti360Path, 'calibration', 'perspective.txt')
    
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)
    TrVeloToCam = {}
    
    for k, v in TrCamToPose.items():
        TrCamkToCam0 = TrCamToPose['image_00'] @ np.linalg.inv(TrCamToPose[k])
        TrCamToVelo =  TrCamkToCam0 @ TrCam0ToVelo
        TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

    intrinsics = get_intrinsics(filePersIntrinsic,cam_id=cam_id)
    print(intrinsics["R_rect"])

    if cam_id==0 or cam_id == 1:
        TrVeloToRect = np.matmul(intrinsics['R_rect'], TrVeloToCam['image_%02d' % cam_id])
    else:
        TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]

    # color map for visualizing depth map
    cm = plt.get_cmap('cool')
    
    data_dir_name = "good_data"
    pcd_dir = os.path.join("data/velodyne_points",data_dir_name)
    imgs_dir = os.path.join("data/image_00",data_dir_name)
    # visualize a set of frame
    # for each frame, load the raw 3D scan and project to image plane
    for file_name in os.listdir(pcd_dir):
        frame,_ = os.path.splitext(file_name)
        frame = int(frame)
        # load bin file into a numpy array
        points = loadVelodyneData(pcd_dir, frame)
        
        # it was x,y,z,intensity its intensity is forced to be 1
        points[:,3] = 1

        # we want velo point to be in homogenous coordinate system[X, Y, Z, 1].
        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:,:3]

        # project to image space
        u,v, depth= cam2image(pointsCam.T,intrinsics)
        u = u.astype(np.int64)
        v = v.astype(np.int64)
        cam_intr = get_intrinsics(filePersIntrinsic,cam_id)
        # prepare depth map for visualization
        depthMap = np.zeros((cam_intr["height"], cam_intr["width"]))
        depthImage = np.zeros((cam_intr["height"], cam_intr["width"], 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<cam_intr["width"]), v>=0), v<cam_intr["height"])
        # visualize points within 30 meters
        mask = np.logical_and(np.logical_and(mask, depth>0), depth<10)
        depthMap[v[mask],u[mask]] = depth[mask]
        layout = (2,1) if cam_id in [0,1] else (1,2)
        fig, axs = plt.subplots(*layout, figsize=(18,12))

        # load RGB image for visualization
        imagePath = os.path.join(imgs_dir,'%010d.png' % frame)
        if not os.path.isfile(imagePath):
            raise RuntimeError('Image file %s does not exist!' % imagePath)

        colorImage = np.array(Image.open(imagePath)) / 255.
        depthImage = cm(depthMap/depthMap.max())[...,:3]
        colorImage[depthMap>0] = depthImage[depthMap>0]

        axs[0].imshow(depthMap, cmap='inferno')
        axs[0].title.set_text('Projected Depth')
        axs[0].axis('off')
        axs[1].imshow(colorImage,cmap='inferno')
        axs[1].title.set_text('Projected Depth Overlaid on Image')
        axs[1].axis('off')
        plt.suptitle('Camera %02d, Frame %010d' % ( cam_id, frame))
        plt.show()

# ================================================================================================================

def loadVelodyneData(raw3DPcdPath,frame=0):
    pcdFile = os.path.join(raw3DPcdPath, '%010d.bin' % frame)
    if not os.path.isfile(pcdFile):
        raise RuntimeError('%s does not exist!' % pcdFile)
    pcd = np.fromfile(pcdFile, dtype=np.float32)
    pcd = np.reshape(pcd,[-1,4])
    return pcd 

def cam2image(points, intrinsics):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(intrinsics["K"][:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int64)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int64)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth

def get_intrinsics(intrinsic_file,cam_id = 0):
        ''' load perspective intrinsics '''
        
        intrinsics_dict = {}
        intrinsic_loaded = False
        width = -1
        height = -1
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_%02d:' % cam_id:
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3,4])
                intrinsic_loaded = True
            elif line[0] == 'R_rect_%02d:' % cam_id:
                R_rect = np.eye(4) 
                R_rect[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)
            elif line[0] == "S_rect_%02d:" % cam_id:
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert(intrinsic_loaded==True)
        assert(width>0 and height>0)
    
        intrinsics_dict["K"] = K
        intrinsics_dict["width"], intrinsics_dict["height"] = width, height
        intrinsics_dict["R_rect"] = R_rect
        return intrinsics_dict

# -====================================================================================================================

if __name__== '__main__':
    main()
    # rgb_pcd()