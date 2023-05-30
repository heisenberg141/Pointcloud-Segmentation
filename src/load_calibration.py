# Utils to load transformation to camera pose to system pose
import os
import numpy as np

def checkfile(filename):
    if not os.path.isfile(filename):
        raise RuntimeError('%s does not exist!' % filename)

def readVariable(fid,name,M,N):
    # rewind
    fid.seek(0,0)
    
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success==0:
      return None
    
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert(len(line) == M*N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat

def loadCalibrationCameraToPose(filename):
    # check file
    checkfile(filename)

    # open file
    fid = open(filename,'r');
     
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
      
    # close file
    fid.close()
    return Tr
    

def loadCalibrationRigid(filename):
    # check file
    checkfile(filename)

    lastrow = np.array([0,0,0,1]).reshape(1,4)
    return np.concatenate((np.loadtxt(filename).reshape(3,4), lastrow))


def loadPerspectiveIntrinsic(filename):
    # check file
    checkfile(filename)

    # open file
    fid = open(filename,'r');

    # read variables
    Tr = {}
    intrinsics = ['P_rect_00', 'R_rect_00', 'P_rect_01', 'R_rect_01']
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    for intrinsic in intrinsics:
        if intrinsic.startswith('P_rect'):
            Tr[intrinsic] = np.concatenate((readVariable(fid, intrinsic, 3, 4), lastrow))
        else:
            Tr[intrinsic] = readVariable(fid, intrinsic, 3, 3)

    # close file
    fid.close()

    return Tr

def GetIntrinsics(cam_id = 0):
        ''' load perspective intrinsics '''
        kitti360Path = "data"
        intrinsic_file = os.path.join(kitti360Path, 'calibration', 'perspective.txt')
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


if __name__=='__main__':
    
    kitti360Path = "data"
   
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    Tr = loadCalibrationCameraToPose(fileCameraToPose)
    print('Loaded %s' % fileCameraToPose)
    print(Tr)

    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    Tr = loadCalibrationRigid(fileCameraToVelo)
    print('Loaded %s' % fileCameraToVelo)
    print(Tr)

    fileSickToVelo = os.path.join(kitti360Path, 'calibration', 'calib_sick_to_velo.txt')
    Tr = loadCalibrationRigid(fileSickToVelo)
    print('Loaded %s' % fileSickToVelo)
    print(Tr)

    filePersIntrinsic = os.path.join(kitti360Path, 'calibration', 'perspective.txt')
    Tr = loadPerspectiveIntrinsic(filePersIntrinsic)
    print('Loaded %s' % filePersIntrinsic)
    print(Tr)