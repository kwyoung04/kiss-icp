import os
import numpy as np
import time
import json
import open3d as o3d
import cv2
import copy
from tqdm import tqdm
from datetime import datetime

from scipy.spatial.transform import Rotation

from my_kiss_icp import KissICP
from kiss_icp.config import KISSConfig
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
#from sophus import SE3d


'''
class Welding_Pose:
    def __init__(self) -> None:
        self.cam_info = dict()
        
        self.tcp_pose = [] # np.identity(4)
        self.images = [] 
        self.pcds = [] 
        self.seg_3D = []
        
        self.state = []    # x, y, z
        
        
        kiss_icp::pipeline::KissICP odometry_;
        kiss_icp::pipeline::KISSConfig config_;
        
    def get_init_info(self, cam_info, tcp_pose, images, pcds, seg_3D):
        self.cam_info = cam_info
        self.tcp_pose = tcp_pose
        self.images = images
        self.pcds = pcds
        self.seg_3D = seg_3D

    def check_number(self):
        l_t = len(self.tcp_pose)
        l_i = len(self.images)
        l_p = len(self.pcds)
        
        if l_t == l_i == l_p :
            return l_t
        
        print("### Number of input data mismatch")
        return -1
    
    def to_cam_pose(self, i):        
        # self.tcp_pose[i] -> np.identity(4)
        # self.cam_info    -> np.identity(4)
       
        initial_cam_pose = self.cam_info

        cam_pose = np.copy(initial_cam_pose)
        cam_pose[:3, 3] += self.tcp_pose[i]
                
        return cam_pose
    
    def state_updata(self, i):
        self.state[i]
        
    
    def feature_prediction(self, i, fm):
        print("### seg_3D to global camera axis")
        src = self.pcds[i]
        tgt = self.pcds[i-1]
        
        pose = self.state ## init_guess
        
        final_pose = fm.run(src, tgt, pose)

        return final_pose

    def merge_points(self):
        print("### seg_3D to global camera axis")
        pass
    
    def run_set(self, cam_info, tcp_pose, images, pcds, seg_3D):
        self.get_init_info(cam_info, tcp_pose, images, pcds, seg_3D) 
        
        n = self.check_number()
        
        global_points = []
        
        fm = Feature_Matching("matching_config") ## matching model & param

        for i in range(n):
            self.to_cam_pose(i) # arg: tcp_pose, return: cam_pose
            
            self.state_updata(i) # arg: cam_pose, return: state
            if i != 1: # fst itr
                self.feature_prediction(i, fm)
                
            points = self.merge_points(seg_3D) # arg: seg_3D, return: points
            global_points.append(points)
        
        return global_points
'''

class OdometryServer:
    def __init__(self, cnt_file):
        ## Data subscribers
        self.pointcloud_sub_ = []
        self.tcp_pose_sub_ = []
        self.data_set_ = []
        
        ## Data publishers
        self.odom_publisher_ = []
        self.frame_publisher_ = []
        self.kpoints_publisher_ = []
        self.map_publisher_ = []
        
        ## KISS-ICP
        config = KISSConfig()
        
        config.data.deskew = False
        config.data.max_range = 100.0 # can be also changed in the CLI
        config.data.min_range = 0.0

        config.mapping.voxel_size = 0.01# <- optional
        config.mapping.max_points_per_voxel = 50

        config.adaptive_threshold.fixed_threshold = 0.03 # <- optional, disables adaptive threshold
        #config.adaptive_threshold.initial_threshold = 0.2
        #config.adaptive_threshold.min_motion_th = 0.1
        
        #config.registration.max_num_iterations = 500 # <- optional
        #config.registration.convergence_criterion = 0.0001 # <- optional

        
        self.target = o3d.geometry.PointCloud()
        
        self.KissICP = KissICP(config)
        
        ## Meta data
        self.num_set = len(cnt_file)
        self._getData(cnt_file)
               
        
    def _getData(self, cnt_file):
        self.data_set_ = cnt_file

    def _saveData(self):
        pass

    def _to_se3(self, xyzrpy):
        x = float(xyzrpy[0])/100
        y = float(xyzrpy[1])/100
        z = float(xyzrpy[2])/100
        
        rx = xyzrpy[3]
        ry = xyzrpy[4]
        rz = xyzrpy[5]
        
        
        # Position vector
        t = np.array([x, y, z])

        # Euler angles to rotation matrix
        r = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
        R = r.as_matrix()

        # SE(3) transformation matrix
        se3_matrix = np.eye(4)
        se3_matrix[:3, :3] = R
        se3_matrix[:3, 3] = t

        return se3_matrix


    def voxel_downsample(self, point_cloud, voxel_size):
        voxel_grid = {}
        for point in point_cloud:
            voxel_index = tuple(np.floor(point / voxel_size).astype(int))
            if voxel_index in voxel_grid:
                voxel_grid[voxel_index].append(point)
            else:
                voxel_grid[voxel_index] = [point]
        
        downsampled_points = []
        for points_in_voxel in voxel_grid.values():
            downsampled_points.append(np.mean(points_in_voxel, axis=0))
        
        return np.array(downsampled_points)

    def RegisterFrame(self, i):
        pointcloud_sub = self.data_set_[i]['pcd']
        #tcp_pose_sub = self._to_se3(self.data_set_[i]['pose']['capture_pose'])
        #tcp_pose_sub = self._to_se3(self.data_set_[i]['pose']['camera_pose'])
 
        #image_sub = self.data_set_[i]['png']

        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(pointcloud_sub)
#
        #_, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
#
        #non_plane_cloud = pcd.select_by_index(inliers, invert=True)
        #plane_cloud = pcd.select_by_index(inliers, invert=False)
        #
        #non_plane_np = np.asarray(non_plane_cloud.points)
        #plane_np = np.asarray(plane_cloud.points)
        #
        ##plane_np = self.voxel_downsample(plane_np, 0.1)
        #
        #
        #
        #combined_np = np.concatenate((plane_np, non_plane_np), axis=0)
        
        
        ## kiss-icp
        frame, keypoints = self.KissICP.register_frame(pointcloud_sub)
        
        point_cloud = o3d.geometry.PointCloud()
        # 
        # numpy_array_data = np.asarray(self.KissICP.local_map.point_cloud())
        # #numpy_array_data = np.asarray(keypoints)
        # point_cloud.points = o3d.utility.Vector3dVector(numpy_array_data)
        # 
        # _, inliers = point_cloud.segment_plane(distance_threshold=(0.01 + j*0.001), ransac_n=7, num_iterations=1000)
        # point_cloud = point_cloud.select_by_index(inliers, invert=True)

        if i > 0:
            numpy_array_data = np.asarray(self.KissICP.local_map.point_cloud())
            selected_points = numpy_array_data[numpy_array_data[:, 2] < (2.36)]
            point_cloud.points = o3d.utility.Vector3dVector(selected_points)
            
            for j in range(1):
                print(j)
                _, inliers = point_cloud.segment_plane(distance_threshold=(0.005), ransac_n=7, num_iterations=1000)
                point_cloud = point_cloud.select_by_index(inliers, invert=True)
                
                file_path = f"/home/eric/github/kiss-icp/test_ply/test_cloud_{i}.ply"
                o3d.io.write_point_cloud(file_path, point_cloud)

    
  
    def PublishFrame(self):
        self._saveData()
  



def read_png_file(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    return img

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def find_edges(point_cloud):
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Compute point cloud normals
    pcd.estimate_normals()

    # KDTree for nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Set threshold distance for edge detection
    threshold = 0.005  # Adjust this value as needed

    # List to store edge points
    edge_points = []

    # Iterate through each point
    for i in range(len(point_cloud)):
        # Find nearest neighbors
        [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 3)  # Adjust k as needed

        # Compute distances to neighbors
        distances = np.linalg.norm(np.asarray(pcd.points)[idx[1:], :] - np.asarray(pcd.points)[idx[0], :], axis=1)

        # Check if any neighbor is beyond threshold distance
        if np.any(distances > threshold):
            edge_points.append(point_cloud[i])

    return np.array(edge_points)

def read_ply_file(ply_file_path, scale=1000):
    ply_data = o3d.io.read_point_cloud(ply_file_path)
    #ply_data = ply_data.voxel_down_sample(voxel_size=0.01)

    points = np.asarray(ply_data.points)
    points = points[~np.isnan(points).any(axis=1)]
    
    scaled_points = points/scale
    scaled_points = scaled_points[scaled_points[:, 2] <= 2.4]
    scaled_points = scaled_points[scaled_points[:, 2] > 1]
    
    
    return scaled_points



def get_sensor_data_only(file_path):
    check_file_existence = os.path.exists(file_path)
    if check_file_existence:
        folder_info = []
        
        files = os.listdir(file_path)
        files.sort()  # 파일 이름의 순서를 정렬
        
        for file in files:
            if file.strip()[-1] == 'y': ## ply 파일만 처리
                data_path = os.path.join(file_path, file)
                file_info = {}
                file_info['pcd'] = find_edges(read_ply_file(data_path))
                folder_info.append(file_info)

        return folder_info
    else:
        return 0



def get_sensor_data(file_path):
    check_file_existence = os.path.exists(file_path)
    if check_file_existence:
        folder_info = []
        
        set_list = os.listdir(file_path)
        
        datetime_list = [datetime.strptime(date_str, '%Y%m%d-%H%M%S') for date_str in set_list]
        sorted_datetime_list = sorted(datetime_list)
        sorted_set_list = [datetime.strftime(date, '%Y%m%d-%H%M%S') for date in sorted_datetime_list]

        for folder_name in sorted_set_list:
            folder_full_path = os.path.join(file_path, folder_name)
            file_info = {}
            
            files = os.listdir(folder_full_path)
            for file in files:
                data_path = os.path.join(folder_full_path, file)
                if file.strip()[-9:] == 'pose.json': ## json
                    file_info['pose'] = read_json_file(data_path)
                
                elif file.strip()[-9:] == 'Meta.json': ## ply
                    file_info['meta'] = read_json_file(data_path)
                
                elif file.strip()[-1] == 'y': ## ply
                    file_info['pcd'] = read_ply_file(data_path)
                
                elif file.strip()[-1] == 'g': ## png
                    file_info['png'] = read_png_file(data_path)
                    pass
                
                else: # other file format
                    return -1

            folder_info.append(file_info)

        return folder_info
    else:
        return 0

def main():    
    
    #cnt_file = get_sensor_data_only("/home/eric/Desktop/hrc_240411/photoneomotionL/01")
    cnt_file = get_sensor_data_only("/home/eric/github/kiss-icp/auto_test/line")
    
    if cnt_file:
        odom = OdometryServer(cnt_file)
        
        #for i in tqdm(range(len(cnt_file))):
        for i in range(len(cnt_file)):
            print(f"########################################## {i}")
            ## Base the robot's pose as the init guess, the icp is operated with pcd to output se3
            odom.RegisterFrame(i)
            
            ## Calculate the location of nodes
            #odom.LookupTransform() 
            

        
        ## Not yet embodied
        #odom.globalBundleAdjustment()
    
        ## Save the final global point
        odom.PublishFrame()
        
    elif cnt_file == -1:
        pass ## error cnt
 

    
    
    
    
    

if __name__ == "__main__":
      main()

