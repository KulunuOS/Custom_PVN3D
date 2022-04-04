#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

# ROS related imports
import open3d as o3d
import rospy
import message_filters
import ros_numpy
from sensor_msgs.msg import PointCloud2, Image
from lib.utils import pcl_helper
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from std_msgs.msg import Header
from rospy_tutorials.msg import Floats
from panda_simulation.srv import getPoses_multiClass as getPoses_srv
from panda_simulation.srv import getPoses_multiClassResponse as getPoses_resp
from scipy.spatial.transform import Rotation as R
#from cv_bridge import CvBridge, CvBridgeError
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') #This is to remove ROS-python from the PYTHONPATH which messes up the Python 3 env this project works with
#bridge = CvBridge()
import os
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pcl
import pickle as pkl
from common import Config
from lib import PVN3D
from lib.utils.sync_batchnorm import convert_model
from lib.utils.pvn3d_eval_utils import cal_frame_poses, cal_frame_poses_lm
from lib.utils.basic_utils import Basic_Utils


''' This script provides a demo node for running inference with PVN3D on images subscribed from
    ROS topics. It has several args that can be turned on or off for publishing the required data '''


## Args to initialize PVN3D
parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
parser.add_argument(
    "-dataset", type=str, default="openDR",
    help="Target dataset, ycb, linemod or opendr. (opendr as default)."
)
parser.add_argument(
    "-cls", type=str, default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can," +
    "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)

#cam_links = {'panda_camera_optical_frame':'panda_camera_link', 'kinect1_optical_link': 'kinect1_link'}
args = parser.parse_args()

if args.dataset == 'linemod':
    config = Config(dataset_name=args.dataset, cls_type=args.cls)
else:
    config = Config(dataset_name=args.dataset)
bs_utils = Basic_Utils(config)


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


## Functions to initialize PVN3D

def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }


def convert_types(img, orig_min, orig_max, tgt_min, tgt_max, tgt_type):

    #info = np.finfo(img.dtype) # Get the information of the incoming image type
    # normalize the data to 0 - 1
    img_out = img / (orig_max-orig_min)   # Normalize by input range

    img_out = (tgt_max - tgt_min) * img_out # Now scale by the output range
    img_out = img_out.astype(tgt_type)
    print(img_out.max())
    #cv2.imshow("Window", img)
    return img_out


def get_normal( cld):
       ''' Open3d based normal estimation '''
       cloud = o3d.geometry.PointCloud()
       cloud.points = o3d.utility.Vector3dVector(cld)

       cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))

       cloud.orient_normals_towards_camera_location()
       n = np.asarray(cloud.normals)
       return n

def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        try:
            checkpoint = torch.load(filename)
        except:
            checkpoint = pkl.load(open(filename, "rb"))
        epoch = checkpoint["epoch"]
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None

## Calculating poses from keypoint predictions
def cal_view_pred_pose(model, data):
    model.eval()
    try:
        #print('Started data acquisition')
        with torch.set_grad_enabled(False): # Data Acquisition
            cu_dt = [item.contiguous().to("cuda", non_blocking=True) for item in data]
            rgb, cld_rgb_nrm, choose = cu_dt#.contiguous()

            # Model Predictions #
            pred_kp_of, pred_rgbd_seg, pred_ctr_of = model(
                cld_rgb_nrm, rgb, choose
            )
            _, classes_rgbd = torch.max(pred_rgbd_seg, -1)
    
            if args.dataset == "ycb":
                pred_cls_ids, pred_pose_lst, pred_kps_lst = cal_frame_poses(
                    cld_rgb_nrm[0][:,:3], classes_rgbd[0], pred_ctr_of[0], pred_kp_of[0], True,
                    config.n_objects, True, args.dataset
                )

            elif args.dataset == "openDR":
                pred_cls_ids, pred_pose_lst, pred_kps_lst = cal_frame_poses(
                    cld_rgb_nrm[0][:,:3], classes_rgbd[0], pred_ctr_of[0], pred_kp_of[0], True,
                    config.n_objects, True, args.dataset
                )
    
            else:
                    pred_pose_lst, pred_kps_lst = cal_frame_poses_lm(
                    cld_rgb_nrm[0][:,:3], classes_rgbd[0], pred_ctr_of[0], pred_kp_of[0], True,
                    config.n_objects, False, 16
                )
    

            print('Prediction Complete...')
            return classes_rgbd.cpu().numpy(), pred_pose_lst, cld_rgb_nrm[0].cpu().numpy(), pred_kps_lst
    except Exception as inst:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('exception: '+str(inst)+' in '+ str(exc_tb.tb_lineno))


## The actual demo starts from here

class PVN3D_ROS(object):

    def __init__(self, config, model, pub_all=False, remove_panda_dpt=False, auto_detect=True):

        self.config = config
        self.model = model
        self.pub_all = pub_all
        self.request = auto_detect
        self.remove_panda_dpt = remove_panda_dpt    #This attribute tells whether to crop out panda links from the original depthImage topic or not.
        self.panda_binImg = None
        self.poses = None
        self.cls_ids = None
        if rospy.has_param('pvn3d_cam'):
            self.cam = rospy.get_param('pvn3d_cam')     # Set the cam, we're getting detections from.
        else:
            self.cam = '/panda/camera'
        ## Publishers
        self.pub1 = rospy.Publisher('/pvn3d_cloud', PointCloud2, queue_size=10)     #Segmented Cloud
        self.pub2 = rospy.Publisher('/pvn3d_pose', PoseArray, queue_size=10)        #Pose predictions of all classes
        self.pub4 = rospy.Publisher('pvn3d_label_bb_img', Image, queue_size=10 )       #Image with class labels projected on it i.e., Segmented Image and the bounding boxes
        self.pub5 = rospy.Publisher('/pvn3d_kps', PointCloud2)                     #Predicted Keypoints
        self.pub6 = rospy.Publisher('/pvn3d_mesh_registered', PointCloud2, queue_size=10)   #Object clouds loaded from pcd files registered to their predicted poses

        ## Subscribers
        self.image_sub = message_filters.Subscriber(str(self.cam)+'/color/image_raw', Image, queue_size=1)    #Image Subscriber -
        self.depth_sub = message_filters.Subscriber(str(self.cam)+'/depth/image_raw', Image, queue_size=1)
        self.cam_frame = 'kinect1_optical_link'
        self.pandaBinImg_sub = rospy.Subscriber('/panda_points_projected/binary', Image, self.project_pandaCB)

        ## Synchronized-time filter for depthImg and rgbImg subscribers
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.ros2torch_CB)

        ## Mesh initilization for ROS demo ##
        self.mesh = {}
        self.corners = {}
        if args.dataset == 'linemod':

            self.mesh['1'] = pcl.load('/home/ahmad3/PVN3D/pvn3d/datasets/linemod/Linemod_preprocessed/models/obj_'+str(config.lm_obj_dict[args.cls])+'.ply')
            self.corners['1'] = np.loadtxt('/home/ahmad3/PVN3D/pvn3d/datasets/linemod/lm_obj_kps/'+str(args.cls)+'/corners.txt')
            self.mesh_pts = np.array(mesh, dtype= np.float32)

        else: # openDR case

            for i in range(1, 11):
                self.mesh[str(i)] = pcl.load('/home/ahmad3/PVN3D/pvn3d/datasets/openDR/openDR_dataset/models/obj_'+str(i)+'.ply')
                self.corners[str(i)] = np.loadtxt('/home/ahmad3/PVN3D/pvn3d/datasets/openDR/openDR_object_kps/'+str(i)+'/corners.txt')

        cam2optical = R.from_euler('zyx',[1.57, 0, 1.57])
        cam2optical = cam2optical.as_dcm()

        for i in self.corners.keys():
            ''' Voxel grid filtering for Mesh points '''
            sor = self.mesh[i].make_voxel_grid_filter()
            sor.set_leaf_size(0.02, 0.02, 0.02)
            self.mesh[i] = np.array(sor.filter())

            ''' Mesh & Corners transformation '''
            self.mesh[i] = np.matmul(cam2optical, self.mesh[i].transpose()).transpose()
            self.corners[i]  = np.matmul(cam2optical, self.corners[i].transpose()).transpose()
            n = self.mesh[i].shape[0]

            #transformation optical to camera link
            self.mesh[i] = np.concatenate(( self.mesh[i][:,2].reshape(n,1) , -self.mesh[i][:, 0].reshape(n,1), -self.mesh[i][:, 1].reshape(n,1) ), axis=1)
            self.corners[i] = np.concatenate(( self.corners[i][:,2].reshape(8,1) , -self.corners[i][:, 0].reshape(8,1), -self.corners[i][:, 1].reshape(8,1) ), axis=1)


    def publishPoses(self, poses, cls_ids):

        #Publishing the poses in the order of [0 - 10] encapsulates both labels and their poses in one ROSMSG
        #This is because of the assumption that labels with pose (0, 0, 0, 0, 0, 0) are undetected in the scene
        poses_msg = [Pose(Point(0,0,0), Quaternion(0,0,0,1))] * 10

        for i, cls_id in enumerate(np.unique( cls_ids[cls_ids.nonzero()] )):
            pose = poses[i]
            rot = R.from_dcm(pose[0:3,0:3])
            quat = rot.as_quat()
            poses_msg[cls_id-1]= Pose(position = Point(x=pose[0,3],y=pose[1,3],z=pose[2,3]), orientation = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3]) )

        print('Publishing poses only...')
        self.pub2.publish(PoseArray(header=Header(stamp=rospy.Time.now(), frame_id=self.cam_frame), poses = poses_msg))


    def publishAll(self, rgb_img, cls_ids, poses, pcld, kps):

        registered_pts = np.zeros((1,3))
        rgb_labeled_bb = rgb_img.copy()
        poses_msg = []

        try:
            ## Iterate and publish for all the available cls_ids in a frame ##

            for i, cls_id in enumerate(np.unique( cls_ids[cls_ids.nonzero()] )):
                mask = np.where(cls_ids[0,:] == cls_id)[0]
                pose = poses[i]

                '''
                #This is to rectify the mistake during training - The following labels are symetric objects in dataset which was not
                # indicated during the training, hence their detected poses are subject to rotational ambiguity along z-axis i.e., because
                # of symmetry they look similar after every 90 degrees of rotation along z-axis


                if int(cls_id) in [1,2,3,5,7]:
                    print('Symmetric class detected. Wrapping yaw...')
                    pose_np = pose.copy()#ros_numpy.numpify(pose)
                    pose_eul = R.from_dcm(pose_np[:3, :3]).as_euler('zyx', degrees=True)
                    print('detected yaw: '+str(pose_eul[0]))
                    if pose_eul[0] > 90:
                        pose_eul[0] = 180 - pose_eul[0]
                    elif pose_eul[0] < -90:
                        pose_eul[0] = 180 + pose_eul[0]
                    print('yaw rectified '+str(pose_eul[0]))
                    pose_np[:3, :3] = R.from_euler('zyx', pose_eul, degrees=True).as_dcm()
                    pose = pose_np#ros_numpy.msgify(Pose, pose_np)'''

                kp = kps[i]
                p2ds = bs_utils.project_p3d(pcld[mask,0:3], 1 , K = config.intrinsic_matrix['custom'] )
                registered_pts = (np.matmul(pose[:3,:3],self.mesh_pts_t.T)+pose[:,3].reshape(3,1)).T
                registered_corners = (np.matmul(pose[:3,:3],self.corners[str(cls_id)].T)+pose[:,3].reshape(3,1)).T
                mesh_p2ds = bs_utils.project_p3d(registered_pts, 1 , K = config.intrinsic_matrix['custom'] )
                kps_2d = bs_utils.project_p3d(kp, 1 , K = config.intrinsic_matrix['custom'] )
                corners_2d = bs_utils.project_p3d(registered_corners, 1 , config.intrinsic_matrix['custom'] )
                rgb_labeled = bs_utils.draw_p2ds(rgb_labeled , p2ds, bs_utils.get_label_color( cls_id=cls_id), 1)
                rgb_kps = bs_utils.draw_p2ds(rgb_kps, kps_2d, (255, 0, 0), 2)
                rgb_labeled_bb = bs_utils.draw_bounding_box(rgb_labeled_bb, corners_2d)

                optical2cam = R.from_euler('zyx',[1.57, 0, 1.57]).as_dcm()
                optical2cam = np.hstack(( optical2cam, np.array([0,0,0]).reshape(3,1) ))

                R_tf = np.matmul(optical2cam.as_dcm(), pose[:3,:3])
                t_tf = pose[:,3]
                t_tf = np.array([ -t_tf[1], -t_tf[2], t_tf[0] ]).reshape(3,1)

                pose_tf = np.matmul( pose.T, optical2cam).T
                pose_tf = np.concatenate((R_tf, t_tf), axis=1)
                pose_tf = pose
                rot = R.from_dcm(pose[0:3,0:3])
                quat = rot.as_quat()

                registered_mesh = (np.matmul(pose_tf[:3,:3],self.mesh[str(cls_id)].T)+pose_tf[:,3].reshape(3,1)).T
                registered_pts = np.concatenate((registered_pts, registered_mesh), axis = 0)
                poses_msg.append(Pose(position = Point(x=pose[0,3],y=pose[1,3],z=pose[2,3]), orientation = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3])))


            cld_registered = pcl.PointCloud()
            cld_registered.from_array(np.array(registered_pts, dtype=np.float32) )
            cloud = pcl.PointCloud()
            kps = pcl.PointCloud()
            kps.from_array(pred_kps[cls])
            cloud.from_array(pcld[:,0:3])
            #colors = 200*np.ones((pcld.shape[0],3))
            colors = pcld[:,3:6]
            #colors[mask[1:],:] = p3d_rgbs[:,3:6]
            #colors[mask,:]= [r,g,b]
            cloud_out = pcl_helper.XYZ_to_XYZRGB(cloud, colors.tolist())
            cloud_msg = pcl_helper.pcl_to_ros(cloud_out, 'camera_optical_link')
            #kp_cld_msg = pcl_helper.pcl_to_ros(kps, 'camera_optical_link')

            print('Publishing All...')
            self.pub1.publish(cloud_msg)
            self.pub2.publish(PoseArray(header=Header(stamp=rospy.Time.now(), frame_id=self.cam_frame), poses = poses_msg))
            self.pub4.publish(ros_numpy.msgify(Image, rgb_labeled_bb,encoding='rgb8'))
            self.pub5.publish(ros_numpy.msgify(Image, rgb_kps,encoding='rgb8'))
            self.pub6.publish(pcl_helper.pcl_to_ros(cld_registered, 'camera_optical_link'))
            print('All published...')


        except Exception as inst:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print('exception: '+str(inst)+' in '+ str(exc_tb.tb_lineno))
            return None

    def ros2torch_CB(self, rgb, depth):

        frame_id = rgb.header.frame_id                    # Check if the recieved msg is from the camera in ros param - Sometimes old msgs in queue are recieved from the previous subscriptions
        if self.request and self.cam_frame == frame_id:   # This just acts as an auto-detect swtch - If turned ON, the callback runs spontaneously in the background and keeps running inference on the incoming img msgs.


            print('TimeSynchronizer subscribed to RGB & Depth topic at: '+ str(rospy.Time.now()))


            try:

                ''' !!Problems wih cvBridge in python 3 '''
                #np_rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding='passthrough')
                #np_dpt = bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
                ''' use  ros_numpy instead '''
                np_rgb = ros_numpy.numpify(rgb)
                np_dpt = ros_numpy.numpify(depth)

                #self.panda_points_img = ros_numpy.numpify(rospy.wait_for_message('/panda_points_projected', Image))

                ''' This step crops out panda-robot-points from depthImage '''
                ''' It can be turned off using remove_panda_dpt arg '''

                # Note: This is a 2nd camera which is not fixed on the robot but looking at the Robot instead.
                if self.remove_panda_dpt:
                    print('Waiting for updated panda binary image...')
                    while self.panda_binImg is None:
                        pass

                    #self.panda_points_img = ros_numpy.numpify(rospy.wait_for_message('/panda_points_projected/binary', Image))
                    ## Since this algorithm only samples points with non-zero depth
                    ## values, we can subtract out all the points representing panda
                    ## in the original depth image by projecting mesh files for all
                    ## panda links using their transforms broadcasted in
                    ## moveit-ROS
                    np_dpt[np.where(np.isnan(np_dpt))] = 0
                    dpt_conv = convert_types(np_dpt,0, 3.0, 0, 255 ,np.int16)       #This temporary conversion from uint to int is to allow negative values after subtraction
                    dpt_diff = dpt_conv - self.panda_binImg.astype(np.int16)
                    dpt_diff[np.where(dpt_diff < 0)] = 0                            #Subtracting binary image of panda-points leaves -ive values on pixels representing panda
                    dpt_panda_removed = convert_types(dpt_diff,0, 255 , 0, 3 ,np.float32)
                else:
                    dpt_panda_removed = np_dpt.copy()

                ## Some dataset had it's rgb channels flipped(to bgr) during training
                ## If detections don't work on the same data during testing
                ## Try flipping the channels
                np_rgb = np.flip(np_rgb, 2)
                #np_rgb = cv2.cvtColor(np_rgb, cv2.COLOR_BGR2GRAY)
                #np_rgb = np.repeat(np.expand_dims( np_rgb, axis = 2), 3, axis = 2)
                #np_rgb[:,:,0] = np_rgb[:,:,2]

                rgb = np.array(np_rgb)[:, :, :3]
                #rgb = rgb[:, :, ::-1].copy()
                rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw
                #print ('RGBShape: ' + str(np_rgb.shape) +'dtype '+str(np_rgb.dtype)+'max_val: '+str(np_rgb.max()))
                #print ('DepthShape: ' + str(np_dpt.shape) +'dtype '+str(np_dpt.dtype)+'max_val: '+str(np.nanmax(np_dpt)))

                if args.dataset == "ycb":
                    K = config.intrinsic_matrix["ycb_K1"]
                elif args.dataset == 'openDR':
                    K = config.intrinsic_matrix["openDR"]
                else:
                    K = config.intrinsic_matrix["custom"]
                #cld, choose = bs_utils.dpt_2_cld(np_dpt, 1, K)

                ## This could be speeded up by subscribing directly to pointcloud topic
                ## and sampling N number of points from it defined in common.config
                cld, choose = bs_utils.dpt_2_cld(dpt_panda_removed, 1, K)
                rgb_lst = []

                for ic in range(rgb.shape[0]):
                    rgb_lst.append(
                        rgb[ic].flatten()[choose].astype(np.float32)
                    )
                rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()
                choose = np.array([choose])
                choose_2 = np.array([i for i in range(len(choose[0, :]))])

                if len(choose_2) < 400:
                    return None
                if len(choose_2) > config.n_sample_points:
                    c_mask = np.zeros(len(choose_2), dtype=int)
                    c_mask[:config.n_sample_points] = 1
                    np.random.shuffle(c_mask)
                    choose_2 = choose_2[c_mask.nonzero()]
                    #print('not wrapped')
                else:
                    leng = len(choose_2)
                    choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')

                cld_rgb = np.concatenate((cld, rgb_pt), axis=1)
                cld_rgb = cld_rgb[choose_2, :]
                cld = cld[choose_2, :]
                normal = bs_utils.get_normal(cld)[:, :3]
                normal[np.isnan(normal)] = 0.0
                cld_rgb_nrm = np.concatenate((cld_rgb, normal), axis=1)
                choose = choose[:, choose_2]

                #print('Cloud NZ: '+str(cld.nonzero()[0].shape))
                #print('Tensor Data loaded to the Model... ')
                #print('cld_rgb_nrm shape '+str(cld_rgb_nrm.shape))
                #print('choose shape'+str( choose[np.newaxis, ...].shape))
                #print('rgb shape '+str(rgb.shape))

                print('ROS - Torch conversion complete!!')
                data_torch = [torch.from_numpy(rgb[np.newaxis, ...].astype(np.float32)), torch.from_numpy(cld_rgb_nrm[np.newaxis, ...].astype(np.float32)),torch.LongTensor(choose[np.newaxis, ...].astype(np.int32))]
                cls_ids, poses, cld, kps = cal_view_pred_pose(self.model, data_torch)
                #print('cls ids NZ shape '+str(cls_ids.nonzero()[0].shape))
                #print('Poses shape'+str(np.array(poses).shape))
                #print('KPs shape '+str(np.array(kps).shape))
                #print('cls_ids shape '+str(cls_ids.shape))
                #print('Cls_ids Detected: '+str( np.unique(cls_ids[cls_ids.nonzero()]) ))
                #cls_ids[cls_ids.nonzero()] -=1
                if len(cls_ids.nonzero()[0]) > 0:
                    print('Cls_ids Detected: '+str( np.unique(cls_ids[0, cls_ids[0,:].nonzero()]) ))
                    if self.pub_all:

                        self.publishAll(np_rgb, cls_ids, poses, cld[:,0:6], kps)
                    else:
                        self.publishPoses(poses, cls_ids)

                    self.poses = poses
                    self.cls_ids = np.unique(cls_ids[0, cls_ids[0,:].nonzero()])
                else:
                    print('No classes detected')
                    self.pub4.publish(ros_numpy.msgify(Image,np_rgb,encoding='rgb8'))

            except Exception as inst:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print('exception: '+str(inst)+' in '+ str(exc_tb.tb_lineno))
                return None

    def getDetections(self, req):

            print('Request for detection received!! Waiting for detection updates...')
            # To get the most recent detections , we reset all the previous poses and labels
            self.poses = None
            self.cls_ids = None
            #self.panda_binImg = None  #Reset this as well in order to get the binImg with latest panda-config in scene

            if rospy.has_param('remove_panda'):             # This param helps other nodes, signal this server whether to remove the scene robot or not.
                self.remove_panda_dpt = rospy.get_param('remove_panda')

            if rospy.has_param('pvn3d_cam'):
                self.cam = rospy.get_param('pvn3d_cam')     # Set the cam, we're getting detections from.
                self.cam_frame = rospy.get_param('pvn3d_cam_frame') # Set camFrame, to publish the poses in.

                # NOTE: In the final tutorial, we continuously switch between the camera fixed on robot and the camera looking at the scene & robot
                # If the testing scenario has one camera only, this param shouldn't exist and the img topics should be the default ones in class def.
                image_sub = message_filters.Subscriber(str(self.cam)+'/color/image_raw', Image, queue_size=1)
                depth_sub = message_filters.Subscriber(str(self.cam)+'/depth/image_raw', Image, queue_size=1)
                self.ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
                self.ts.registerCallback(self.ros2torch_CB)
                rospy.sleep(1)

            # Finally set this, in order to get the TSCallback working.
            self.request = True

            while True:             ## Wait for the time synchronizer to completely go through the ros2torch callback - This is roughly equal to the inference speed
                if self.cls_ids is not None:  #Whenever we get any detection updates...

                    self.request = False            ## Reset the detection request so that the callback doesn't go in auto-detect mode.
                    self.panda_binImg = None  #Reset this as well in order to get the binImg with latest panda-config in scene
                    poses_msg = []
                    for pose in self.poses:
                        rot = R.from_dcm(pose[0:3,0:3])
                        quat = rot.as_quat()
                        poses_msg.append( Pose(position = Point(x=pose[0,3],y=pose[1,3],z=pose[2,3]), orientation = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3]) ) )
                    pose_arr_msg = PoseArray(header=Header(stamp=rospy.Time.now(), frame_id=self.cam_frame), poses = poses_msg)
                    return getPoses_resp(pose_arr_msg , Floats( self.cls_ids.tolist() ))

    def project_pandaCB(self, binImg): #Works in the background, updates the binImg as an attr. of this demo
        print('Binary Image for Panda recieved at detection Node...')
        self.panda_binImg = ros_numpy.numpify(binImg)

''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' Main function for stand-alone PVN3D demo on ROS'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''

def main():

    rospy.init_node('pvn3d_LM_pred')
    '''
    ## Dataset and Cls_id args ##
    if args.dataset == "ycb":
        pass
    else:
        pass'''

    ## Model Initialization ##
    model = PVN3D(
        num_classes=config.n_objects, pcld_input_channels=6, pcld_use_xyz=True,
        num_points=config.n_sample_points, num_kps=config.n_keypoints
    ).cuda()
    model = convert_model(model)
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )
    model = nn.DataParallel(model)

    print('Stand-alone Demo started...')
    while not rospy.is_shutdown():


        demo = PVN3D_ROS(config, model, pub_all=True, auto_detect=True)

        rospy.spin()


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Server Function for working with grasp & assembly clients
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def detection_server():
    rospy.init_node('pvn3d_detection_server')

    ## Model Initialization ##
    model = PVN3D(
        num_classes=config.n_objects, pcld_input_channels=6, pcld_use_xyz=True,
        num_points=config.n_sample_points, num_kps=config.n_keypoints
    ).cuda()
    model = convert_model(model)
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )
    model = nn.DataParallel(model)
    demo = PVN3D_ROS(config, model, pub_all=True, auto_detect=False)

    while not rospy.is_shutdown():

        print('PVN3d Detection server initialized. Waiting for Requests...')
        server = rospy.Service('get_poses_pvn3d', getPoses_srv, demo.getDetections )
        server.spin()


if __name__ == "__main__":
    #main()                 #Runs detections and publishes continuously
    detection_server()      #Runs detections on request-only and returns predictions as a response
