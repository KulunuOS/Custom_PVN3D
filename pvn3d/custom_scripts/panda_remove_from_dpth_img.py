import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
import open3d as o3d
import tf
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import GetModelState
import ros_numpy
import numpy as np
from scipy.spatial.transform import Rotation as R
import collections


''' This sript runs a node that listens camera and robot transforms in world-frame in order 
    to remove points from Franka Panda's links visible in a depth image. It transforms each
    link's mesh to it's respective pose and projects it to 2D to get a binary image.
    It then subtracts this binary image from the original depth image '''

    ## It can be imported and initialized as a Node object in another script or can
    ## be run from  the main()

def project_p3d( p3d, cam_scale, K):
        if p3d.shape[1]<4:
            p3d = p3d * cam_scale
            p2d = np.dot(p3d, K.T)
            p2d_3 = p2d[:, 2]
            p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
            p2d[:, 2] = p2d_3
            p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
            return p2d

def convert_types(img, orig_min, orig_max, tgt_min, tgt_max, tgt_type):

    #info = np.finfo(img.dtype) # Get the information of the incoming image type
    # normalize the data to 0 - 1
    img_out = img / (orig_max-orig_min)   # Normalize by input range

    img_out = (tgt_max - tgt_min) * img_out # Now scale by the output range
    img_out = img_out.astype(tgt_type)
    print(img_out.max())
    #cv2.imshow("Window", img)
    return img_out


def draw_p2ds( img, p2ds, color, rad):
    h, w = img.shape[0], img.shape[1]
    for pt_2d in p2ds:
        pt_2d[0] = np.clip(pt_2d[0], 0, w)
        pt_2d[1] = np.clip(pt_2d[1], 0, h)
        if p2ds.shape[1]>2:
            img = cv2.circle(
                cv2.UMat(img), (pt_2d[0], pt_2d[1]), rad, (int(pt_2d[2]), int(pt_2d[3]), int(pt_2d[4])) , -1
            )
        else:
            img = cv2.circle(
                cv2.UMat(img), (pt_2d[0], pt_2d[1]), rad, color, -1
            )
        '''
        img = cv2.circle(
            img, (pt_2d[0], pt_2d[1]), rad, color, -1
        )'''
    return img.get()


class Panda_pts(object):

    def __init__(self, cam_sub, cam_info_sub, link_list)
    rospy.init_node('panda_dpt_pts')
    self.listener = tf.TransformListener()
    self.cam_sub = cam_sub
    #self.link_list = ['link0','link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'camera_link']
    self.link_list = link_list
    #all_link_meshes = {'link0':[], 'link1':[], 'link2':[], 'link3':[], 'link4':[],'link5':[], 'link6':[], 'hand':[]  }
    self.bridge = CvBridge()
    self.intrinsic_mat = np.array(rospy.wait_for_message(cam_info_sub, CameraInfo).P).reshape(3,4)


    def get_bin_img(self):
        ## This is for a gazebo-simulated camera
        try:
            rospy.wait_for_service('gazebo/get_model_state')
            client = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
            resp1 = client('kinect_ros', 'world')
            break

        except Exception as inst:
            print('Error in gazebo/get_link_state service request: ' + str(inst) )
            print('No camera found')

        dpt_img = rospy.wait_for_message(self.cam_sub, Image)
        cv_im = self.bridge.imgmsg_to_cv2(dpt_img, desired_encoding='passthrough')
        cv_dpt = cv_im.copy().astype(np.float32)

        ## Camera poses from ROS Message to Numpy arrays
        cam_pos = np.array([resp1.pose.position.x, resp1.pose.position.y, resp1.pose.position.z]).reshape(3,1)
        cam_quat = np.array([resp1.pose.orientation.x, resp1.pose.orientation.y, resp1.pose.orientation.z, resp1.pose.orientation.w])

        rotMat = R.from_quat(cam_quat).as_dcm()
        cam_in_world = np.hstack (( rotMat, cam_pos ))
        cam_in_world = np.vstack(( cam_in_world , [0,0,0,1] ))

        panda_points = np.empty((1,4))

        for i in self.link_list:

        #for i in ['link1']:

            (trans, rot) = self.listener.lookupTransform('world', '/panda_'+i , rospy.Time(0))
            link_pos = np.array(trans)
            link_quat = np.array(rot)
            l_rotMat = R.from_quat(link_quat).as_dcm()
            link_in_world = np.hstack (( l_rotMat, link_pos.reshape(3,1) ))
            link_in_world = np.vstack(( link_in_world, [0, 0, 0, 1] ))

            ## Mesh Transformation of each link to camera coordinates
            link_in_cam = np.dot(np.linalg.inv(cam_in_world), link_in_world  )
            link_pts = np.asarray(o3d.io.read_point_cloud('~/panda_sim_ws/src/franka_ros/franka_description/meshes/visual/'+ i +'.pcd').points)
            link_pts = np.hstack(( link_pts, np.ones((link_pts.shape[0], 1)) ))
            link_pts_tf = np.matmul(link_in_cam, link_pts.T ).T
            panda_points = np.vstack(( panda_points , link_pts_tf ))

            ### Perspective projection of each link onto Depth Image
            cam2optical = R.from_euler('zyx',[1.57, 0, 1.57] ).as_dcm()
            cam2optical = np.hstack(( np.vstack(( cam2optical , [0,0,0] )) , np.array([[0],[0],[0],[1]]) ))
            panda_points = np.matmul( cam2optical, panda_points.T ).T

            bin_img = np.zeros((480, 640)).astype(np.uint8)
            panda_points_2D = project_p3d(panda_points[:,:3] , 1, self.intrinsic_mat[:3,:3])

            print('Projecting panda-points...Might take a few seconds....')
            panda_points_img = draw_p2ds(bin_img, panda_points_2D, (255,255,255), 1)
            return panda_points_img, panda_points


    def publishBinImg(self, binImg):
        self.pubBin = rospy.Publisher('/panda_points_projected/binary', Image)
        self.pubBin.publish(ros_numpy.msgify(Image, binImg.astype(np.uint8), encoding='mono8') )


    def publishDiffImg(self, dptImg, binImg):
        self.pubDiff = rospy.Publisher('/panda_points_projected/diff', Image)
        dptImg[np.where(np.isnan(cv_dpt))] = 0
        dpt_conv = convert_types(dptImg,0, 3.0, 0, 255 ,np.int16)
        dpt_diff = dpt_conv - binImg.astype(np.int16)
        dpt_diff[np.where(dpt_diff < 0)] = 0
        pub.publish(ros_numpy.msgify(Image, dpt_diff.astype(np.uint8), encoding='mono8') )

if __name__ == "__main__":

    while not rospy.is_shutdown():
        rospy.init_node('panda_dpt_pts')
        cam_topic ='/kinect1/color/image_raw'
        cam_info_topic = '/kinect1/color/camera_info'
        link_list = ['link0','link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'camera_link']
        if rospy.has_param('remove_panda') and rospy.get_param('remove_panda'):
            panda = Panda_pts(cam_topic, cam_info_topic, link_list)
            panda_points_img, panda_points = panda.get_bin_img()
            panda.publishBinImg(panda_points_img)
