import numpy as np
import rospy
import tf2_ros
import sys
from Panda_pts import Panda_pts
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion,TransformStamped, Transform, Vector3
from sensor_msgs.msg import Image
import ros_numpy
from rospy_tutorials.msg import Floats
#from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from panda_simulation.srv import computeGrasps_multiClass, getPoses_multiClass
from scipy.spatial.transform import Rotation as R

rospy.init_node('demo_cranfield_assembly')
camSub = '/kinect1/depth/image_raw'
camInfoSub = '/kinect1/color/camera_info'
link_list = ['link0','link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'camera_link', 'rightfinger', 'leftfinger']

''' This script runs a second demo that acts as a client to the detection server in demo_ros.py
    This is to facilitate loading predefined grasps and transforming them relative to PVN3D poses
    It also acts as a server for moveit-based clients that request for grasps to execute '''
    ## The name 'cranfield' is a bit ambiguous as the original demo was supposed to be for
    ## a cranfield assembly task. Nevertheless it's a generic grasp server, that requests
    ## poses from PVN3D server and returns grasps which can then be used to performs any
    ## assembly or evaluation demo

class Demo_CF(object):

    def __init__(self, sub):

        #self.sub = rospy.Subscriber(sub, PoseArray, self.pvn3d_CB)                  ## Could also be used instead of /get_poses_pvn3d service - Old implementation
        self.poses = None
        self.labels = None
        self.grasp_pub = rospy.Publisher('/grasps_pvn3d', PoseArray, queue_size=10)
        self.cls_dict = {1:'piston', 2:'round_peg', 3:'square_peg', 4:'pendulum', 5:'pendulum_head', 6:'separator', 7:'shaft', 8:'face_plate', 9:'valve_tappet', 10:'bolt'}

        print('Node: demo_cranfield_assembly, waiting for /get_poses_pvn3d service proxy...')
        rospy.wait_for_service('get_poses_pvn3d')
        print('Successfully connected to /get_poses_pvn3d!!')
        self.grasp_server = rospy.Service('compute_grasps', computeGrasps_multiClass, self.get_grasps_handle)
        self.poses_client = rospy.ServiceProxy('get_poses_pvn3d', getPoses_multiClass)
        #self.listener = tf.TransformListener()

    def pvn3d_CB(self, poses):
        self.poses = poses.poses

    def get_detections(self):
        '''
        rob_binImg, _ = robot_pts.get_bin_img()
        robot_pts.publishBinImg(rob_binImg)
        robot_pts.publishDiffImg(rob_binImg)'''

        #poses = rospy.wait_for_message('/pvn3d_pose', PoseArray).poses
        ## NEW IMPLEMENTATION - The detections are only updated when requested by the poses_client
        resp = self.poses_client(True)
        self.poses = resp.poses.poses
        self.labels = resp.objectLabels.data


        ## OLD IMPLEMENTATION - The detections update on subscriber callback
        '''
        if self.poses is not None:

            self.labels = []
            for i, pose in enumerate(self.poses):
                position = (pose.position.x, pose.position.y, pose.position.z)
                orientation = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w )
                if position!=(0,0,0) and orientation!=(0,0,0,1):
                    self.labels.append(i+1)
            return self.poses, self.labels
        else:
            return [],[] '''

    '''
    def transform_poses(self, poses, transform):

        transformed_poses = [0] * len(poses)
        for i, pose in enumerate(self.poses)
            transformed_poses[i] = do_transform_pose(pose, transform)

        return transformed_poses'''

    def transform_poses(self, poses, transform):

        transform_np = ros_numpy.numpify(transform.transform)

        transformed_poses = [0] * len(poses)
        for i, pose in enumerate(self.poses):

            pose_np = ros_numpy.numpify(pose)
            transformed_poses[i] = np.matmul(transform_np, pose_np)

        return transformed_poses

    def arrange_poses_euc_dist(self):
        '''This function is to arrange the poses, labels and their graps based on their euclidean distance from the camera frame
           This ensures robot always picks up the closest object first'''

        euc_dist = []
        for i, lb in enumerate(self.labels):

            euc_dist.append( np.linalg.norm(ros_numpy.numpify(self.poses[i])[0:3,3]) )

        euc_dist = np.array(euc_dist)
        sorting = np.argsort(euc_dist)
        print(sorting)
        self.poses = np.array(self.poses)[sorting].tolist()
        self.labels = np.array(self.labels)[sorting].tolist()

    def get_grasps_handle(self, req):

        print('compute_grasps request recieved...')
        self.get_detections() # Request an update on detections everytime, we recieve a computeGrasp request.
        self.arrange_poses_euc_dist() #Arrange so that robot always picks up the closest object first

        # The cam frame of the camera used for detections i.e., There are two cameras being used for the demo
        if rospy.has_param('pvn3d_cam_frame'):
            self.cam_frame = rospy.get_param('pvn3d_cam_frame')

        try:
            tf_stamped = tf_buffer.lookup_transform('world', self.cam_frame , rospy.Time(0))
            poses_in_world = self.transform_poses(self.poses, tf_stamped)

            grasp_msgs = []
            grasp_labels = []
            grasp_offs_X = []
            grasp_offs_Y = []
            grasp_offs_yaw = []

            if self.labels is not None:

                for i, lb in enumerate(self.labels):

                    preDefined_grasps = np.loadtxt('./datasets/openDR/dataset_config/preDefined_grasps/'+str(int(lb))+'.txt')


                    print('Loaded '+str(preDefined_grasps.shape[0])+' grasps for '+ self.cls_dict[lb])

                    for j in range(preDefined_grasps.shape[0]):

                        gr_rot = R.from_euler('zyx', preDefined_grasps[j, 3:6] ).as_dcm()
                        gr_pos = preDefined_grasps[j, 0:3].reshape(3,1)

                        #obj_pos = np.array([poses_in_world[i].position.x, poses_in_world[i].position.y, poses_in_world[i].position.z])
                        #obj_rot = R.from_quat(np.array([poses_in_world[i].orientation.x, poses_in_world[i].orientation.y, poses_in_world[i].orientation.z, poses_in_world[i].orientation.w]) ).as_dcm()

                        grasp_in_obj = np.vstack(( np.hstack(( gr_rot, gr_pos)), np.array([0,0,0,1]) ))
                        #obj_in_world = np.hstack(( obj_rot, obj_pos ))
                        obj_in_world = poses_in_world[i]
                        grasp_in_world = np.matmul( obj_in_world, grasp_in_obj )

                        grasp_msgs.append(ros_numpy.msgify(Pose, grasp_in_world ))
                        #grasp_msgs.append(ros_numpy.msgify(Pose, self.poses ))
                        grasp_labels.append(lb) ## Labeling every grasp with its respective class label.
                        grasp_offs_X.append(preDefined_grasps[j, 6])
                        grasp_offs_Y.append(preDefined_grasps[j, 7])
                        grasp_offs_yaw.append(preDefined_grasps[j, 8])

                print(grasp_labels)
                self.grasp_pub.publish(PoseArray(header=Header(stamp=rospy.Time.now(), frame_id='world'), poses = grasp_msgs) )
                return PoseArray(header=Header(stamp=rospy.Time.now(), frame_id='world'), poses = grasp_msgs), Floats(grasp_labels), Floats(grasp_offs_X), Floats(grasp_offs_Y), Floats(grasp_offs_yaw)

        except Exception as inst:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print('exception: '+str(inst)+' in '+ str(exc_tb.tb_lineno))

    def main(self):

        while not rospy.is_shutdown():

            poses, labels = self.get_detections()

            #print('Detected labels: '+str(labels))



            #rospy.spin()

if __name__=="__main__":

        rospy.loginfo('compute_grasps service initialized')
        tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        #robot_pts = Panda_pts(camSub, camInfoSub, link_list)
        demo = Demo_CF('/pvn3d_pose')
        #demo.main()
        demo.grasp_server.spin()
	

