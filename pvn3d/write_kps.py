import sys
import os
import lib.utils.pcl_helper as pch
sys.path.remove('/opt/ros/noetic/lib/python3/dist-packages')		#This is to remove ROS-python from the PYTHONPATH which messes up the Python 3 env this project works with
import lib.utils.basic_utils as bs_utl #Basic_Utils
import pcl
import numpy as np
import argparse


''' This script loads a pcd or ply file of an object model and runs Farthest-point sampling on it to save a txt file with keypoints in Object coordinates '''
parser = argparse.ArgumentParser(description="Arg parser")


parser.add_argument(
    "-dataset",
    type=str,
    default='openDR',
    help="Define dataset you want to save the pointcloud to",
)


parser.add_argument(
    "-cls_id",
    type=int ,
    default=0,
    help="Class ID of the object ",
)

parser.add_argument(
    "-n_kps",
    type=int ,
    default=16,
    help="No. of required keypoints per class",
)
args = parser.parse_args()

dataset_folder = {'openDR':'openDR_dataset', 'linemod':'Linemod_preprocessed', 'ycb':'ycb_dataset', 'CrankSlider': 'CrankSlider_dataset', 'Adapt':'Adapt_dataset'}
kps_folder = {'openDR':'openDR_object_kps', 'linemod':'lm_object_kps', 'ycb':'ycb_object_kps', 'CrankSlider': 'CrankSlider_object_kps', 'Adapt': 'Adapt_object_kps'}


cls_ids = args.cls_id

if args.dataset == 'openDR':
	cls_ids = [1,2,3,4,5,6,7,8,9,10]
elif args.dataset == 'CrankSlider':
	cls_ids = [1,2,3,4,5,6,7,8]
elif args.dataset == 'Adapt':
	cls_ids = [1,2,3]


for cls_id in cls_ids:
	if not os.path.exists('./datasets/'+args.dataset+'/'+ kps_folder[args.dataset]+'/'+ str(cls_id)):
		os.makedirs('./datasets/'+args.dataset+'/'+ kps_folder[args.dataset]+'/'+ str(cls_id))
	print('Writing kps for class '+str(cls_id))
	pcd = pcl.load('./datasets/'+args.dataset+'/'+dataset_folder[args.dataset]+'/models/obj_'+str(cls_id)+'.pcd')
	pts = np.array(pcd)
	fps_idx = bs_utl.farthestPointSampling(pts, args.n_kps)



	colors = 100*np.ones((pts.shape[0],3))
	colors[fps_idx] = np.array([255, 0 , 0])

	colors2 = 100*np.ones((pts[fps_idx].shape[0],3))
	#colors[:] = np.array([255, 0 , 0])
	colors = colors.tolist()

	#cloud_kps = pch.XYZ_to_XYZRGB(pts[fps_idx], colors2)
	cloud_kps = pch.XYZ_to_XYZRGB(pts, colors)

	np.savetxt('./datasets/'+args.dataset+'/'+kps_folder[args.dataset]+'/'+str(cls_id)+'/farthest_'+str(args.n_kps)+'.txt', pts[fps_idx])	#Keypoints
	pcd = pcl.save(cloud_kps,'./datasets/'+args.dataset+'/'+kps_folder[args.dataset]+'/'+str(cls_id)+'/farthest_'+str(args.n_kps)+'.pcd')	#PCD file with keypoints added to the actual cloud
