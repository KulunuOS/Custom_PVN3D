import sys
import lib.utils.pcl_helper as pch
sys.path.remove('/opt/ros/noetic/lib/python3/dist-packages')		#This is to remove ROS-python from the PYTHONPATH which messes up the Python 3 env this project works with
import pcl
import lib.utils.basic_utils as bs_utl
import numpy as np
import argparse
import os

''' This script loads a pcd or ply file of an object model and extract the corners of it's bounding-box and saves it to a txt file '''

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
	print('Writing corners for class id '+str(cls_id))
	pcd = pcl.load('./datasets/'+args.dataset+'/'+dataset_folder[args.dataset]+'/models/obj_'+str(cls_id)+'.pcd')
	pts = np.array(pcd)
	#fps_idx = bs_utl.farthestPointSampling(pts, 8)
	corners = []
	for i in range(0,2):	
	
		if (i == 0):
			x = np.min(pts[:,0])		
		else:
			x = np.max(pts[:,0])

		for j in range(0,2):
		
			if (j == 0):
				y = np.min(pts[:,1])		
			else:
				y = np.max(pts[:,1])
		
			for k in range(0,2):
		
				if (k == 0):
					z = np.min(pts[:,2])		
				else:
					z = np.max(pts[:,2])
	
				corners.append([x,y,z])


	corners = np.array(corners)
	#print(corners.shape)
	pts = np.vstack(( pts, corners ))
	colors = 100*np.ones((pts.shape[0],3))
	colors[-8:-1,:] = np.array([255, 0 , 0])

	colors = colors.tolist()

	cloud_corners = pch.XYZ_to_XYZRGB(pts, colors)


	np.savetxt('./datasets/'+args.dataset+'/'+kps_folder[args.dataset]+'/'+str(cls_id)+'/corners.txt', corners)	#corners
	pcd = pcl.save(cloud_corners,'./datasets/'+args.dataset+'/'+kps_folder[args.dataset]+'/'+str(cls_id)+'/corners.pcd')  ##PCD file with corners added to the actual cloud
