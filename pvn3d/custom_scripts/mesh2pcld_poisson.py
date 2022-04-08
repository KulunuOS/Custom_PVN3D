import open3d as o3d
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="Arg parser")

''' This script loads a .stl .ply or .obj mesh file of the object and runs poisson re-construction on it to sample a preset number of points from the mesh '''
''' This is used for dataset generation when object pointclouds are reprojected into an image at their respective poses, to draw the class-label images from scratch '''

parser.add_argument(
    "-dataset",
    type=str,
    default='openDR',
    help="Define dataset you want to save the pointcloud to",
)


parser.add_argument(
    "-obj_id",
    type=int ,
    default=0,
    help="Class ID of the object ",
)

parser.add_argument(
    "-mesh_file",
    type=str ,
    default=0,
    help="path of the .stl .ply or .obj mesh file ",
)
args = parser.parse_args()


mesh = o3d.io.read_triangle_mesh(str(args.mesh_file))
poisson_pcld = mesh.sample_points_poisson_disk(number_of_points=30000)    #This is the amount that was found to be dense enough i.e., after reprojection into 2D, there are no gaps or blank pixels in the object image

o3d.io.write_point_cloud('./datasets/'+str(args.dataset)+'/'+str(args.dataset)+'_dataset/models/obj_'+str(args.obj_id)+'.ply', poisson_pcld )  


