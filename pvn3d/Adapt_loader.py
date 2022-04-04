# Imports
#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import sys
import open3d as o3d
import torch
import pickle as pkl
import torch.nn as nn
import argparse
from PIL import Image
import os
import numpy as np
from lib.utils.basic_utils import Basic_Utils
from common import Config
config = Config(dataset_name='Adapt')
bs_utils = Basic_Utils(config)

from lib.utils.sync_batchnorm import convert_model
from lib import PVN3D
from lib.utils.pvn3d_eval_utils import cal_frame_poses
import pcl
import lib.utils.pcl_helper as pch
from matplotlib import pyplot as plt
import json
import time
import pandas as pd
# Procedure

# get rgb image
# get depth image
# get cld choose norms ??
# data = (rgb, cld_choose_nrms)
# cls_ids, poses, cld, kps = cal_view_pred_pose(model, data) -----< check cal_frame_poses usage

## Calculating poses from keypoint predictions
#parser = argparse.ArgumentParser(description="Arg parser")

dataset = 'Adapt'
checkpoint = '/home/kulunu/PVN3D/pvn3d/train_log/Adapt/checkpoints/pvn3d_8_kps.pth.tar'
#dat_dir = '/home/kulunu/Desktop/ADAPT_Competition/ADAPT_Pose_Estimation/ADAPT_FB2/validation/'
#dat_dir = '/home/kulunu/PVN3D/pvn3d/datasets/Adapt/Adapt_dataset'
dat_dir = '/home/kulunu/Desktop/ADAPT_Competition/ADAPT_Pose_Estimation_test_split/adapt_fb2_test/test/'
mesh_files_path = '/home/kulunu/PVN3D/pvn3d/datasets/Adapt/Adapt_dataset/models'
corners_txt_path = '/home/kulunu/PVN3D/pvn3d/datasets/Adapt/Adapt_object_kps/'

#test_set = '/home/kulunu/PVN3D/pvn3d/datasets/Adapt/Adapt_dataset/test.txt'

scene_id_leading_zeros = 6
img_id_leading_zeros = 6
rgb_format = '.jpg'
dpt_format = '.png'
cam_scale = 1
use_given_K = True




def get_normal(self, cld):
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
        #print("==> Loading from checkpoint '{}'".format(filename))
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
        #print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None

## Calculating poses from keypoint predictions
def cal_view_pred_pose(model, data):
    model.eval()
    """"
    if model.training:
        print('model state in training')
    else:
        print('model state not in training')
    """
    try:
        #print('Started data acquisition')
        with torch.set_grad_enabled(False): # Data Acquisition
            
            cu_dt = [item.contiguous().to("cuda", non_blocking=True) for item in data]
            rgb, cld_rgb_nrm, choose = cu_dt#.contiguous()
            
            # The 3 main per-point predictions for all seeds
            #  Mk = 307200 * 3 [offset to Kps of each object 1,2,3]
            #  Ms = 307200 * 3 [0 or 1 representatioon whther each point belongs to 1,2,3]
            #  Mc = 307200 * 3 [offset center of each object 1,2,3]

            pred_kp_of, pred_rgbd_seg, pred_ctr_of = model(
                cld_rgb_nrm, rgb, choose
            )
            print('The shape of pred_kp_of :', np.shape(pred_kp_of))
            print('The shape of pred_seg :', np.shape(pred_rgbd_seg))
            print('The shape of pred_ctr_of :', np.shape(pred_ctr_of[0][0]))


            _, classes_rgbd = torch.max(pred_rgbd_seg, -1)
            print('classes_rgbd :', np.shape(classes_rgbd) )
            print('shape of pcld :', np.shape(cld_rgb_nrm[0][:,:3]) )
  
            if dataset == "Adapt":
                
                pred_cls_ids, pred_pose_lst, pred_kps_lst = cal_frame_poses(
                    cld_rgb_nrm[0][:,:3], classes_rgbd[0], pred_ctr_of[0], pred_kp_of[0], True,
                    config.n_objects,False, ds='Adapt'
                )

            else:
                print('Error')

            print('predicted_cls_ids :',pred_cls_ids)
            print('predicted_pose_list :',np.shape(pred_pose_lst))
            print('predicted_kps_list :',np.shape(pred_kps_lst))

            print('Prediction Complete...')
            return classes_rgbd.cpu().numpy(), pred_pose_lst, cld_rgb_nrm[0].cpu().numpy(), pred_kps_lst
    except Exception as inst:
        _, _, exc_tb = sys.exc_info()
        print('exception: '+str(inst)+' in '+ str(exc_tb.tb_lineno))

class Adapt_dataset():

    def __init__(self, scene_id):

        self.dir = dat_dir + str(scene_id).zfill(scene_id_leading_zeros)
        #self.dir = dat_dir
        self.rgb_dir = self.dir +'/rgb'  
        self.dpt_dir = self.dir+'/depth'    
        self.mesh = {}
        self.corners = {}

        for i in range(1, 4):
                self.mesh[str(i)] = np.array(pcl.load(mesh_files_path+'/obj_'+str(i)+'.pcd'))
                self.corners[str(i)] = np.loadtxt(corners_txt_path+str(i)+'/corners.txt')



    # The function return an item at a time
    def get_item(self,im_id):  
        
        # read rgb images from directory and load to a numpy array
        with Image.open(os.path.join(self.rgb_dir,str(im_id).zfill(img_id_leading_zeros)+ rgb_format)) as ri:
            rgb = np.array(ri)[:, :, :3]
        # convert from [height x width x channel] to [channel x height x width]
        rgb = np.transpose(rgb, (2, 0, 1))
        

        # read depth images from directory and load to a numpy array
        with Image.open(os.path.join(self.dpt_dir,str(im_id).zfill(img_id_leading_zeros)+ dpt_format)) as di:
            dpt = np.array(di)
            dpt = dpt/65535 #depth images are 16bit (0-65535)        
            dpt = dpt*30.0   #input range of kinect
        
        # depth image is converted to a point-cloud
        if use_given_K :
                    with open(os.path.join(self.dir,'scene_camera.json'), "r") as f:
                        mat = json.load(f)
                        K = np.array(mat[str(im_id)]['cam_K']).reshape(3,3)
        else :
             K = np.array([[615.1397026909722, 0.0, 322.78683465719223], [0.0, 615.4323641853988, 244.22407517651774], [0.0, 0.0, 1.0]]) #camera insintric
            # K = np.array([[554.25469119, 0.,         320.5],[0.,        554.25469119,  240.5],[0.,        0.,         1.]])
        
        cld, choose = bs_utils.dpt_2_cld(dpt, cam_scale, K)
        rgb_lst = []

        for ic in range(rgb.shape[0]):
            rgb_lst.append(
                rgb[ic].flatten()[choose].astype(np.float32)
            )
        rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()
        choose = np.array([choose])
        choose_2 = np.array([i for i in range(len(choose[0, :]))])

        if len(choose_2) < 400:
            print("Valid points are less than 400")
            return None
        if len(choose_2) > config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
            print("Valid points are more than configured")
        else:
            leng = len(choose_2)
            choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')
            print("Valid points are optimum")

        cld_rgb = np.concatenate((cld, rgb_pt), axis=1)
        cld_rgb = cld_rgb[choose_2, :]
        cld = cld[choose_2, :]
        normal = bs_utils.get_normal(cld)[:, :3]
        normal[np.isnan(normal)] = 0.0
        cld_rgb_nrm = np.concatenate((cld_rgb, normal), axis=1)
        choose = choose[:, choose_2]
        
        #print('Torch conversion complete!!')
        data_torch = [torch.from_numpy(rgb[np.newaxis, ...].astype(np.float32)), torch.from_numpy(cld_rgb_nrm[np.newaxis, ...].astype(np.float32)),torch.LongTensor(choose[np.newaxis, ...].astype(np.int32))]
        #cls_ids, poses, cld, kps = cal_view_pred_pose(self.model, data_torch)
        

        return data_torch , rgb, self.mesh, self.corners,K
        
# This could be the function main()
def evaluate(scn_id, im_id,model):   

    item = Adapt_dataset(scn_id)
    data_torch , rgb_img, mesh, corners,K = item.get_item(im_id)
    # data_torch , rgb_img, mesh, corners,K = item.get_item(obj_id,img_id,scene_id)

    cls_ids, poses, cld, kps = cal_view_pred_pose(model, data_torch)
    pcld = cld[:,0:6]

    #print(np.unique(cls_ids))
    #print(np.shape(poses))
    #print(poses)

    registered_pts = np.zeros((1,3))
    rgb_labeled_bb = rgb_img.copy()
    rgb_labeled_bb = np.transpose(rgb_labeled_bb, (1, 2, 0)) 
    #print('rgb_shape', np.shape(rgb_labeled_bb))
    poses_msg = []

    return cls_ids, poses, pcld, kps,mesh,corners,K, rgb_labeled_bb
#print(corners)

#write json file
def show_result(cls_ids, poses, kps,mesh,corners,pcld,K,rgb_labeled_bb):

    for i, cls_id in enumerate(np.unique(cls_ids[cls_ids.nonzero()] )):
        
        #print (cls_id)
        mask = np.where(cls_ids[0,:] == cls_id)[0]
        pose = np.asarray(poses[i])

        kp = kps[i]

        this_mesh = mesh[str(i+1)]
        this_mesh = np.asarray(this_mesh)
        #print('mesh :',this_mesh)
        
        
        this_corners = corners[str(i+1)]
        this_corners = np.asarray(this_corners)
        #print('corners :',this_corners)

        #print(np.shape(pose[:3,:3]))
        print('used', K)
        p2ds = bs_utils.project_p3d(pcld[mask,0:3], 1 , K=K )
        registered_pts = (np.matmul(pose[:3,:3],this_mesh.T)+pose[:,3].reshape(3,1)).T
        registered_corners = (np.matmul(pose[:3,:3],this_corners.T)+pose[:,3].reshape(3,1)).T
        mesh_p2ds = bs_utils.project_p3d(registered_pts, 1 , K=K  )
        kps_2d = bs_utils.project_p3d(kp, 1 , K=K )
        corners_2d = bs_utils.project_p3d(registered_corners, 1 ,K=K )
        rgb_labeled = bs_utils.draw_p2ds(rgb_labeled_bb , p2ds, bs_utils.get_label_color( cls_id=cls_id), 1)
        print('rgb_labeled', np.shape(rgb_labeled))
        rgb_kps = bs_utils.draw_p2ds(rgb_labeled, kps_2d, (255, 0, 0), 2)
        print('rgb_kps', np.shape(rgb_kps))
        rgb_labeled_bb = bs_utils.draw_bounding_box(rgb_kps, corners_2d)
        print('rgb_labeled', np.shape(rgb_labeled_bb))

    plt.imshow(rgb_labeled_bb, interpolation='nearest')
    plt.show()
   

def get_json():
    f = open('/home/kulunu/Desktop/ADAPT_Competition/ADAPT_Pose_Estimation_test_split/adapt_fb2_test/test_targets_adapt.json')
    info = json.load(f)
    
    submissions = []
    for i,im in enumerate(info):
        
        print('Writing for test img_id',i)
        im = info[i]
        im_id = im['im_id']
        n_cnt = im['inst_count']
        cls_id = im['obj_id']
        scn_id = im['scene_id']

        cls_ids, poses,_, _,_,_,_,_ = evaluate(scn_id,im_id)
        cls_id = int(cls_id)

        if cls_id in np.unique(cls_ids):
            print(cls_id)
            print(np.unique(cls_ids))
            print(np.unique(cls_ids).tolist().index(cls_id))
            j = np.unique(cls_ids).tolist().index(cls_id) - 1
         
            pose = poses[j]
            Rt = pose[:3,:3].tolist() 
            Tt = pose[:,3].tolist()
        else:
           
            pose = []
            Rt = [0,0,0] 
            Tt = []
            
        submissions.append(
            {   
                "img_id": im_id,
                "obj_id": cls_id, 
                "cam_R_m2c": Rt,
                "cam_t_m2c": Tt
            } 
        )
    with open('/home/kulunu/Desktop/ADAPT_Competition/submission.json', 'w') as fp:
        json.dump(submissions, fp)

#cls_ids, poses, pcld, kps,mesh,corners,K ,rgb_labeled_bb= evaluate(scn_id= '',im_id=11)
#show_result(cls_ids, poses, kps,mesh,corners,pcld,K,rgb_labeled_bb)


def get_csv(model):
    " generate a CSV File as result. one line = scene_id, im_id, obj_id, score, R, t, time "

    
    # for item in test data test and save a line
    f = open('/home/kulunu/Desktop/ADAPT_Competition/ADAPT_Pose_Estimation_test_split/adapt_fb2_test/test_targets_adapt.json')
    info = json.load(f)

    for i,im in enumerate(info):
    #for i in range(4073,8399):
        print('Writing for test img_id',i)
        start = time.process_time()

        """
        if model.training:
            print('model state in training')
        else:
            print('model state not in training')
        """
        im = info[i]
        im_id = im['im_id']
        n_cnt = im['inst_count']
        cls_id = im['obj_id']
        scn_id = im['scene_id']
        
        item = Adapt_dataset(scn_id)
        data_torch ,_,_,_,_= item.get_item(im_id)
        cls_ids,poses,_,_ = cal_view_pred_pose(model, data_torch)
        cls_id = int(cls_id)

        if cls_id in np.unique(cls_ids):
            print(cls_id)
            print(np.unique(cls_ids))
            print(np.unique(cls_ids).tolist().index(cls_id))
            j = np.unique(cls_ids).tolist().index(cls_id) - 1
         
            pose = poses[j]
            Rt = tuple(pose[:3,:3].flatten('C'))
            Tt = tuple(pose[:,3].flatten('C'))

            score = 0.5
            t = time.process_time() - start
            sub = pd.DataFrame([[scn_id,
                                im_id,
                                cls_id,
                                score, 
                                ' '.join(map(str,Rt)), 
                                ' '.join(map(str,Tt)), t ]],
                                columns=None)
            if i == 0:
                sub.to_csv('/home/kulunu/Desktop/ADAPT_Competition/submission.csv',header=['scene_id','im_id','obj_id','score','R','t','time'],index=False, mode='w')
            else:
                sub.to_csv('/home/kulunu/Desktop/ADAPT_Competition/submission.csv',header=None,index=False, mode='a')

def main():
    #Load the model
    model = PVN3D(
                num_classes=config.n_objects, pcld_input_channels=6, pcld_use_xyz=True,
                num_points=config.n_sample_points, num_kps=config.n_keypoints
                    ).cuda()
    model = convert_model(model)
    model.cuda()
    if checkpoint is not None:
        checkpoint_status = load_checkpoint(
            model, None, filename=checkpoint[:-8]
        )
    model = nn.DataParallel(model)
    
    """"
    if model.training:
        print('model state in training')
    else:
        print('model state not in training')
    """
    #get_csv(model)
    cls_ids, poses, pcld, kps,mesh,corners,K, rgb_labeled_bb = evaluate(0,1,model)
    show_result(cls_ids, poses, kps,mesh,corners,pcld,K,rgb_labeled_bb)

if __name__ == "__main__":
    main()