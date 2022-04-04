from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import open3d as o3d
import sys
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, Image
from lib.utils import pcl_helper
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation as R
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import open3d as o3d
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import pprint
import os.path as osp
import os
import argparse
import time
import shutil
import pcl
import tqdm
from lib.utils.etw_pytorch_utils.viz import *
from lib import PVN3D
from datasets.openDR.openDR_dataset import openDR_Dataset
from lib.loss import OFLoss, FocalLoss
from common import Config
from lib.utils.sync_batchnorm import convert_model
from lib.utils.warmup_scheduler import CyclicLR
from lib.utils.pvn3d_eval_utils import TorchEval
import lib.utils.etw_pytorch_utils as pt_utils
import resource
from collections import namedtuple
from lib.utils.basic_utils import Basic_Utils

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))
config = Config(dataset_name='openDR')
bs_utl = Basic_Utils(config)

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-weight_decay",
    type=float,
    default=0,
    help="L2 regularization coeff [default: 0.0]",
)
parser.add_argument(
    "-lr", type=float, default=1e-2, help="Initial learning rate [default: 1e-2]"
)
parser.add_argument(
    "-lr_decay",
    type=float,
    default=0.5,
    help="Learning rate decay gamma [default: 0.5]",
)
parser.add_argument(
    "-decay_step",
    type=float,
    default=2e5,
    help="Learning rate decay step [default: 20]",
)
parser.add_argument(
    "-bn_momentum",
    type=float,
    default=0.9,
    help="Initial batch norm momentum [default: 0.9]",
)
parser.add_argument(
    "-bn_decay",
    type=float,
    default=0.5,
    help="Batch norm momentum decay gamma [default: 0.5]",
)
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to start from"
)
parser.add_argument(
    "-epochs", type=int, default=1000, help="Number of epochs to train for"
)
parser.add_argument(
    "-run_name",
    type=str,
    default="sem_seg_run_1",
    help="Name for run in tensorboard_logger",
)

parser.add_argument("--test", action="store_true")
parser.add_argument("--cal_metrics", action="store_true")

lr_clip = 1e-5
bnm_clip = 1e-2


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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


def save_checkpoint(
        state, is_best, filename="checkpoint", bestname="model_best",
        bestname_pure='pvn3d_best'
):
    filename = "{}_{}_kps.pth.tar".format(filename, config.n_keypoints)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "{}.pth.tar".format(bestname))
        shutil.copyfile(filename, "{}.pth.tar".format(bestname_pure))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
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


def model_fn_decorator(
    criterion, criterion_of, test=False
):
    modelreturn = namedtuple("modelreturn", ["preds", "loss", "acc"])
    teval = TorchEval('openDR')

    def model_fn(
        model, data, epoch=0, is_eval=False, is_test=False, finish_test=False
    ):
        if finish_test:
            teval.cal_auc('openDR')
            return None
        if is_eval:
            model.eval()
        with torch.set_grad_enabled(not is_eval):
            cu_dt = [item.to("cuda", non_blocking=True) for item in data]
            rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, ctr_targ_ofst, \
                cls_ids, rts, labels, kp_3ds, ctr_3ds = cu_dt

            pred_kp_of, pred_rgbd_seg, pred_ctr_of = model(
                cld_rgb_nrm, rgb, choose
            )

            #print('labels '+ str(np.unique(labels.cpu().numpy())))
            loss_rgbd_seg = criterion(
                pred_rgbd_seg.view(labels.numel(), -1),
                labels.view(-1)
            ).sum()
            loss_kp_of = criterion_of(
                pred_kp_of, kp_targ_ofst, labels,
            ).sum()
            loss_ctr_of = criterion_of(
                pred_ctr_of, ctr_targ_ofst, labels,
            ).sum()
            w = [2.0, 1.0, 1.0]
            loss = loss_rgbd_seg * w[0] + loss_kp_of * w[1] + \
                   loss_ctr_of * w[2]

            _, classes_rgbd = torch.max(pred_rgbd_seg, -1)
            acc_rgbd = (
                classes_rgbd == labels
            ).float().sum() / labels.numel()

            if is_test:
                pred_pose_lst,pred_kps_lst,_,_2 = teval.eval_pose_parallel(
                    pcld, rgb, classes_rgbd,
                    pred_ctr_of, ctr_targ_ofst,
                    labels, epoch,
                    cls_ids, rts, pred_kp_of,
                    min_cnt=1, use_p2d=False, use_ctr_clus_flter=True,
                    use_ctr=True, ds_type="openDR"
                )

        return modelreturn(
            (pred_kp_of, pred_rgbd_seg, pred_ctr_of,  pred_pose_lst, cld_rgb_nrm, rts, rgb, pred_kps_lst), loss,
            {
                "acc_rgbd": acc_rgbd.item(),
                "loss": loss.item(),
                "loss_rgbd_seg": loss_rgbd_seg.item(),
                "loss_kp_of": loss_kp_of.item(),
                "loss_ctr_of": loss_ctr_of.item(),
                "loss_target": loss.item(),
            }
        )

    return model_fn


class Trainer(object):
    r"""
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    """

    def __init__(
        self,
        model,
        model_fn,
        optimizer,
        checkpoint_name="ckpt",
        best_name="best",
        lr_scheduler=None,
        bnm_scheduler=None,
        viz=None,
    ):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model,
            model_fn,
            optimizer,
            lr_scheduler,
            bnm_scheduler,
        )

        self.checkpoint_name, self.best_name = checkpoint_name, best_name

        self.training_best, self.eval_best = {}, {}
        self.viz = viz

    def eval_epoch(self, d_loader, is_test=False):
        self.model.eval()

        eval_dict = {}
        total_loss = 0.0
        count = 1.0
        for i, data in tqdm.tqdm(
            enumerate(d_loader), leave=False, desc="val"
        ):



            while not rospy.is_shutdown():
                self.optimizer.zero_grad()

                preds, loss, eval_res = self.model_fn(
                    self.model, data, is_eval=True, is_test=is_test
                )


                pcld = preds[4].cpu().numpy()[0]
                pcld = np.array(pcld, dtype=np.float32)
                rgb_img = preds[6].cpu().numpy()[0].transpose()
                rgb_img = rgb_img.transpose(1, 0, 2)
                pose = np.array(preds[3])
                gt_pose = preds[5].cpu().numpy()

                pred_kps = preds[7]#.cpu().numpy()[0]
                #print(pred_kps)
                _ , classes_rgbd = torch.max(preds[1], -1)
                #classes_rgbd  = preds[1]
                classes_rgbd = classes_rgbd.cpu().numpy()
                rgb_img = np.array(rgb_img, dtype=np.uint8)
                '''
                print('gt_pose '+str(gt_pose.shape))
                print('pose '+str(np.array(pose).shape))
                print('kps '+ str(np.array(pred_kps).shape))
                print('classes rgbd '+str(classes_rgbd.shape))
                print('unique labels '+str(np.unique(classes_rgbd)))
                print('')'''
                print('pose '+str(np.array(pose).shape))
                #print('pose[0] '+str(pose[0,:,:]) )
                rgb_labeled = rgb_img.copy()
                pose_msg = []
                pose_msg2 = []
                mask = np.zeros((1), dtype=np.uint32)
                rgb = np.zeros((pose.shape[0],3), dtype=np.uint32)
                rgb_mask = []
                for cls in range(pose.shape[0]):

                    rot = R.from_dcm(pose[cls, 0:3,0:3])
                    quat = rot.as_quat()
    
                    rot_gt = R.from_dcm(gt_pose[0,cls,0:3,0:3])
                    quat_gt = rot_gt.as_quat()

                    registered_corners = np.matmul(pose[cls, 0:3,0:3], corners_t[cls,:,:].T)
                    registered_corners += pose[cls,:3,3].reshape(3,1)

                    # Image with all the Bounding boxes - Takes longer per iteration
                    corners_2d = bs_utl.project_p3d(registered_corners.T, 1 , K = config.intrinsic_matrix['custom'] )
                    rgb_labeled_bb = bs_utl.draw_bounding_box(rgb_labeled, corners_2d)
                    '''
                    pose_msg.append(Pose(position = Point(x=pose[cls,0,3],y=pose[cls,1,3],z=pose[cls, 2,3]), orientation = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3])))
                    pose_msg2.append(Pose(position = Point(x=gt_pose[0,cls,0,3],y=gt_pose[0,cls,1,3],z=gt_pose[0,cls,2,3]), orientation = Quaternion(x=quat_gt[0],y=quat_gt[1],z=quat_gt[2],w=quat_gt[3])))
                    '''

                    this_mask = np.where(classes_rgbd[0,:]==cls+1)[0]
                    rgb_mask.append(this_mask.shape[0])
                    this_rgb = bs_utl.get_label_color( cls_id=cls+1 )
                    rgb[cls,:] = np.array([this_rgb[0], this_rgb[1], this_rgb[2] ])
                    mask = np.concatenate((mask, this_mask), axis = 0  )

                cls = 3
                if pose.shape[0]>cls:
                    rot = R.from_dcm(pose[cls, 0:3,0:3])
                    quat = rot.as_quat()
                    rot_gt = R.from_dcm(gt_pose[0,cls,0:3,0:3])
                    quat_gt = rot_gt.as_quat()

                    pose_msg.append(Pose(position = Point(x=pose[cls,0,3],y=pose[cls,1,3],z=pose[cls, 2,3]), orientation = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3])))
                    pose_msg2.append(Pose(position = Point(x=gt_pose[0,cls,0,3],y=gt_pose[0,cls,1,3],z=gt_pose[0,cls,2,3]), orientation = Quaternion(x=quat_gt[0],y=quat_gt[1],z=quat_gt[2],w=quat_gt[3])))
                    rgb = np.repeat(rgb, rgb_mask, axis=0)
                    p3d_rgbs = np.concatenate( (pcld[mask[1:],0:3], rgb) , axis= 1 )
                    #p2ds = bs_utl.project_p3d(pcld[mask,0:3], 1 , K = config.intrinsic_matrix['openDR'] )
                    p2ds_rgb = bs_utl.project_p3d(p3d_rgbs, 1 , K = config.intrinsic_matrix['openDR'] )
                    #print(p2ds.shape)
                    r = rgb[0,0]#np.random.randint(0,255)
                    g = rgb[1,0]#np.random.randint(0,255)
                    b = rgb[2,0]#np.random.randint(0,255)
                    #registered_pts = (np.matmul(pose[:3,:3],mesh_pts_t.T)+pose[:,3].reshape(3,1)).T
                    '''
                    # Image with single bounding box of the selected the cls - for faster visualization
                    registered_corners = np.matmul(pose[cls, 0:3,0:3], corners_t[cls,:,:].T)
                    registered_corners+= pose[cls,:3,3].reshape(3,1)
                    corners_2d = bs_utl.project_p3d(registered_corners.T, 1 , K = config.intrinsic_matrix['custom'] )'''
                    #mesh_p2ds = bs_utl.project_p3d(registered_pts, 1 , K = config.intrinsic_matrix['custom'] )
                    kps_2d = bs_utl.project_p3d(pred_kps[cls], 1 , K = config.intrinsic_matrix['custom'] )

                    ## Projects 2D points from all labels together with their respective colors
                    rgb_labeled = bs_utl.draw_p2ds(rgb_labeled, p2ds_rgb, (r, g, b), 1)
                    #rgb_labeled_kps = bs_utl.draw_p2ds(rgb_labeled, kps_2d, (0, 255, 0), 2)
                    #rgb_labeled_bb = bs_utl.draw_bounding_box(rgb_labeled_kps, corners_2d)
                    #rgb_labeled_axis = bs_utl.draw_axis(rgb_labeled, gt_pose[0:3,0:3], gt_pose[:,3], config.intrinsic_matrix['custom'])
                    #rgb_labeled = rgb_img
                    #rgb_labeled_kps = bs_utl.draw_p2ds( rgb_labeled ,kps_2d, (255,0,0),2 )
                    #print(rgb_labeled.max())
                    #print(rgb_labeled.min())
                    #print(pcld.shape)
                    #rgb_labeled.dtype = np.uint16
                    #print(rgb_labeled.shape)



                    cloud = pcl.PointCloud()
                    kps = pcl.PointCloud()
                    kps.from_array(pred_kps[cls])
                    cloud.from_array(pcld[:,0:3])
                    #colors = 200*np.ones((pcld.shape[0],3))
                    #colors[mask[1:],:] = p3d_rgbs[:,3:6]
                    colors = pcld[:,3:6]
                    #colors[mask,:]= [r,g,b]
                    cloud_out = pcl_helper.XYZ_to_XYZRGB(cloud, colors.tolist())
                    cloud_msg = pcl_helper.pcl_to_ros(cloud_out, 'world')
                    kp_cld_msg = pcl_helper.pcl_to_ros(kps, 'world')
                    #cld_registered = pcl.PointCloud()

                    #cld_registered.from_array(np.array(registered_pts, dtype=np.float32) )

                    pub.publish(cloud_msg)
                    pub2.publish(PoseArray(header=Header(stamp=rospy.Time.now(), frame_id='world'), poses = pose_msg))
                    pub3.publish(PoseArray(header=Header(stamp=rospy.Time.now(), frame_id='world'), poses = pose_msg2))
                    pub4.publish(ros_numpy.msgify(Image,rgb_labeled, encoding='rgb8'))
                    pub5.publish(kp_cld_msg)
                    #pub6.publish(pcl_helper.pcl_to_ros(cld_registered, 'world'))
                    '''
                    try:
                        usrIn = input('Enter to continue to next Test Image...')
                    except:
                        usrIn = None'''
                break
            if 'loss_target' in eval_res.keys():
                total_loss += eval_res['loss_target']
            else:
                total_loss += loss.item()

            count += 1
            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]
        if is_test:
            self.model_fn(
                self.model, data, is_eval=True, is_test=is_test, finish_test=True
            )

        return total_loss / count, eval_dict

    def train(
        self,
        start_it,
        start_epoch,
        n_epochs,
        train_loader,
        test_loader=None,
        best_loss=0.0,
        log_epoch_f = None
    ):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """

        def is_to_eval(epoch, it):
            if it < 300 * 100:
                eval_frequency = 870
            elif it < 400 * 100:
                eval_frequency = (20 * 100)
            elif it < 500 * 100:
                eval_frequency = (12 * 100)
            elif it < 600 * 100:
                eval_frequency = (8 * 100)
            elif it < 800 * 100:
                eval_frequency = (4 * 100)
            else:
                eval_frequency = (2 * 100)
            to_eval = (it % eval_frequency) == 0
            return to_eval, eval_frequency

        it = start_it
        _, eval_frequency = is_to_eval(0, it)

        with tqdm.trange(start_epoch, n_epochs + 1, desc="epochs") as tbar, tqdm.tqdm(
            total=eval_frequency, leave=False, desc="train"
        ) as pbar:

            for epoch in tbar:
                # Reset numpy seed.
                # REF: https://github.com/pytorch/pytorch/issues/5059
                np.random.seed()
                if log_epoch_f is not None:
                    print('Epoch Started...')
                    os.system("echo {} > {}".format(epoch, log_epoch_f))
                for ibs, batch in enumerate(train_loader):
                    print('Batch '+ str(ibs))
                    self.model.train()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step(it)

                    if self.bnm_scheduler is not None:
                        self.bnm_scheduler.step(it)

                    self.optimizer.zero_grad()
                    _, loss, res = self.model_fn(self.model, batch)
                    print(res)
                    loss.backward()
                    self.optimizer.step()

                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()

                    if self.viz is not None:
                        self.viz.update("train", it, res)

                    eval_flag, eval_frequency = is_to_eval(epoch, it)
                    if eval_flag:
                        pbar.close()

                        if test_loader is not None:
                            val_loss, res = self.eval_epoch(test_loader)

                            if self.viz is not None:
                                self.viz.update("val", it, res)

                            is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            save_checkpoint(
                                checkpoint_state(
                                    self.model, self.optimizer, val_loss, epoch, it
                                ),
                                is_best,
                                filename=self.checkpoint_name,
                                bestname=self.best_name+'_%.4f'%val_loss,
                                bestname_pure=self.best_name,
                            )
                            info_p = self.checkpoint_name.replace(
                                '.pth.tar','_epoch.txt'
                            )
                            os.system(
                                'echo {} {} >> {}'.format(
                                    it, val_loss, info_p
                                )
                            )

                        pbar = tqdm.tqdm(
                            total=eval_frequency, leave=False, desc="train"
                        )
                        pbar.set_postfix(dict(total_it=it))

                    self.viz.flush()

        return best_loss


if __name__ == "__main__":
    args = parser.parse_args()
    rospy.init_node('pvn3d_openDR_pred')
    pub = rospy.Publisher('/eval/pvn3d_cloud', PointCloud2)
    pub2 = rospy.Publisher('/eval/pvn3d_pose', PoseArray)
    pub3 = rospy.Publisher('/eval/pvn3d_gt_pose', PoseArray )
    pub4 = rospy.Publisher('/eval/pvn3d_label_img', Image )
    pub5 = rospy.Publisher('/eval/pvn3d_kps', PointCloud2)
    pub6 = rospy.Publisher('/eval/pvn3d_mesh_registered', PointCloud2)
    #pub6 = rospy.Publisher('/pvn3d_mesh_registered', PointCloud2)
    #mesh = pcl.load('/home/ahmad3/piston_from_cad_poisson.pcd')
    #sor = mesh.make_voxel_grid_filter()
    #sor.set_leaf_size(0.02, 0.02, 0.02)
    #mesh_ds = sor.filter()
    #mesh_pts = np.array(mesh, dtype= np.float32)
    cam2optical = R.from_euler('zyx',[1.57, 0,1.57])
    cam2optical = cam2optical.as_dcm()
    corners_t = np.zeros((10,8,3))
    for i in range(0,10):
        corners = np.loadtxt('./PVN3D/pvn3d/datasets/openDR/openDR_object_kps/'+str(i+1)+'/corners.txt')
        corners_t[i] = np.matmul(cam2optical, corners.transpose()).transpose()
        corners_t[i] = np.concatenate(( corners_t[i,:,2].reshape(8,1) , -corners_t[i,:, 0].reshape(8,1), -corners_t[i,:, 1].reshape(8,1) ), axis=1)
    '''

    mesh_pts_t = np.matmul(cam2optical, mesh_pts.transpose()).transpose()
    corners_t = np.matmul(cam2optical, corners.transpose()).transpose()
    n = mesh_pts_t.shape[0]
    mesh_pts_t = np.concatenate(( mesh_pts_t[:,2].reshape(n,1) , -mesh_pts_t[:, 0].reshape(n,1), -mesh_pts_t[:, 1].reshape(n,1) ), axis=1)
    corners_t = np.concatenate(( corners_t[:,2].reshape(8,1) , -corners_t[:, 0].reshape(8,1), -corners_t[:, 1].reshape(8,1) ), axis=1)
    '''

    test_ds = openDR_Dataset('test')
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
        num_workers=20
    )

    model = PVN3D(
        num_classes=config.n_classes, pcld_input_channels=6, pcld_use_xyz=True,
        num_points=config.n_sample_points, num_kps=config.n_keypoints
    ).cuda()
    model = convert_model(model)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    cur_mdl_pth = os.path.join(config.log_model_dir, 'pvn3d.pth.tar')
    if args.checkpoint is None and os.path.exists(cur_mdl_pth):
        args.checkpoint = cur_mdl_pth
    # load status from checkpoint
    if args.checkpoint is not None:

        checkpoint_status = load_checkpoint(
            model, optimizer, filename=args.checkpoint[:-8]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status
            print('Loading from checkpoint...' + 'It: '+str(it) + 'Epoch: '+str(start_epoch)+'Loss: '+str(best_loss))
    model = nn.DataParallel(
        model
    )

    lr_scheduler = CyclicLR(
        optimizer, base_lr=1e-5, max_lr=1e-3,
        step_size=config.n_total_epoch * config.num_mini_batch_per_epoch // 6,
        mode = 'triangular'
    )

    bnm_lmbd = lambda it: max(
        args.bn_momentum
        * args.bn_decay ** (int(it * config.mini_batch_size / args.decay_step)),
        bnm_clip,
    )
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bnm_lmbd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    model_fn = model_fn_decorator(
        nn.DataParallel(FocalLoss(gamma=2)),
        nn.DataParallel(OFLoss()),
        args.test,
    )

    viz = CmdLineViz()

    viz.text(pprint.pformat(vars(args)))

    checkpoint_fd = config.log_model_dir
    if not os.path.exists(checkpoint_fd):
        os.system('mkdir -p {}'.format(checkpoint_fd))

    trainer = Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name = os.path.join(checkpoint_fd, "pvn3d"),
        best_name = os.path.join(checkpoint_fd, "pvn3d_best"),
        lr_scheduler = lr_scheduler,
        bnm_scheduler = bnm_scheduler,
        viz = viz,
    )

    start = time.time()
    val_loss, res = trainer.eval_epoch(
        test_loader, is_test=True
    )


