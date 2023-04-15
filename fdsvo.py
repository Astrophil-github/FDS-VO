

import sys
sys.path.append("/home/a/FDS-VO")
import cv2
import copy
from glob import glob
import math
from matplotlib import pyplot as plt
import numpy as np
import os
from time import time
from tqdm import tqdm

from libs.geometry.camera_modules import SE3
import libs.datasets as Dataset
from libs.deep_models.deep_models import DeepModel
from libs.general.frame_drawer import FrameDrawer
from libs.general.timer import Timer
from libs.matching.keypoint_sampler import KeypointSampler
from libs.matching.depth_consistency import DepthConsistency
from libs.tracker import EssTracker, PnpTracker
from libs.general.utils import *
from libs.YOLOv5.YOLOv5 import YOLOv5
import numpy as np



class FDSVO():
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration reading from yaml file
        """
        # configuration
        self.cfg = cfg

        # tracking stage 跟踪阶段
        self.tracking_stage = 0

        # predicted global poses 预测全球姿势
        self.global_poses = {0: SE3()}
        # print(self.global_poses) # {0: <libs.geometry.camera_modules.SE3 object at 0x7f0cd3a89390>}

        # reference data and current data 参考数据和当前数据
        self.initialize_data()

        self.setup()

    def setup(self):
        """Reading configuration and setup, including
        读取配置和设置，包括

            - Timer
            - Dataset
            - Tracking method
            - Keypoint Sampler
            - Deep networks
            - Deep layers
            - Visualizer
        """
        # timer
        self.timers = Timer()

        # intialize dataset
        self.dataset = Dataset.datasets[self.cfg.dataset](self.cfg)
        # print(self.cfg.dataset) # kitti_odom
        # print(self.dataset) # <libs.datasets.kitti.KittiOdom object at 0x7f9692e79910>
        
        # get tracking method 获取跟踪方法
        self.tracking_method = self.cfg.tracking_method # tracking method [hybrid, PnP]
        self.initialize_tracker()

        # initialize keypoint sampler 初始化关键点采样器
        self.kp_sampler = KeypointSampler(self.cfg)
        
        # Deep networks
        self.deep_models = DeepModel(self.cfg)
        self.deep_models.initialize_models()
        if self.cfg.online_finetune.enable:
            self.deep_models.setup_train()
        
        # Depth consistency 深度一致性
        if self.cfg.kp_selection.depth_consistency.enable:
            self.depth_consistency_computer = DepthConsistency(self.cfg, self.dataset.cam_intrinsics)

        # visualization interface 可视化界面
        self.drawer = FrameDrawer(self.cfg.visualization)

        # YOLOv5
        self.initialize_object_models()

        # pose_check列表，记录被矫正的帧
        self.pose_check_list = []


    def initialize_object_models(self):
        '''初始化目标检测模型'''
        if self.cfg.deep_object.enable:
            if self.cfg.deep_object.network == 'YOLOv5':
                self.deep_object_model = YOLOv5(self.cfg)
            else:
                pass

    def handle_kp(self, kp_sel_outputs):
        '''
        kp处理函数，用于处理在移动物体上的kp
        '''
        # a = len(kp_sel_outputs['kp1_best'][0])
        del_list = []
        coors = np.array(self.cur_data['coor'])
        coors = coors.reshape(int(len(coors)/4), 4)
        for kp_index in range(len(kp_sel_outputs['kp1_best'][0])):
            for coor in coors:
                if kp_sel_outputs['kp1_best'][0][kp_index][0] > coor[0] and \
                        kp_sel_outputs['kp1_best'][0][kp_index][1] > coor[1] and\
                        kp_sel_outputs['kp1_best'][0][kp_index][0] < coor[2] and\
                        kp_sel_outputs['kp1_best'][0][kp_index][1] < coor[3]:
                    del_list.append(kp_index)

                    break

        kp_sel_outputs['kp1_best'] = np.delete(kp_sel_outputs['kp1_best'][0], del_list, axis=0)[np.newaxis, :]
        kp_sel_outputs['kp2_best'] = np.delete(kp_sel_outputs['kp2_best'][0], del_list, axis=0)[np.newaxis, :]

        self.cur_data['kp_best'] = np.delete(self.cur_data['kp_best'], del_list, axis=0)
        self.ref_data['kp_best'] = np.delete(self.ref_data['kp_best'], del_list, axis=0)

        return kp_sel_outputs

    def pose_check(self):
        '''pose校验函数，加入动力学模型进行pose校验'''
        pass

        
    def initialize_data(self):
        """
        initialize data of current view and reference view
        初始化当前视图和参考视图的数据
        """
        self.ref_data = {}
        self.cur_data = {}

    def initialize_tracker(self):
        """Initialize tracker初始化跟踪器
        """
        if self.tracking_method == 'hybrid':
            self.e_tracker = EssTracker(self.cfg, self.dataset.cam_intrinsics, self.timers)
            self.pnp_tracker = PnpTracker(self.cfg, self.dataset.cam_intrinsics)
        elif self.tracking_method == 'PnP':
            self.pnp_tracker = PnpTracker(self.cfg, self.dataset.cam_intrinsics)
        elif self.tracking_method == 'deep_pose':
            return
        else:
            assert False, "Wrong tracker is selected, choose from [hybrid, PnP, deep_pose]"

    def update_global_pose(self, new_pose, scale=1.):
        """update estimated poses w.r.t global coordinate system
        更新w.r.t全球坐标系

        Args:
            new_pose (SE3): new pose
            scale (float): scaling factor
        """
        if self.cfg.pose_check.enable and self.tracking_stage >= 2:
            self.pose1 = new_pose
            if self.cfg.pose_check.mode == 'simple':
                dist = np.linalg.norm(self.pose1.t - self.pose0.t)
                if dist > self.cfg.pose_check.allowed_band[0]:
                    new_pose = self.pose0

                    print('{}时刻的pose0与pose1的距离{}超出阈值{}，采用上一时刻的new_pose'.format(self.cur_data['timestamp'], dist, self.cfg.pose_check.allowed_band[0]))
                    self.pose_check_list.append(self.cur_data['timestamp'])


            elif self.cfg.pose_check.mode == 'dynamics':
                pass
            else:
                raise ValueError("cfg.pose_check.mode is wrong!")
        self.cur_data['pose'].t = self.cur_data['pose'].R @ new_pose.t * scale \
                            + self.cur_data['pose'].t
        self.cur_data['pose'].R = self.cur_data['pose'].R @ new_pose.R
        self.global_poses[self.cur_data['id']] = copy.deepcopy(self.cur_data['pose'])

        if self.cfg.pose_check.enable:
            self.pose0 = new_pose

    def tracking(self):
        """Tracking using both Essential matrix and PnP
        Essential matrix for rotation and translation direction;
            *** triangluate depth v.s. CNN-depth for translation scale ***
        PnP if Essential matrix fails

        使矩用旋转和平移方向的基本阵和PnP基本矩阵进行跟踪；
            ***三角形深度v.s.CNN深度转换比例***
        如果基本矩阵失效，则为PnP
        """
        # First frame
        if self.tracking_stage == 0:
            # initial pose
            if self.cfg.directory.gt_pose_dir is not None:
                self.cur_data['pose'] = SE3(self.dataset.gt_poses[self.cur_data['id']])
            else:
                self.cur_data['pose'] = SE3()
            return

        # Second to last frames
        elif self.tracking_stage >= 1:
            ''' keypoint selection '''
            if self.tracking_method in ['hybrid', 'PnP']:
                # Depth consistency (CNN depths + CNN pose) 深度一致性  默认是False
                if self.cfg.kp_selection.depth_consistency.enable:
                    self.depth_consistency_computer.compute(self.cur_data, self.ref_data)

                # kp_selection  默认用的是local_bestN方法，但是由于光溜一致性问题，所以改用了bestN
                self.timers.start('kp_sel', 'tracking')
                kp_sel_outputs = self.kp_sampler.kp_selection(self.cur_data, self.ref_data)
                if kp_sel_outputs['good_kp_found']:
                    self.kp_sampler.update_kp_data(self.cur_data, self.ref_data, kp_sel_outputs)
                self.timers.end('kp_sel')

            ''' Pose estimation '''
            # Initialize hybrid pose
            hybrid_pose = SE3()
            E_pose = SE3()

            if self.cfg.kp_handle.enable and self.cfg.deep_object.enable:
                kp_sel_outputs = self.handle_kp(kp_sel_outputs)

            if not(kp_sel_outputs['good_kp_found']):
                print("No enough good keypoints, constant motion will be used!")
                pose = self.ref_data['motion']
                # pose = self.ref_data['pose']
                self.update_global_pose(pose, 1)
                return


            ''' E-tracker '''
            if self.tracking_method in ['hybrid']:
                # Essential matrix pose
                self.timers.start('E-tracker', 'tracking')
                # 传入的三个数据分别是npy类型和bool类型
                e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                                self.ref_data[self.cfg.e_tracker.kp_src],
                                self.cur_data[self.cfg.e_tracker.kp_src],
                                not(self.cfg.e_tracker.iterative_kp.enable)) # pose: from cur->ref

                # Astrophil 这里使用ORB特征进行
                if self.cfg.use_orb.enable:
                    def find_feature_matches(img_1, img_2):
                        orb = cv2.ORB_create()

                        kp1 = orb.detect(img_1)
                        kp2 = orb.detect(img_2)

                        kp1, des1 = orb.compute(img_1, kp1)
                        kp2, des2 = orb.compute(img_2, kp2)

                        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

                        matches = bf.match(des1, des2)

                        min_distance = matches[0].distance
                        max_distance = matches[0].distance

                        for x in matches:
                            if x.distance < min_distance:
                                min_distance = x.distance
                            if x.distance > max_distance:
                                max_distance = x.distance

                        # print("Max dist:", max_distance)
                        # print("Min dist:", min_distance)

                        good_match = []

                        for x in matches:
                            if x.distance <= max(2 * min_distance, 30.0):
                                good_match.append(x)
                        # return kp1, kp2, good_match

                        pts1 = []
                        pts2 = []
                        for i in range(int(len(matches))):
                            pts1.append(kp1[matches[i].queryIdx].pt)
                            pts2.append(kp2[matches[i].trainIdx].pt)

                        pts1 = np.int32(pts1)
                        pts2 = np.int32(pts2)
                        return pts1,pts2

                    keypoint_1, keypoint_2 = find_feature_matches(self.ref_data['img'], self.cur_data['img'])
                    print("keypoint_1  ", len(keypoint_1), "keypoint_2  ", len(keypoint_2))
                    e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                                    keypoint_1,
                                    keypoint_2,
                                    not(self.cfg.e_tracker.iterative_kp.enable)) # pose: from cur->ref

                E_pose = e_tracker_outputs['pose']
                self.timers.end('E-tracker')

                # Rotation
                hybrid_pose.R = E_pose.R

                # save inliers 节省内存
                self.ref_data['inliers'] = e_tracker_outputs['inliers']

                # scale recovery  尺度回归
                if np.linalg.norm(E_pose.t) != 0:
                    self.timers.start('scale_recovery', 'tracking')
                    scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, False)
                    scale = scale_out['scale']
                    if self.cfg.scale_recovery.kp_src == 'kp_depth':
                        self.cur_data['kp_depth'] = scale_out['cur_kp_depth']
                        self.ref_data['kp_depth'] = scale_out['ref_kp_depth']
                        self.cur_data['rigid_flow_mask'] = scale_out['rigid_flow_mask']
                    if scale != -1:
                        hybrid_pose.t = E_pose.t * scale
                    self.timers.end('scale_recovery')

                # Iterative keypoint refinement 迭代关键点求精
                if np.linalg.norm(E_pose.t) != 0 and self.cfg.e_tracker.iterative_kp.enable:
                    self.timers.start('E-tracker iter.', 'tracking')
                    # Compute refined keypoint
                    self.e_tracker.compute_rigid_flow_kp(self.cur_data,
                                                         self.ref_data,
                                                         hybrid_pose)

                    e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                                self.ref_data[self.cfg.e_tracker.iterative_kp.kp_src],
                                self.cur_data[self.cfg.e_tracker.iterative_kp.kp_src],
                                True) # pose: from cur->ref
                    E_pose = e_tracker_outputs['pose']

                    # Rotation
                    hybrid_pose.R = E_pose.R

                    # save inliers
                    self.ref_data['inliers'] = e_tracker_outputs['inliers']

                    # scale recovery
                    if np.linalg.norm(E_pose.t) != 0 and self.cfg.scale_recovery.iterative_kp.enable:
                        scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, True)
                        scale = scale_out['scale']
                        if scale != -1:
                            hybrid_pose.t = E_pose.t * scale
                    else:
                        hybrid_pose.t = E_pose.t * scale
                    self.timers.end('E-tracker iter.')

            ''' PnP-tracker '''
            if self.tracking_method in ['PnP', 'hybrid']:
                # PnP if Essential matrix fail
                if np.linalg.norm(E_pose.t) == 0 or scale == -1:
                    self.timers.start('pnp', 'tracking')
                    pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                                    self.ref_data[self.cfg.pnp_tracker.kp_src],
                                    self.cur_data[self.cfg.pnp_tracker.kp_src],
                                    self.ref_data['depth'],
                                    not(self.cfg.pnp_tracker.iterative_kp.enable)
                                    ) # pose: from cur->ref
                    
                    # Iterative keypoint refinement
                    if self.cfg.pnp_tracker.iterative_kp.enable:
                        self.pnp_tracker.compute_rigid_flow_kp(self.cur_data, self.ref_data, pnp_outputs['pose'])
                        pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                                    self.ref_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                                    self.cur_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                                    self.ref_data['depth'],
                                    True
                                    ) # pose: from cur->ref

                    self.timers.end('pnp')

                    # use PnP pose instead of E-pose
                    hybrid_pose = pnp_outputs['pose']
                    self.tracking_mode = "PnP"

            ''' Deep-trazacker '''
            if self.tracking_method in ['deep_pose']:
                hybrid_pose = SE3(self.ref_data['deep_pose'])
                self.tracking_mode = "DeepPose"

            ''' Summarize data '''
            # update global poses
            self.ref_data['pose'] = copy.deepcopy(hybrid_pose)
            self.ref_data['motion'] = copy.deepcopy(hybrid_pose)
            pose = self.ref_data['pose']
            self.update_global_pose(pose, 1)

    def update_data(self, ref_data, cur_data):
        """Update data
        
        Args:
            ref_data (dict): reference data
            cur_data (dict): current data
        
        Returns:
            ref_data (dict): updated reference data
            cur_data (dict): updated current data
        """
        for key in cur_data:
            if key == "id":
                ref_data['id'] = cur_data['id']
            else:
                if ref_data.get(key, -1) is -1:
                    ref_data[key] = {}
                ref_data[key] = cur_data[key]
        
        # Delete unused flow to avoid data leakage
        ref_data['flow'] = None
        cur_data['flow'] = None
        ref_data['flow_diff'] = None
        return ref_data, cur_data

    def load_raw_data(self):
        """load image data and (optional) GT/precomputed depth data
        加载图像数据和（可选）GT/预计算深度数据
        """
        # Reading image
        self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])

        # Reading/Predicting depth
        if self.dataset.data_dir['depth_src'] is not None:
            self.cur_data['raw_depth'] = self.dataset.get_depth(self.cur_data['timestamp'])
    
    def deep_model_inference(self):
        """deep model prediction
        深层模型推理
        """
        if self.tracking_method in ['hybrid', 'PnP']:
            # Single-view Depth prediction
            if self.dataset.data_dir['depth_src'] is None:
                self.timers.start('depth_cnn', 'deep inference')
                if self.tracking_stage > 0 and \
                    self.cfg.online_finetune.enable and self.cfg.online_finetune.depth.enable:
                        img_list = [self.cur_data['img'], self.ref_data['img']]
                else:
                    img_list = [self.cur_data['img']]

                self.cur_data['raw_depth'] = \
                    self.deep_models.forward_depth(imgs=img_list)
                self.cur_data['raw_depth'] = cv2.resize(self.cur_data['raw_depth'],
                                                    (self.cfg.image.width, self.cfg.image.height),
                                                    interpolation=cv2.INTER_NEAREST
                                                    )

                # save the Depth data
                '''Astrophil'''
                if self.cfg.save_img.depth.enable:
                    # 判断是否存在文件夹，没有就生成
                    depth_path = self.cfg.save_img.depth.path + self.cfg.seq + '/'
                    if not os.path.exists(depth_path):
                        os.makedirs(depth_path)
                    # 保存
                    cv2.imwrite(r'{}{:010d}.png'.format(depth_path, self.cur_data['timestamp']), self.cur_data['raw_depth'])

                self.timers.end('depth_cnn')

            # else:
            #     img_name = "{:010d}.png".format(self.cur_data['timestamp'])
            #     depth_path = os.path.join(self.dataset.data_dir['depth'], img_name)
            #     scale = 1
            #     depth = cv2.imread(depth_path, -1)
            #     img_h, img_w = [192, 640]
                # self.cur_data['raw_depth'] = cv2.resize(depth,
                #                    (img_w, img_h),
                #                    interpolation=cv2.INTER_NEAREST
                #                    )



            self.cur_data['depth'] = preprocess_depth(self.cur_data['raw_depth'], self.cfg.crop.depth_crop, [self.cfg.depth.min_depth, self.cfg.depth.max_depth])

            # Two-view flow
            if self.tracking_stage >= 1:
                self.timers.start('flow_cnn', 'deep inference')
                flows = self.deep_models.forward_flow(
                                        self.cur_data,
                                        self.ref_data,
                                        forward_backward=self.cfg.deep_flow.forward_backward)

                # save the flow data
                '''Astrophil'''
                if self.cfg.save_img.flow.enable:
                    np.save(r'{}{}/{}-{}.npy'.format(
                        self.cfg.save_img.flow.path, self.cfg.seq, self.ref_data['id'], self.cur_data['id']),
                        flows[(self.ref_data['id'], self.cur_data['id'])]
                    )

                # Store flow
                self.ref_data['flow'] = flows[(self.ref_data['id'], self.cur_data['id'])].copy()
                if self.cfg.deep_flow.forward_backward:
                    self.cur_data['flow'] = flows[(self.cur_data['id'], self.ref_data['id'])].copy()
                    self.ref_data['flow_diff'] = flows[(self.ref_data['id'], self.cur_data['id'], "diff")].copy()

                self.timers.end('flow_cnn')

        # Relative camera pose
        if self.tracking_stage >= 1 and self.cfg.deep_pose.enable:
            self.timers.start('pose_cnn', 'deep inference')
            # Deep pose prediction
            pose = self.deep_models.forward_pose(
                        [self.ref_data['img'], self.cur_data['img']]
                        )
            self.ref_data['deep_pose'] = pose # from cur->ref
            self.timers.end('pose_cnn')

        # object
        if self.cfg.deep_object.enable:
            self.cur_data['object_img'], self.cur_data['coor'] = self.deep_object_model.forward(self.cur_data['img'])

    def main(self):
        """Main program
        """
        print("==> Start FDS-VO")
        print("==> Running sequence: {}".format(self.cfg.seq))

        if self.cfg.no_confirm:
            start_frame = 0
        else:
            '''Astrophil'''
            # start_frame = int(input("Start with frame: "))
            start_frame = 0



        for img_id in tqdm(range(start_frame, len(self.dataset), self.cfg.frame_step)):
            self.timers.start('FDS-VO')
            self.tracking_mode = "Ess. Mat."

            """ Data reading """
            # Initialize ids and timestamps 初始化ID和时间戳
            self.cur_data['id'] = img_id
            self.cur_data['timestamp'] = self.dataset.get_timestamp(img_id)

            # Read image data and (optional) precomputed depth data 读取图像数据和（可选）预计算深度数据
            self.timers.start('data_loading')
            self.load_raw_data()
            self.timers.end('data_loading')

            # Deep model inferences 深层模型推理
            self.timers.start('deep_inference')
            self.deep_model_inference()
            self.timers.end('deep_inference')

            """ Visual odometry """
            self.timers.start('tracking')
            self.tracking()
            self.timers.end('tracking')

            """ Online Finetuning """
            if self.tracking_stage >= 1 and self.cfg.online_finetune.enable:
                self.deep_models.finetune(self.ref_data['img'], self.cur_data['img'],
                                      self.ref_data['pose'].pose,
                                      self.dataset.cam_intrinsics.mat,
                                      self.dataset.cam_intrinsics.inv_mat)

            """ Visualization """
            if self.cfg.visualization.enable:
                self.timers.start('visualization')
                self.drawer.main(self)
                self.timers.end('visualization')

            """ Update reference and current data """
            self.ref_data, self.cur_data = self.update_data(
                                    self.ref_data,
                                    self.cur_data,
            )

            self.tracking_stage += 1

            self.timers.end('FDS-VO')

        print("=> Finish!")



        """ Display & Save result """
        print("The result is saved in [{}].".format(self.cfg.directory.result_dir))
        # Save trajectory map
        print("Save VO map.")
        map_png = "{}/map.png".format(self.cfg.directory.result_dir)
        cv2.imwrite(map_png, self.drawer.data['traj'])

        # Save trajectory txt
        traj_txt = "{}/{}.txt".format(self.cfg.directory.result_dir, self.cfg.seq)
        self.dataset.save_result_traj(traj_txt, self.global_poses)

        # save finetuned model
        if self.cfg.online_finetune.enable and self.cfg.online_finetune.save_model:
            self.deep_models.save_model()

        # Output experiement information
        self.timers.time_analysis()


        print("被矫正的帧数列如：\n", self.pose_check_list)
