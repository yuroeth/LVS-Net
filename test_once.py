import os
import cv2
import time
import json
import random
import inspect
import argparse
import numpy as np
from tqdm import tqdm

from dataloaders import make_data_loader
from models.sync_batchnorm.replicate import patch_replication_callback
from models.vs_net import *
from utils.loss import loss_dict
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils import utils
from torch.autograd import Variable
import poselib
import transforms3d.quaternions as txq
import matplotlib.pyplot as plt
 
import os.path as osp

from configs import *

import warnings
warnings.filterwarnings("ignore")

class Trainer(object):
    def __init__(self, cfg, image_id):
        self.cfg = cfg

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.cfg["log_tb_dir"])
        self.summary.create_summary()
        # Define Dataloader
        kwargs = {"num_workers": cfg["num_workers"], "pin_memory": True}
        self.val_loader = make_data_loader(
            cfg, True, image_id, **kwargs)

        # read landmark centers
        if cfg["landmark"] == "point":
            nfile = "id2centers.json"
        elif cfg["landmark"] == "line":
            nfile = "id2lines.json"
        else:
            raise NotImplementedError
        self.id2center = np.array(json.load(
            open(osp.join(cfg["data_dir"], nfile)))).astype(np.float64)

        self.coding_book = torch.zeros(
            (len(self.id2center), cfg["seg_channel"]), dtype=torch.float32).cuda()
        torch.nn.init.xavier_uniform(self.coding_book)

        print("coding book size = {}".format(self.coding_book.shape))
        # generate color map
        unique_label = np.arange(len(self.id2center))
        unique_label = unique_label.astype(
            np.int64) * 6364136223846793005 + 1442695040888963407
        color_map = np.zeros((unique_label.shape[0], 3), np.uint8)
        color_map[:, 0] = np.bitwise_and(unique_label, 0xff)
        color_map[:, 1] = np.bitwise_and(np.right_shift(unique_label, 4), 0xff)
        color_map[:, 2] = np.bitwise_and(np.right_shift(unique_label, 8), 0xff)
        self.color_map = np.array(color_map)

        self.coding_book = Variable(self.coding_book, requires_grad=True)

        # Define network
        model = VSNet(backbone=cfg["backbone"],
                        seg_decoder=cfg["seg_decoder"],
                        vertex_decoder=cfg["vertex_decoder"],
                        seg_channel=cfg["seg_channel"],
                        vertex_channel=cfg["vertex_channel"],
                        output_stride=cfg["out_stride"],
                        sync_bn=cfg["sync_bn"])

        train_params = [{"params": model.get_1x_lr_params(), "lr": cfg["lr"]},
                        {"params": model.get_10x_lr_params(),
                         "lr": cfg["lr"] * 10},
                        {"params": self.coding_book, "lr": cfg["lr"] * 10}
                        ]

        # Define Optimizer
        if cfg["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(train_params, momentum=cfg["momentum"],
                                        weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"])
        elif cfg["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(train_params, lr=cfg["lr"],
                                         weight_decay=cfg["weight_decay"], amsgrad=True)
        else:
            raise NotImplementedError

        # Define Criterion
        self.seg_criterion = loss_dict[cfg["seg_loss_type"]]
        self.vertex_criterion = loss_dict[cfg["vertex_loss_type"]]
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        # self.evaluator = Evaluator(
        #     self.coding_book.shape[0], cfg["vertex_channel"])

        # Define lr scheduler
        # self.scheduler = LR_Scheduler(mode=cfg["lr_scheduler"], base_lr=cfg["lr"],
        #                               num_epochs=cfg["epochs"], iters_per_epoch=len(
        #                                   self.train_loader),
        #                               lr_step=cfg["lr_step"])

        self.model = torch.nn.DataParallel(self.model)
        patch_replication_callback(self.model)
        self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = {"mIoU": 0.0, "Acc": 0.0, "Acc": 0.0,
                          "FWIoU": 0.0, "translation_median": 1000}
        if cfg["resume"] is not None and cfg["resume"] == True:
            print(os.path.isfile(cfg["resume_checkpoint"]))
            if not os.path.isfile(cfg["resume_checkpoint"]):
                raise RuntimeError("=> no checkpoint found at {}" .format(
                    cfg["resume_checkpoint"]))
            checkpoint = torch.load(cfg["resume_checkpoint"])
            cfg.opt["start_epoch"] = checkpoint["epoch"] - 1
            self.model.module.load_state_dict(checkpoint["state_dict"])
            # if not cfg["ft"]:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            if "coding_book" in checkpoint.keys():
                assert self.coding_book.shape == checkpoint["coding_book"].shape
                self.coding_book = checkpoint["coding_book"]
            else:
                print("Alert! coding book does not exist in the checkpoint")
            print("=> loaded checkpoint {} (epoch {})"
                  .format(cfg["resume"], checkpoint["epoch"]))
        
        self.image_id = image_id

        save_dir = osp.join(cfg["log_tb_dir"], "cm_degree_metrics")
        if not osp.isdir(save_dir):
            os.makedirs(save_dir)
        self.save_path = osp.join(save_dir, "image_{}.npz".format(self.image_id))
        image_dir = osp.join(cfg["log_tb_dir"], "line_segs")
        if not osp.isdir(image_dir):
            os.makedirs(image_dir)
        self.image_path = osp.join(image_dir, "image_{}.png".format(self.image_id))

    def validation(self, epoch):
        print("=================================")
        print("validation")
        print("=================================")
        self.model.eval()
        # self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        num_iter_val = len(self.val_loader)

        test_loss = 0.0
        num_images = 0
        ten_count = []
        five_count = []
        three_count = []
        one_count = []
        translation_list = []
        angular_list = []
        reproject_list = []
        test_seg_loss = 0.0
        test_ver_loss = 0.0
        for i, data in enumerate(tbar):
            # debug
            # if i != self.image_id:
            #     continue
            image, seg_target, vertex_target = [d.cuda() for d in data[:3]]
            valid_mask = data[-1].cuda()
            pose_target, camera_k_matrix, ori_img = data[3:]
            seg_target = seg_target.long()
            valid_mask = (seg_target.detach() > 0).float()
            with torch.no_grad():
                seg_pred, vertex_pred, seg_pred_x4s = self.model(
                    image)

                loss_seg = 0
                if self.cfg["seg_decoder"]:
                    loss_seg = self.seg_criterion(seg_pred, seg_target, self.coding_book,
                                                  margin=self.cfg["seg_loss_margin"],
                                                  seg_k=self.cfg["seg_k"],
                                                  valid_mask=valid_mask)
                    test_seg_loss += loss_seg.item()
                    self.summary.add_scalar(
                        "val/loss_seg_iter", loss_seg.item(), i + num_iter_val * epoch)

                loss_vertex = 0
                if self.cfg["vertex_decoder"]:
                    loss_vertex = self.vertex_criterion(vertex_pred, vertex_target,
                                                        valid_mask)
                    test_ver_loss += loss_vertex.item()
                    self.summary.add_scalar(
                        "val/loss_vertex_iter", loss_vertex.item(), i + num_iter_val * epoch)

                loss = 0
                if self.cfg["seg_decoder"]:
                    loss += loss_seg
                if self.cfg["vertex_decoder"]:
                    loss += loss_vertex * self.cfg["vertex_loss_ratio"]

                test_loss += loss.item()
                tbar.set_description("Test loss: %.9f" % (test_loss / (i + 1)))
                self.summary.add_scalar(
                    "val/total_loss_iter", loss.item(), i + num_iter_val * epoch)

                global_step = i * \
                    self.cfg["val_batch_size"] + image.data.shape[0]

                # evaluate seg_pred
                seg_target = seg_target.detach().squeeze()
                if self.cfg["seg_decoder"]:
                    seg_pred, knn = utils.evaluate_segmentation(seg_pred_x4s,
                                                                self.coding_book, seg_target.size(), self.cfg["use_own_nn"])
                else:
                    seg_pred = seg_target

                num_images = i * self.cfg["val_batch_size"] + image.data.shape[0]

                # np.savetxt(self.cfg["log_tb_dir"] + "/vertex_pred_saved.txt", vertex_pred.cpu().numpy())
                # np.savetxt(self.cfg["log_tb_dir"] + "/seg_pred_saved.txt", seg_pred.cpu().numpy())

                # # evaluate vertex
                # pt3d_filter, pt2d_filter, _ = utils.evaluate_vertex_v2(vertex_pred, seg_pred,
                #                                                        self.id2center, inlier_thresh=0.999,
                #                                                        min_mask_num=self.cfg["val_label_filter_threshsold"])

                # evaluate atf
                # print('validating image: {}'.format(i))
                line3d_filter, line2d_filter, _, _ = utils.evaluate_afm(vertex_pred, seg_pred, self.id2center, threshold=0.1, 
                                                                    min_mask_num=self.cfg["val_label_filter_threshsold"])
                
                camera_k_matrix = camera_k_matrix.squeeze().numpy()
                # translation_distance, angular_distance, error = 1e9, 1e9, 1e9
                if line2d_filter.shape[0] > 6:
                    camera_dict = {'model': 'SIMPLE_PINHOLE', 'width': 960, 'height': 540, 'params': [camera_k_matrix[0][0], camera_k_matrix[0][2], camera_k_matrix[1][2]]}
                    line2d_1 = [line2d_filter[i][:2].reshape(2,1).astype(np.float64) for i in range(len(line2d_filter))]
                    line2d_2 = [line2d_filter[i][2:].reshape(2,1).astype(np.float64) for i in range(len(line2d_filter))]
                    line3d_1 = [line3d_filter[i][:3].reshape(3,1).astype(np.float64) for i in range(len(line2d_filter))]
                    line3d_2 = [line3d_filter[i][3:].reshape(3,1).astype(np.float64) for i in range(len(line2d_filter))]
                    camera_pose, info = poselib.estimate_absolute_pose_pnpl([], [], line2d_1, line2d_2, line3d_1, line3d_2, camera_dict, {'max_epipolar_error': 8.0})
                    pose_pred = np.zeros((3,4))
                    pose_pred[:3, :3] = txq.quat2mat(camera_pose.q)
                    pose_pred[:3, 3] = camera_pose.t
                    translation_distance, angular_distance = utils.cm_degree_metric(
                        pose_pred, pose_target)
                    print(translation_distance, angular_distance, info)
                    np.savez(self.save_path, trans_dist=translation_distance, ang_dist=angular_distance, q=camera_pose.q, t=camera_pose.t, ransac_result=info)

                    inlier_lines = info['inliers_lines']
                    outlier_lines = [not item for item in inlier_lines]
                    inliers = line2d_filter[inlier_lines]
                    outliers = line2d_filter[outlier_lines]
                    width = 960
                    height = 540
                    plt.rcParams['figure.figsize'] = (9.6,5.4)
                    plt.plot([inliers[:, 0], inliers[:, 2]], [inliers[:, 1], inliers[:, 3]], 'r-')
                    plt.plot([outliers[:, 0], outliers[:, 2]], [outliers[:, 1], outliers[:, 3]], 'b-')
                    plt.xlim([0, width])
                    plt.ylim([height, 0])
                    plt.axis('off')
                    plt.savefig(self.image_path, dpi=100)

                # if pt2d_filter.shape[0] > 6:
                #     # pnp
                #     ret, pose_pred = utils.pnp(
                #         pt3d_filter, pt2d_filter, camera_k_matrix)
                #     error = utils.reproject_error(
                #         pt3d_filter, pt2d_filter, pose_pred, camera_k_matrix)
                #     translation_distance, angular_distance = utils.cm_degree_metric(
                #         pose_pred, pose_target)
                #     print(translation_distance, angular_distance, error, i)
                # ten_count.append(translation_distance <
                #                  10 and angular_distance < 10)
                # five_count.append(translation_distance <
                #                   5 and angular_distance < 5)
                # three_count.append(translation_distance <
                #                    3 and angular_distance < 3)
                # one_count.append(translation_distance <
                #                  1 and angular_distance < 1)
                # translation_list.append(translation_distance)
                # angular_list.append(angular_distance)
                # reproject_list.append(error)

                # Add batch sample into evaluator
                if self.cfg["seg_decoder"]:
                    # self.evaluator.add_seg_batch(seg_target, seg_pred)
                    if self.cfg["visualize_segmentation"]:
                        self.summary.visualize_seg_image(ori_img, seg_pred, seg_target,
                                                         epoch, i, global_step, self.color_map)

                if self.cfg["vertex_decoder"]:
                    # evaluate vertex_pred
                    vertex_target, vertex_pred = vertex_target.squeeze(), vertex_pred.squeeze()
                    # self.evaluator.add_vertex_batch(vertex_target, vertex_pred)

                    # vertex acc的计算
                    if self.cfg["visualize_voting"]:
                        # resize
                        _, h, w = vertex_pred.shape
                        vertex_pred = torch.sign(vertex_pred) * torch.exp(-torch.abs(vertex_pred))
                        vertex_pred[0,:,:] *= w
                        vertex_pred[1,:,:] *= h
                        vertex_target = torch.sign(vertex_target) * torch.exp(-torch.abs(vertex_target))
                        vertex_target[0,:,:] *= w
                        vertex_target[1,:,:] *= h        
                        if self.cfg["visualize_landmark"] != None and self.cfg["visualize_landmark"]:
                            self.summary.visualize_vertex_image(ori_img, vertex_pred, vertex_target,
                                                                epoch, i, global_step, pt2d_filter, True)
                        else:
                            self.summary.visualize_vertex_image(ori_img, vertex_pred, vertex_target,
                                                                epoch, i, global_step)

        # mIoU, Acc, Acc_class, FWIoU = self.summary.visualize_seg_evaluator(
        #     self.evaluator, epoch, "val/seg/")
        # print("Validation:")
        # print("[Epoch: %d, numImages: %5d]" % (epoch, num_images))
        # print("Loss: %.9f" % (test_loss / num_iter_val))
        # print("Seg Loss: %.9f" % (test_seg_loss / num_iter_val))
        # print("Ver Loss: %.9f" % (test_ver_loss / num_iter_val))
        # self.summary.add_scalar("val/total_loss_epoch",
        #                         test_loss / num_iter_val, epoch)
        # self.summary.add_scalar("val/total_seg_epoch",
        #                         test_seg_loss / num_iter_val, epoch)
        # self.summary.add_scalar("val/total_ver_epoch",
        #                         test_ver_loss / num_iter_val, epoch)
        # self.summary.add_scalar("val/pnp/10cm_epoch",
        #                         np.mean(ten_count), epoch)
        # self.summary.add_scalar("val/pnp/5cm_epoch",
        #                         np.mean(five_count), epoch)
        # self.summary.add_scalar("val/pnp/3cm_epoch",
        #                         np.mean(three_count), epoch)
        # self.summary.add_scalar("val/pnp/1cm_epoch", np.mean(one_count), epoch)
        # self.summary.add_scalar(
        #     "val/pnp/translation_median_epoch", np.median(translation_list), epoch)
        # self.summary.add_scalar(
        #     "val/pnp/angular_median_epoch", np.median(angular_list), epoch)

        # new_pred = {"mIoU": mIoU.item(), "Acc": Acc.item(), "Acc_class": Acc_class.item(), "FWIoU": FWIoU.item(),
        #             "10cm": np.mean(ten_count),
        #             "5cm": np.mean(five_count), "3cm": np.mean(three_count), "1cm": np.mean(one_count),
        #             "translation_median": np.median(translation_list), "angular_list": np.median(angular_list)}
        # print(new_pred)
        # if new_pred["translation_median"] < self.best_pred["translation_median"]:
        #     is_best = True
        #     self.best_pred = new_pred
        #     self.saver.save_checkpoint({
        #         "epoch": epoch + 1,
        #         "state_dict": self.model.module.state_dict(),
        #         "optimizer": self.optimizer.state_dict(),
        #         "best_pred": self.best_pred,
        #         "coding_book": self.coding_book
        #     }, is_best, save_model=self.cfg["save_model"])


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Landmark Segmentation Training")
    parser.add_argument("--dataset", type=str, default="cambridge_loc",
                        choices=["7scenes_loc", "cambridge_loc"], help="experiment config file")
    parser.add_argument("--scene", type=str, default="KingsCollege",
                        help="experiment scene")
    parser.add_argument("--gpu-id", type=str, default="",
                        help="experiment gpu id")
    parser.add_argument("--use-aug", type=str, default="",
                        choices=["", "true", "false"], help="experiment use aug")
    parser.add_argument("--config", type=str, default=None,
                        help="experiment config file")
    parser.add_argument("--debug", type=str, default="",
                        choices=["", "true", "false"], help="debug")
    parser.add_argument("--resume", type=str, default="true",
                        choices=["", "true", "false"], help="resume")
    parser.add_argument("--landmark", type=str, default="line",
                        choices=["point", "line"], help="landmark type")
    parser.add_argument("--experiment", type=str, default="line0508/",
                        help="experiment name [end with '/']")
    parser.add_argument("--visualize_seg", action='store_true')
    parser.add_argument("--visualize_voting", action='store_true')
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--image_id", type=int, default=0)
    args = parser.parse_args()

    debug = None

    if args.debug != "":
        debug = (args.debug == "true")

    if args.dataset == "7scenes_loc":
        cfg = SevenScenesLocConfig(args.config, debug)
    elif args.dataset == "cambridge_loc":
        cfg = CambridgeLocConfig(args.config, debug)
    if args.scene != "":
        cfg.opt["scene"] = args.scene
    if args.gpu_id != "":
        cfg.opt["devices"] = args.gpu_id
    if args.use_aug == "true":
        cfg.opt["use_aug"] = True
    if args.resume == "true":
        cfg.opt["resume"] = True
        # cfg.opt["resume_checkpoint"] = cfg["export_dir"] + \
        #     '/ckpts/checkpoint-backup.pth.tar'
    if args.landmark != "":
        cfg.opt["landmark"] = args.landmark
    if args.experiment != "":
        cfg.opt["experiment"] = args.experiment
    if args.visualize_seg:
        cfg.opt["visualize_segmentation"] = True
    if args.visualize_voting:
        cfg.opt["visualize_voting"] = True
    if args.base_dir != "":
        cfg.opt["base_dir"] = args.base_dir

    cfg.print_opt()
    cfg.set_environmental_variables()

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    trainer = Trainer(cfg, args.image_id)
    print("Starting Epoch:", trainer.cfg["start_epoch"])
    print("Total Epoches:", trainer.cfg["epochs"])
    trainer.validation(trainer.cfg["start_epoch"])

    trainer.summary.close()

if __name__ == "__main__":
    main()
