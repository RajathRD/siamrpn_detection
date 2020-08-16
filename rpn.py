from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.input_size = cfg.TRAIN.SCALES[0]
        self.template_size = cfg.TRAIN.query_size
        # define the convrelu layers processing input feature map
        # self.mix_Conv = nn.Sequential(
        #         nn.Conv2d(self.din, 512, 3, 1, 1, bias=True),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True)
        #     )
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)

        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        print ("rpn_cls_prob", rpn_cls_prob.size())
        print ("rpn_cls_score", rpn_cls_score.size())

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
        print("rpn_bbox_pred", rpn_bbox_pred.size())
        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0


        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))

            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
        return rois, self.rpn_loss_cls, self.rpn_loss_box

class _siamRPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_siamRPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.input_size = cfg.TRAIN.SCALES[0]
        self.template_size = cfg.TRAIN.query_size
        self.score_displacement = int((self.input_size - self.template_size) / self.feat_stride)
        # define the convrelu layers processing input feature map
        # self.mix_Conv = nn.Sequential(
        #         nn.Conv2d(self.din, 512, 3, 1, 1, bias=True),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True)
        #     )
        self.RPN_Conv_1 = nn.Conv2d(self.din, 256, 3)
        self.RPN_Conv_2 = nn.Conv2d(self.din, 256, 3)

        # define bg/fg classifcation score layer
        self.anchor_num = len(self.anchor_scales) * len(self.anchor_ratios)
        self.nc_score_out = self.anchor_num * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score_1 = nn.Conv2d(self.din, 256*self.nc_score_out, 3, 1, 0)
        self.RPN_cls_score_2 = nn.Conv2d(self.din, 256, 3, 1, 0)

        # define anchor box offset prediction layer]
        
        self.nc_bbox_out = self.anchor_num * 4 # 4(coords) * 9 (anchors)

        self.RPN_bbox_pred_1 = nn.Conv2d(self.din, 256*self.nc_bbox_out, 3, 1, 0)
        self.RPN_bbox_pred_2 = nn.Conv2d(self.din, 256, 3, 1, 0)

        self.regress_adjust = nn.Conv2d(self.nc_bbox_out, self.nc_bbox_out, 1)
        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, detect_feat, query_feat, im_info, gt_boxes, num_boxes):
        batch_size = query_feat.size(0)
        # return feature map after convrelu layer
        rpn_cls_score1 = self.RPN_cls_score_1(query_feat)
        rpn_bbox_pred1 = self.RPN_bbox_pred_1(query_feat)

        rpn_cls_score2 = self.RPN_cls_score_2(detect_feat)
        rpn_bbox_pred2 = self.RPN_bbox_pred_2(detect_feat)


        # print ("rpn_cls_score1", rpn_cls_score1.size())
        # print ("rpn_cls_score2", rpn_cls_score2.size())
        # print ("rpn_bbox_pred1", rpn_bbox_pred1.size())
        # print ("rpn_bbox_pred2", rpn_bbox_pred2.size())

        score_filters = rpn_cls_score1.view(-1, 256, 6, 6)
        conv_scores = rpn_cls_score2.view(1, -1, rpn_cls_score2.size()[2], rpn_cls_score2.size()[3])
        
        rpn_cls_score = F.conv2d(conv_scores, score_filters, groups=batch_size)
        # print ("rpn_cls_score", rpn_cls_score.size())
        rpn_cls_score = rpn_cls_score.reshape(batch_size, 2 * self.anchor_num, rpn_cls_score.size()[2], rpn_cls_score.size()[3])
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        
        # print ("rpn_cls_score", rpn_cls_score.size())
        # get rpn offsets to the anchor boxes
        
        kernel_regression = rpn_bbox_pred1.view(-1, 256, 6, 6)
        conv_reg = rpn_bbox_pred2.reshape(1, -1, rpn_bbox_pred2.size()[2], rpn_bbox_pred2.size()[3])
        rpn_bbox_pred = F.conv2d(conv_reg, kernel_regression, groups=batch_size)
        rpn_bbox_pred = rpn_bbox_pred.reshape(batch_size, 4 * self.anchor_num, rpn_bbox_pred.size()[2], rpn_bbox_pred.size()[3])
        # print("rpn_bbox_pred", rpn_bbox_pred.size())
        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0


        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))

            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            # print ("rpn_cls_score", rpn_cls_score, "\nrpn_label", rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            # print ("rpn_targets", rpn_bbox_targets, rpn_bbox_targets.size())
            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
        return rois, self.rpn_loss_cls, self.rpn_loss_box
