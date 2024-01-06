"""
Note: 1. Given the projected [px, py], we use bilinear interpolation to collection the image feature representation
Author: Yuhang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImgFeatAggregator(nn.Module):
    def __init__(self, interp_method = 'bilinear'):
        super(ImgFeatAggregator, self).__init__()
        assert interp_method in ['bilinear', 'bicubic', 'nearest']
        self.inter_method = interp_method

    def interp_imgfeat(self, imgfeat, grid_px_py):
        '''
        Given an imgfeat, we use bilinear sampling to img feat centres around [px, py]
        TODO: double check [px, py], which direction is along the height and which direction is along the width
        :param imgfeat: [N, C, Hin, Win]
        :param grid_px_py: [N, Hout, Wout, 2]
        :return: [N, C, Hout, Wout],
        '''
        sampled_grid_feat = F.grid_sample(input=imgfeat,
                                          grid=grid_px_py,
                                          mode=self.inter_method,
                                          padding_mode='zeros')

        return sampled_grid_feat

    def feature_sampling(self, mlvl_feats, reference_points, pc_range, img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()
        reference_points_3d = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        # reference_points (B, num_queries, 4)
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        B, num_query = reference_points.size()[:2]
        num_cam = lidar2img.size(1)
        reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
        reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
        eps = 1e-5
        mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        reference_points_cam = (reference_points_cam - 0.5) * 2
        mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 1:2] > -1.0)
                & (reference_points_cam[..., 1:2] < 1.0))
        mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
        mask = torch.nan_to_num(mask)
        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            B, N, C, H, W = feat.size()
            feat = feat.view(B * N, C, H, W)
            reference_points_cam_lvl = reference_points_cam.view(B * N, num_query, 1, 2)
            sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
            sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
            sampled_feats.append(sampled_feat)
        sampled_feats = torch.stack(sampled_feats, -1)
        sampled_feats = sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
        return reference_points_3d, sampled_feats, mask

    def forward(self, input_img):
        if len(input_img.shape) == 4:
            if self.extractor_name in ['ResNet50', 'ResNet34', 'ResNet18']:
                return self.imgfeat_extractor(input_img)['imgfeat']
            elif self.extractor_name == 'LoFTR':
                batch = {'image0': input_img, 'image1': input_img}
                feat_coarse, feat_fine = self.imgfeat_extractor(batch)
                return feat_coarse, feat_fine
                # breakpoint()
                # if self.extract_layer == 'coarse':
                #     return self.featdim_match_layer(feat_coarse)
                # else:
                #     return self.featdim_match_layer(feat_fine)
        else:
            batch_size, view_num, channel_num, height, width = input_img.shape
            input_img = torch.reshape(input_img, shape=[batch_size*view_num, channel_num,height, width])
            if self.extractor_name in ['ResNet50', 'ResNet34', 'ResNet18']:
                img_feat = self.imgfeat_extractor(input_img)['imgfeat']
            elif self.extractor_name == 'LoFTR':
                batch = {'image0': input_img, 'image1': input_img}
                feat_coarse, feat_fine = self.imgfeat_extractor(batch)
                if self.extract_layer == 'coarse':
                    img_feat = self.featdim_match_layer(feat_coarse)
                else:
                    img_feat = self.featdim_match_layer(feat_fine)

            img_feat_dim, img_feat_height, img_feat_width = img_feat.shape[1:]
            img_feat = torch.reshape(img_feat, shape=[batch_size, view_num, img_feat_dim,
                                                      img_feat_height, img_feat_width])

            return img_feat