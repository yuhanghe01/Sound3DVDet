import sys
sys.path.append('/homes/yuhe/PycharmProjects/sound-spaces-2.0/3DSSDet/LoFTR-master/')
sys.path.append('/home/yuhang/pycharm/sound-spaces-2.0/SoundNeRAF/sound-spaces-2.0/3DSSDet/LoFTR-master/')
from src.loftr import LoFTR, default_cfg
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


class RGBImgFeatExtractor(nn.Module):
    def __init__(self, extractor_name = 'ResNet50', extract_layer = 'layer2.3.relu_2'):
        super(RGBImgFeatExtractor, self).__init__()
        assert extractor_name in ['ResNet18', 'ResNet34', 'ResNet50', 'LoFTR']
        # assert extract_layer in ['layer2.3.relu_2', 'coarse', 'fine']
        self.extractor_name = extractor_name
        self.extract_layer = extract_layer

        self.imgfeat_extractor = self.get_imgfeat_extractor()

    def get_imgfeat_extractor(self):
        if self.extractor_name == 'ResNet18':
            return self.obtain_resnet18_pretrained_model()
        elif self.extractor_name == 'ResNet34':
            return self.obtain_resnet34_pretrained_model()
        elif self.extractor_name == 'ResNet50':
            return self.obtain_resnet50_pretrained_model()
        elif self.extractor_name == 'LoFTR':
            return self.obtain_LoFTR_pretrained_model()

    def obtain_resnet50_pretrained_model(self):
        '''The mapping between extract_layer name and feature size is:
        'layer1': torch.Size([1, 256, 56, 56]),
        'layer2', torch.Size([1, 512, 28, 28]),
        'layer3', torch.Size([1, 1024, 14, 14]),
        'layer4', torch.Size([1, 2048, 7, 7])
        '''
        resnet50_model = models.resnet50(pretrained=True)
        resnet50_model = create_feature_extractor(resnet50_model, {self.extract_layer: 'imgfeat'})
        # remove the last fc-layer
        # resnet50_model = nn.Sequential(*list(resnet50_model.children())[:-1])

        for param_tmp in resnet50_model.parameters():
            param_tmp.requires_grad = False

        resnet50_model.eval()

        return resnet50_model

    def obtain_resnet34_pretrained_model(self):
        '''The mapping between extract_layer name and feature size is:
        'layer1': torch.Size([1, 64, 56, 56]),
        'layer2', torch.Size([1, 128, 28, 28]),
        'layer3', torch.Size([1, 256, 14, 14]),
        'layer4', torch.Size([1, 512, 7, 7])
        '''
        resnet34_model = models.resnet34(pretrained=True)
        resnet34_model = create_feature_extractor(resnet34_model, {self.extract_layer: 'imgfeat'})
        # remove the last fc-layer
        # resnet34_model = nn.Sequential(*list(resnet34_model.children())[:-1])

        for param_tmp in resnet34_model.parameters():
            param_tmp.requires_grad = False

        resnet34_model.eval()

        return resnet34_model

    def obtain_resnet18_pretrained_model(self):
        '''The mapping between extract_layer name and feature size is:
        'layer1': torch.Size([1, 64, 56, 56]),
        'layer2', torch.Size([1, 128, 28, 28]),
        'layer3', torch.Size([1, 256, 14, 14]),
        'layer4', torch.Size([1, 512, 7, 7])
        '''
        resnet18_model = models.resnet18(pretrained=True)
        resnet18_model = create_feature_extractor(resnet18_model, {self.extract_layer: 'imgfeat'})
        # remove the last fc-layer
        # resnet18_model = nn.Sequential(*list(resnet18_model.children())[:-1])

        for param_tmp in resnet18_model.parameters():
            param_tmp.requires_grad = False

        resnet18_model.eval()

        return resnet18_model

    def obtain_LoFTR_pretrained_model(self):
        _default_cfg = deepcopy(default_cfg)
        _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
        matcher = LoFTR(config=_default_cfg)
        matcher.load_state_dict(
            torch.load("/homes/yuhe/InternData/loftr_pretrained_model/weights/indoor_ds_new.ckpt")['state_dict'])
        matcher = matcher.eval()

        #[256, 64, 64] for coarse
        #[128, 256, 256] for fine
        if self.extract_layer == 'coarse':
            self.featdim_match_layer = nn.Linear(in_features=256, out_features=512)
        else:
            self.featdim_match_layer = nn.Linear(in_features=128, out_features=512)

        return matcher

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

import cv2
import os
import numpy as np

def normalize_an_img(input_img):
    '''
    The Input Image, the channel is in R-G-B order
    :param input_img: [H, W, 3], float32, torch.tensor
    :return: normalized tensor, in [-1, 1] range
    '''
    MEAN = 255. * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    STD = 255. * torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    input_img = input_img.permute(-1, 0, 1)

    output_img = (input_img - MEAN[:, None, None]) / STD[:, None,
                                                     None]  # [channel, height, width]

    return output_img

def prepare_an_img(img_name):
    assert os.path.exists(img_name)
    img = cv2.imread(img_name, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = cv2.resize(img, (224, 224))
    img = torch.from_numpy(img)
    img = normalize_an_img(img)

    return img

def main():
    device = torch.device('cuda:0')
    ResNet50_extractor = RGBImgFeatExtractor(extract_layer='layer1')
    ResNet50_extractor = ResNet50_extractor.to(device)

    rgb_filename = '/homes/yuhe/Desktop/H/Sound3DVDet_data/reorganize/rgbimg_filename_list.txt'
    rgb_filename_list = [line_tmp.rstrip('\n') for line_tmp in open(rgb_filename).readlines()]

    for img_id, rgb_filename in enumerate(rgb_filename_list):
        print('Processed {} imgs!'.format(img_id)) if img_id % 100 == 0 else None
        img = prepare_an_img(rgb_filename)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)

        resnet_embed = ResNet50_extractor(img)
        resnet_embed = resnet_embed.detach().cpu().numpy()
        resnet_embed = np.squeeze(resnet_embed)

        rgbimg_basename = os.path.basename(rgb_filename)
        coarse_feat_savebasename = rgbimg_basename.replace('.png', '_res50_256.npy')
        save_dir = os.path.dirname(rgb_filename)
        save_dir = save_dir.replace('/homes/yuhe/Downloads',
                                    '/homes/yuhe/Desktop/H/Sound3DVDet_data')

        np.save(os.path.join(save_dir, coarse_feat_savebasename), resnet_embed)


    print('Done!')


if __name__ == '__main__':
    main()


