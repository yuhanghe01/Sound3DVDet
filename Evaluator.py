import os.path

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import pickle

eps = np.finfo(np.float).eps
from scipy.optimize import linear_sum_assignment
import yaml
import glob
import scipy
from scipy.spatial.distance import cdist
import sys
sys.path.append('LoFTR-master')

import Sound3DVDetNet

class Sound3DVDetEvaluator:
    def __init__(self, dist_thred_list = [0.3, 0.6, 0.8],
                 view_num = 4,
                 test_data_dir = None,
                 device = 'cuda:0',
                 pretrained_model = None,
                 config_filename = None,
                 class_num = 5,
                 class_score_thred=0.5):
        # super(Sound3DVDetEvaluator).__init__()
        self.dist_thred_list = dist_thred_list
        self.view_num = view_num

        self.device = device

        assert os.path.exists(test_data_dir)
        self.test_file_list = glob.glob(os.path.join(test_data_dir, '*.pickle'))
        assert len(self.test_file_list) > 0

        self.pretrained_model = pretrained_model
        self.config_filename = config_filename
        self.class_num = class_num
        self.class_score_thred = class_score_thred

        self.init_model()
        self.init_eval_dict()

        # if imgfeat_extract_type == 'LoFTR' and imgfeat_extract_layer == 'coarse':
        #     img_feat_key = 'rgb_feat_loftr_coarse'
        # elif imgfeat_extract_type == 'LoFTR' and imgfeat_extract_layer == 'fine':
        #     img_feat_key = 'rgb_feat_loftr_fine'
        # elif imgfeat_extract_type == 'ResNet50':
        #     img_feat_key = 'rgb_feat_resnet50'
        # else:
        #     raise ValueError('Unkown Pre-extracted Image Feature!')
        # self.img_feat_key = 'rgb_feat_resnet50'
        self.img_feat_key = 'rgb_feat_loftr_coarse'

    def init_eval_dict(self):
        self.eval_dict = dict()
        self.eval_dict['mAP'] = 0.
        self.eval_dict['mAR'] = 0.
        self.eval_dict['mALE'] = 0.

        for class_label in range(self.class_num):
            self.eval_dict[class_label] = dict()
            self.eval_dict[class_label]['AP'] = 0.
            self.eval_dict[class_label]['AR'] = 0.
            self.eval_dict[class_label]['ALE'] = 0.
            for dist_thred in self.dist_thred_list:
                self.eval_dict[class_label][dist_thred] = dict()
                self.eval_dict[class_label][dist_thred]['TP_num'] = 0
                self.eval_dict[class_label][dist_thred]['TN_num'] = 0
                self.eval_dict[class_label][dist_thred]['FN_num'] = 0
                self.eval_dict[class_label][dist_thred]['LE'] = list()


    def init_model(self):
        assert os.path.exists(self.config_filename)
        with open(self.config_filename) as f:
            input_args = yaml.safe_load(f)
        #TODO: may need to change args to reflect different model variants
        #input_args['IMGFEAT_EXTRACT_CONFIG']['extractor_name'] = 'ResNet50'

        net = Sound3DVDetNet.Sound3DVDet(
            input_soundfeat_channel_num=input_args['SOUND3DVDET_CONFIG']['soundfeat_channel_num'],
            ss_query_feat_dim=input_args['SOUND3DVDET_CONFIG']['ss_query_feat_dim'],
            class_num=input_args['SOUND3DVDET_CONFIG']['class_num'],
            transformer_layernum=input_args['TRANSFORMER_CONFIG']['transformer_layer_num'],
            transformer_config=input_args['TRANSFORMER_CONFIG'],
            ss_query_num=input_args['SOUND3DVDET_CONFIG']['ss_query_num'],
            img_height=input_args['SOUND3DVDET_CONFIG']['img_height'],
            img_width=input_args['SOUND3DVDET_CONFIG']['img_width'],
            device=self.device,
            multiview_num=input_args['SOUND3DVDET_CONFIG']['multiview_num'],
            feat_aggreg_method=input_args['SOUND3DVDET_CONFIG']['feat_aggmethod'],
            aggregate_img_feat=input_args['SOUND3DVDET_CONFIG']['agg_imgfeat'],
            aggregate_sound_feat=input_args['SOUND3DVDET_CONFIG']['agg_soundfeat'],
            rgbfeat_extractor_name=input_args['IMGFEAT_EXTRACT_CONFIG']['extractor_name'],
            rgbfeat_extract_layer_name=input_args['IMGFEAT_EXTRACT_CONFIG']['extract_layer_name'])

        net = net.to(self.device)
        model_state = net.state_dict()
        pretrained_model = torch.load(self.pretrained_model)
        model_state.update(pretrained_model['model'])
        net.load_state_dict(model_state)

        net.to(device=self.device)
        net.eval()

        self.model = net

    def parse_one_data(self, input_multiview_data_filename):
        with open(input_multiview_data_filename, 'rb') as handle:
            input_multiview_data = pickle.load(handle)
        rgb_feat = input_multiview_data[self.img_feat_key]
        logmel_gccphat = input_multiview_data['logmel_gccphat_feat']
        ss_pos_camcoord = input_multiview_data['ss_pos_camcoord']
        ss_label = input_multiview_data['ss_label']
        cam_pos = input_multiview_data['cam_pos']
        cam_rot = input_multiview_data['cam_rot']

        #iterate over views
        refview_img_feat_list = list()
        remainview_img_feat_list = list()
        refview_logmel_gccphat_feat_list = list()
        remainview_logmel_gccphat_feat_list = list()

        refview_campos_list = list()
        refview_camrot_list = list()

        remainview_campos_list = list()
        remainview_camrot_list = list()

        refview_ss_pos_camcoord_list = list()
        remainview_ss_pos_camcoord_list = list()

        for view_id in range(self.view_num):
            refview_id = view_id
            remain_view_ids = np.arange(self.view_num).tolist()
            remain_view_ids.remove(refview_id)

            refview_img_feat = rgb_feat[refview_id, :, :, :]
            remainview_img_feat = rgb_feat[remain_view_ids, :, :, :]

            refview_logmel_gccphat = logmel_gccphat[refview_id, :, :, :]
            remainview_logmel_gccphat = logmel_gccphat[remain_view_ids, :, :, :]

            refview_ss_pos_camcoord = ss_pos_camcoord[refview_id, :]
            remainview_ss_pos_camcoord = ss_pos_camcoord[remain_view_ids, :]

            refview_campos = cam_pos[refview_id, :]
            remainview_campos = cam_pos[remain_view_ids, :]

            refview_camrot = cam_rot[refview_id, :]
            remainview_camrot = cam_rot[remain_view_ids, :]

            refview_img_feat_list.append(refview_img_feat)
            remainview_img_feat_list.append(remainview_img_feat)

            refview_logmel_gccphat_feat_list.append(refview_logmel_gccphat)
            remainview_logmel_gccphat_feat_list.append(remainview_logmel_gccphat)

            refview_campos_list.append(refview_campos)
            refview_camrot_list.append(refview_camrot)

            remainview_campos_list.append(remainview_campos)
            remainview_camrot_list.append(remainview_camrot)

            refview_ss_pos_camcoord_list.append(refview_ss_pos_camcoord)
            remainview_ss_pos_camcoord_list.append(remainview_ss_pos_camcoord)


        refview_img_feat = np.stack(refview_img_feat_list, axis=0)
        remainview_img_feat = np.stack(remainview_img_feat_list, axis=0)

        refview_logmel_gccphat = np.stack(refview_logmel_gccphat_feat_list, axis=0)
        remainview_logmel_gccphat = np.stack(remainview_logmel_gccphat_feat_list, axis=0)

        refview_campos = np.stack(refview_campos_list, axis=0)
        remainview_campos = np.stack(remainview_campos_list, axis=0)

        refview_camrot = np.stack(refview_camrot_list, axis=0)
        remainview_camrot = np.stack(remainview_camrot_list, axis=0)

        refview_ss_pos_camcoord = np.stack(refview_campos_list, axis=0)
        remainview_ss_pos_camcoord = np.stack(remainview_ss_pos_camcoord_list, axis=0)

        return refview_img_feat, remainview_img_feat, \
               refview_logmel_gccphat, remainview_logmel_gccphat, \
               refview_campos, remainview_campos, \
               refview_camrot, remainview_camrot, \
               refview_ss_pos_camcoord, remainview_ss_pos_camcoord, \
               ss_label


    def construct_test_data(self):
        with open(self.test_data_filename, 'rb') as handle:
            self.test_file_list = pickle.load(handle)['data_list']

    def distance_between_spherical_coordinates_rad(self, az1, ele1, az2, ele2):
        """
        Angular distance between two spherical coordinates
        MORE: https://en.wikipedia.org/wiki/Great-circle_distance
        :return: angular distance in degrees
        """
        dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
        # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
        dist = np.clip(dist, -1, 1)
        dist = np.arccos(dist) * 180 / np.pi

        return dist


    def inference(self, refview_img_feat, remainview_img_feat,
               refview_logmel_gccphat, remainview_logmel_gccphat,
               refview_campos, remainview_campos,
               refview_camrot, remainview_camrot):

        refview_img_feat = torch.from_numpy(refview_img_feat).to(self.device)
        remainview_img_feat = torch.from_numpy(remainview_img_feat).to(self.device)
        refview_logmel_gccphat = torch.from_numpy(refview_logmel_gccphat).to(self.device)
        remainview_logmel_gccphat = torch.from_numpy(remainview_logmel_gccphat).to(self.device)

        refview_campos = torch.from_numpy(refview_campos).to(self.device)
        remainview_campos = torch.from_numpy(remainview_campos).to(self.device)

        refview_camrot = torch.from_numpy(refview_camrot).to(self.device)
        remainview_camrot = torch.from_numpy(remainview_camrot).to(self.device)

        output_dict = self.model(refview_campos, refview_camrot, remainview_campos, remainview_camrot,
                                 refview_img_feat, refview_logmel_gccphat,
                                 remainview_img_feat, remainview_logmel_gccphat )


        pred_ss_3Dpos = output_dict['sound3dvdet_output']['pred_ss3Dpos']
        pred_class_logits = output_dict['sound3dvdet_output']['pred_ssclasslogits']
        pred_class_score = F.softmax(pred_class_logits, dim=-1)

        pred_ss_3Dpos = pred_ss_3Dpos.detach().cpu().numpy()
        pred_class_score = pred_class_score.detach().cpu().numpy()

        return pred_ss_3Dpos, pred_class_score

    def update_eval_dict(self, gt_refview_ss_pos_camcoord, gt_ss_label,
                         pred_refview_ss_pos_camcoord, pred_ss_class_score):
        '''
        :param gt_refview_ss_pos_camcoord: [view_num, query_num, 3]
        :param gt_ss_label: [view_num, query_num]
        :param pred_refview_ss_pos_camcoord: [view_num, query_num, 3]
        :param pred_ss_class_score: [view_num, query_num, class_num]
        :return: None, updateh eval dict internally
        '''
        # pred_ss_label = np.argsort(pred_ss_class_score, axis=-1)[:,:,-1]

        pred_ss_label = np.ones([4,16],np.int32)*5

        for view_id in range(self.view_num):
            for query_id in range(16):
                init_ss_label = np.argsort(pred_ss_class_score[view_id, query_id, :])[-1]
                if init_ss_label != 5 and pred_ss_class_score[view_id, query_id, init_ss_label] >= self.class_score_thred:
                    pred_ss_label[view_id, query_id] = init_ss_label


        #iter over all classes
        class_label_list = np.unique(gt_ss_label).tolist()
        for class_label in class_label_list:
            #if it is background label, ignore it
            if class_label == -1:
                continue
            gt_ss_3Dpos = gt_refview_ss_pos_camcoord[np.where(gt_ss_label == class_label)[0]]
            gt_ss_3Dpos = np.squeeze(gt_ss_3Dpos)

            if len(gt_ss_3Dpos.shape) == 1:
                gt_ss_3Dpos = np.expand_dims(gt_ss_3Dpos, axis=0)

            #no detections for this class
            if len(np.where(pred_ss_label==class_label)[0]) == 0:
                for dist_thred in self.dist_thred_list:
                    self.eval_dict[class_label][dist_thred]['FN_num'] = gt_ss_3Dpos.shape[0]
            else:
                pred_ss_3Dpos = pred_refview_ss_pos_camcoord[np.where(pred_ss_label == class_label)[0], np.where(pred_ss_label == class_label)[1]]
                dist_matrice= cdist(gt_ss_3Dpos, pred_ss_3Dpos, 'euclidean')
                row_ind, col_ind = linear_sum_assignment(dist_matrice)

                # print('min_dist_matrices = {}'.format(np.min(dist_matrice)))

                common_ind_len = min(row_ind.shape[0], col_ind.shape[0])
                for match_id in range(common_ind_len):
                    row_id, col_id = row_ind[match_id], col_ind[match_id]
                    euc_dist = dist_matrice[row_id, col_id]

                    for dist_thred in self.dist_thred_list:
                        if dist_thred >= euc_dist:
                            self.eval_dict[class_label][dist_thred]['TP_num'] += 1
                            self.eval_dict[class_label][dist_thred]['LE'].append(euc_dist)
                        else:
                            self.eval_dict[class_label][dist_thred]['TN_num'] += 1

                if common_ind_len < row_ind.shape[0]:
                    for dist_thred in self.dist_thred_list:
                        self.eval_dict[class_label][dist_thred]['FN_num'] += row_ind.shape[0] - common_ind_len
                if common_ind_len < col_ind.shape[0]:
                    for dist_thred in self.dist_thred_list:
                        self.eval_dict[class_label][dist_thred]['TN_num'] += col_ind.shape[0] - common_ind_len

    def summarize_eval_result(self):
        AP_list = list()
        AR_list = list()
        ALE_list = list()
        for class_id in range(self.class_num):
            TP_num = 0
            TN_num = 0
            FN_num = 0
            LE_list = list()

            for dist_thred in self.dist_thred_list:
                TP_num += self.eval_dict[class_id][dist_thred]['TP_num']
                TN_num += self.eval_dict[class_id][dist_thred]['TN_num']
                FN_num += self.eval_dict[class_id][dist_thred]['FN_num']
                LE_list.append(np.sum(np.array(self.eval_dict[class_id][dist_thred]['LE'],np.float32))/(self.eval_dict[class_id][dist_thred]['TP_num']+eps))

            self.eval_dict[class_id]['AP'] = TP_num/(TP_num+TN_num+eps)
            self.eval_dict[class_id]['AR'] = TP_num/(TP_num+FN_num+eps)
            assert len(LE_list) > 0
            self.eval_dict[class_id]['ALE'] = np.sum(np.array(LE_list,np.float32))/(len(self.dist_thred_list)+eps)

            AP_list.append(self.eval_dict[class_id]['AP'])
            AR_list.append(self.eval_dict[class_id]['AR'])
            ALE_list.append(self.eval_dict[class_id]['ALE'])

        self.eval_dict['mAP'] = np.sum(np.array(AP_list,np.float32))/self.class_num
        self.eval_dict['mAR'] = np.sum(np.array(AR_list, np.float32)) / self.class_num
        self.eval_dict['mALE'] = np.sum(np.array(ALE_list, np.float32)) / self.class_num


        print('*****Eval Result*******')
        for class_id in range(self.class_num):
            print('class: {}, AP: {};\tAR: {};\tLE: {};'.format(class_id,
                                                                self.eval_dict[class_id]['AP'],
                                                                self.eval_dict[class_id]['AR'],
                                                                self.eval_dict[class_id]['ALE']))
        print('Overall Result:')
        print('mAP: {};\t mAR: {};\t mALE: {}'.format(self.eval_dict['mAP'], self.eval_dict['mAR'], self.eval_dict['mALE']))
        print('*******Done!**********')




    def evaluate(self):
        for file_id, filename in enumerate(self.test_file_list):
            if file_id % 10 == 0:
                print('Processed {}/{} file!'.format(file_id, len(self.test_file_list)))
            assert os.path.exists(filename)

            parsed_data = self.parse_one_data(filename)

            pred_ss_3Dpos, pred_class_score = self.inference(parsed_data[0], parsed_data[1],
                                                             parsed_data[2], parsed_data[3],
                                                             parsed_data[4], parsed_data[5],
                                                             parsed_data[6], parsed_data[7])

            self.update_eval_dict(parsed_data[8], parsed_data[10], pred_ss_3Dpos, pred_class_score)

        self.summarize_eval_result()

def main():
    #Eval Parameter Configuration
    dist_tred_list = [0.3, 0.7, 1.0]
    view_num = 4
    pretrained_model = 'sound3dvdet_model_epoch_100.pth'
    config_filename = 'config.yaml'
    class_num = 5
    class_score_thred = 0.7


    sound3dvdet_evaluator = Sound3DVDetEvaluator(dist_thred_list=dist_tred_list,
                                                 view_num=view_num,
                                                 test_data_dir=test_data_dir,
                                                 pretrained_model=pretrained_model,
                                                 config_filename=config_filename,
                                                 class_num=class_num,
                                                 class_score_thred=class_score_thred)
    sound3dvdet_evaluator.evaluate()


if __name__ == '__main__':
    main()

