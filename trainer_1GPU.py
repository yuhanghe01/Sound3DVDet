"""
Train script of Sound3DVDet
"""
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
os.putenv('CUDA_VISIBLE_DEVICES','0')

import Sound3DVDetNet_bak
import DataProvider
import Loss


def add_gt_to_output_dict( output_dict, ss_pos_camcoords, ss_labels, view_num, transformer_layer_num):
    output_dict['sound3dvdet_output']['gt_ss3Dpos'] = ss_pos_camcoords[:,0,:]
    output_dict['sound3dvdet_output']['gt_ssclasslabels'] = ss_labels[:,0,:]

    output_dict['refview_output']['gt_ss3Dpos'] = ss_pos_camcoords[:,0,:]
    output_dict['refview_output']['gt_ssclasslabels'] = ss_labels[:,0,:]

    for layer_id in range(transformer_layer_num):
        output_dict['transformer_interm_output']['layer_{}'.format(layer_id)]['gt_ss3Dpos'] = ss_pos_camcoords[:,0,:]
        output_dict['transformer_interm_output']['layer_{}'.format(layer_id)]['gt_ssclasslabels'] = ss_labels[:,0,:]

    for view_id in range(view_num-1):
        output_dict['multiview_output']['view_{}'.format(view_id)]['gt_ss3Dpos'] = ss_pos_camcoords[:,view_id+1,:]
        output_dict['multiview_output']['view_{}'.format(view_id)]['gt_ssclasslabels'] = ss_labels[:,view_id+1,:]

    return output_dict

def sanitize_config(config):
    assert config['TRAIN_OPTION']['pred_mono'] == config['DATASET_OPTION']['use_mono']

def train(input_args):
    device = torch.device(type="cuda", index=0)
    #step1: initialise the neural network
    net = Sound3DVDetNet.Sound3DVDet(input_soundfeat_channel_num=input_args['SOUND3DVDET_CONFIG']['soundfeat_channel_num'],
                                     ss_query_feat_dim=input_args['SOUND3DVDET_CONFIG']['ss_query_feat_dim'],
                                     class_num=input_args['SOUND3DVDET_CONFIG']['class_num'],
                                     transformer_layernum=input_args['TRANSFORMER_CONFIG']['transformer_layer_num'],
                                     transformer_config=input_args['TRANSFORMER_CONFIG'],
                                     ss_query_num=input_args['SOUND3DVDET_CONFIG']['ss_query_num'],
                                     img_height=input_args['SOUND3DVDET_CONFIG']['img_height'],
                                     img_width=input_args['SOUND3DVDET_CONFIG']['img_width'],
                                     device=device,
                                     multiview_num=input_args['SOUND3DVDET_CONFIG']['multiview_num'],
                                     feat_aggreg_method=input_args['SOUND3DVDET_CONFIG']['feat_aggmethod'],
                                     aggregate_img_feat=input_args['SOUND3DVDET_CONFIG']['agg_imgfeat'],
                                     aggregate_sound_feat=input_args['SOUND3DVDET_CONFIG']['agg_soundfeat'],)
    net = net.to(device)
    net.train()

    #step2: initialise the train data loader
    #TODO: need to fill in the details
    print('Construct data loader ...')
    trainset = DataProvider.Sound3DVDetDataset_DiverseSS()

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=input_args['TRAIN_CONFIG']['batch_size'],
                                               num_workers = 4,
                                               shuffle=True,)

    #step3: construct the optimizer
    if input_args['TRAIN_CONFIG']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=input_args['TRAIN_CONFIG']['init_lr'],
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=input_args['TRAIN_CONFIG']['weight_decay'],
                                     amsgrad=True)

    elif input_args['TRAIN_CONFIG']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(),
                                     lr=input_args['TRAIN_CONFIG']['init_lr'],
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=input_args['TRAIN_CONFIG']['weight_decay'],
                                     amsgrad=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=input_args['TRAIN_CONFIG']['lr_decay_epochs'],
                                                   gamma=input_args['TRAIN_CONFIG']['lr_decay_gamma'])

    #step4: construct Loss object
    loss_calculator = Loss.SSDetLoss(num_classes = input_args['SOUND3DVDET_CONFIG']['class_num'],
                                     eos_coef = input_args['TRAIN_CONFIG']['no_object_weight'],
                                     cost_class_weight = input_args['TRAIN_CONFIG']['ce_weight'],
                                     cost_pos_weight = input_args['TRAIN_CONFIG']['l1_weight'],
                                     deeply_supervise = input_args['TRAIN_CONFIG']['deeply_supervise'],
                                     multiview_num = input_args['SOUND3DVDET_CONFIG']['multiview_num'],
                                     transformer_layer_num = input_args['TRANSFORMER_CONFIG']['transformer_layer_num'],
                                     )

    loss_calculator = loss_calculator.to(device)

    print('begin training')

    loss_val_list = list()
    #step5: start to train
    for ep in range(1, input_args['TRAIN_CONFIG']['train_epochs'] + 1):
        for idx, data_tmp in enumerate(train_loader):
            rgb_feat = data_tmp[0]
            logmel_gccphat = data_tmp[1]
            ss_pos_camcoords = data_tmp[2]
            ss_labels = data_tmp[3]
            cam_pos = data_tmp[4]  # [N,viewnum,3]
            cam_rot = data_tmp[5]  # [N,viewnum,4]

            ref_camera_pos = cam_pos[:, 0, :]
            ref_camera_rot = cam_rot[:, 0, :]
            multiview_camera_pos = cam_pos[:, 1:, :]
            multiview_camera_rot = cam_rot[:, 1:, :]
            ref_img_feat = rgb_feat[:, 0, :, :, :]
            ref_micarray_input = logmel_gccphat[:, 0, :, :, :]
            multiview_img_feat = rgb_feat[:, 1:, :, :, :]
            multiview_micarray_input = logmel_gccphat[:, 1:, :, :, :]

            ref_camera_pos = ref_camera_pos.to(device)
            ref_camera_rot = ref_camera_rot.to(device)
            multiview_camera_pos = multiview_camera_pos.to(device)
            multiview_camera_rot = multiview_camera_rot.to(device)
            ref_img_feat = ref_img_feat.to(device)
            ref_micarray_input = ref_micarray_input.to(device)
            multiview_img_feat = multiview_img_feat.to(device)
            multiview_micarray_input = multiview_micarray_input.to(device)

            # with torch.autograd.set_detect_anomaly(True):
            output_dict = net(ref_camera_pos,
                              ref_camera_rot,
                              multiview_camera_pos,
                              multiview_camera_rot,
                              ref_img_feat,
                              ref_micarray_input,
                              multiview_img_feat,
                              multiview_micarray_input)

            ss_pos_camcoords = ss_pos_camcoords.to(device)
            ss_labels = ss_labels.to(device)

            #update output_dict, we input the ground truth to the appropriate position
            output_dict = add_gt_to_output_dict(output_dict,
                                                ss_pos_camcoords=ss_pos_camcoords,
                                                ss_labels=ss_labels,
                                                view_num=input_args['SOUND3DVDET_CONFIG']['multiview_num'],
                                                transformer_layer_num=input_args['TRANSFORMER_CONFIG']['transformer_layer_num'])

            loss = loss_calculator(output_dict)[0]
            # loss = torch.sum(output_dict['sound3dvdet_output']['pred_ss3Dpos'])


            # if args['TRAIN_OPTION']['add_param_l2regu_loss']:
            #     l2_norm_loss = utils.get_param_l2regu_loss(model=net)
            #     loss += args['TRAIN_OPTION']['param_l2regu_loss_weight']*l2_norm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Finished one step training!')

            current_lr = optimizer.param_groups[0]['lr']

            if idx % 1 == 0:
                print("== step: [{:3}/{}] | epoch: [{}/{}] | loss: {:.6f} | lr = {}".format(idx + 1,
                                                                                 len(train_loader),
                                                                                 ep,
                                                                                 input_args['TRAIN_CONFIG']['train_epochs'],
                                                                                 loss.detach().cpu().numpy(),
                                                                                 current_lr))

            if ((ep-1)*len(train_loader) + idx)%input_args['TRAIN_CONFIG']['train_epochs'] == 0:
                loss_val_list.append(loss.detach().cpu().numpy())

        if ep % input_args['TRAIN_CONFIG']['save_every_n_epochs'] == 0:
            model_save_basename = 'sound3dvdet_model_epoch_{}.pth'.format(ep)
            outdict = {
                'epoch': ep,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),}

            torch.save(outdict, os.path.join(input_args['TRAIN_CONFIG']['checkpoint_dir'], model_save_basename))

        lr_scheduler.step()

    #dump the loss curve
    np.save(os.path.join(input_args['TRAIN_CONFIG']['checkpoint_dir'], 'loss_curve.npy'),
            np.array(loss_val_list, np.float32))

    print("\n=======  Training Finished  ======= \n")


def sanitize_config(config):
    if config['IMGFEAT_EXTRACT_CONFIG']['extractor_name'] == 'LoFTR':
        assert config['IMGFEAT_EXTRACT_CONFIG']['extract_layer_name'] in ['fine', 'coarse']
    else:
        assert config['IMGFEAT_EXTRACT_CONFIG']['extractor_name'] in ['ResNet18', 'ResNet34', 'ResNet50']


def main():
    yaml_config_filename = 'config.yaml'
    assert os.path.exists(yaml_config_filename)
    with open(yaml_config_filename) as f:
        args_config = yaml.safe_load(f)

    sanitize_config(args_config)
    train(args_config)

if __name__ == '__main__':
    main()