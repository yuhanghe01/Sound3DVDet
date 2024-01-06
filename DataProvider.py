import os
import torch
import json
import glob
import random
import numpy as np
import cv2
import math
from scipy.signal import fftconvolve
import librosa
import itertools as it
import WaveSpectrumConverter
import data_utils
import MicArrayFeat
import pickle

class Sound3DVDetDataset_FixedSS(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir='/mnt/nas/yuhang/3DSSDet_Data_efficient_multiss_large',
                 ss_num = 5,
                 multiview_num = 3,
                 seed_sound_dir = '/home/yuhang/pycharm/sound-spaces-2.0/data/sounds/1s_all',
                 sampling_rate = 20001,
                 use_pano_img = True,
                 all_seed_sound_telephone = True,
                 test_set_ratio = 0.30,
                 turn_angle = 30,
                 mode = 'train',
                 room_scene_name = None,
                 RIR_dim = 20001,
                 random_seed = 100,
                 cate_name = 'disc'):
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.root_dir = root_dir
        self.ss_num = ss_num
        self.cate_name = cate_name
        self.multiview_num = multiview_num
        self.turn_angle = turn_angle
        self.turn_angle_radian = math.pi / (180. / self.turn_angle)
        self.seed_sound_dir = seed_sound_dir
        self.sampling_rate = sampling_rate
        self.use_pano_img = use_pano_img
        self.RIR_dim = RIR_dim
        self.all_seed_sound_telephone = all_seed_sound_telephone
        self.test_set_ratio = test_set_ratio
        assert mode in ['train', 'test']
        self.mode = mode
        self.room_scene_name = room_scene_name
        self.get_seedsound_filename_list()
        self.get_seed_sound_list()
        self.data_processor = data_utils.DataProcessor()
        self.micarray_processor = MicArrayFeat.MicArrayFeat()
        self.prepare_data()
        self.wavespectrum_converter = WaveSpectrumConverter.WaveSpectrumConverter()
        self.ss_grid_map = self.data_processor.get_grid_center_coords()

    def prepare_data(self):
        self.multiview_list = list()

        datadir_list = self.get_datadir_list()

        for datadir_tmp in datadir_list:
            self.parse_one_multiview_multiss( datadir_tmp,
                                              self.cate_name)

        assert len(self.multiview_list) > 0
        random.shuffle(self.multiview_list)
        print('multiview num = {}'.format(len(self.multiview_list)))
        self.train_multiview_list = self.multiview_list[0:int(len(self.multiview_list)*0.8)]
        self.test_multiview_list = self.multiview_list[int(len(self.multiview_list)*0.8):]

    def get_datadir_list(self):
        datadir_list = list()
        phyobj_name = 'wall'
        for scene_name in os.listdir(self.root_dir):
            phyobj_name_list = os.listdir(os.path.join(self.root_dir, scene_name))
            if phyobj_name not in phyobj_name_list:
                continue
            data_dirs = os.listdir(os.path.join(self.root_dir, scene_name, phyobj_name))
            if len(data_dirs) == 0:
                continue
            for data_dir_tmp in data_dirs:
                dir_tmp = os.path.join(self.root_dir, scene_name, phyobj_name, data_dir_tmp)
                datadir_list.append(dir_tmp)

        return datadir_list

    def get_micarray_audiofeat(self, micarray_RIR):
        '''
        Given a input microphone array RIR, we compute the audio feature
        :param micarray_RIR: [4, RIR_dim]
        :return: [H, W, 10] feature
        '''
        audio_list = list()
        for RIR_tmp in micarray_RIR:
            #TODO: may need to change to seed sound, if there are multiple sound sources in the environment
            audio_list.append(self.convolve_monoRIR_get_audio(self.seed_sound_list[0],
                                                              RIR_tmp))
        micarray_audio = np.stack(audio_list, axis=0)

        # logmel_gccphat_feat = self.data_processor.get_micarray_feat(micarray_audio)
        logmel_gccphat_feat = self.micarray_processor.get_micarray_feat(micarray_audio) #[10, 512, 512]

        return logmel_gccphat_feat

    def parse_one_multiview_multiss(self, input_dir, category_name):
        assert category_name in ['homo', 'disc']
        #step1: decide if homo RIR exists
        if category_name == 'homo':
            homo_RIR_list = glob.glob( os.path.join(input_dir, 'novel_view_micarray_RIR{}_*.npy'.format('mono')))
        else:
            homo_RIR_list = glob.glob(os.path.join(input_dir, 'novel_view_micarray_RIR{}_*.npy'.format('disc')))

        if len(homo_RIR_list) > 0:
            recordness_list = list()
            rgb_filename_list = list()
            for RIR_homo_filename in homo_RIR_list:
                RIR_homo_basename = os.path.basename(RIR_homo_filename)
                if category_name == 'homo':
                    img_id = int(RIR_homo_basename.replace('novel_view_micarray_RIRmono_','').replace('.npy',''))
                else:
                    img_id = int(RIR_homo_basename.replace('novel_view_micarray_RIRdisc_', '').replace('.npy', ''))
                rgb_img_basename = 'novel_view_rgb{}.png'.format(img_id)
                rgb_img_filename = os.path.join( os.path.dirname(RIR_homo_filename),
                                                 rgb_img_basename )
                assert os.path.exists(rgb_img_filename)

                rgb_filename_list.append(rgb_img_filename)

                if category_name == 'homo':
                    recordness_basename = 'novel_view_mono_recordness_{}.npy'.format(img_id)
                else:
                    recordness_basename = 'novel_view_disc_recordness_{}.npy'.format(img_id)

                recordness_filename = os.path.join( os.path.dirname(RIR_homo_filename),
                                                    recordness_basename )
                assert os.path.exists(recordness_filename)
                recordness_list.append(np.load(recordness_filename))

            recordness_homo = np.stack(recordness_list, axis=0)

            if category_name == 'homo':
                ss_pos_homo_filename = os.path.join( input_dir, 'ss_pos_mono.npy' )
            else:
                ss_pos_homo_filename = os.path.join( input_dir, 'ss_pos_disc.npy' )

            assert os.path.exists(ss_pos_homo_filename)
            ss_pos_homo = np.load(ss_pos_homo_filename)

            #extract multiview info
            each_ss_viewed_time = np.sum(recordness_homo, axis=0, keepdims=False).astype(np.int32)
            each_img_views_ssnum = np.sum(recordness_homo, axis=1, keepdims=False).astype(np.int32)

            ss_candidates = np.where(each_ss_viewed_time >= self.multiview_num )[0].tolist()
            img_candidates = np.where(each_img_views_ssnum >= self.ss_num )[0].tolist()

            if len(ss_candidates) < self.multiview_num or len(img_candidates) < self.ss_num:
                return None

            for img_combination in it.combinations(img_candidates, r=self.multiview_num):
                for ss_combination in it.combinations(ss_candidates, r=self.ss_num):
                    recordness_homo_sub = list()
                    for img_id in img_combination:
                        for ss_id in ss_combination:
                            recordness_homo_sub.append(recordness_homo[img_id, ss_id])
                    recordness_homo_sub = np.array(recordness_homo_sub, np.bool8)
                    if np.all(recordness_homo_sub == True):
                        one_multiview_list = list()
                        for view_id in img_combination:
                            cam_pos_filename = os.path.join(input_dir, 'novel_view_curpos_{}.npy'.format(view_id))
                            assert os.path.exists(cam_pos_filename)
                            cam_rot_filename = os.path.join(input_dir, 'novel_view_currot_{}.npy'.format(view_id))
                            assert os.path.exists(cam_rot_filename)
                            depth_filename = os.path.join(input_dir, 'novel_view_depth{}.npy'.format(view_id))
                            assert os.path.exists(depth_filename)
                            rgb_filename = os.path.join(input_dir, 'novel_view_rgb{}.png'.format(view_id))
                            assert os.path.exists(rgb_filename)
                            ss_coords = list()

                            if category_name == 'homo':
                                micarray_RIR_filename = os.path.join(input_dir,
                                                                     'novel_view_micarray_RIRmono_{}.npy'.format(
                                                                         view_id))
                            else:
                                micarray_RIR_filename = os.path.join(input_dir,
                                                                     'novel_view_micarray_RIRdisc_{}.npy'.format(
                                                                         view_id))

                            # due to historical reasons, the RIR-file may not exist
                            if not os.path.exists(micarray_RIR_filename):
                                break

                            for ss_id in ss_combination:
                                world_coord = ss_pos_homo[ss_id]

                                cam_coord = self.data_processor.world2camera_transform_rotasarray(world_coord,
                                                                                       [np.load(cam_pos_filename),
                                                                                        np.load(cam_rot_filename)])
                                img_coord = self.data_processor.get_2D_pos_from_absolute_3D_rotasarray(world_coord,
                                                                                            [np.load(cam_pos_filename),
                                                                                             np.load(cam_rot_filename)])

                                #to get the RIR position, we need the recordness file
                                if category_name == 'homo':
                                    recordness_basename = 'novel_view_mono_recordness_{}.npy'.format(view_id)
                                else:
                                    recordness_basename = 'novel_view_disc_recordness_{}.npy'.format(view_id)

                                recordness_filename = os.path.join(input_dir, recordness_basename)
                                assert os.path.exists(recordness_filename)
                                recordness = np.load(recordness_filename)
                                RIR_pos_id = np.sum(recordness[0:ss_id+1])-1

                                # print('cam_coord shape = {}'.format(cam_coord.shape))
                                ss_coords.append({'wolrd_coord': world_coord,
                                                  'cam_coord': cam_coord,
                                                  'img_coord': img_coord,
                                                  'micarray_RIR': {'RIR_pos_id': RIR_pos_id,
                                                                   'micarray_RIR_filename': micarray_RIR_filename}})

                            if not len(ss_coords) == self.ss_num:
                                continue
                            oneview_dict = dict()
                            oneview_dict['rgb_filename'] = rgb_filename
                            oneview_dict['depth_filename'] = depth_filename
                            oneview_dict['cam_pos'] = np.load(cam_pos_filename)
                            oneview_dict['cam_rot'] = np.load(cam_rot_filename)
                            oneview_dict['ss_coords'] = ss_coords

                            one_multiview_list.append(oneview_dict)

                        #print('one_multiview_list len = {}'.format(len(one_multiview_list)))
                        if len(one_multiview_list) == self.multiview_num:
                            # print('one_multiview_list len = {}'.format(len(one_multiview_list)))
                            self.multiview_list.append(one_multiview_list)

                        if len(self.multiview_list) % 100 == 0:
                            print('cached {} multiviews!'.format(len(self.multiview_list)))

                        if len(self.multiview_list) >  100:
                            break

        return None

    def parse_one_multiview_oness(self, input_dir):
        #step1: decide if homo RIR exists
        rgb_img_list = list()
        for filename in os.listdir(input_dir):
            if 'novel_view_rgb' in filename:
                rgb_img_list.append(filename)

        multiview_list = list()

        for rgb_imgname in rgb_img_list:
            one_view_dict = dict()
            one_view_dict['rgb_img_filename'] = os.path.join(input_dir, rgb_imgname)
            RIR_mono_filename = rgb_imgname.replace('novel_view_rgb', 'novel_view_micarray_RIRmono_').replace('.png','.npy')
            # assert os.path.exists(os.path.join(input_dir, RIR_mono_filename))
            # RIR_disc_filename = rgb_imgname.replace('novel_view_rgb', 'novel_view_micarray_RIRdisc_').replace('.png','.npy')

            # assert os.path.exists(os.path.join(input_dir, RIR_disc_filename))
            one_view_dict['RIR_mono_filename'] = os.path.join(input_dir, RIR_mono_filename )
            # one_view_dict['RIR_disc_filename'] = os.path.join(input_dir, RIR_disc_filename )

            ss_pos_mono_filename = 'ss_pos_mono.npy'
            if not os.path.exists(os.path.join(input_dir, ss_pos_mono_filename)):
                return None
            assert os.path.exists(os.path.join(input_dir, ss_pos_mono_filename))
            one_view_dict['ss_pos_mono_filename'] = os.path.join(input_dir, ss_pos_mono_filename)

            # ss_pos_disc_filename = 'ss_pos_disc.npy'
            # assert os.path.exists(os.path.join(input_dir, ss_pos_disc_filename))
            # one_view_dict['ss_pos_disc_filename'] = os.path.join(input_dir, ss_pos_disc_filename)

            cam_pos_filename = rgb_imgname.replace('novel_view_rgb', 'novel_view_curpos_').replace('.png','.npy')
            assert os.path.exists(os.path.join(input_dir, cam_pos_filename))
            one_view_dict['cam_pos_filename'] = os.path.join(input_dir, cam_pos_filename)

            cam_rot_filename = rgb_imgname.replace('novel_view_rgb', 'novel_view_currot_').replace('.png','.npy')
            assert os.path.exists(os.path.join(input_dir, cam_rot_filename))
            one_view_dict['cam_rot_filename'] = os.path.join(input_dir, cam_rot_filename)

            multiview_list.append(one_view_dict)

        return multiview_list

    def get_seedsound_filename_list(self):
        seed_sound_list = list()
        if self.all_seed_sound_telephone:
            for i in range(10):
                seed_sound_list.append('telephone.wav')
        else:
            seed_sound_list.append('telephone.wav')
            seed_sound_list.append('siren.wav')
            seed_sound_list.append('horn_beeps.wav')
            seed_sound_list.append('computer_beeps.wav')
            seed_sound_list.append('c_fan.wav')
            seed_sound_list.append('water_waves_1.wav')
            seed_sound_list.append('fireplace.wav')
            seed_sound_list.append('big_door.wav')
            seed_sound_list.append('alarm.wav')
            seed_sound_list.append('person_1.wav')

        self.seed_sound_filename_list = seed_sound_list

    def convolve_binauralRIR_get_audio(self, current_source_sound, binaural_rir):
        '''
        Given the precomputed RIR, convolve them with the input seed sound to get the audio heard at this particular position
        :param current_source_sound: 1D seed sound
        :param input_RIRs: a list of two, left-ear and right-ear RIR
        :return: two audios, left and right
        '''
        binaural_rir = np.transpose(binaural_rir)
        binaural_convolved = np.array([fftconvolve(current_source_sound, binaural_rir[:, channel]
                                                   ) for channel in range(binaural_rir.shape[-1])])
        audiogoal = binaural_convolved[:, :self.sampling_rate]

        return audiogoal

    def convolve_monoRIR_get_audio(self, current_source_sound, mono_rir):
        mono_rir = np.squeeze(mono_rir)
        mono_audio_convolved = np.array(fftconvolve(current_source_sound, mono_rir))

        mono_audio_convolved = mono_audio_convolved[0:self.sampling_rate]

        return mono_audio_convolved

    def get_seed_sound_list(self):
        seed_sound_list = list()
        for i in range(self.ss_num):
            seed_sound_filename = os.path.join(self.seed_sound_dir, self.seed_sound_filename_list[i])
            assert os.path.exists(seed_sound_filename)

            seed_sound, sr = librosa.load(seed_sound_filename, sr=self.sampling_rate)
            seed_sound_list.append(seed_sound)

        self.seed_sound_list = seed_sound_list

    def compute_valid_length(self):
        return 48
        # if self.mode == 'train':
        #     return len(self.train_multiview_list)
        # else:
        #     return len(self.test_multiview_list)

    def parse_one_multiview_data(self, input_multiview_data):
        rgb_feat_list = list()
        cam_pos_list = list()
        cam_rot_list = list()
        logmel_gccphat_feat_list = list()
        ss_pos_camcoord_list = list()
        ss_grid_pos_list = list()
        for one_view_dict in input_multiview_data:
            rgb_filename = one_view_dict['rgb_filename']
            rgb_feat = self.prepare_an_img(rgb_filename)
            depth_filename = one_view_dict['depth_filename']
            cam_pos = one_view_dict['cam_pos']
            cam_rot = one_view_dict['cam_rot']

            logmel_gccphat_feats = list()
            ss_pos_camcoords = list()

            #add received sound across all sound sources
            micarray_received_all_sound = list()
            for ss_id in range(len(one_view_dict['ss_coords'])):
                micarray_RIR_filename = one_view_dict['ss_coords'][ss_id]['micarray_RIR']['micarray_RIR_filename']
                RIR_pos_id = one_view_dict['ss_coords'][ss_id]['micarray_RIR']['RIR_pos_id']
                micarray_RIR = np.load(micarray_RIR_filename)[RIR_pos_id, :, :]

                micarray_received_sound = list()

                for RIR_tmp in micarray_RIR:
                    #TODO: need to decide how to decide seed_sound for each sound source
                    micarray_received_sound.append(self.convolve_monoRIR_get_audio(self.seed_sound_list[ss_id],
                                                                                   RIR_tmp))

                micarray_received_all_sound.append(micarray_received_sound)

            micarray_received_all_sound = np.array(micarray_received_all_sound, np.float32) #[ss_num, micarray_num, d]
            micarray_received_all_sound = np.sum(micarray_received_all_sound, axis=0, keepdims=False )
            logmel_gccphat_feat = self.micarray_processor.get_micarray_feat(micarray_received_all_sound)

            # #get each ss info
            # for ss_id in range(len(one_view_dict['ss_coords'])):
            #     micarray_RIR_filename = one_view_dict['ss_coords'][ss_id]['micarray_RIR']['micarray_RIR_filename']
            #     RIR_pos_id = one_view_dict['ss_coords'][ss_id]['micarray_RIR']['RIR_pos_id']
            #     micarray_RIR = np.load(micarray_RIR_filename)[RIR_pos_id,:,:]
            #     logmel_gccphat_feat = self.get_micarray_audiofeat(micarray_RIR)
            #     logmel_gccphat_feats.append(logmel_gccphat_feat)
            #     ss_pos_camcoords.append(one_view_dict['ss_coords'][ss_id]['cam_coord'])

            # logmel_gccphat_feats = np.stack(logmel_gccphat_feats, axis=0)
            ss_grid_pos, pos_shift = self.data_processor.localize_ss_gridmap(ss_pos_camcoords, self.ss_grid_map)
            ss_pos_camcoords = np.stack(ss_pos_camcoords, axis=0)

            rgb_feat_list.append(rgb_feat)
            cam_pos_list.append(cam_pos)
            cam_rot_list.append(cam_rot)
            ss_grid_pos_list.append(ss_grid_pos)
            ss_pos_camcoord_list.append(ss_pos_camcoords)
            logmel_gccphat_feat_list.append(logmel_gccphat_feat)

        rgb_feat = torch.stack(rgb_feat_list, dim=0)
        cam_pos = np.array(cam_pos_list, np.float32)
        cam_rot = np.array(cam_rot_list, np.float32)
        ss_pos_camcoord = np.array(ss_pos_camcoord_list, np.float32)
        logmel_gccphat_feat = np.array(logmel_gccphat_feat_list, np.float32)

        #return order: RGB Image Feat, GCCPhat Feat, SS_Pos_camcoord, Cam_Pos, Cam_Rot
        return rgb_feat.to(torch.float32), \
               torch.from_numpy(logmel_gccphat_feat).to(torch.float32), \
               torch.from_numpy(ss_pos_camcoord).to(torch.float32), \
               torch.from_numpy(cam_pos).to(torch.float32), \
               torch.from_numpy(cam_rot).to(torch.float32)

    def parse_one_jsonfile(self, input_json_filename):
        assert os.path.exists(input_json_filename)
        with open(input_json_filename, 'r') as f:
            input_dict = json.load(f)

        return input_dict

    def get_all_json_file(self, input_dir):
        subdir_list = glob.glob(os.path.join(input_dir, '*'))
        json_file_list = list()

        for subdir_name in subdir_list:
            json_file_basename = os.path.basename(subdir_name) + '.json'
            assert os.path.exists(
                os.path.join(subdir_name, json_file_basename))
            json_file_list.append(
                os.path.join(subdir_name, json_file_basename))

        assert len(json_file_list) > 0

        return json_file_list

    def get_img_pretrained_feat(self, input_img_list):
        img_feat = np.zeros(shape=[len(input_img_list), 512],dtype=np.float32)
        for row_idx, img_name in enumerate(input_img_list):
            feat_name = img_name.replace('.png', '.npy')
            assert os.path.exists(feat_name)
            feat_tmp = np.load(feat_name)
            feat_tmp = feat_tmp.astype(np.float32)
            img_feat[row_idx, :] = feat_tmp

        return img_feat

    def normalize_an_img(self, input_img):
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

    def prepare_an_img(self, img_name):
        assert os.path.exists(img_name)
        img = cv2.imread(img_name, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img)
        img = self.normalize_an_img(img)

        return img

    def __len__(self):
        return self.compute_valid_length()

    def __getitem__(self, index):
        if self.mode == 'train':
            rgb_feat, logmel_gccphat, ss_pos_camcoord, cam_pos, cam_rot = self.parse_one_multiview_data(self.train_multiview_list[index])
        else:
            rgb_feat, logmel_gccphat, ss_pos_camcoord, cam_pos, cam_rot = self.parse_one_multiview_data(self.test_multiview_list[index])

        return rgb_feat, logmel_gccphat, ss_pos_camcoord, cam_pos, cam_rot

class Sound3DVDetDataset_DiverseSS(torch.utils.data.Dataset):
    """
    In our experiment, we choose to fix multiview-num by very the sound source number (ss_num), it better reflects real
    scenario, in which we can't resontrain how many sound sources are in the scene, but we can decide how many views to take
    """
    def __init__(self,
                 root_dir='/mnt/nas/yuhang/3DSSDet_Data_efficient_multiss_large',
                 ss_num_list = [1,2,3,4,5,6,7,8,9,10],
                 multiview_num = 3,
                 seed_sound_dir = '/home/yuhang/pycharm/sound-spaces-2.0/data/sounds/1s_all',
                 sampling_rate = 20001,
                 use_pano_img = True,
                 all_seed_sound_telephone = True,
                 test_set_ratio = 0.30,
                 turn_angle = 30,
                 mode = 'train',
                 room_scene_name = None,
                 RIR_dim = 20001,
                 random_seed = 100,
                 cate_name = 'disc',
                 sample_num_each_datadir = 10,
                 max_ss_num = 10,
                 imgfeat_extract_type='ResNet'):
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.root_dir = root_dir
        self.ss_num_list = ss_num_list
        self.cate_name = cate_name
        self.multiview_num = multiview_num
        self.turn_angle = turn_angle
        self.turn_angle_radian = math.pi / (180. / self.turn_angle)
        self.seed_sound_dir = seed_sound_dir
        self.sampling_rate = sampling_rate
        self.use_pano_img = use_pano_img
        self.RIR_dim = RIR_dim
        self.max_ss_num = max_ss_num
        self.all_seed_sound_telephone = all_seed_sound_telephone
        self.test_set_ratio = test_set_ratio
        assert mode in ['train', 'test']
        self.mode = mode
        self.room_scene_name = room_scene_name
        self.sample_num_each_datadir = sample_num_each_datadir
        assert imgfeat_extract_type in ['LoFTR', 'ResNet']
        self.imgfeat_extract_type = imgfeat_extract_type
        self.get_seedsound_filename_list()
        self.get_seed_sound_list()
        self.data_processor = data_utils.DataProcessor()
        self.micarray_processor = MicArrayFeat.MicArrayFeat()
        self.prepare_data()
        self.wavespectrum_converter = WaveSpectrumConverter.WaveSpectrumConverter()
        self.ss_grid_map = self.data_processor.get_grid_center_coords()

    def prepare_data(self):
        self.multiview_list = list()
        self.train_multiview_list = list()
        self.test_multiview_list = list()
        root_dir = '/mnt/nas/yuhang/3DSSDet_Data_reorganised/'
        ss_classlabel_num = 5
        multiview_num = 5
        ss_num = 5

        sample_num_each_cate_min = 100
        sample_num_each_cate_max = 1000

        # physical_objs = ['wall', 'ceiling', 'curtain', 'bed', 'tv_monitor',
        #                  'chest_of_drawers', 'cushion', 'table', 'lighting',
        #                  'table', 'door', 'shower', 'sink', 'mirror', 'stool',
        #                  'toilet', 'cabinet', 'counter', 'sofa', 'furniture',
        #                  'chair', 'shelving', 'appliances', 'picture', 'objects',
        #                  'towel']

        physical_objs = ['wall', 'ceiling',
                         'table', 'door', 'cabinet',
                         'chair'] #removed: 'lighting', 'mirror', 'furniture', 'objects', 'picture', 'appliances', 'towel'
                                                # 'cushion', 'shelving', 'sofa'

        # sucess_objs = list()
        # for ss_num_tmp in range(1, ss_num+1):
        #     for obj_name in physical_objs:
        #         #load disc
        #         save_dir_name_disc = 'ssnum{}_viewnum{}_objname{}_catenamedisc'.format(ss_num_tmp,
        #                                                                                multiview_num,
        #                                                                                obj_name)
        #         pkl_filename = os.path.join(root_dir, save_dir_name_disc, 'sound3dvdet_data.pickle')
        #
        #         if not os.path.exists(pkl_filename):
        #             print('file does not exist: {}'.format(pkl_filename))
        #         else:
        #             with open(pkl_filename, 'rb') as handle:
        #                 data_list_tmp = pickle.load(handle)['data_list']
        #                 if len(data_list_tmp) < sample_num_each_cate_min:
        #                     print('length does not meet: disc: obj: {}, ss_num: {}, len: {}'.format(obj_name, ss_num_tmp,
        #                                                                                             len(data_list_tmp)))
        #                 else:
        #                     #self.multiview_list.extend(data_list_tmp[0:min(len(data_list_tmp), sample_num_each_cate_max)])
        #                     data_list_tmp = data_list_tmp[0:min(len(data_list_tmp), sample_num_each_cate_max)]
        #                     random.shuffle(data_list_tmp)
        #                     self.train_multiview_list.extend(data_list_tmp[0:int(0.8*len(data_list_tmp))])
        #                     self.test_multiview_list.extend(data_list_tmp[int(0.8 * len(data_list_tmp)):])
        #                     sucess_objs.append(obj_name+'disc'+str(ss_num_tmp))
        #
        #         #load homo
        #         save_dir_name_disc = 'ssnum{}_viewnum{}_objname{}_catenamehomo'.format(ss_num_tmp,
        #                                                                                multiview_num,
        #                                                                                obj_name)
        #
        #         pkl_filename = os.path.join(root_dir, save_dir_name_disc, 'sound3dvdet_data.pickle')
        #
        #         if not os.path.exists(pkl_filename):
        #             print('file does not exist: {}'.format(pkl_filename))
        #         else:
        #             with open(pkl_filename, 'rb') as handle:
        #                 data_list_tmp = pickle.load(handle)['data_list']
        #                 if len(data_list_tmp) < sample_num_each_cate_min:
        #                     print('length does not meet: homo: obj: {}, ss_num: {}, len: {}'.format(obj_name, ss_num_tmp,
        #                                                                                             len(data_list_tmp)))
        #                 else:
        #                     # self.multiview_list.extend(data_list_tmp[0:min(len(data_list_tmp), sample_num_each_cate_max)])
        #                     # sucess_objs.append(obj_name + 'homo' + str(ss_num_tmp))
        #                     data_list_tmp = data_list_tmp[0:min(len(data_list_tmp), sample_num_each_cate_max)]
        #                     random.shuffle(data_list_tmp)
        #                     self.train_multiview_list.extend(data_list_tmp[0:int(0.8*len(data_list_tmp))])
        #                     self.test_multiview_list.extend(data_list_tmp[int(0.8 * len(data_list_tmp)):])

        # print('find obj info: {}'.format(sucess_objs))
        # breakpoint()





        # with open('/mnt/nas/yuhang/3DSSDet_Data_reorganised/ssnum6_viewnum5_objnametoilet_catenamedisc/sound3dvdet_data.pickle','rb') as handle:
        #     data_tmp = pickle.load(handle)['data_list']
        #     self.multiview_list.extend(data_tmp)
        #
        # with open('/mnt/nas/yuhang/3DSSDet_Data_reorganised/ssnum2_viewnum5_objnamechest_of_drawers_catenamedisc/sound3dvdet_data.pickle','rb') as handle:
        #     data_tmp = pickle.load(handle)['data_list']
        #     self.multiview_list.extend(data_tmp)

        # random.shuffle(self.train_multiview_list)
        # random.shuffle(self.test_multiview_list)


        with open(os.path.join(root_dir, 'train_info_with_sslabel.pickle'), 'rb') as handle:
            self.train_multiview_list = pickle.load(handle)['data_list'][0:5000]

        with open(os.path.join(root_dir, 'test_info_with_sslabel.pickle'), 'rb') as handle:
            self.test_multiview_list = pickle.load(handle)['data_list'][0:1000]

        # assert len(self.train_multiview_list) > 0
        # assert len(self.test_multiview_list) > 0
        #
        # with open(os.path.join(root_dir, 'train_info.pickle'), 'wb') as handle:
        #     pickle.dump({'data_list': self.train_multiview_list}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(os.path.join(root_dir, 'test_info.pickle'), 'wb') as handle:
        #     pickle.dump({'data_list': self.test_multiview_list}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # self.train_multiview_list = self.multiview_list[0:int(len(self.multiview_list)*0.8)]
        # self.test_multiview_list = self.multiview_list[int(len(self.multiview_list)*0.8):]

    def prepare_data_bak(self):
        self.multiview_list = list()
        with open('/mnt/nas/yuhang/3DSSDet_Data_reorganised/ssnum6_viewnum5_objnametoilet_catenamedisc/sound3dvdet_data.pickle','rb') as handle:
            data_tmp = pickle.load(handle)['data_list']
            self.multiview_list.extend(data_tmp)

        with open('/mnt/nas/yuhang/3DSSDet_Data_reorganised/ssnum2_viewnum5_objnamechest_of_drawers_catenamedisc/sound3dvdet_data.pickle','rb') as handle:
            data_tmp = pickle.load(handle)['data_list']
            self.multiview_list.extend(data_tmp)

        self.train_multiview_list = self.multiview_list[0:int(len(self.multiview_list)*0.8)]
        self.test_multiview_list = self.multiview_list[int(len(self.multiview_list)*0.8):]

    def prepare_data_bak(self):
        self.multiview_list = list()

        physical_objs = ['wall', 'ceiling', 'curtain', 'bed', 'tv_monitor',
                         'chest_of_drawers', 'cushion', 'table', 'lighting',
                         'table', 'door', 'shower', 'sink', 'mirror', 'stool',
                         'toilet', 'cabinet', 'counter', 'sofa', 'furniture',
                         'chair', 'shelving', 'appliances', 'picture', 'objects',
                         'towel']

        ss_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        view_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sample_num_each_datadir_list = [3, 3, 3, 3, 10, 10, 10, 10, 10, 10]
        for physical_obj in physical_objs:
            datadir_list = self.get_datadir_list(phyobj_name=physical_obj)
            print('data dir num: {}'.format(len(datadir_list)))
            for ss_num_id, ss_num_requirement in enumerate(ss_num_list):
                for view_num_requirement in view_num_list:
                    sample_num_requirement_list = [sample_num_each_datadir_list[ss_num_id]]
                    for sample_num_requirement in sample_num_requirement_list:
                        for cate_name in ['homo', 'disc']:
                            ss_num_multiview_list_tmp = list()
                            print('Processing ssnum{}_viewnum{}_objname{}_catename{}'.format(ss_num_requirement,
                                                                                             view_num_requirement,
                                                                                             physical_obj,
                                                                                             cate_name))
                            for datadir_id, datadir_tmp in enumerate(datadir_list):
                                multiview_list_for_ssnum = self.parse_one_multiview_multiss( datadir_tmp,
                                                                                             category_name=cate_name,
                                                                                             ss_num_requirement=ss_num_requirement,
                                                                                             view_num_requirement=view_num_requirement,
                                                                                             sample_num_requirement=sample_num_requirement,)

                                ss_num_multiview_list_tmp.extend(multiview_list_for_ssnum) if len(multiview_list_for_ssnum) > 0 else None

                            save_dir = '/mnt/nas/yuhang/3DSSDet_Data_reorganised/'
                            save_subdir = 'ssnum{}_viewnum{}_objname{}_catename{}'.format(ss_num_requirement,
                                                                               view_num_requirement,
                                                                               physical_obj,
                                                                                          cate_name)
                            if len(ss_num_multiview_list_tmp) > 0:
                                random.shuffle(ss_num_multiview_list_tmp)
                                save_dir_tmp = os.path.join(save_dir, save_subdir)
                                os.makedirs(save_dir_tmp) if not os.path.exists(save_dir_tmp) else None
                                save_filename = os.path.join(save_dir_tmp, 'sound3dvdet_data.pickle')
                                save_dict = {'data_list': ss_num_multiview_list_tmp}
                                with open(save_filename, 'wb') as handle:
                                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # print('for ss_num: {}, get samples: {}'.format(ss_num_requirement,
                #                                                len(ss_num_multiview_list_tmp)))
                # self.multiview_list.extend(ss_num_multiview_list_tmp) if len(ss_num_multiview_list_tmp) > 0 else None

        breakpoint()
        assert len(self.multiview_list) > 0
        random.shuffle(self.multiview_list)
        print('Finally we have obtained multiview num = {}'.format(len(self.multiview_list)))
        self.train_multiview_list = self.multiview_list[0:int(len(self.multiview_list)*0.8)]
        self.test_multiview_list = self.multiview_list[int(len(self.multiview_list)*0.8):]

    def get_datadir_list(self, phyobj_name='wall'):
        datadir_list = list()
        # phyobj_name = 'wall'
        for scene_name in os.listdir(self.root_dir):
            phyobj_name_list = os.listdir(os.path.join(self.root_dir, scene_name))
            if phyobj_name not in phyobj_name_list:
                continue
            data_dirs = os.listdir(os.path.join(self.root_dir, scene_name, phyobj_name))
            if len(data_dirs) == 0:
                continue
            for data_dir_tmp in data_dirs:
                dir_tmp = os.path.join(self.root_dir, scene_name, phyobj_name, data_dir_tmp)
                datadir_list.append(dir_tmp)

        return datadir_list

    def get_micarray_audiofeat(self, micarray_RIR):
        '''
        Given a input microphone array RIR, we compute the audio feature
        :param micarray_RIR: [4, RIR_dim]
        :return: [H, W, 10] feature
        '''
        audio_list = list()
        for RIR_tmp in micarray_RIR:
            #TODO: may need to change to seed sound, if there are multiple sound sources in the environment
            audio_list.append(self.convolve_monoRIR_get_audio(self.seed_sound_list[0],
                                                              RIR_tmp))
        micarray_audio = np.stack(audio_list, axis=0)

        # logmel_gccphat_feat = self.data_processor.get_micarray_feat(micarray_audio)
        logmel_gccphat_feat = self.micarray_processor.get_micarray_feat(micarray_audio) #[10, 512, 512]

        return logmel_gccphat_feat

    def parse_one_multiview_multiss(self,
                                    input_dir,
                                    category_name,
                                    ss_num_requirement,
                                    view_num_requirement,
                                    sample_num_requirement):
        multiview_list = list()
        # assert category_name in ['homo', 'disc']
        #step1: decide if homo RIR exists
        if category_name == 'homo':
            homo_RIR_list = glob.glob( os.path.join(input_dir, 'novel_view_micarray_RIR{}_*.npy'.format('mono')))
        else:
            homo_RIR_list = glob.glob(os.path.join(input_dir, 'novel_view_micarray_RIR{}_*.npy'.format('disc')))

        if len(homo_RIR_list) > 0:
            recordness_list = list()
            rgb_filename_list = list()
            for RIR_homo_filename in homo_RIR_list:
                RIR_homo_basename = os.path.basename(RIR_homo_filename)
                if category_name == 'homo':
                    img_id = int(RIR_homo_basename.replace('novel_view_micarray_RIRmono_','').replace('.npy',''))
                else:
                    img_id = int(RIR_homo_basename.replace('novel_view_micarray_RIRdisc_', '').replace('.npy', ''))
                rgb_img_basename = 'novel_view_rgb{}.png'.format(img_id)
                rgb_img_filename = os.path.join( os.path.dirname(RIR_homo_filename),
                                                 rgb_img_basename )
                assert os.path.exists(rgb_img_filename)

                rgb_filename_list.append(rgb_img_filename)

                if category_name == 'homo':
                    recordness_basename = 'novel_view_mono_recordness_{}.npy'.format(img_id)
                else:
                    recordness_basename = 'novel_view_disc_recordness_{}.npy'.format(img_id)

                recordness_filename = os.path.join( os.path.dirname(RIR_homo_filename),
                                                    recordness_basename )
                assert os.path.exists(recordness_filename)
                recordness_list.append(np.load(recordness_filename))

            recordness_homo = np.stack(recordness_list, axis=0)

            if category_name == 'homo':
                ss_pos_homo_filename = os.path.join( input_dir, 'ss_pos_mono.npy' )
            else:
                ss_pos_homo_filename = os.path.join( input_dir, 'ss_pos_disc.npy' )

            assert os.path.exists(ss_pos_homo_filename)
            ss_pos_homo = np.load(ss_pos_homo_filename)

            #extract multiview info
            each_ss_viewed_time = np.sum(recordness_homo, axis=0, keepdims=False).astype(np.int32)
            each_img_views_ssnum = np.sum(recordness_homo, axis=1, keepdims=False).astype(np.int32)

            ss_candidates = np.where(each_ss_viewed_time >= view_num_requirement )[0].tolist()
            img_candidates = np.where(each_img_views_ssnum >= ss_num_requirement )[0].tolist()

            if len(ss_candidates) < view_num_requirement or len(img_candidates) < ss_num_requirement:
                return list()

            for img_combination in it.combinations(img_candidates, r=view_num_requirement):
                for ss_combination in it.combinations(ss_candidates, r=ss_num_requirement):
                    recordness_homo_sub = list()
                    for img_id in img_combination:
                        for ss_id in ss_combination:
                            recordness_homo_sub.append(recordness_homo[img_id, ss_id])
                    recordness_homo_sub = np.array(recordness_homo_sub, np.bool8)
                    if np.all(recordness_homo_sub == True):
                        one_multiview_list = list()
                        for view_id in img_combination:
                            cam_pos_filename = os.path.join(input_dir, 'novel_view_curpos_{}.npy'.format(view_id))
                            assert os.path.exists(cam_pos_filename)
                            cam_rot_filename = os.path.join(input_dir, 'novel_view_currot_{}.npy'.format(view_id))
                            assert os.path.exists(cam_rot_filename)
                            depth_filename = os.path.join(input_dir, 'novel_view_depth{}.npy'.format(view_id))
                            assert os.path.exists(depth_filename)
                            rgb_filename = os.path.join(input_dir, 'novel_view_rgb{}.png'.format(view_id))
                            assert os.path.exists(rgb_filename)
                            ss_coords = list()

                            if category_name == 'homo':
                                micarray_RIR_filename = os.path.join(input_dir,
                                                                     'novel_view_micarray_RIRmono_{}.npy'.format(
                                                                         view_id))
                            else:
                                micarray_RIR_filename = os.path.join(input_dir,
                                                                     'novel_view_micarray_RIRdisc_{}.npy'.format(
                                                                         view_id))

                            # due to historical reasons, the RIR-file may not exist
                            if not os.path.exists(micarray_RIR_filename):
                                break

                            for ss_id in ss_combination:
                                world_coord = ss_pos_homo[ss_id]

                                cam_coord = self.data_processor.world2camera_transform_rotasarray(world_coord,
                                                                                       [np.load(cam_pos_filename),
                                                                                        np.load(cam_rot_filename)])
                                img_coord = self.data_processor.get_2D_pos_from_absolute_3D_rotasarray(world_coord,
                                                                                            [np.load(cam_pos_filename),
                                                                                             np.load(cam_rot_filename)])

                                #to get the RIR position, we need the recordness file
                                if category_name == 'homo':
                                    recordness_basename = 'novel_view_mono_recordness_{}.npy'.format(view_id)
                                else:
                                    recordness_basename = 'novel_view_disc_recordness_{}.npy'.format(view_id)

                                recordness_filename = os.path.join(input_dir, recordness_basename)
                                assert os.path.exists(recordness_filename)
                                recordness = np.load(recordness_filename)
                                RIR_pos_id = np.sum(recordness[0:ss_id+1])-1

                                # print('cam_coord shape = {}'.format(cam_coord.shape))
                                ss_coords.append({'wolrd_coord': world_coord,
                                                  'cam_coord': cam_coord,
                                                  'img_coord': img_coord,
                                                  'micarray_RIR': {'RIR_pos_id': RIR_pos_id,
                                                                   'micarray_RIR_filename': micarray_RIR_filename}})

                            if not len(ss_coords) == ss_num_requirement:
                                continue
                            oneview_dict = dict()
                            oneview_dict['rgb_filename'] = rgb_filename
                            oneview_dict['depth_filename'] = depth_filename
                            oneview_dict['cam_pos'] = np.load(cam_pos_filename)
                            oneview_dict['cam_rot'] = np.load(cam_rot_filename)
                            oneview_dict['ss_coords'] = ss_coords
                            oneview_dict['ss_num'] = ss_num_requirement

                            one_multiview_list.append(oneview_dict)

                        if len(one_multiview_list) == view_num_requirement:
                            multiview_list.append(one_multiview_list)
                            if len(multiview_list) >= sample_num_requirement:
                                return multiview_list
                            # if len(multiview_list) % 100 == 0:
                            #     print('cached {} multiviews!'.format(len(multiview_list)))

        return multiview_list

    def parse_one_multiview_oness(self, input_dir):
        #step1: decide if homo RIR exists
        rgb_img_list = list()
        for filename in os.listdir(input_dir):
            if 'novel_view_rgb' in filename:
                rgb_img_list.append(filename)

        multiview_list = list()

        for rgb_imgname in rgb_img_list:
            one_view_dict = dict()
            one_view_dict['rgb_img_filename'] = os.path.join(input_dir, rgb_imgname)
            RIR_mono_filename = rgb_imgname.replace('novel_view_rgb', 'novel_view_micarray_RIRmono_').replace('.png','.npy')
            # assert os.path.exists(os.path.join(input_dir, RIR_mono_filename))
            # RIR_disc_filename = rgb_imgname.replace('novel_view_rgb', 'novel_view_micarray_RIRdisc_').replace('.png','.npy')

            # assert os.path.exists(os.path.join(input_dir, RIR_disc_filename))
            one_view_dict['RIR_mono_filename'] = os.path.join(input_dir, RIR_mono_filename )
            # one_view_dict['RIR_disc_filename'] = os.path.join(input_dir, RIR_disc_filename )

            ss_pos_mono_filename = 'ss_pos_mono.npy'
            if not os.path.exists(os.path.join(input_dir, ss_pos_mono_filename)):
                return None
            assert os.path.exists(os.path.join(input_dir, ss_pos_mono_filename))
            one_view_dict['ss_pos_mono_filename'] = os.path.join(input_dir, ss_pos_mono_filename)

            # ss_pos_disc_filename = 'ss_pos_disc.npy'
            # assert os.path.exists(os.path.join(input_dir, ss_pos_disc_filename))
            # one_view_dict['ss_pos_disc_filename'] = os.path.join(input_dir, ss_pos_disc_filename)

            cam_pos_filename = rgb_imgname.replace('novel_view_rgb', 'novel_view_curpos_').replace('.png','.npy')
            assert os.path.exists(os.path.join(input_dir, cam_pos_filename))
            one_view_dict['cam_pos_filename'] = os.path.join(input_dir, cam_pos_filename)

            cam_rot_filename = rgb_imgname.replace('novel_view_rgb', 'novel_view_currot_').replace('.png','.npy')
            assert os.path.exists(os.path.join(input_dir, cam_rot_filename))
            one_view_dict['cam_rot_filename'] = os.path.join(input_dir, cam_rot_filename)

            multiview_list.append(one_view_dict)

        return multiview_list

    def get_seedsound_filename_list(self):
        seed_sound_list = list()
        if self.all_seed_sound_telephone:
            for i in range(10):
                seed_sound_list.append('telephone.wav')
        else:
            seed_sound_list.append('telephone.wav')
            seed_sound_list.append('siren.wav')
            seed_sound_list.append('alarm.wav')
            seed_sound_list.append('fireplace.wav')
            seed_sound_list.append('horn_beeps.wav')
            seed_sound_list.append('c_fan.wav')
            seed_sound_list.append('computer_beeps.wav')
            seed_sound_list.append('c_fan.wav')
            seed_sound_list.append('water_waves_1.wav')
            seed_sound_list.append('fireplace.wav')
            seed_sound_list.append('big_door.wav')
            seed_sound_list.append('alarm.wav')
            seed_sound_list.append('person_1.wav')

        self.seed_sound_filename_list = seed_sound_list

    def convolve_binauralRIR_get_audio(self, current_source_sound, binaural_rir):
        '''
        Given the precomputed RIR, convolve them with the input seed sound to get the audio heard at this particular position
        :param current_source_sound: 1D seed sound
        :param input_RIRs: a list of two, left-ear and right-ear RIR
        :return: two audios, left and right
        '''
        binaural_rir = np.transpose(binaural_rir)
        binaural_convolved = np.array([fftconvolve(current_source_sound, binaural_rir[:, channel]
                                                   ) for channel in range(binaural_rir.shape[-1])])
        audiogoal = binaural_convolved[:, :self.sampling_rate]

        return audiogoal

    def convolve_monoRIR_get_audio(self, current_source_sound, mono_rir):
        mono_rir = np.squeeze(mono_rir)
        mono_audio_convolved = np.array(fftconvolve(current_source_sound, mono_rir))

        mono_audio_convolved = mono_audio_convolved[0:self.sampling_rate]

        return mono_audio_convolved

    def get_seed_sound_list(self):
        seed_sound_list = list()
        for i in range(max(self.ss_num_list)):
            seed_sound_filename = os.path.join(self.seed_sound_dir, self.seed_sound_filename_list[i])
            assert os.path.exists(seed_sound_filename)

            seed_sound, sr = librosa.load(seed_sound_filename, sr=self.sampling_rate)
            seed_sound_list.append(seed_sound)

        self.seed_sound_list = seed_sound_list

    def compute_valid_length(self):
        if self.mode == 'train':
            return len(self.train_multiview_list)
        else:
            return len(self.test_multiview_list)

    def parse_one_multiview_data(self, input_multiview_data):
        rgb_feat_list = list()
        cam_pos_list = list()
        cam_rot_list = list()
        logmel_gccphat_feat_list = list()
        ss_pos_camcoord_list = list()
        # ss_grid_pos_list = list()
        ss_labels_list = list()
        for one_view_dict in input_multiview_data:
            rgb_filename = one_view_dict['rgb_filename']
            rgb_feat = self.prepare_an_img(rgb_filename)
            depth_filename = one_view_dict['depth_filename']
            cam_pos = one_view_dict['cam_pos']
            cam_rot = one_view_dict['cam_rot']

            logmel_gccphat_feats = list()
            ss_pos_camcoords = list()
            ss_labels = list()

            ss_class_labels = one_view_dict['ss_class_labels']

            #add received sound across all sound sources
            micarray_received_all_sound = list()
            for ss_id in range(len(one_view_dict['ss_coords'])):
                micarray_RIR_filename = one_view_dict['ss_coords'][ss_id]['micarray_RIR']['micarray_RIR_filename']
                RIR_pos_id = one_view_dict['ss_coords'][ss_id]['micarray_RIR']['RIR_pos_id']
                micarray_RIR = np.load(micarray_RIR_filename)[RIR_pos_id, :, :]

                ss_class_label = ss_class_labels[ss_id]

                micarray_received_sound = list()

                for RIR_tmp in micarray_RIR:
                    #TODO: need to decide how to decide seed_sound for each sound source
                    # micarray_received_sound.append(self.convolve_monoRIR_get_audio(self.seed_sound_list[ss_id],
                    #                                                                RIR_tmp))
                    micarray_received_sound.append(self.convolve_monoRIR_get_audio(self.seed_sound_list[ss_class_label],
                                                                                   RIR_tmp))

                micarray_received_all_sound.append(micarray_received_sound)

                #cam_coord_tmp = torch.from_numpy(one_view_dict['ss_coords'][ss_id]['cam_coord']).to(torch.float32)
                #ss_pos_camcoords.append({'gt_ss_3Dpos':cam_coord_tmp})
                ss_pos_camcoords.append(one_view_dict['ss_coords'][ss_id]['cam_coord'])

                #TODO: need to decide each ss class id
                #ss_labels.append({'labels':ss_id})
                ss_labels.append(ss_class_label)

            #pad ss camcoord and ss label
            for pad_id in range(self.max_ss_num - len(one_view_dict['ss_coords'])):
                ss_pos_camcoords.append(np.array([0.,0.,0.],np.float32))
                ss_labels.append(-1)
            micarray_received_all_sound = np.array(micarray_received_all_sound, np.float32) #[ss_num, micarray_num, d]
            micarray_received_all_sound = np.sum(micarray_received_all_sound, axis=0, keepdims=False )
            logmel_gccphat_feat = self.micarray_processor.get_micarray_feat(micarray_received_all_sound)

            # #get each ss info
            # for ss_id in range(len(one_view_dict['ss_coords'])):
            #     micarray_RIR_filename = one_view_dict['ss_coords'][ss_id]['micarray_RIR']['micarray_RIR_filename']
            #     RIR_pos_id = one_view_dict['ss_coords'][ss_id]['micarray_RIR']['RIR_pos_id']
            #     micarray_RIR = np.load(micarray_RIR_filename)[RIR_pos_id,:,:]
            #     logmel_gccphat_feat = self.get_micarray_audiofeat(micarray_RIR)
            #     logmel_gccphat_feats.append(logmel_gccphat_feat)
            #     ss_pos_camcoords.append(one_view_dict['ss_coords'][ss_id]['cam_coord'])

            # logmel_gccphat_feats = np.stack(logmel_gccphat_feats, axis=0)
            #ss_grid_pos, pos_shift = self.data_processor.localize_ss_gridmap(ss_pos_camcoords, self.ss_grid_map)
            ss_pos_camcoords = np.stack(ss_pos_camcoords, axis=0)
            ss_labels = np.stack(ss_labels, axis=0)

            rgb_feat_list.append(rgb_feat)
            cam_pos_list.append(cam_pos)
            cam_rot_list.append(cam_rot)
            #ss_grid_pos_list.append(ss_grid_pos)
            ss_pos_camcoord_list.append(ss_pos_camcoords)
            ss_labels_list.append(ss_labels)
            logmel_gccphat_feat_list.append(logmel_gccphat_feat)

        rgb_feat = torch.stack(rgb_feat_list, dim=0)
        cam_pos = np.array(cam_pos_list, np.float32)
        cam_rot = np.array(cam_rot_list, np.float32)
        ss_pos_camcoord = np.array(ss_pos_camcoord_list, np.float32)
        ss_label = np.array(ss_labels_list, np.int32)
        logmel_gccphat_feat = np.array(logmel_gccphat_feat_list, np.float32)

        #return order: RGB Image Feat, GCCPhat Feat, SS_Pos_camcoord, Cam_Pos, Cam_Rot
        # return rgb_feat.to(torch.float32), \
        #        torch.from_numpy(logmel_gccphat_feat).to(torch.float32), \
        #        torch.from_numpy(ss_pos_camcoord).to(torch.float32), \
        #        torch.from_numpy(cam_pos).to(torch.float32), \
        #        torch.from_numpy(cam_rot).to(torch.float32)

        return rgb_feat.to(torch.float32), \
               torch.from_numpy(logmel_gccphat_feat).to(torch.float32), \
               torch.from_numpy(ss_pos_camcoord).to(torch.float32), \
               torch.from_numpy(ss_label).to(torch.long), \
               torch.from_numpy(cam_pos).to(torch.float32), \
               torch.from_numpy(cam_rot).to(torch.float32)

    def parse_one_jsonfile(self, input_json_filename):
        assert os.path.exists(input_json_filename)
        with open(input_json_filename, 'r') as f:
            input_dict = json.load(f)

        return input_dict

    def get_all_json_file(self, input_dir):
        subdir_list = glob.glob(os.path.join(input_dir, '*'))
        json_file_list = list()

        for subdir_name in subdir_list:
            json_file_basename = os.path.basename(subdir_name) + '.json'
            assert os.path.exists(
                os.path.join(subdir_name, json_file_basename))
            json_file_list.append(
                os.path.join(subdir_name, json_file_basename))

        assert len(json_file_list) > 0

        return json_file_list

    def get_img_pretrained_feat(self, input_img_list):
        img_feat = np.zeros(shape=[len(input_img_list), 512],dtype=np.float32)
        for row_idx, img_name in enumerate(input_img_list):
            feat_name = img_name.replace('.png', '.npy')
            assert os.path.exists(feat_name)
            feat_tmp = np.load(feat_name)
            feat_tmp = feat_tmp.astype(np.float32)
            img_feat[row_idx, :] = feat_tmp

        return img_feat

    def normalize_an_img(self, input_img):
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

    def prepare_an_img(self, img_name):
        assert os.path.exists(img_name)
        if self.imgfeat_extract_type == 'ResNet':
            img = cv2.imread(img_name, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = cv2.resize(img, (224, 224))
            img = torch.from_numpy(img)
            img = self.normalize_an_img(img)
        else:
            #LoFTR uses a simple way to process RGB image
            img_raw = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            img = torch.from_numpy(img_raw)[None] / 255.

        return img

    def __len__(self):
        return self.compute_valid_length()

    def __getitem__(self, index):
        if self.mode == 'train':
            rgb_feat, logmel_gccphat, ss_pos_camcoord, ss_label, cam_pos, cam_rot = self.parse_one_multiview_data(self.train_multiview_list[index])
        else:
            rgb_feat, logmel_gccphat, ss_pos_camcoord, ss_label, cam_pos, cam_rot = self.parse_one_multiview_data(self.test_multiview_list[index])

        return rgb_feat, logmel_gccphat, ss_pos_camcoord, ss_label, cam_pos, cam_rot


def get_config(yaml_config_filename):
    import yaml
    with open(yaml_config_filename) as f:
        config_dict = yaml.safe_load(f)

    return config_dict

def main():
    seq_dataset = Sound3DVDetDataset_DiverseSS()

    data_loader = torch.utils.data.DataLoader(seq_dataset,
                                              batch_size=1,
                                              num_workers=0,
                                              shuffle=False,)

    for data_tmp in data_loader:
        rgb_feat = data_tmp[0]
        logmel_gccphat = data_tmp[1]
        ss_pos_camcoord_dict = data_tmp[2]
        ss_label_dict = data_tmp[3]
        cam_pos = data_tmp[4]
        cam_rot = data_tmp[5]
        breakpoint()

if __name__ == '__main__':
    main()