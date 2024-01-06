import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import CoordTransform
import ImgFeatAggregator
import MicArraySSDet


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        #self attention layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        #learnable K, V, Q Matrix
        self.k_linear = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.q_linear = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.v_linear = nn.Linear(in_features=self.d_model, out_features=self.d_model)

        # FFN layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = self.get_activation_fn(activation)
        self.normalize_before = normalize_before

    def get_activation_fn(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    def with_pos_embed(self, tensor, pos = None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, token_embed, key_padding_mask = None, attn_mask = None):
        query_embed = self.q_linear(token_embed)
        key_embed = self.k_linear(token_embed)
        val_embed = self.v_linear(token_embed)

        self_attn_output = self.self_attn(query = query_embed, key = key_embed, value=val_embed,
                              key_padding_mask = key_padding_mask,
                              attn_mask=attn_mask )[0]
        multihead_outpout = token_embed + self.dropout(self_attn_output)
        multihead_outpout = self.norm1(multihead_outpout)

        #FFN
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(multihead_outpout))))
        ffn_output = multihead_outpout + self.dropout2(ffn_output)
        ffn_output = self.norm2(ffn_output)

        return ffn_output

    def forward_pre(self, token_embed, key_padding_mask = None, attn_mask = None):
        token_embed = self.norm1(token_embed)
        query_embed = self.q_linear(token_embed)
        key_embed = self.k_linear(token_embed)
        val_embed = self.v_linear(token_embed)

        self_attn_output = self.self_attn(query = query_embed, key = key_embed, value=val_embed,
                              key_padding_mask = key_padding_mask,
                              attn_mask=attn_mask )[0]
        multihead_outpout = token_embed + self.dropout(self_attn_output)
        multihead_outpout = self.norm2(multihead_outpout)

        #FFN
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(multihead_outpout))))
        ffn_output = multihead_outpout + self.dropout(ffn_output)

        return ffn_output

    def forward(self, token_embed):
        if self.normalize_before:
            return self.forward_pre(token_embed)
        else:
            return self.forward_post(token_embed)


class Conv2DModule(nn.Module):
    def __init__(self, in_channels = 100, out_channels = 100, kernel_size = [3,3], stride = 1, padding = 0, bias = True):
        super(Conv2DModule, self).__init__()
        self.conv2d_module = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=stride,
                                                     padding=padding,
                                                     bias=bias),
                                           nn.BatchNorm2d(num_features=out_channels),
                                           nn.ReLU())

    def forward(self, input_feat):
        return self.conv2d_module(input_feat)

class SS3DDet_Encoder(nn.Module):
    def __init__(self, input_feat_dim = 10, grid_feat_dim = 512):
        super(SS3DDet_Encoder, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.grid_feat_dim = grid_feat_dim
        self.construct_encoder_nn()

    def construct_encoder_nn(self):
        self.encoder_nn = nn.Sequential(Conv2DModule(in_channels=self.input_feat_dim,
                                                     out_channels=128,
                                                     kernel_size=[3,3],
                                                     stride=[2,2],
                                                     padding=[1,1],
                                                     bias=True),
                                         Conv2DModule(in_channels=128,
                                                      out_channels=256,
                                                      kernel_size=[3,3],
                                                      stride=[2,2],
                                                      padding=[1,1],
                                                      bias=True),
                                        Conv2DModule(in_channels=256,
                                                     out_channels=self.grid_feat_dim,
                                                     kernel_size=[3,3],
                                                     stride=[1,1],
                                                     padding=[1,1],
                                                     bias=True))

    def forward(self, input_feat):
        return self.encoder_nn(input_feat)


class SS3DVDet_Head(nn.Module):
    def __init__(self, class_num = 5, ss_query_dim = 512):
        super(SS3DVDet_Head, self).__init__()
        self.class_num = class_num
        self.ss_query_dim = ss_query_dim

        #predict offset for each class separately
        self.ss_pos_regress_head = nn.Sequential(nn.Linear(in_features=self.ss_query_dim,
                                                           out_features=self.ss_query_dim//2),
                                                 nn.BatchNorm1d(num_features=self.ss_query_dim//2),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=self.ss_query_dim//2,
                                                           out_features=3),
                                                 nn.Sigmoid())

        self.classification_head = nn.Sequential(nn.Linear(in_features=self.ss_query_dim,
                                                           out_features=self.class_num+1),)

    def inverse_sigmoid(self, x, eps=1e-5):
        """Inverse function of sigmoid.
        """
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def forward(self, input_ss_query_feat_ori):
        '''
        given a set of input_ss_query_feat, we use Linear layer to joint predict its 3D position and
        :param input_ss_query_feat: [B, token_num, query_dim]
        :return: [B, token_num, 3] for ss 3D pos, [B, token_num, class_num + 1] for classification
        '''
        input_ss_query_feat = input_ss_query_feat_ori.clone()
        batch_size = input_ss_query_feat.shape[0]
        token_num = input_ss_query_feat.shape[1]
        query_feat_dim = input_ss_query_feat.shape[2]

        input_ss_query_feat = torch.reshape(input_ss_query_feat,
                                            shape=[batch_size*token_num, query_feat_dim])

        ss_query_3D_pos = self.ss_pos_regress_head(input_ss_query_feat)
        ss_query_3D_pos = torch.reshape(ss_query_3D_pos, shape=[batch_size, token_num, -1])
        ss_query_3D_pos = self.inverse_sigmoid(ss_query_3D_pos)

        classify_logits = self.classification_head(input_ss_query_feat)
        classify_logits = torch.reshape(classify_logits, shape=[batch_size, token_num, -1])

        return ss_query_3D_pos, classify_logits

class Sound3DVDet(nn.Module):
    def __init__(self, input_soundfeat_channel_num = 10,
                 ss_query_feat_dim = 512,
                 class_num = 10,
                 transformer_layernum = 6,
                 transformer_config = None,
                 ss_query_num = 16,
                 img_height = 512,
                 img_width = 512,
                 device = 'cpu:0',
                 multiview_num = 10,
                 feat_aggreg_method = 'add',
                 aggregate_img_feat = True,
                 aggregate_sound_feat = True,
                 rgbfeat_extractor_name = 'LoFTR',
                 rgbfeat_extract_layer_name = 'fine'):
        super(Sound3DVDet, self).__init__()
        self.input_soundfeat_channel_num = input_soundfeat_channel_num
        self.ss_query_feat_dim = ss_query_feat_dim
        self.class_num = class_num
        self.transformer_layernum = transformer_layernum
        self.transformer_config = transformer_config
        self.ss_query_num = ss_query_num

        self.img_height = img_height
        self.img_width = img_width

        self.device = device
        self.multiview_num = multiview_num

        self.aggregate_img_feat = aggregate_img_feat
        self.aggregate_sound_feat = aggregate_sound_feat

        assert feat_aggreg_method in ['mean', 'add']
        self.feat_aggreg_method = feat_aggreg_method

        self.rgbfeat_extractor_name = rgbfeat_extractor_name
        self.rgbfeat_extract_layer_name = rgbfeat_extract_layer_name

        self.construct_Transformer_NN()

        self.sound3dvdet_head = SS3DVDet_Head(class_num=self.class_num,
                                             ss_query_dim=self.ss_query_feat_dim)

        self.coord_transform = CoordTransform.CoordTransform(device = self.device)
        self.imgfeat_aggregator = ImgFeatAggregator.ImgFeatAggregator()
        self.ssquery_generator = MicArraySSDet.MicArraySSDetector()
        # self.rgb_imgfeat_extractor = RGBImgFeatExtractor.RGBImgFeatExtractor(extractor_name=self.rgbfeat_extractor_name,
        #                                                                      extract_layer=self.rgbfeat_extract_layer_name)

        self.get_feat_dim_match_layer()


    def get_feat_dim_match_layer(self):
        if self.rgbfeat_extract_layer_name == 'coarse' and self.rgbfeat_extractor_name == 'LoFTR':
            self.featdim_match_layer = nn.Linear(in_features=256, out_features=512)
        elif self.rgbfeat_extract_layer_name == 'fine' and self.rgbfeat_extractor_name == 'LoFTR':
            self.featdim_match_layer = nn.Linear(in_features=128, out_features=512)
        else:
            self.featdim_match_layer = nn.Identity()

    def feat_dim_match(self, input_2D_feat):
        if len(input_2D_feat.shape) == 4:
            batch_size, channel_num, height, width = input_2D_feat.shape
            input_2D_feat = torch.permute(input_2D_feat, dims=[0,2,3,1])
            input_2D_feat = torch.reshape(input_2D_feat, shape=[batch_size*height*width, channel_num])
            input_2D_feat = self.featdim_match_layer(input_2D_feat)
            input_2D_feat = torch.reshape(input_2D_feat, shape=[batch_size, height, width, -1])

            input_2D_feat = torch.permute(input_2D_feat, dims=[0,3,1,2])
        else:
            batch_size, view_num, channel_num, height, width = input_2D_feat.shape
            input_2D_feat = torch.permute(input_2D_feat, dims=[0,1,3,4,2])
            input_2D_feat = torch.reshape(input_2D_feat, shape=[batch_size*view_num*height*width, channel_num])
            input_2D_feat = self.featdim_match_layer(input_2D_feat)
            input_2D_feat = torch.reshape(input_2D_feat, shape=[batch_size, view_num, height, width, -1])

            input_2D_feat = torch.permute(input_2D_feat, dims=[0,1,4,2,3])

        return input_2D_feat


    def construct_Transformer_NN(self):
        self.transformer_layers = nn.ModuleList()

        for i in range(self.transformer_layernum):
            self.transformer_layers.append(TransformerEncoderLayer(d_model=self.transformer_config['d_model'],
                                                                   nhead=self.transformer_config['nhead'],
                                                                   dim_feedforward=self.transformer_config['dim_feedforward'],
                                                                   dropout=self.transformer_config['dropout'],
                                                                   normalize_before=self.transformer_config['normalize_before']))
            # self.transformer_layers.append(TransformerEncoderLayer(d_model=512,
            #                                                        nhead=8,
            #                                                        dim_feedforward=512,
            #                                                        dropout=0.1,
            #                                                        normalize_before=True))


    def aggregate_multiview_feat(self, ss_queries, ref_camera_pos, ref_camera_rot,
                                 multiview_camera_pos, multiview_camera_rot,
                                 ref_img_feat, multiview_img_feat,
                                 ref_sound_feat, multiview_sound_feat):
        ss_queries_3D_pos = self.sound3dvdet_head(ss_queries)[0]  # [N, token_num, 3]
        aggregated_img_feat_list = list()
        aggregated_sound_feat_list = list()

        #forward to img feat extractor
        ref_img_feat = self.feat_dim_match( ref_img_feat )
        multiview_img_feat = self.feat_dim_match( multiview_img_feat )
        # ref_img_feat = self.featdim_match_layer(ref_img_feat)
        # multiview_img_feat = self.featdim_match_layer(multiview_img_feat)
        # ref_img_feat = self.rgb_imgfeat_extractor(ref_img_feat)
        # multiview_img_feat = self.rgb_imgfeat_extractor(multiview_img_feat)

        batch_size = ss_queries.shape[0]
        for batch_id in range(batch_size):
            ss_imgplane_pos_list = list()
            img_feat_list = list()
            sound_feat_list = list()
            # step1: project query 3D pos to current image plane
            ref_ss3Dpos_cameracoord = ss_queries_3D_pos[batch_id, :, :]
            ref_camera_pos_tmp = ref_camera_pos[batch_id, :]
            ref_camera_rot_tmp = ref_camera_rot[batch_id, :]
            current_imgplane_pos, is_inside = self.coord_transform.project_cameraCoord_point_to_imgPlane(
                ref_ss3Dpos_cameracoord)

            ss_imgplane_pos_list.append(current_imgplane_pos)
            img_feat_list.append(ref_img_feat[batch_id, :, :, :])
            sound_feat_list.append(ref_sound_feat[batch_id, :, :, :])
            # step2: project query 3D pos to neighboring image plane
            for view_id in range(self.multiview_num - 1):
                target_camera_pos = multiview_camera_pos[batch_id, view_id, :]
                target_camera_rot = multiview_camera_rot[batch_id, view_id, :]
                target_ss3Dpos_cameracoord = self.coord_transform.transform_cameraA_to_cameraB(ref_ss3Dpos_cameracoord,
                                                                                               ref_camera_pos_tmp,
                                                                                               ref_camera_rot_tmp,
                                                                                               target_camera_pos,
                                                                                               target_camera_rot)

                target_imgplane_pos, is_inside = self.coord_transform.project_cameraCoord_point_to_imgPlane(
                    target_ss3Dpos_cameracoord)

                ss_imgplane_pos_list.append(target_imgplane_pos)
                img_feat_list.append(multiview_img_feat[batch_id, view_id, :, :, :])
                sound_feat_list.append(multiview_sound_feat[batch_id, view_id, :, :, :])

            ss_imgplane_pos = torch.stack(ss_imgplane_pos_list, dim=0)
            ss_imgplane_pos = torch.reshape(ss_imgplane_pos,
                                            shape=[self.multiview_num,
                                                   np.sqrt(self.ss_query_num).astype(np.int32),
                                                   np.sqrt(self.ss_query_num).astype(np.int32),
                                                   2])

            img_feat = torch.stack(img_feat_list, dim=0)
            aggregated_img_feat = self.imgfeat_aggregator(img_feat, ss_imgplane_pos)
            aggregated_img_feat = torch.reshape(aggregated_img_feat, shape=[self.multiview_num, self.ss_query_num, -1])

            if self.feat_aggreg_method == 'add':
                aggregated_img_feat = torch.sum(aggregated_img_feat, dim=0, keepdim=False)
            elif self.feat_aggreg_method == 'mean':
                aggregated_img_feat = torch.mean(aggregated_img_feat, dim=0, keepdim=False)
            else:
                raise ValueError('Unknown Img Feature Aggregation Method!')

            aggregated_img_feat_list.append(aggregated_img_feat)

            #aggregate sound feat
            sound_feat = torch.stack(sound_feat_list, dim=0)
            aggregated_sound_feat = self.imgfeat_aggregator(sound_feat, ss_imgplane_pos)
            aggregated_sound_feat = torch.reshape(aggregated_sound_feat, shape=[self.multiview_num, self.ss_query_num, -1])

            if self.feat_aggreg_method == 'add':
                aggregated_sound_feat = torch.sum(aggregated_sound_feat, dim=0, keepdim=False)
            elif self.feat_aggreg_method == 'mean':
                aggregated_sound_feat = torch.mean(aggregated_sound_feat, dim=0, keepdim=False)
            else:
                raise ValueError('Unknown Img Feature Aggregation Method!')

            aggregated_sound_feat_list.append(aggregated_sound_feat)

        aggregated_img_feat = torch.stack(aggregated_img_feat_list, dim=0)
        aggregated_sound_feat = torch.stack(aggregated_sound_feat_list, dim=0)

        return aggregated_img_feat, aggregated_sound_feat


    # def forward(self,ref_camera_pos, ref_camera_rot, multiview_camera_pos, multiview_camera_rot,
    #             ref_img_feat, ref_sound_feat,
    #             multiview_img_feat, multiview_sound_feat):
    def forward(self, ref_camera_pos, ref_camera_rot, multiview_camera_pos, multiview_camera_rot,
                ref_img_feat, ref_micarray_input,
                multiview_img_feat, multiview_micarray_input):
        '''
        Given the input multiview image feature, and microphone array feature, we sequentially predict a bunch of sound
        source queries, which is further fed to sound source detection head for joint localising and classification
        :param ss_queries:
        :param ref_camera_pos:
        :param ref_camera_rot:
        :param multiview_camera_pos:
        :param multiview_camera_rot:
        :param ref_img_feat:
        :param ref_sound_feat:
        :param multiview_img_feat:
        :param multiview_sound_feat:
        :return:
        '''
        output_dict = dict()
        #step0: generate sound source queries
        ss_queries, ref_sound_feat = self.ssquery_generator(ref_micarray_input)
        output_dict['refview_output'] = dict()
        ref_ss3Dpos, ref_ssclasslogits = self.sound3dvdet_head(ss_queries)
        output_dict['refview_output']['pred_ss3Dpos'] = ref_ss3Dpos
        output_dict['refview_output']['pred_ssclasslogits'] = ref_ssclasslogits

        #step1: generate multiview_micarray_feat
        output_dict['multiview_output'] = dict()
        multiview_sound_feat_list = list()
        for view_id in range(self.multiview_num-1):
            ss_queries_view, novelview_sound_feat = self.ssquery_generator(multiview_micarray_input[:, view_id, :,:,:])
            multiview_sound_feat_list.append(novelview_sound_feat)
            view_ss3Dpos, view_ssclasslogits = self.sound3dvdet_head(ss_queries_view)
            output_dict['multiview_output']['view_{}'.format(view_id)] = dict()
            output_dict['multiview_output']['view_{}'.format(view_id)]['pred_ss3Dpos'] = view_ss3Dpos
            output_dict['multiview_output']['view_{}'.format(view_id)]['pred_ssclasslogits'] = view_ssclasslogits

        multiview_sound_feat = torch.stack(multiview_sound_feat_list, dim=1)

        output_dict['transformer_interm_output'] = dict()
        #step2: feed sound source queries to Transformer nn
        for layer_id, transformer_layer in enumerate(self.transformer_layers):
            ss_queries = transformer_layer(ss_queries) #[N, token_num, query_dim]
            #update ss_queries

            aggregated_img_feat, aggregated_sound_feat = self.aggregate_multiview_feat(ss_queries, ref_camera_pos, ref_camera_rot,
                                                                                       multiview_camera_pos, multiview_camera_rot,
                                                                                       ref_img_feat, multiview_img_feat,
                                                                                       ref_sound_feat, multiview_sound_feat)

            if self.aggregate_img_feat:
                ss_queries += aggregated_img_feat
            if self.aggregate_sound_feat:
                ss_queries += aggregated_sound_feat

            trans_ss3Dpos, trans_ssclasslogits = self.sound3dvdet_head(ss_queries)
            output_dict['transformer_interm_output']['layer_{}'.format(layer_id)] = dict()
            output_dict['transformer_interm_output']['layer_{}'.format(layer_id)]['pred_ss3Dpos'] = trans_ss3Dpos
            output_dict['transformer_interm_output']['layer_{}'.format(layer_id)]['pred_ssclasslogits'] = trans_ssclasslogits

        #step3: feed sound source queries to sound3dvdet head for position regression and category classification
        ss_3D_pos, ss_class_logits = self.sound3dvdet_head(ss_queries)


        output_dict['sound3dvdet_output'] = dict()
        output_dict['sound3dvdet_output']['pred_ss3Dpos'] = ss_3D_pos
        output_dict['sound3dvdet_output']['pred_ssclasslogits'] = ss_class_logits

        return output_dict


if __name__ == '__main__':
    import os
    import DataProvider_bak

    seq_dataset = DataProvider.Sound3DVDetDataset()

    data_loader = torch.utils.data.DataLoader(seq_dataset,
                                              batch_size=10,
                                              shuffle=False,)
    os.putenv('CUDA_VISIBLE_DEVICES','0')
    device = torch.device('cuda:0')
    sound3ddet = Sound3DVDet(device=device, multiview_num=5)
    sound3ddet = sound3ddet.train().to(device)
    input_feat = torch.ones(size=[2,10, 512, 512], dtype=torch.float32).to(device)

    for data_tmp in data_loader:
        rgb_feat = data_tmp[0]
        logmel_gccphat = data_tmp[1]
        ss_pos_camcoords = data_tmp[2]
        ss_labels = data_tmp[3]
        cam_pos = data_tmp[4] #[N,viewnum,3]
        cam_rot = data_tmp[5] #[N,viewnum,4]

        ref_camera_pos = cam_pos[:,0,:]
        ref_camera_rot = cam_rot[:,0,:]
        multiview_camera_pos = cam_pos[:,1:,:]
        multiview_camera_rot = cam_rot[:,1:,:]
        ref_img_feat = rgb_feat[:,0,:,:,:]
        ref_micarray_input = logmel_gccphat[:,0,:,:,:]
        multiview_img_feat = rgb_feat[:,1:,:,:,:]
        multiview_micarray_input = logmel_gccphat[:,1:,:,:,:]
        ref_camera_pos = ref_camera_pos.to(device)
        ref_camera_rot = ref_camera_rot.to(device)
        multiview_camera_pos = multiview_camera_pos.to(device)
        multiview_camera_rot = multiview_camera_rot.to(device)
        ref_img_feat = ref_img_feat.to(device)
        ref_micarray_input = ref_micarray_input.to(device)
        multiview_img_feat = multiview_img_feat.to(device)
        multiview_micarray_input = multiview_micarray_input.to(device)

        # ss_pos, ss_class_logits = sound3ddet(ref_camera_pos,
        #                                      ref_camera_rot,
        #                                      multiview_camera_pos,
        #                                      multiview_camera_rot,
        #                                      ref_img_feat,
        #                                      ref_micarray_input,
        #                                      multiview_img_feat,
        #                                      multiview_micarray_input)

        output_dict = sound3ddet(ref_camera_pos,
                                             ref_camera_rot,
                                             multiview_camera_pos,
                                             multiview_camera_rot,
                                             ref_img_feat,
                                             ref_micarray_input,
                                             multiview_img_feat,
                                             multiview_micarray_input)

        breakpoint()

        # ref_camera_pos, ref_camera_rot, multiview_camera_pos, multiview_camera_rot,
        # ref_img_feat, ref_micarray_input,
        # multiview_img_feat, multiview_micarray_input
