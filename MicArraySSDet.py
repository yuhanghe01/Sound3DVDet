import torch
import os
import torch.nn as nn
import torch.nn.functional as F

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, stride, padding ):
        super(Conv2DBlock, self).__init__()
        self.conv_op = nn.Sequential( nn.Conv2d(in_channels=in_channels,
                                                out_channels=output_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding),
                                      nn.BatchNorm2d(num_features=output_channels),
                                      nn.ReLU() )

    def forward(self, input_feat):
        return self.conv_op(input_feat)

class MicArraySSDetector(nn.Module):
    def __init__(self, input_channel_dim: int = 10, ss_pred_num: int = 10, TF_map_height: int = 512,
                 TF_map_width: int = 512 ):
        super(MicArraySSDetector, self).__init__()
        self.input_channel_dim = input_channel_dim
        self.ss_pred_num = ss_pred_num
        self.TF_map_height = TF_map_height
        self.TF_map_width = TF_map_width

        #the input feature size is [10, 512, 512], 10 channels
        self.encoder_config = [{'in_channel': 10, 'out_channel': 32, 'kernel_size': [3,3], 'stride': 2, 'padding': [1,1]},  #[32, 256, 256]
                               {'in_channel': 32, 'out_channel': 64, 'kernel_size': [3,3], 'stride': 2, 'padding': [1,1]},  #[64,128, 128]
                               {'in_channel': 64, 'out_channel': 128, 'kernel_size': [3,3], 'stride': 2, 'padding': [1,1]},  #[128, 64, 64]
                               {'in_channel': 128, 'out_channel': 256, 'kernel_size': [3,3], 'stride': 2, 'padding': [1,1]},  #[256,32, 32]
                               {'in_channel': 256, 'out_channel': 256, 'kernel_size': [3, 3], 'stride': 2, 'padding': [1,1]},  #[256, 16,16]
                               {'in_channel': 256, 'out_channel': 512, 'kernel_size': [3, 3], 'stride': 2, 'padding': [1,1]},  # [512, 8, 8]
                               ]

        self.construct_Encoder()
        self.micarray_feat_mapper = nn.Linear(in_features=self.TF_map_width//2, out_features=self.TF_map_width)

    def construct_Encoder(self):
        self.conv_encoder0 = Conv2DBlock(in_channels=self.encoder_config[0]['in_channel'],
                                         output_channels=self.encoder_config[0]['out_channel'],
                                         kernel_size=self.encoder_config[0]['kernel_size'],
                                         stride=self.encoder_config[0]['stride'],
                                         padding=self.encoder_config[0]['padding'])

        self.conv_encoder1 = Conv2DBlock(in_channels=self.encoder_config[1]['in_channel'],
                                         output_channels=self.encoder_config[1]['out_channel'],
                                         kernel_size=self.encoder_config[1]['kernel_size'],
                                         stride=self.encoder_config[1]['stride'],
                                         padding=self.encoder_config[1]['padding'])

        self.conv_encoder2 = Conv2DBlock(in_channels=self.encoder_config[2]['in_channel'],
                                         output_channels=self.encoder_config[2]['out_channel'],
                                         kernel_size=self.encoder_config[2]['kernel_size'],
                                         stride=self.encoder_config[2]['stride'],
                                         padding=self.encoder_config[2]['padding'])

        self.conv_encoder3 = Conv2DBlock(in_channels=self.encoder_config[3]['in_channel'],
                                         output_channels=self.encoder_config[3]['out_channel'],
                                         kernel_size=self.encoder_config[3]['kernel_size'],
                                         stride=self.encoder_config[3]['stride'],
                                         padding=self.encoder_config[3]['padding'])

        self.conv_encoder4 = Conv2DBlock(in_channels=self.encoder_config[4]['in_channel'],
                                         output_channels=self.encoder_config[4]['out_channel'],
                                         kernel_size=self.encoder_config[4]['kernel_size'],
                                         stride=self.encoder_config[4]['stride'],
                                         padding=self.encoder_config[4]['padding'])

        self.conv_encoder5 = Conv2DBlock(in_channels=self.encoder_config[5]['in_channel'],
                                         output_channels=self.encoder_config[5]['out_channel'],
                                         kernel_size=self.encoder_config[5]['kernel_size'],
                                         stride=self.encoder_config[5]['stride'],
                                         padding=self.encoder_config[5]['padding'])


    def get_micarray_feat_rep(self, input_feat):
        batch_size, channel_num, feat_height, feat_width = input_feat.shape
        input_feat = torch.permute(input_feat, dims=[0,2,3,1])
        input_feat = torch.reshape(input_feat, shape=[batch_size*feat_height*feat_width,channel_num])
        micarray_feat_rep = self.micarray_feat_mapper(input_feat)

        micarray_feat_rep = torch.reshape(micarray_feat_rep, shape=[batch_size, feat_height, feat_width, -1])
        micarray_feat_rep = torch.permute(micarray_feat_rep, dims=[0,3,1,2]) #[N,C,H,W]

        return micarray_feat_rep

    def forward(self, input_micarray_feat):
        '''
        the input feat is the concatenation of current-pos mono-audio and goal-pos anechoic mono-audio. The mono-audio is
        represented in time-frequency, including both real and imaginary part. So there are total 4 channels.
        :param input_micarray_feat: [B, 10, 512, 512]
        :return: predicted RIR: [B, 512, 512], [B, 512, 512]; and encoder embedding [B, 512]
        '''
        #step1: feed to encoder
        encoder0_feat = self.conv_encoder0(input_micarray_feat)
        encoder1_feat = self.conv_encoder1(encoder0_feat)
        encoder2_feat = self.conv_encoder2(encoder1_feat)
        encoder3_feat = self.conv_encoder3(encoder2_feat) #[N, 32, 32, 256]
        encoder4_feat = self.conv_encoder4(encoder3_feat) #[N, 16, 16, 256]
        encoder5_feat = self.conv_encoder5(encoder4_feat) #[B, 8, 8, 512]

        breakpoint()

        #during our implementation, we just need 4x4 = 16 source source queries
        encoder5_feat = F.avg_pool2d(encoder5_feat, kernel_size=[2,2], stride=[2,2])
        batch_size, feat_dim = encoder5_feat.shape[0:2]

        ss_object_queries = torch.reshape(encoder5_feat, shape=[batch_size, -1, feat_dim])

        micarray_feat_rep = self.get_micarray_feat_rep(encoder4_feat)

        return ss_object_queries, micarray_feat_rep


def main():
    micarray_ssdetector = MicArraySSDetector()
    micarray_ssdetector.train()
    import os
    import numpy as np
    os.putenv('CUDA_VISIBLE_DEVICES', '0')
    device = torch.device('cuda:0')
    micarray_ssdetector = micarray_ssdetector.to(device)
    input_feat = torch.from_numpy(np.random.rand(2,10,512,512).astype(np.float32)).to(torch.float32).to(device)

    output, micarray_feat_rep = micarray_ssdetector(input_feat)

    breakpoint()

if __name__ == '__main__':
    main()




