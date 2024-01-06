"""
Note: Part of the code was borrowed from: https://github.com/skmhrk1209/GANSynth/blob/master/spectral_ops.py
Note: I refractor it to be Pytorch supported!
"""
import torch
import numpy as np
import os
import pickle
from scipy.signal import fftconvolve
import librosa

class MicArrayFeat:
    def __init__(self,
                 sampling_rate=20001,
                 spectrum_shape=[256, 256],
                 waveform_len_secs = 1.,
                 seed_sound_dir = '/homes/yuhe/Downloads/1s_all',
                 one_ss_category = True,
                 ss_num = 5):
        self.sampling_rate = sampling_rate
        self.spectrum_shape = spectrum_shape  # [time_steps, freq_bins]
        self.waveform_len_secs = waveform_len_secs
        self.seed_sound_dir = seed_sound_dir
        self.one_ss_category = one_ss_category
        self.ss_num = ss_num
        self.waveform_length = int(sampling_rate*waveform_len_secs)

        # for stft
        self.n_fft = (self.spectrum_shape[1] - 1) * 2
        self.n_fft_2mel = (self.spectrum_shape[1]*2 - 1)*2
        self.hop_length = self.waveform_length // self.spectrum_shape[1]

        breakpoint()

        #self.get_seed_sound()
        self.get_linear2mel_weight_matrix()

    def get_seed_sound(self):
        # seed_sound_filename = os.path.join(self.seed_sound_dir, 'telephone.wav')
        seed_sound_filename_list = list()
        seed_sound_filename_list.append(os.path.join(self.seed_sound_dir, 'c_creak.wav'))
        seed_sound_filename_list.append(os.path.join(self.seed_sound_dir, 'c_fan.wav'))
        seed_sound_filename_list.append(os.path.join(self.seed_sound_dir, 'person_0.wav'))
        seed_sound_filename_list.append(os.path.join(self.seed_sound_dir, 'radio_static.wav'))
        seed_sound_filename_list.append(os.path.join(self.seed_sound_dir, 'come_to_office.wav'))
        assert os.path.exists(seed_sound_filename_list[0])
        assert os.path.exists(seed_sound_filename_list[1])
        assert os.path.exists(seed_sound_filename_list[2])
        assert os.path.exists(seed_sound_filename_list[3])
        assert os.path.exists(seed_sound_filename_list[4])

        seed_sound_list = list()
        for ss_id in range(self.ss_num):
            seed_sound, sr = librosa.load(seed_sound_filename_list[ss_id],sr=self.sampling_rate)
            seed_sound_list.append(seed_sound)

        self.seed_sounds = seed_sound_list

    def get_linear2mel_weight_matrix(self):
        #the obtained weight is normalized already
        self.linear2mel_weight = librosa.filters.mel(sr=self.sampling_rate,
                                                     n_fft=self.n_fft_2mel,
                                                     n_mels=self.spectrum_shape[1])

    def get_stft(self, input_wave, mel_trans = False):
        stft_spectrum_feat = librosa.core.stft(np.asfortranarray(input_wave),
                                               n_fft=self.n_fft_2mel if mel_trans else self.n_fft,
                                               hop_length=self.hop_length,
                                               win_length=2*self.hop_length,
                                               window='hann')

        if stft_spectrum_feat.shape[1] > self.spectrum_shape[1]:
            stft_spectrum_feat = stft_spectrum_feat[:,0:self.spectrum_shape[1]]

        return stft_spectrum_feat

    def get_melscale_spectrum(self, linear_spectrum):
        '''
        :param linear_spectrum: the spectrum obtained by stft transform
        :return: spectrum in mel-scale
        '''
        mag_spectra = np.abs(linear_spectrum) ** 2
        mel_spectra = np.dot(self.linear2mel_weight, mag_spectra)
        log_mel_spectra = librosa.power_to_db(mel_spectra)

        return log_mel_spectra

    def normalize(self, inputs, mean, stddev):
        return (inputs - mean) / stddev

    def convert_wave2spectrum(self, waveforms):
        '''
        :param waveforms: [N, T], N waveforms of the same size
        :return: logmel spectrum and instaneous frequency (IF)
        '''
        stfts = torch.stft(input=waveforms,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           window=torch.hann_window(self.n_fft, periodic=True),
                           return_complex=True)

        stfts = stfts[..., 1:]

        spectrum_magnitude = torch.abs(stfts)
        spectrum_phase = torch.angle(stfts)

        mel_magnitude_spectrum = torch.tensordot(spectrum_magnitude,
                                                 self.linear2mel_weight_matrix,
                                                 dims=1)

        mel_phase_spectrum = torch.tensordot(spectrum_phase,
                                             self.linear2mel_weight_matrix,
                                             dims=1)

        mel_magnitude_spectrum = torch.log(mel_magnitude_spectrum + 1.0e-6)

        return mel_magnitude_spectrum, mel_phase_spectrum

    def convolve_binauralRIR_get_audio(self, binaural_rir, ss_index = 0):
        '''
        Given the precomputed RIR, convolve them with the input seed sound to get the audio heard at this particular position
        :param current_source_sound: 1D seed sound
        :param input_RIRs: a list of two, left-ear and right-ear RIR
        :return: two audios, left and right
        '''
        assert ss_index <= len(self.seed_sounds)

        if self.one_ss_category:
            current_source_sound = self.seed_sounds[0]
        else:
            current_source_sound = self.seed_sounds[ss_index]

        binaural_rir = np.transpose(binaural_rir)
        binaural_convolved = np.array([fftconvolve(current_source_sound, binaural_rir[:, channel]
                                                   ) for channel in range(binaural_rir.shape[-1])])
        binaural_audio_convolved = binaural_convolved[:, 0:int(self.sampling_rate*self.waveform_len_secs)]

        return binaural_audio_convolved

    def convolve_monoRIR_get_audio(self, mono_rir, ss_index = 0):
        assert ss_index <= len(self.seed_sounds)
        mono_rir = np.squeeze(mono_rir)

        if self.one_ss_category:
            current_source_sound = self.seed_sounds[0]
        else:
            current_source_sound = self.seed_sounds[ss_index]

        mono_audio_convolved = np.array(fftconvolve(current_source_sound, mono_rir))

        mono_audio_convolved = mono_audio_convolved[0:int(self.sampling_rate*self.waveform_len_secs)]

        return mono_audio_convolved

    def get_gccphat_feat(self, leftear_wave, rightear_wave):
        '''
        Refered the source code: https://github.com/sharathadavanne/seld-dcase2021/blob/master/cls_feature_class.py
        '''
        #we don't compute the gccphat also in log-mel scale
        leftear_sptrum = librosa.core.stft(leftear_wave,
                                           n_fft=self.n_fft,
                                           hop_length=self.hop_length,
                                           win_length=self.n_fft,
                                           window='hann')
        leftear_sptrum = leftear_sptrum[:,1:]

        rightear_sptrum = librosa.core.stft(rightear_wave,
                                           n_fft=self.n_fft,
                                           hop_length=self.hop_length,
                                           win_length=self.n_fft,
                                           window='hann')

        rightear_sptrum = rightear_sptrum[:,1:]

        R = np.conj(rightear_sptrum, leftear_sptrum)
        cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
        gccphat_feat = np.concatenate((cc[:, -self.spectrum_shape[1] // 2:], cc[:, :self.spectrum_shape[1] // 2]),
                                      axis=-1)

        return gccphat_feat

    def get_micarray_feat(self, input_wave_array):
        '''
        currently, the input micarray is a four-channel waveforms of 1s length. The output is a 10-channel 2D feature: which
        contains 4-channel mel-scale TF frequency and inter-channel gcc-phat feature (any 2 out of 4, so it is 6 channels)
        :param input_wave_array: [4, sampling_rate]
        :return: [10, H, W]
        '''
        melscale_feat_list = list()
        for channel_id in range(input_wave_array.shape[0]):
            stft_feat = self.get_stft(input_wave_array[channel_id], mel_trans=True)
            melscale_feat = self.get_melscale_spectrum(stft_feat)
            melscale_feat_list.append(melscale_feat)
        melscale_feat = np.stack(melscale_feat_list, axis=0) #[6, H, W]

        #get gcc-phat
        gcc_phat_feat1 = self.get_gccphat_feat(input_wave_array[0, :], input_wave_array[1, :])
        gcc_phat_feat2 = self.get_gccphat_feat(input_wave_array[0, :], input_wave_array[2, :])
        gcc_phat_feat3 = self.get_gccphat_feat(input_wave_array[0, :], input_wave_array[3, :])
        gcc_phat_feat4 = self.get_gccphat_feat(input_wave_array[1, :], input_wave_array[2, :])
        gcc_phat_feat5 = self.get_gccphat_feat(input_wave_array[1, :], input_wave_array[3, :])
        gcc_phat_feat6 = self.get_gccphat_feat(input_wave_array[2, :], input_wave_array[3, :])

        gcc_phat_feat = np.stack([gcc_phat_feat1,
                                  gcc_phat_feat2,
                                  gcc_phat_feat3,
                                  gcc_phat_feat4,
                                  gcc_phat_feat5,
                                  gcc_phat_feat6], axis=0)

        micarray_feat = np.concatenate([melscale_feat, gcc_phat_feat], axis=0) #[10, H, W]

        return micarray_feat

def process_oneroom(root_dir, room_name, wave_cliplen=20001, spectrogram_shape=[512, 512], sampling_rate = 20001):
    spectrum_manager = SpectrumManager()

    pickle_filename = os.path.join(root_dir, room_name, '{}_TFrep.pickle'.format(room_name))
    assert os.path.exists(pickle_filename)

    with open(pickle_filename, 'rb') as handle:
        explore_dict = pickle.load(handle)

    step_num = len(explore_dict['explore_info'].keys()) - 2

    save_dir = os.path.dirname(pickle_filename)

    ss_num = 5

    mono_IF_list = list()
    mono_mag_list = list()
    binaural_IF_list = list()
    binaural_mag_list = list()
    gccphat_feat_list = list()

    for step_id in range(step_num):
        if step_id % 10 == 0:
            print('Processed {}/{} for {}'.format(step_id, step_num, room_name))
        step_key = 'step_{}'.format(step_id)
        accu_mono_audio = np.zeros([ss_num, sampling_rate], np.float32)
        accu_binaural_audio = np.zeros([ss_num, 2, sampling_rate], np.float32)

        for ss_id in range(ss_num):
            mono_RIR_tmp = explore_dict['explore_info'][step_key]['sound_rir']['ss_{}'.format(ss_id)]['mono_rir']
            mono_RIR_tmp = mono_RIR_tmp[0:wave_cliplen]
            mono_audio = spectrum_manager.convolve_monoRIR_get_audio( mono_RIR_tmp, ss_id )
            accu_mono_audio[ss_id, :] = mono_audio
            binaural_RIR_tmp = explore_dict['explore_info'][step_key]['sound_rir']['ss_{}'.format(ss_id)]['binaural_rir']
            binaural_RIR_tmp = binaural_RIR_tmp[:, 0:wave_cliplen]
            binaural_audio = spectrum_manager.convolve_binauralRIR_get_audio(binaural_RIR_tmp, ss_id)
            accu_binaural_audio[ss_id,:,:] = binaural_audio

        accu_mono_audio = np.cumsum(accu_mono_audio, axis=0)
        accu_binaural_audio = np.cumsum(accu_binaural_audio, axis=0)

        gccphat_feat = np.zeros(shape=[ss_num, spectrogram_shape[0], spectrogram_shape[1]], dtype=np.float32)

        for ss_id in range(ss_num):
            leftear_audio = accu_binaural_audio[ss_id,0,:]
            rightear_audio = accu_binaural_audio[ss_id,1,:]
            gccphat_feat_tmp = spectrum_manager.get_gccphat_feat(leftear_audio, rightear_audio)
            gccphat_feat[ss_id, :, :] = gccphat_feat_tmp

        gccphat_feat_list.append(gccphat_feat)
        gccphat_feat_savename = os.path.join(save_dir, 'gccphat_step{}.npy'.format(step_id))
        np.save(gccphat_feat_savename, gccphat_feat)
        explore_dict['explore_info'][step_key]['sound_rir']['gccphat_savename'] = os.path.basename(
            gccphat_feat_savename)

        mono_magnitude, mono_IF = spectrum_manager.convert_wave2spectrum(torch.from_numpy(accu_mono_audio))
        mono_magnitude = mono_magnitude.numpy()
        mono_IF = mono_IF.numpy()
        accu_binaural_audio = np.reshape(accu_binaural_audio, newshape=[-1,accu_binaural_audio.shape[-1]])
        binaural_magnitude, binaural_IF = spectrum_manager.convert_wave2spectrum(torch.from_numpy(accu_binaural_audio))
        binaural_magnitude = binaural_magnitude.numpy()
        binaural_magnitude = np.reshape(binaural_magnitude, newshape=[-1, 2, binaural_magnitude.shape[-2], binaural_magnitude.shape[-1]])
        binaural_IF = binaural_IF.numpy()
        binaural_IF = np.reshape(binaural_IF, newshape=[-1, 2, binaural_IF.shape[-2], binaural_IF.shape[-1]])

        mono_spec_mag_savename = os.path.join(save_dir, 'monospec_mag_step{}.npy'.format(step_id))
        binaural_spec_mag_savename = os.path.join(save_dir, 'binauralspec_mag_step{}.npy'.format(step_id))

        np.save(mono_spec_mag_savename, mono_magnitude)
        np.save(binaural_spec_mag_savename, binaural_magnitude)

        mono_spec_IF_savename = os.path.join(save_dir, 'monospec_IF_step{}.npy'.format(step_id))
        binaural_spec_IF_savename = os.path.join(save_dir, 'binauralspec_IF_step{}.npy'.format(step_id))

        np.save(mono_spec_IF_savename, mono_IF)
        np.save(binaural_spec_IF_savename, binaural_IF)

        mono_mag_list.append(mono_magnitude)
        mono_IF_list.append(mono_IF)

        binaural_mag_list.append(binaural_magnitude)
        binaural_IF_list.append(binaural_IF)

        # update the exploredict
        explore_dict['explore_info'][step_key]['sound_rir']['mono_mag_savename'] = os.path.basename(
            mono_spec_mag_savename)
        explore_dict['explore_info'][step_key]['sound_rir']['mono_IF_savename'] = os.path.basename(
            mono_spec_IF_savename)

        explore_dict['explore_info'][step_key]['sound_rir']['binaural_mag_savename'] = os.path.basename(
            binaural_spec_mag_savename)
        explore_dict['explore_info'][step_key]['sound_rir']['binaural_IF_savename'] = os.path.basename(
            binaural_spec_IF_savename)

    MONO_IF = np.stack(mono_IF_list, axis=0)
    MONO_MAG = np.stack(mono_mag_list, axis=0)

    BINA_IF = np.stack(binaural_IF_list, axis=0)
    BINA_MAG = np.stack(binaural_mag_list, axis=0)

    GCC_PHAT = np.stack(gccphat_feat_list, axis=0)
    GCC_PHAT_mean = np.mean(GCC_PHAT, axis=0, keepdims=False)
    GCC_PHAT_std = np.std(GCC_PHAT, axis=0, keepdims=False)
    GCC_PHAT_std[np.where(np.abs(GCC_PHAT_std) < 0.00001)] = 0.00001

    mono_mag_mean = np.mean(MONO_MAG, axis=0, keepdims=False)
    mono_mag_std = np.std(MONO_MAG, axis=0, keepdims=False)

    # avoid std value to be 0.
    mono_mag_std[np.where(np.abs(mono_mag_std) < 0.00001)] = 0.00001

    mono_IF_mean = np.mean(MONO_IF, axis=0, keepdims=False)
    mono_IF_std = np.std(MONO_IF, axis=0, keepdims=False)
    mono_IF_std[np.where(np.abs(mono_IF_std) < 0.00001)] = 0.00001

    binaural_mag_mean = np.mean(BINA_MAG, axis=0, keepdims=False)
    binaural_mag_std = np.std(BINA_MAG, axis=0, keepdims=False)
    binaural_mag_std[np.where(np.abs(binaural_mag_std) < 0.00001)] = 0.00001

    binaural_IF_mean = np.mean(BINA_IF, axis=0, keepdims=False)
    binaural_IF_std = np.std(BINA_IF, axis=0, keepdims=False)

    binaural_IF_std[np.where(np.abs(binaural_IF_std) < 0.00001)] = 0.00001

    explore_dict['mono_mag_mean'] = mono_mag_mean
    explore_dict['mono_mag_std'] = mono_mag_std
    explore_dict['mono_IF_mean'] = mono_IF_mean
    explore_dict['mono_IF_std'] = mono_IF_std

    explore_dict['binaural_mag_mean'] = binaural_mag_mean
    explore_dict['binaural_mag_std'] = binaural_mag_std
    explore_dict['binaural_IF_mean'] = binaural_IF_mean
    explore_dict['binaural_IF_std'] = binaural_IF_std

    explore_dict['gccphat_mean'] = GCC_PHAT_mean
    explore_dict['gccphat_std'] = GCC_PHAT_std

    #compute the single mean/std for mono/binaural audio
    mono_mag_mean_scalar = np.mean(MONO_MAG.reshape([-1]))
    mono_mag_std_scalar = np.std(MONO_MAG.reshape([-1]))
    mono_IF_mean_scalar = np.mean(MONO_IF.reshape([-1]))
    mono_IF_std_scalar = np.std(MONO_IF.reshape([-1]))

    explore_dict['mono_mag_mean_scalar'] = mono_mag_mean_scalar
    explore_dict['mono_mag_std_scalar'] = mono_mag_std_scalar
    explore_dict['mono_IF_mean_scalar'] = mono_IF_mean_scalar
    explore_dict['mono_IF_std_scalar'] = mono_IF_std_scalar

    binaural_mag_mean_scalar = np.mean(BINA_MAG.reshape([-1]))
    binaural_mag_std_scalar = np.std(BINA_MAG.reshape([-1]))
    binaural_IF_mean_scalar = np.mean(BINA_IF.reshape([-1]))
    binaural_IF_std_scalar = np.std(BINA_IF.reshape([-1]))

    explore_dict['binaural_mag_mean_scalar'] = binaural_mag_mean_scalar
    explore_dict['binaural_mag_std_scalar'] = binaural_mag_std_scalar
    explore_dict['binaural_IF_mean_scalar'] = binaural_IF_mean_scalar
    explore_dict['binaural_IF_std_scalar'] = binaural_IF_std_scalar

    gccphat_mean_scalar = np.mean(GCC_PHAT.reshape([-1]))
    gccphat_std_scalar = np.std(GCC_PHAT.reshape([-1]))

    explore_dict['gccphat_mean_scalar'] = gccphat_mean_scalar
    explore_dict['gccphat_std_scalar'] = gccphat_std_scalar

    explore_dict_savename = os.path.join(root_dir, room_name, '{}_TFrep_v2.pickle'.format(room_name))
    # with open(explore_dict_savename, 'wb') as handle:
    #     pickle.dump(explore_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    spectrum_manager = MicArrayFeat()
    breakpoint()
    root_dir = '/mnt/nas/yuhang/mp3d_rgba_stepsie1.0_0206/'
    room_name_list = os.listdir(root_dir)

    spectrogram_shape = [512, 512]

    for room_name in room_name_list:
        print('processing {}'.format(room_name))
        if not room_name == '17DRP5sb8fy':
            continue
        process_oneroom(root_dir,
                        room_name,
                        spectrogram_shape=spectrogram_shape)

    print('Done!')
