"""
Note: Part of the code was borrowed from: https://github.com/skmhrk1209/GANSynth/blob/master/spectral_ops.py
Note: I refractor it to be Pytorch supported!
"""
import torch
import tensorflow as tf
import numpy as np
import os
import pickle
import torch.nn.functional as F
import torch.nn as nn
import InstanFreqLoss

class WaveSpectrumConverter(nn.Module):
    def __init__(self, sampling_rate = 16000, spectrum_shape = [512, 512], waveform_length = 20001):
        super(WaveSpectrumConverter, self).__init__()
        self.sampling_rate = sampling_rate
        self.spectrum_shape = spectrum_shape #[time_steps, freq_bins]
        self.waveform_length = waveform_length
        self.get_linear2mel_weight_matrix()

        self.torch_pi = nn.Parameter(torch.from_numpy(np.array([np.pi],np.float32)), requires_grad=False)

        #for stft
        self.n_fft = (self.spectrum_shape[1]-1) * 2
        self.hop_length = self.waveform_length//self.spectrum_shape[1]

        #for instaneous frequency map calculator
        self.instanfreq_calculator = InstanFreqLoss.IntanFreq()

    def get_linear2mel_weight_matrix(self):
        # this matrix can be constant by graph optimization `Constant Folding`
        # since there are no Tensor inputs
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.spectrum_shape[1],
            num_spectrogram_bins=self.spectrum_shape[1],
            sample_rate=self.sampling_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sampling_rate / 2.0)

        with tf.Session() as sess:
            linear_to_mel_weight_matrix = linear_to_mel_weight_matrix.eval(session=sess)

        self.linear2mel_weight_matrix = nn.Parameter(torch.from_numpy(linear_to_mel_weight_matrix),
                                                     requires_grad=False)
        self.mel2linear_weight_matrix  = nn.Parameter(torch.linalg.pinv(torch.from_numpy(linear_to_mel_weight_matrix)),
                                                      requires_grad=False)

    def normalize(self, inputs, mean, stddev):
        return (inputs - mean)/stddev

    def convert_wave2spectrum_noIF(self, waveforms ):
        '''
        :param waveforms: [N, T], N waveforms of the same size
        :return: logmel spectrum and instaneous frequency (IF)
        '''
        stfts = torch.stft(input=waveforms,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          window=torch.hann_window(self.n_fft, periodic=True),
                          return_complex=True)

        if stfts.shape[1] > self.spectrum_shape[1] + 1:
            stfts = stfts[:,0:self.spectrum_shape[1]+1]
        stfts = stfts[...,1:]

        spectrum_magnitude = torch.abs(stfts)
        spectrum_phase = torch.angle(stfts)

        mel_magnitude_spectrum = torch.tensordot(spectrum_magnitude,
                                                 self.linear2mel_weight_matrix,
                                                 dims=1)

        mel_phase_spectrum = torch.tensordot(spectrum_phase,
                                             self.linear2mel_weight_matrix,
                                             dims=1)

        mel_magnitude_spectrum = torch.log(mel_magnitude_spectrum + 1.0e-6)
        mel_magnitude_spectrum = self.normalize(mel_magnitude_spectrum, -3.76, 10.05)

        mel_instantaneous_frequencies = mel_phase_spectrum
        #mel_instantaneous_frequencies = self.instanfreq_calculator(mel_phase_spectrum, time_axis=2)
        # mel_instantaneous_frequencies = self.instantaneous_frequency(mel_phase_spectrum, axis=-2)
        mel_instantaneous_frequencies = self.normalize(mel_instantaneous_frequencies, 0.0, 1.0)

        if mel_magnitude_spectrum.shape[1] > self.spectrum_shape[0]:
            mel_magnitude_spectrum = mel_magnitude_spectrum[:, 0:self.spectrum_shape[0],:]
            mel_instantaneous_frequencies = mel_instantaneous_frequencies[:,0:self.spectrum_shape[0],:]

        return mel_magnitude_spectrum, mel_instantaneous_frequencies

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

        if stfts.shape[2] > self.spectrum_shape[1] + 1:
            stfts = stfts[:,:,0:self.spectrum_shape[1]+1]

        stfts = stfts[...,1:]

        spectrum_magnitude = torch.abs(stfts)
        spectrum_phase = torch.angle(stfts)

        mel_magnitude_spectrum = torch.tensordot(spectrum_magnitude,
                                                 self.linear2mel_weight_matrix,
                                                 dims=1)

        mel_phase_spectrum = torch.tensordot(spectrum_phase,
                                             self.linear2mel_weight_matrix,
                                             dims=1)

        mel_magnitude_spectrum = torch.log(mel_magnitude_spectrum + 1.0e-6)
        #mel_magnitude_spectrum = self.normalize(mel_magnitude_spectrum, -3.76, 10.05)

        # mel_instantaneous_frequencies = mel_phase_spectrum

        mel_instantaneous_frequencies = self.instanfreq_calculator(mel_phase_spectrum, time_axis=2)
        # mel_instantaneous_frequencies = self.instantaneous_frequency(mel_phase_spectrum, axis=-2)
        #mel_instantaneous_frequencies = self.normalize(mel_instantaneous_frequencies, 0.0, 1.0)

        if mel_magnitude_spectrum.shape[1] > self.spectrum_shape[0]:
            mel_magnitude_spectrum = mel_magnitude_spectrum[:, 0:self.spectrum_shape[0],:]
            mel_instantaneous_frequencies = mel_instantaneous_frequencies[:,0:self.spectrum_shape[0],:]

        return mel_magnitude_spectrum, mel_instantaneous_frequencies

    def unnormalize(self, inputs, mean, stddev):
        return inputs*stddev + mean

    def convert_spectrum2wave(self, log_mel_magnitude_spectrograms, mel_instantaneous_frequencies):
        # log_mel_magnitude_spectrograms = self.unnormalize(logmel_magnitude_spectrum, -3.76, 10.05)
        # mel_instantaneous_frequencies = self.unnormalize(mel_instantaneous_frequencies, 0.0, 1.0)
        mel_magnitude_spectrum = torch.exp(log_mel_magnitude_spectrograms)
        # mel_phase_spectrum = mel_instantaneous_frequencies
        mel_phase_spectrum = torch.cumsum(mel_instantaneous_frequencies*np.pi, dim=2)

        magnitudes = torch.tensordot(mel_magnitude_spectrum, self.mel2linear_weight_matrix, dims=1)
        #magnitudes = torch.reshape(magnitudes, magnitudes.set_shape(mel_magnitude_spectrum.shape[:-1].concatenate(self.mel2linear_weight_matrix.shape[-1:])))
        phase_spectrum = torch.tensordot(mel_phase_spectrum, self.mel2linear_weight_matrix, dims=1)
        #phase_spectrum = torch.reshape(phase_spectrum, phase_spectrum.shape[:-1].concatenate(self.mel2linear_weight_matrix.shape[-1:]))

        stfts = torch.complex(magnitudes, imag=0.*magnitudes) * torch.complex(torch.cos(phase_spectrum), torch.sin(phase_spectrum))
        # discard_dc
        stfts = F.pad(stfts, pad=[1,0])

        waveforms = torch.istft(input=stfts,
                                n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                window=torch.hann_window(window_length=self.n_fft, periodic=True).to(log_mel_magnitude_spectrograms.device))

        if waveforms.shape[1] < self.waveform_length:
            waveforms = F.pad(waveforms, [0, self.waveform_length-waveforms.shape[1]])
        elif waveforms.shape[1] > self.waveform_length:
            waveforms = waveforms[:,0:self.waveform_length]

        return waveforms

    def convert_spectrum2wave_noIF(self, log_mel_magnitude_spectrograms, mel_instantaneous_frequencies):
        # log_mel_magnitude_spectrograms = self.unnormalize(logmel_magnitude_spectrum, -3.76, 10.05)
        # mel_instantaneous_frequencies = self.unnormalize(mel_instantaneous_frequencies, 0.0, 1.0)

        mel_magnitude_spectrum = torch.exp(log_mel_magnitude_spectrograms)
        mel_phase_spectrum = mel_instantaneous_frequencies
        # mel_phase_spectrum = torch.cumsum(mel_instantaneous_frequencies*np.pi, dim=-2)

        magnitudes = torch.tensordot(mel_magnitude_spectrum, self.mel2linear_weight_matrix, dims=1)
        #magnitudes = torch.reshape(magnitudes, magnitudes.set_shape(mel_magnitude_spectrum.shape[:-1].concatenate(self.mel2linear_weight_matrix.shape[-1:])))
        phase_spectrum = torch.tensordot(mel_phase_spectrum, self.mel2linear_weight_matrix, dims=1)
        #phase_spectrum = torch.reshape(phase_spectrum, phase_spectrum.shape[:-1].concatenate(self.mel2linear_weight_matrix.shape[-1:]))

        stfts = torch.complex(magnitudes, imag=0.*magnitudes) * torch.complex(torch.cos(phase_spectrum), torch.sin(phase_spectrum))
        # discard_dc
        stfts = F.pad(stfts, pad=[1,0])

        waveforms = torch.istft(input=stfts,
                                n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                window=torch.hann_window(window_length=self.n_fft, periodic=True).to(log_mel_magnitude_spectrograms.device))

        if waveforms.shape[1] < self.waveform_length:
            waveforms = F.pad(waveforms, [0, self.waveform_length-waveforms.shape[1]])
        elif waveforms.shape[1] > self.waveform_length:
            waveforms = waveforms[:,0:self.waveform_length]

        return waveforms

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def process_oneroom( root_dir, room_name, wave_cliplen = 20001, spectrogram_shape = [128, 256]):
    pickle_filename = os.path.join(root_dir, room_name, '{}.pickle'.format(room_name))
    if not os.path.exists(pickle_filename):
        return None

    with open(pickle_filename, 'rb') as handle:
        explore_dict = pickle.load(handle)

    step_num = len(explore_dict['explore_info'].keys()) - 2

    save_dir = os.path.dirname(pickle_filename)

    wavespectrum_converter = WaveSpectrumConverter()

    ss_num = 5

    mono_IF_list = list()
    mono_mag_list = list()
    binaural_IF_list = list()
    binaural_mag_list = list()

    for step_id in range(step_num):
        if step_id % 10 == 0:
            print('Processed {}/{} for {}'.format(step_id, step_num, room_name))
        step_key = 'step_{}'.format(step_id)
        mono_magnitude = np.zeros(shape=[ss_num, spectrogram_shape[0], spectrogram_shape[1]])
        mono_IF = np.zeros(shape=[ss_num, spectrogram_shape[0], spectrogram_shape[1]])
        binaural_magnitude = np.zeros(shape=[ss_num, 2, spectrogram_shape[0], spectrogram_shape[1]])
        binaural_IF = np.zeros(shape=[ss_num, 2, spectrogram_shape[0], spectrogram_shape[1]])
        mono_RIR = np.zeros([ss_num, wave_cliplen], np.float32)
        binaural_RIR = np.zeros([ss_num, 2, wave_cliplen], np.float32)
        for ss_id in range(ss_num):
            mono_RIR_tmp = explore_dict['explore_info'][step_key]['sound_rir']['ss_{}'.format(ss_id)]['mono_rir']
            mono_RIR_tmp = mono_RIR_tmp[0:wave_cliplen]
            mono_RIR[ss_id, :] = mono_RIR_tmp
            binaural_RIR_tmp = explore_dict['explore_info'][step_key]['sound_rir']['ss_{}'.format(ss_id)]['binaural_rir']
            binaural_RIR_tmp = binaural_RIR_tmp[:,0:wave_cliplen]
            binaural_RIR[ss_id,:,:] = binaural_RIR_tmp
            binaural_RIR = np.squeeze(binaural_RIR)

        mono_RIR = np.cumsum(mono_RIR, axis=0)
        binaural_RIR = np.cumsum(binaural_RIR, axis=0)

        #IF magnitude and phase
        mono_mag, mono_phase = wavespectrum_converter.convert_wave2spectrum(torch.from_numpy(mono_RIR).to(torch.float32))

        #non-IF magnitude and phase
        mono_mag_noIF, mono_phase_noIF = wavespectrum_converter.convert_wave2spectrum_noIF(torch.from_numpy(mono_RIR).to(torch.float32))
        np.save('/home/yuhang/mono_mag_IF.npy', mono_mag.detach().cpu().numpy())
        np.save('/home/yuhang/mono_phase_IF.npy', mono_phase.detach().cpu().numpy())
        np.save('/home/yuhang/mono_mag_noIF.npy', mono_mag_noIF.detach().cpu().numpy())
        np.save('/home/yuhang/mono_phase_noIF.npy', mono_phase_noIF.detach().cpu().numpy())
        np.save('/home/yuhang/mono_RIR.npy', mono_RIR)
        mono_RIR_recovered = wavespectrum_converter.convert_spectrum2wave(mono_mag, mono_phase)

        breakpoint()
        binaural_mag_leftear, binaural_phase_leftear = wavespectrum_converter.convert_wave2spectrum(torch.from_numpy(binaural_RIR[:,0,:]).to(torch.float32))
        binaural_mag_rightear, binaural_phase_rightear = wavespectrum_converter.convert_wave2spectrum(torch.from_numpy(binaural_RIR[:,1,:]).to(torch.float32))

        binaural_mag = np.stack((binaural_mag_leftear.detach().numpy(), binaural_mag_rightear.detach().numpy()), axis=1)
        binaural_phase = np.stack((binaural_phase_leftear.detach().numpy(), binaural_phase_rightear.detach().numpy()), axis=1)

        mono_spec_mag_savename = os.path.join(save_dir, 'monospec_mag_step{}.npy'.format(step_id))
        binaural_spec_mag_savename = os.path.join(save_dir, 'binauralspec_mag_step{}.npy'.format(step_id))

        np.save(mono_spec_mag_savename, mono_mag.detach().numpy())
        np.save(binaural_spec_mag_savename, binaural_mag)

        mono_spec_IF_savename = os.path.join(save_dir, 'monospec_IF_step{}.npy'.format(step_id))
        binaural_spec_IF_savename = os.path.join(save_dir, 'binauralspec_IF_step{}.npy'.format(step_id))

        np.save(mono_spec_IF_savename, mono_phase.detach().numpy())
        np.save(binaural_spec_IF_savename, binaural_phase)

        mono_mag_list.append(mono_mag.detach().numpy())
        mono_IF_list.append(mono_phase.detach().numpy())

        binaural_mag_list.append(binaural_mag)
        binaural_IF_list.append(binaural_phase)

        #update the exploredict
        explore_dict['explore_info'][step_key]['sound_rir']['mono_mag_savename'] = os.path.basename(mono_spec_mag_savename)
        explore_dict['explore_info'][step_key]['sound_rir']['mono_IF_savename'] = os.path.basename(mono_spec_IF_savename)

        explore_dict['explore_info'][step_key]['sound_rir']['binaural_mag_savename'] = os.path.basename(binaural_spec_mag_savename)
        explore_dict['explore_info'][step_key]['sound_rir']['binaural_IF_savename'] = os.path.basename(binaural_spec_IF_savename)

    MONO_IF = np.stack(mono_IF_list, axis=0)
    MONO_MAG = np.stack(mono_mag_list, axis=0)

    BINA_IF = np.stack(binaural_IF_list, axis=0)
    BINA_MAG = np.stack(binaural_mag_list, axis=0)

    mono_mag_mean = np.mean(MONO_MAG, axis=0, keepdims=False)
    mono_mag_std = np.std(MONO_MAG, axis=0, keepdims=False)

    #avoid std value to be 0.
    mono_mag_std[np.where(np.abs(mono_mag_std)<0.00001)] = 0.00001

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

    explore_dict_savename = os.path.join(root_dir, room_name, '{}_TFrep.pickle'.format(room_name))
    with open(explore_dict_savename, 'wb') as handle:
        pickle.dump(explore_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    root_dir = '/mnt/nas/yuhang/mp3d_rgba_rendering_1026/'
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