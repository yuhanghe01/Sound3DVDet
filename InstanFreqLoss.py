"""
NOTE: Implement Instaneous Frequency Loss. Coder refer to: https://github.com/ss12f32v/GANsynth-pytorch/blob/master/phase_operation.py
     Paper refer to: https://arxiv.org/pdf/1808.06719.pdf
"""
import torch
import torch.nn as nn
import numpy as np


class IntanFreq(nn.Module):
    def __init__(self):
        super(IntanFreq, self).__init__()
        pass

    def diff(self, x, axis):
        """Take the finite difference of a tensor along an axis.
        Args:
        x: Input tensor of any dimension.
        axis: Axis on which to take the finite difference.
        Returns:
        d: Tensor with size less than x by 1 along the difference dimension.
        Raises:
        ValueError: Axis out of range for tensor.
        """
        # dim_on_axis = x.shape[axis]
        begin_back = [0] * len(x.shape)
        end_back = [x.shape[0], x.shape[1], x.shape[2]]
        end_back[axis] -= 1

        begin_front = [0] * len(x.shape)
        begin_front[axis] += 1
        end_front = [x.shape[0], x.shape[1], x.shape[2]]

        slice_front = x[begin_front[0]:end_front[0], begin_front[1]:end_front[1], begin_front[2]:end_front[2]]
        slice_back = x[begin_back[0]:end_back[0], begin_back[1]:end_back[1], begin_back[2]:end_back[2]]

        d = slice_front - slice_back

        return d

    def unwrap(self, p, axis=-1):
        """Unwrap a cyclical phase tensor.
        Args:
        p: Phase tensor.
        discont: Float, size of the cyclic discontinuity.
        axis: Axis of which to unwrap.
        Returns:
        unwrapped: Unwrapped tensor of same size as input.
        """
        dd = self.diff(p, axis=axis)
        # ddmod = np.mod(dd + np.pi, 2.0 * np.pi) - np.pi  # ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
        ddmod = torch.remainder(dd + np.pi, 2.0 * np.pi) - np.pi
        # idx = np.logical_and(np.equal(ddmod, -np.pi),np.greater(dd, 0))
        # ddmod = np.where(idx, np.ones_like(ddmod) * np.pi, ddmod)
        # ph_correct = ddmod - dd
        idx = torch.logical_and(torch.eq(ddmod, -np.pi), torch.greater(dd, 0))
        ddmod = torch.where(idx, torch.ones_like(ddmod) * np.pi, ddmod)
        ph_correct = ddmod - dd

        # ddmod = np.where(idx, np.zeros_like(ddmod), dd)  # ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
        # ph_cumsum = np.cumsum(ph_correct, axis=axis)  # ph_cumsum = tf.cumsum(ph_correct, axis=axis)
        ph_cumsum = torch.cumsum(ph_correct, dim=axis)

        #
        # breakpoint()
        ph_cumsum_fake = torch.zeros_like(p)
        ph_cumsum_fake[:, :, 1:] = ph_cumsum
        ph_cumsum = ph_cumsum_fake
        unwrapped = p + ph_cumsum

        return unwrapped

    def instantaneous_frequency(self, phase_angle, time_axis):
        """Transform a fft tensor from phase angle to instantaneous frequency.
        Unwrap and take the finite difference of the phase. Pad with initial phase to
        keep the tensor the same size.
        Args:
        phase_angle: Tensor of angles in radians. [Batch, Time, Freqs]
        time_axis: Axis over which to unwrap and take finite difference.
        Returns:
        dphase: Instantaneous frequency (derivative of phase). Same size as input.
        """
        phase_unwrapped = self.unwrap(phase_angle, axis=time_axis)
        dphase = self.diff(phase_unwrapped, axis=time_axis)
        slice_begin = [0] * len(phase_unwrapped.shape)
        slice_end = [phase_unwrapped.shape[0], phase_unwrapped.shape[1], phase_unwrapped.shape[2]]
        slice_end[time_axis] = 1

        # Add an initial phase to dphase
        phase_slice = phase_unwrapped[slice_begin[0]:slice_end[0],
                      slice_begin[1]:slice_end[1],
                      slice_begin[2]:slice_end[2]]

        dphase = torch.cat([phase_slice, dphase], dim=time_axis) / torch.pi
        # dphase = np.concatenate([phase_slice, dphase], axis=time_axis) / np.pi

        return dphase

    def forward(self, phase_angle, time_axis):
        return self.instantaneous_frequency(phase_angle, time_axis)


class InstanFreqLoss(nn.Module):
    def __init__(self):
        super(InstanFreqLoss, self).__init__()
        self.instfreq_calculator = InstanFreqLoss()

    def forward(self, phase_angle1, phase_angle2, time_axis):
        instanfreq1 = self.instfreq_calculator(phase_angle1, time_axis)
        instanfreq2 = self.instfreq_calculator(phase_angle2, time_axis)

        return torch.mean(torch.abs(instanfreq1 - instanfreq2))