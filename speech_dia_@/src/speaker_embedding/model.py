import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
from pyannote.core import SlidingWindow

from typing_extensions import Literal
from warnings import warn

from typing import Optional, List
from speaker_embedding.task import TaskType, TaskOutput

RESOLUTION_CHUNK = 'chunk'
# = SincNet ========================================================================
class SincConv1d(nn.Module):
    """Sinc-based 1D convolution

    Parameters
    ----------
    in_channels : `int`
        Should be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    stride : `int`, optional
        Defaults to 1.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    min_low_hz: `int`, optional
        Defaults to 50.
    min_band_hz: `int`, optional
        Defaults to 50.

    Usage
    -----
    Same as `torch.nn.Conv1d`

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio. "Speaker Recognition from raw waveform with
    SincNet". SLT 2018. https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, in_channels, out_channels, kernel_size,
                 sample_rate=16000, min_low_hz=50, min_band_hz=50,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super().__init__()

        if in_channels != 1:
            msg = (
                f'SincConv1d only supports one input channel. '
                f'Here, in_channels = {in_channels}.'
            )
            raise ValueError(msg)
        self.in_channels = in_channels

        self.out_channels = out_channels

        if kernel_size % 2 == 0:
            msg = (
                f'SincConv1d only support odd kernel size. '
                f'Here, kernel_size = {kernel_size}.'
            )
            raise ValueError(msg)
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv1d does not support bias.')
        if groups > 1:
            raise ValueError('SincConv1d does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Half Hamming half window
        n_lin=torch.linspace(0, self.kernel_size / 2 - 1,
                             steps=int((self.kernel_size / 2)))
        self.window_= 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (kernel_size, 1)
        # Due to symmetry, I only need half of the time axes
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate


    def forward(self, waveforms):
        """Get sinc filters activations

        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.

        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:,0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Equivalent to Eq.4 of the reference paper
        # I just have expanded the sinc and simplified the terms.
        # This way I avoid several useless computations.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class SincNet(nn.Module):
    """SincNet (learnable) feature extraction

    Parameters
    ----------
    waveform_normalize : `bool`, optional
        Standardize waveforms (to zero mean and unit standard deviation) and
        apply (learnable) affine transform. Defaults to True.
    instance_normalize : `bool`, optional
        Standardize internal representation (to zero mean and unit standard
        deviation) and apply (learnable) affine transform. Defaults to True.


    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio. "Speaker Recognition from raw waveform with
    SincNet". SLT 2018. https://arxiv.org/abs/1808.00158

    """

    @staticmethod
    def get_alignment(**kwargs):
        return 'strict'

    @staticmethod
    def get_resolution(sample_rate=16000,
                       kernel_size=[251, 5, 5],
                       stride=[1, 1, 1],
                       max_pool=[3, 3, 3], **kwargs):
        """
        """

        # https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
        padding = 0
        receptive_field, jump, start = 1, 1, 0.5
        for ks, s, mp in zip(kernel_size, stride, max_pool):
            # increase due to (Sinc)Conv1d
            receptive_field += (ks - 1) * jump
            start += ((ks - 1) / 2 - padding) * jump
            jump *= s
            # increase in receptive field due to MaxPool1d
            receptive_field += (mp - 1) * jump
            start += ((mp - 1) / 2 - padding) * jump
            jump *= mp

        return SlidingWindow(duration=receptive_field / sample_rate,
                             step=jump / sample_rate,
                             start=0.)


    def __init__(self,
                 waveform_normalize=True,
                 sample_rate=16000,
                 min_low_hz=50,
                 min_band_hz=50,
                 out_channels=[80, 60, 60],
                 kernel_size=[251, 5, 5],
                 stride=[1, 1, 1],
                 max_pool=[3, 3, 3],
                 instance_normalize=True,
                 activation='leaky_relu',
                 dropout=0.):
        super().__init__()

        self.dimension = out_channels[-1]

        # check parameters values
        n_layers = len(out_channels)
        if len(kernel_size) != n_layers:
            msg = (f'out_channels ({len(out_channels):d}) and kernel_size '
                   f'({len(kernel_size):d}) should have the same length.')
            raise ValueError(msg)
        if len(stride) != n_layers:
            msg = (f'out_channels ({len(out_channels):d}) and stride '
                   f'({len(stride):d}) should have the same length.')
            raise ValueError(msg)
        if len(max_pool) != n_layers:
            msg = (f'out_channels ({len(out_channels):d}) and max_pool '
                   f'({len(max_pool):d}) should have the same length.')
            raise ValueError(msg)

        # Waveform normalization
        self.waveform_normalize = waveform_normalize
        if self.waveform_normalize:
            self.waveform_normalize_ = torch.nn.InstanceNorm1d(1, affine=True)

        # SincNet-specific parameters
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Conv1D parameters
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1d_ = nn.ModuleList([])

        # Max-pooling parameters
        self.max_pool = max_pool
        self.max_pool1d_ = nn.ModuleList([])

        # Instance normalization
        self.instance_normalize = instance_normalize
        if self.instance_normalize:
            self.instance_norm1d_ = nn.ModuleList([])

        config = zip(self.out_channels,
                     self.kernel_size,
                     self.stride,
                     self.max_pool)

        for i, (out_channels, kernel_size, stride, max_pool) in enumerate(config):

            # 1D convolution
            if i > 0:
                conv1d = nn.Conv1d(in_channels,
                                   out_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=0,
                                   dilation=1,
                                   groups=1,
                                   bias=True)
            else:
                conv1d = SincConv1d(1,
                                    out_channels,
                                    kernel_size,
                                    sample_rate=self.sample_rate,
                                    min_low_hz=self.min_low_hz,
                                    min_band_hz=self.min_band_hz,
                                    stride=stride,
                                    padding=0,
                                    dilation=1,
                                    bias=False,
                                    groups=1)
            self.conv1d_.append(conv1d)

            # 1D max-pooling
            max_pool1d = nn.MaxPool1d(max_pool, stride=max_pool,
                                      padding=0, dilation=1)
            self.max_pool1d_.append(max_pool1d)

            # 1D instance normalization
            if self.instance_normalize:
                instance_norm1d = nn.InstanceNorm1d(out_channels, affine=True)
                self.instance_norm1d_.append(instance_norm1d)

            in_channels = out_channels

        # Activation function
        self.activation = activation
        if self.activation == 'leaky_relu':
            self.activation_ = nn.LeakyReLU(negative_slope=0.2)
        else:
            msg = (
                f'Only "leaky_relu" activation is supported.'
            )
            raise ValueError(msg)

        # Dropout
        self.dropout = dropout
        if self.dropout:
            self.dropout_ = nn.Dropout(p=self.dropout)

    def forward(self, waveforms):
        """Extract SincNet features

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1)
            Batch of waveforms

        Returns
        -------
        features : (batch_size, n_frames, out_channels[-1])
        """

        output = waveforms.transpose(1, 2)

        # standardize waveforms
        if self.waveform_normalize:
            output = self.waveform_normalize_(output)

        layers = zip(self.conv1d_, self.max_pool1d_)
        for i, (conv1d, max_pool1d) in enumerate(layers):

            output = conv1d(output)
            if i == 0:
                output = torch.abs(output)

            output = max_pool1d(output)

            if self.instance_normalize:
                output = self.instance_norm1d_[i](output)

            output = self.activation_(output)

            if self.dropout:
                output = self.dropout_(output)

        return output.transpose(1, 2)

    
    # def dimension():
    #     doc = "Output features dimension."
    #     def fget(self):
    #         return self.out_channels[-1]
    #     return locals()
    # dimension = property(**dimension())

# = TDNN ===========================================================================
class TemporalPooling(nn.Module):
    """Pooling strategy over temporal sequences."""

    @staticmethod
    def create(method: Literal['sum', 'max', 'last', 'stats']):
        """Pooling strategy factory. returns an instance of `TemporalPooling` given its name.

        Parameters
        ----------
        method : {'sum', 'max', 'last', 'stats', 'x-vector'}
            Temporal pooling strategy. The `x-vector` method name
            for stats pooling (equivalent to `stats`) is kept for
            retrocompatibility but it will be removed in a future version.
        Returns
        -------
        output : nn.Module
            The temporal pooling strategy object
        """
        if method == 'sum':
            klass = SumPool
        elif method == 'max':
            klass = MaxPool
        elif method == 'last':
            klass = LastPool
        elif method == 'stats':
            klass = StatsPool
        elif method == 'x-vector':
            klass = StatsPool
            warn("`x-vector` is deprecated and will be removed in a future version. Please use `stats` instead")
        else:
            raise ValueError(f"`{method}` is not a valid temporal pooling method")
        return klass()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("TemporalPooling subclass must implement `forward`")

class SumPool(TemporalPooling):
    """Calculate pooling as the sum over a sequence"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, hidden_size)
            A batch of sequences.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, hidden_size)
        """
        return x.sum(dim=1)

class MaxPool(TemporalPooling):
    """Calculate pooling as the maximum over a sequence"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, hidden_size)
            A batch of sequences.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, hidden_size)
        """
        return x.max(dim=1)[0]


class LastPool(TemporalPooling):
    """Calculate pooling as the last element of a sequence"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, hidden_size)
            A batch of sequences.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, hidden_size)
        """
        return x[:, -1]

class StatsPool(TemporalPooling):
    """Calculate pooling as the concatenated mean and standard deviation of a sequence"""

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, hidden_size)
            A batch of sequences.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, 2 * hidden_size)
        """
        mean, std = torch.mean(x, dim=1), torch.std(x, dim=1)
        return torch.cat((mean, std), dim=1)

class TDNN(nn.Module):
    def __init__(self, context: list, input_channels: int, output_channels: int, full_context: bool = True):
        """
        Implementation of a 'Fast' TDNN layer by exploiting the dilation argument of the PyTorch Conv1d class

        Due to its fastness the context has gained two constraints:
            * The context must be symmetric
            * The context must have equal spacing between each consecutive element

        For example: the non-full and symmetric context {-3, -2, 0, +2, +3} is not valid since it doesn't have
        equal spacing; The non-full context {-6, -3, 0, 3, 6} is both symmetric and has an equal spacing, this is
        considered valid.

        :param context: The temporal context
        :param input_channels: The number of input channels
        :param output_channels: The number of channels produced by the temporal convolution
        :param full_context: Indicates whether a full context needs to be used
        """
        super(TDNN, self).__init__()
        self.full_context = full_context
        self.input_dim = input_channels
        self.output_dim = output_channels

        context = sorted(context)
        self.check_valid_context(context, full_context)

        if full_context:
            kernel_size = context[-1] - context[0] + 1 if len(context) > 1 else 1
            self.temporal_conv = weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size))
        else:
            # use dilation
            delta = context[1] - context[0]
            self.temporal_conv = weight_norm(
                nn.Conv1d(input_channels, output_channels, kernel_size=len(context), dilation=delta))

    def forward(self, x):
        """
        :param x: is one batch of data, x.size(): [batch_size, sequence_length, input_channels]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, len(valid_steps), output_dim]
        """
        x = self.temporal_conv(torch.transpose(x, 1, 2))
        return F.relu(torch.transpose(x, 1, 2))

    @staticmethod
    def check_valid_context(context: list, full_context: bool):
        """
        Check whether the context is symmetrical and whether and whether the passed
        context can be used for creating a convolution kernel with dil

        :param full_context: indicates whether the full context (dilation=1) will be used
        :param context: The context of the model, must be symmetric if no full context and have an equal spacing.
        """
        if full_context:
            assert len(context) <= 2, "If the full context is given one must only define the smallest and largest"
            if len(context) == 2:
                assert context[0] + context[-1] == 0, "The context must be symmetric"
        else:
            assert len(context) % 2 != 0, "The context size must be odd"
            assert context[len(context) // 2] == 0, "The context contain 0 in the center"
            if len(context) > 1:
                delta = [context[i] - context[i - 1] for i in range(1, len(context))]
                assert all(delta[0] == delta[i] for i in range(1, len(delta))), "Intra context spacing must be equal!"


class XVectorNet(nn.Module):
    """
    X-Vector neural network architecture as defined by https://www.danielpovey.com/files/2018_icassp_xvectors.pdf

    Parameters
    ----------
    input_dim : int, default 24
        dimension of the input frames
    embedding_dim : int, default 512
        dimension of latent embeddings
    """

    @property
    def dimension(self):
        return self.embedding_dim

    def __init__(self, input_dim: int = 24, embedding_dim: int = 512):
        super(XVectorNet, self).__init__()
        frame1 = TDNN(context=[-2, 2], input_channels=input_dim, output_channels=512, full_context=True)
        frame2 = TDNN(context=[-2, 0, 2], input_channels=512, output_channels=512, full_context=False)
        frame3 = TDNN(context=[-3, 0, 3], input_channels=512, output_channels=512, full_context=False)
        frame4 = TDNN(context=[0], input_channels=512, output_channels=512, full_context=True)
        frame5 = TDNN(context=[0], input_channels=512, output_channels=1500, full_context=True)
        self.tdnn = nn.Sequential(frame1, frame2, frame3, frame4, frame5, StatsPool())
        self.segment6 = nn.Linear(3000, embedding_dim)
        self.segment7 = nn.Linear(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_intermediate: Optional[str] = None):
        """Calculate X-Vector network activations.
           Return the requested intermediate layer without computing unnecessary activations.

        Parameters
        ----------
        x : (batch_size, n_frames, out_channels)
            Batch of frames
        return_intermediate : 'stats_pool' | 'segment6' | 'segment7' | None
            If specified, return the activation of this specific layer.
            segment6 and segment7 activations are returned before the application of non linearity.

        Returns
        -------
        activations :
            (batch_size, 3000)               if return_intermediate == 'stats_pool'
            (batch_size, embedding_dim)      if return_intermediate == 'segment6' | 'segment7' | None
        """

        x = self.tdnn(x)

        if return_intermediate == 'stats_pool':
            return x

        x = self.segment6(x)

        if return_intermediate == 'segment6':
            return x

        x = self.segment7(F.relu(x))

        if return_intermediate == 'segment7':
            return x

        return F.relu(x)

# = Embedding ======================================================================
class Embedding(nn.Module):
    """Embedding

    Parameters
    ----------
    n_features : `int`
        Input dimension.
    batch_normalize : `boolean`, optional
        Apply batch normalization. This is more or less equivalent to
        embedding whitening.
    unit_normalize : `boolean`, optional
        Normalize embeddings. Defaults to False.
    """

    def __init__(self, n_features, batch_normalize=False, unit_normalize=False):
        super().__init__()

        self.n_features = n_features
        # batch normalization ~= embeddings whitening.

        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.batch_normalize_ = nn.BatchNorm1d(
                n_features, eps=1e-5, momentum=0.1, affine=False)

        self.unit_normalize = unit_normalize

        #khanhdtq
        self.dimension = self.n_features 

    def forward(self, embedding):

        if self.batch_normalize:
            embedding = self.batch_normalize_(embedding)

        if self.unit_normalize:
            norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
            embedding = embedding / norm

        return embedding
    
    # def dimension():
    #     doc = "Output dimension."
    #     def fget(self):
    #         return self.n_features
    #     return locals()
    # dimension = property(**dimension())

class SincTDNN(nn.Module):

    def __init__(self, specifications, **architecture_params):
        super().__init__()
        self.specifications = specifications
        self.resolution_ = self.get_resolution(**architecture_params)
        self.alignment_ = self.get_alignment(**architecture_params)
        self.init(**architecture_params)

    @property
    def task(self):
        """Type of task addressed by the model

        Shortcut for self.specifications['task']
        """
        return self.specifications['task']

    @staticmethod
    def get_alignment(sincnet=None, **kwargs):
        """
        """

        if sincnet is None:
            sincnet = dict()

        return SincNet.get_alignment(**sincnet)

    supports_packed = False

    @staticmethod
    def get_resolution(sincnet : Optional[dict] = None, **kwargs):
        """Get sliding window used for feature extraction

        Parameters
        ----------
        sincnet : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
        """

        return RESOLUTION_CHUNK

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        # return Model.dimension.fget(self)
    @property
    def n_features(self):
        """Number of input features

        Shortcut for self.specifications['X']['dimension']

        Returns
        -------
        n_features : `int`
            Number of input features
        """
        return self.specifications['X']['dimension']

    def init(self,
             sincnet : Optional[dict] = None,
             tdnn : Optional[dict] = None,
             embedding : Optional[dict] = None):
        """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output

        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters.
        tdnn : `dict`, optional
            X-Vector Time-Delay neural network parameters.
            Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        n_features = self.n_features

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if n_features != 1:
            raise ValueError('SincNet only supports mono waveforms. '
                             f'Here, waveform has {n_features} channels.')
        self.sincnet_ = SincNet(**sincnet)
        n_features = self.sincnet_.dimension

        if tdnn is None:
            tdnn = dict()
        self.tdnn = tdnn
        self.tdnn_ = XVectorNet(n_features, **tdnn)
        n_features = self.tdnn_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
        else:
            self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
            self.activation_ = self.task.default_activation
    
    def forward(self, waveforms: torch.Tensor, **kwargs):
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms

        Returns
        -------
        output : `torch.Tensor`
            Final network output or intermediate network output
            (only when `return_intermediate` is provided).
        """

        output = self.sincnet_(waveforms)

        return_intermediate = 'segment6' if self.task.is_representation_learning else None
        output = self.tdnn_(output, return_intermediate=return_intermediate)

        if self.task.is_representation_learning:
            return self.embedding_(output)

        return self.activation_(self.linear_(output))
