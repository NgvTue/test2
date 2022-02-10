import warnings
from typing import Optional
from typing import Union
from typing import Text
from pathlib import Path
import os 

import torch
import numpy as np

from pyannote_xxx.core import SlidingWindow
from pyannote_xxx.core import SlidingWindowFeature

from .model import RESOLUTION_FRAME
from .model import RESOLUTION_CHUNK

# from pyannote.audio.augmentation import Augmentation
from .features import FeatureExtraction


# from pyannote.audio.applications.config import load_config
# from pyannote.audio.applications.config import load_specs
# from pyannote.audio.applications.config import load_params

from .config import load_config, load_specs, load_params
class Pretrained(FeatureExtraction):
    """

    Parameters
    ----------
    validate_dir : Path
        Path to a validation directory.
    epoch : int, optional
        If provided, force loading this epoch.
        Defaults to reading epoch in validate_dir/params.yml.
    augmentation : Augmentation, optional
    duration : float, optional
        Use audio chunks with that duration. Defaults to the fixed duration
        used during training, when available.
    step : float, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.25.
    device : optional
    return_intermediate : optional
    """

    # TODO: add progress bar (at least for demo purposes)

    def __init__(self, weights_pt: Path = None,
                    cfg_path: str = None, 
                    epoch: int = None,
                    augmentation  = None,
                    duration: float = None,
                    step: float = None,
                    batch_size: int = 32,
                    device: Optional[Union[Text, torch.device]] = None,
                    return_intermediate = None,
                    progress_hook=None):

        try:
            weights_pt = Path(weights_pt)
        except TypeError as e:
            msg = (
                f'"weights_pt" must be str, bytes or os.PathLike object, '
                f'not {type(weights_pt).__name__}.'
            )
            raise TypeError(msg)

        self.weights_pt = weights_pt

        config_yml = Path(os.path.join(cfg_path, 'config.yml'))
        config = load_config(config_yml, training=False)
        # print(config)
        # use feature extraction from config.yml configuration file
        self.feature_extraction_ = config['feature_extraction']

        super().__init__(augmentation=augmentation,
                         sample_rate=self.feature_extraction_.sample_rate)

        self.feature_extraction_.augmentation = self.augmentation

        specs_yml = Path(os.path.join(cfg_path, 'specs.yml'))
        specifications = load_specs(specs_yml)

        if epoch is None:
            params_yml = Path(os.path.join(cfg_path, 'params.yml'))
            params = load_params(params_yml)
            self.epoch_ = params['epoch']
            # keep track of pipeline parameters
            self.pipeline_params_ = params.get('params', {})
        else:
            self.epoch_ = epoch

        self.preprocessors_ = config['preprocessors']

        model = config['get_model_from_specs'](specifications)
        model.load_state_dict(
            torch.load(self.weights_pt,
                       map_location=lambda storage, loc: storage))

        # defaults to using GPU when available
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # send model to device
        self.model_ = model.eval().to(self.device)

        # initialize chunks duration with that used during training
        self.duration = getattr(config['task'], 'duration', None)

        # override chunks duration by user-provided value
        if duration is not None:
            # warn that this might be sub-optimal
            if self.duration is not None and duration != self.duration:
                msg = (
                    f'Model was trained with {self.duration:g}s chunks and '
                    f'is applied on {duration:g}s chunks. This might lead '
                    f'to sub-optimal results.'
                )
                warnings.warn(msg)
            # do it anyway
            self.duration = duration

        if step is None:
            step = 0.25
        self.step = step
        self.chunks_ = SlidingWindow(duration=self.duration,
                                     step=self.step * self.duration)

        self.batch_size = batch_size

        self.return_intermediate = return_intermediate
        self.progress_hook = progress_hook

    @property
    def classes(self):sliding_window

    def get_resolution(self):

        resolution = self.model_.resolution

        # model returns one vector per input frame
        if resolution == RESOLUTION_FRAME:
            resolution = self.feature_extraction_.sliding_window

        # model returns one vector per input window
        if resolution == RESOLUTION_CHUNK:
            resolution = self.chunks_

        return resolution

    def get_features(self, y, sample_rate):

        features = SlidingWindowFeature(
            self.feature_extraction_.get_features(y, sample_rate),
            self.feature_extraction_.sliding_window)

        return self.model_.slide(features,
                                 self.chunks_,
                                 batch_size=self.batch_size,
                                 device=self.device,
                                 return_intermediate=self.return_intermediate,
                                 progress_hook=self.progress_hook).data

    def get_context_duration(self):
        # FIXME: add half window duration to context?
        return self.feature_extraction_.get_context_duration()