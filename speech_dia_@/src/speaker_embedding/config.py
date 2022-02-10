#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

import warnings
from functools import partial

from pathlib import Path
import shutil
from typing import Text
from typing import Dict
import yaml

import collections
from .task import Task
from soundfile import SoundFile
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import FileFinder
# from pyannote.audio.features.utils import get_audio_duration
def get_audio_duration(current_file):
    """Return audio file duration

    Parameters
    ----------
    current_file : dict
        Dictionary given by pyannote.database.

    Returns
    -------
    duration : float
        Audio file duration.
    """

    with SoundFile(current_file['audio'], 'r') as f:
        duration = float(f.frames) / f.samplerate

    return duration

def merge_cfg(pretrained_cfg, cfg):
    for k, v in cfg.items():

        # case where the user purposedly set a section value to "null"
        # this might happen when fine-tuning a pretrained model
        if v is None:
            _ = pretrained_cfg.pop(k, None)

        # if v is a dictionary, go deeper and merge recursively
        elif isinstance(v, collections.abc.Mapping):
            pretrained_cfg[k] = merge_cfg(pretrained_cfg.get(k, {}), v)

        # in any other case, override pretrained_cfg[k] by cfg[k]
        else:
            pretrained_cfg[k] = v

    return pretrained_cfg


def load_specs(specs_yml: Path) -> Dict:
    """

    Returns
    -------
    specs : Dict
        ['task']
        [and others]
    """

    with open(specs_yml, 'r') as fp:
        specifications = yaml.load(fp, Loader=yaml.SafeLoader)
    specifications['task'] = Task.from_str(specifications['task'])
    return specifications

def load_params(params_yml: Path) -> Dict:

    with open(params_yml, 'r') as fp:
        params = yaml.load(fp, Loader=yaml.SafeLoader)

    return params
