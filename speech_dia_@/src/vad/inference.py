# from .my_pretrained import Pretrained
from .my_pretrained import Pretrained 
from pyannote_xxx.audio.utils.signal import Binarize
from pyannote_xxx.core import Segment, Annotation
from pyannote_xxx.core import SlidingWindow, SlidingWindowFeature
import yaml
from pathlib import Path
import os 
import numpy as np 

class SADInfer:
    def __init__(self, weight_pt, sample_rate=16000, device='cpu'):

        self.sample_rate = sample_rate

        current_dir = os.path.dirname(os.path.realpath(__file__))
        print(current_dir + "dsdsada")
        self.sad = Pretrained(weights_pt=weight_pt, cfg_path=current_dir, 
                                device=device)
        print("here")
        with open(os.path.join(current_dir, 'params.yml'), 'r') as fp:
            params = yaml.load(fp, Loader=yaml.SafeLoader)

        for k, v in params['params'].items():
            params[k] = float(v)
        print(params)
        self.binarize = Binarize(offset=params['offset'], 
                                onset=params['onset'], 
                                log_scale=True, 
                                min_duration_off=params['min_duration_off'], 
                                min_duration_on=params['min_duration_on'])

    def speech_detection(self, y): 

        # Compute prediction scores
        features = self.sad.get_features(y, self.sample_rate)
        sad_scores = SlidingWindowFeature(features, self.sad.get_resolution()) 

        # Apply threshold 
        hypothesis = self.binarize.apply(sad_scores, dimension=1)

        segments = []
        for s in hypothesis:
            segments.append(Segment(s.start, s.end))
        return segments
