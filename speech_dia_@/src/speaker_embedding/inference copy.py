import numpy as np 
import torch
from .model import SincTDNN
from ..utils.task import TaskOutput, TaskType, Task 
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
import pescador
import soundfile as sf
import librosa 
import wave 
from pyannote.audio.applications.config import load_specs
import yaml 
import os 
from pathlib import Path 

sample_rate = 16000
weight_pt = 'checkpoints/emb.pt'

def generate_inference_segments(y, infer_seg_len, overlap=0.25):
    original_len = len(y)
    # assert original_len > infer_seg_len

    if original_len > infer_seg_len:
        infer_segs = []
        start_ = 0
        while (start_ + infer_seg_len) <= original_len:
            infer_seg = y[start_: (start_ + infer_seg_len)]
            infer_segs.append(infer_seg)
            
            start_ += int(infer_seg_len * (1 - overlap))
            
        return infer_segs 
    
    offset = (infer_seg_len - original_len) // 2
    seg = np.pad(np.squeeze(y, 1), (offset, infer_seg_len - offset - original_len), 'constant')
    return [seg[:, np.newaxis]]

class SpkrEmbInfer:
    def __init__(self, weight_pt, 
                    embedding_dim=512,
                    batch_size=4,
                    duration=2., step=0.025, 
                    sample_rate=sample_rate,
                    device='cpu'):

        self.sample_rate = sample_rate 
        self.duration = duration 
        self.context = 0. 
        self.raw_audio_sliding_window = SlidingWindow(start=-.5/sample_rate,
                                    duration=1./sample_rate,
                                    step=1./sample_rate)       
        # self.chunks_ =  SlidingWindow(duration=duration, step=step * duration)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        specs_yml = Path(os.path.join(current_dir, 'specs.yml'))
        specifications = load_specs(specs_yml)
        with open(os.path.join(current_dir, 'config.yml'), 'r') as fp:
            cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        arch_params = cfg['architecture']['params']
        print(specifications)
        print(arch_params)
        model = SincTDNN(
            specifications,
            **arch_params
        )
        model.load_state_dict(torch.load(weight_pt, map_location=lambda storage, loc: storage))        
        self.model = model.eval().to(device)
        self.embedding_dim = embedding_dim 
        self.batch_size = batch_size
        self.device = device

    # def get_segment_embedding_from_wav(self, y):
    #     # apply model 
    #     features = SlidingWindowFeature(y, self.raw_audio_sliding_window)    
    #     support = features.extent

    #     if support.duration < self.chunks_.duration:
    #         chunks = [support]
    #         fixed = support.duration
    #     else:
    #         chunks = list(self.chunks_(support, align_last=True))
    #         fixed = self.chunks_.duration        
    #     batches = pescador.maps.buffer_stream(
    #         iter({'X': features.crop(window, mode='center', fixed=fixed)}
    #                 for window in chunks),
    #         self.batch_size, partial=True)

    #     fX = []
    #     for batch in batches:
    #         tX = torch.tensor(batch['X'], dtype=torch.float32, device='cpu')
    #         ftX = self.model(tX).detach().to('cpu').numpy()
    #         fX.append(ftX)

    #     fX = np.vstack(fX) 

    #     return SlidingWindowFeature(fX, self.chunks_).data          

    def get_segment_embedding_from_wav(self, y):
        segments = generate_inference_segments(y, int(self.duration * self.sample_rate))

        batches = pescador.maps.buffer_stream(
            iter({'X': segment} for segment in segments),
            self.batch_size, partial=True)
        
        fX = []
        for batch in batches:
            tX = torch.tensor(batch['X'], dtype=torch.float32, device=self.device)
            ftX = self.model(tX).detach().to('cpu').numpy()
            fX.append(ftX)

        fX = np.vstack(fX) 

        fX = np.mean(np.vstack(fX), axis=0, keepdims=False)

        return fX 

    def get_segment_embedding_from_file(self, segment, current_file):
        """Offline inference
        Parameters
        ----------
        fpath: path to .wav file
        segments: (start, end) of one voiced segment (obtained from SAD) 
        Returns
        -------
        data: (n_subsegments, embedding_dim)
        """                    
        
        # extend segment on both sides with requested context
        xsegment = Segment(
            max(0, segment.start - self.context),
            min(current_file['duration'], segment.end + self.context))       

        # obtain (augmented) waveform on this extended segment
        (seg_start, seg_end), = self.raw_audio_sliding_window.crop(xsegment, mode='center',
                                    fixed=xsegment.duration, return_ranges=True)
            
        y_seg = current_file['waveform'][seg_start: seg_end] 

        embs = self.get_segment_embedding_from_wav(y_seg)   
        return embs 
