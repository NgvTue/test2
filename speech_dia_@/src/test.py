from speaker_embedding.inference import SpkrEmbInfer
from vad.inference import  SADInfer
import sys
from pyannote.audio.tasks.segmentation.speaker_change_detection import SpeakerChangeDetection
from pyannote.audio import Model 

from pytorch_lightning import Trainer
pretrained = Model.from_pretrained("/home/tuenguyen/speech/speech_dia_@/version_6/checkpoints/epoch=17-step=182645.ckpt")
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pyannote.audio import Inference
from pyannote.core import notebook
inference = Inference(pretrained)
test_file = "/home/tuenguyen/speech/speech_dia/Vin_localmeeting/trim_12s.wav"
vad_probability = inference(test_file)
# print(vad_probability.frames)
print(vad_probability.__dict__)

figure, ax = plt.subplots()
print(dir(notebook))
notebook.plot_feature(vad_probability, ax=ax)

# save to file
figure.savefig('annotation.png')

