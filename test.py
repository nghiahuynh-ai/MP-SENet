from models.generator import MPNet
import torch
import json
import time
from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft


with open('config.json') as f:
    data = f.read()
    
global h
json_config = json.loads(data)
h = AttrDict(json_config)

net = MPNet(h)
print(net)
print(sum(p.numel() for p in net.parameters()))

x = torch.rand(1, 16000)
print(x.shape)
noisy_amp, noisy_pha, noisy_com = mag_pha_stft(x, 400, 100, 400, 0.3)
print(noisy_amp.shape)
s = time.time()
amp_g, pha_g, com_g = net(noisy_amp, noisy_pha)
e = time.time()
audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
print(audio_g.shape)
print(e-s)