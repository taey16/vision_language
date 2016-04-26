# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

log_filename = \
  '/storage/attribute/checkpoints/tshirts_shirts_blous_knit_jacket_onepiece_skirts_coat_cardigan_vest_pants_leggings_shoes_bags_swimwears_hat_panties_bra_801544_40000_seq_length14/resception_ep29_bs16_flipfalse_croptrue_original_init_gamma0.100000_lstm_tanh_hid512_lay2_drop2.000000e-01_adam_lr1.000000e-03_seed0.90_start541152_every45096_finetune0_cnnlr1.000000e-03_cnnwc1.000000e-05_retrain_iter0/'

f = file('%s/tsne.txt' % log_filename, 'r').read().split('\n')


matplotlib.rc('font',family='AppleGothic')
plt.figure(figsize=(16,16));
datax = []
datay = []
for line in f[:-1]:
  label, x, y = line.split()
  x = float(x)
  y = float(y)
  datax.append(x)
  datay.append(y)
  plt.annotate(label.decode('utf-8'), xy = (x, y), xytext = (0, 0), textcoords = 'offset points')

plt.scatter(datax, datay, color='white');
plt.grid(True);
plt.savefig('tsne.pdf')

"""
import numpy as np
import matplotlib.pyplot as plt
#from nltk.corpus import stopwords
#STOPWORDS = stopwords.words('english')

f = file('tsne.txt', 'r').read().split('\n')

datax = []
datay = []
for line in f:
  label, x, y = line.split()
  #if label in STOPWORDS: continue
  x = float(x)
  y = float(y)
  datax.append(x)
  datay.append(y)
  plt.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points')

plt.scatter(datax, datay, color='white')
plt.savefig('tsne.pdf')
# plt.show()
"""
