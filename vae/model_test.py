from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


vae = load_model('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\vae\\vae_mlp_mnist.h5')

test = 1