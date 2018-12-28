# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:46:02 2018

@author: jbk48
"""

import tensorflow as tf
import CycleGAN
import os


cyclegan = CycleGAN.CycleGAN("apple2orange")
cyclegan.train(epochs=200, batch_size=1, sample_interval=200)
