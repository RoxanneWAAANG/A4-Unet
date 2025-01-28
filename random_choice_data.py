# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:59:02 2023

@author: Pilot Crysi
"""

import os
import math
import shutil
import random

train_hq = '/home/fyp1/carvana-image-masking-challenge/train_hq/'
train_masks = '/home/fyp1/carvana-image-masking-challenge/train_masks/'

test_kfold = '/home/fyp1/carvana-image-masking-challenge/test_kfold/'
test_kfold_masks = '/home/fyp1/carvana-image-masking-challenge/test_kfold_masks/'

percentage, train_list = 0.1, os.listdir(train_hq)

for i in range(math.floor(percentage * len(train_list))):
    name = random.choice(train_list)
    train_list.remove(name)
    shutil.move(train_hq + name, test_kfold + name)
    shutil.move(train_masks + name[0:-4] + '_mask.gif', test_kfold_masks + name[0:-4] + '_mask.gif')

