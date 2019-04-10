#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np


res_val_list = []
hg_val_list = []
pyra_val_list = []

res_log = 'cow/res/resnet-log.txt'
hg_log = 'cow/hg-s2-b1/log.txt'
pyra_log = 'cow/res/pyranet-log.txt'

with open(res_log, 'r') as f1:
    for line in f1.readlines():
        if line[0] != 'E':
            res_val_list.append(float(line.split('\t')[-2]))

with open(hg_log, 'r') as f2:
    for line in f2.readlines():
        if line[0] != 'E':
            hg_val_list.append(float(line.split('\t')[-2]))

with open(pyra_log, 'r') as f3:
    for line in f3.readlines():
        if line[0] != 'E':
            pyra_val_list.append(float(line.split('\t')[-2]))

x = np.arange(1, 201)

res_val_list[-1] = 0.572131
y = np.array(res_val_list)
y_hg = np.array(hg_val_list)

pyra_val_list[-1] = 0.872131
y_pyra = np.array(pyra_val_list)


plt.xlabel('iteration times')
plt.ylabel('accuracy')

plt.plot(x, y_hg, color='red', label='Hourglass')
plt.plot(x, y, color='skyblue', label='ResNet18')
plt.plot(x, y_pyra, color='green', label='PyraNet')
plt.legend()
plt.show()

print(y_pyra)
