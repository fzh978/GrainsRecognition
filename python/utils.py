#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
获取归一化颜色分量，属于初级性能，没有太高的识别率容易受光线影响
归一化后只需要2个特征量，因为B=1-G-R
'''

import cv2
import numpy as np
from class_preprocess import my_Preprocess
import math

# 输入彩色和二值化图像即可
def get_Hue(input_img, input_img_bin):
	total_valid_pixels = 0
	sum_H=0
	height, width = input_img_bin.shape
	img_hsv=cv2.cvtColor(input_img,cv2.COLOR_BGR2HSV)
	
	# 遍历像素
	for y in range(height):
		for x in range(width):
			if input_img_bin[y, x] > 0:
				total_valid_pixels += 1
				sum_H+=img_hsv[y,x][0]
		
	sum_H_avg=float(sum_H)/total_valid_pixels    #在opencv中H为实际H/2
	# 实质上将0-180值范围映射到cos(2pi*2*sum_H_avg/360)和cos(2pi*2*sum_H_avg/360)
	sum_H_avg_periodically=math.pi * sum_H_avg / 90

	return [(1+math.cos(sum_H_avg_periodically))/2,(1+math.sin(sum_H_avg_periodically))/2]  #归一化到（0，1） sin貌似更好

# 线性标准化，返回list
def normalize_from_list(input_list):
	len_of_line = len(input_list[0])
	res = []
	print len(input_list)
	for row in input_list:
		a1 = np.asarray(row, dtype=np.float64)
		a2 = np.zeros((len_of_line,), dtype=np.float64)
		# 标准化到[-1,+1]
		cv2.normalize(a1, a2, -1.0, 1.0, cv2.NORM_MINMAX)
		res.append(a2.tolist())
		
	return res
	
	
if __name__ == '__main__':
#	prefix="lvdou"
#	for i in range(1,7):
#		pic_prep = my_Preprocess("../data/grains/"+prefix+"/"+str(i)+".jpg")
#		print get_Hue(pic_prep.get_img(), pic_prep.get_img_binary())
	
	a = [[1, 2, 3,4], [3, 4, 5], [6, 7, 8]]
	print normalize_from_list(a)
	
