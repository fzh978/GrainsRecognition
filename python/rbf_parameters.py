#!/usr/bin/env python
# -*- coding: utf-8 -*-

from class_features import my_Features
from class_svm import my_SVM
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

svm = my_SVM()
# svm_param_C, svm_param_gamma = 2.82842712475, 0.25

# trained_model_filename = "../data/cv2_svm_model.xml"

features_train = my_Features('../data/grain_list.csv', '../data/grain_features.csv')
#  features_train.load_itemlist()
#  features_train.save_features()
features_train.load_saved_features()

train_y, train_x = features_train.get_features_y_x()

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
cv = ShuffleSplit(n_splits=10, test_size=0.02, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(train_x, train_y)
for train_index, test_index in cv.split(train_x, train_y):
      print("TRAIN:", train_index, "TEST:", test_index)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#train_number = len(train_x)

# opencv only support np.array array
#train_x = np.asarray(train_x, np.float32)
#train_y = np.asarray(train_y, np.float32)


