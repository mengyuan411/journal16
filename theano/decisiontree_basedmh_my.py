import numpy as np
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T
#from mp import Layer, MLP, gradient_updates_momentum
from sklearn.cross_validation import train_test_split
from upsampling import upsample
from cps_generator import cps_generator
import os
import urllib
import traceback
import socket
import sys
import time
import glob
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn import ensemble
from IPython.display import Image
from numpy import *
import numpy as np
from scipy.stats import itemfreq
import random


FAST = 0
SLOW = 1
def cal_accuracy(Y_true,Y_pre):
    if len(Y_true) != len(Y_pre):
        print "Y_true and Y_pre 's length are mismatch!"
    num = len(Y_true)
    no = 0
    matched = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    # fast : +  slow : -
    for i in Y_true:
        if i == FAST and Y_pre[no] == FAST:
            tp = tp + 1
        elif i == SLOW and Y_pre[no] == FAST:
            fp = fp + 1
        elif i == FAST and Y_pre[no] == SLOW:
            fn = fn + 1
        elif i == SLOW and Y_pre[no] == SLOW:
            tn = tn + 1
        else:
            a =1 # do nothing
        no = no + 1
    return float(tp+tn)/float(num),float(tp)/float(tp + fn),float(fp)/float(fp+tn),float(tn)/float(fp+tn),float(fn)/float(tp+fn)



#w=float(sys.argv[1])
#depth=float(sys.argv[2])
#msl=float(sys.argv[3])
tree_num=int(sys.argv[1])
leaf_size=int(sys.argv[2])

attributes = []
labels = []
with open('../scikit-cake/test2/attris_out.txt','r') as f, open('../scikit-cake/test2/labels_out.txt','r') as g:
  attributes_np = np.load(f)
  labels_np = np.load(g)
  print(len(list(attributes_np)))
  print(len(list(labels_np))) 
f.close()
g.close()
labels_new = []
attributes_new = []
timestamps = []
for elem in labels_np:
  if elem >= 0.0125:
    labels_new.append(0)
  else:
    labels_new.append(1)
#for elem in attributes_np:
#  timestamps.append(elem[7])
#start = min(timestamps)
#for elem in attributes_np:
#  elem[7] = str((int(elem[7])-int(start))%86400)
labels_np_new = np.array(labels_new)
#bias = np.ones((attributes_np.shape[0],1))
#attributes_final = np.concatenate((attributes_np,bias),axis=1)
X_train, X_test, y_train, y_test = train_test_split(attributes_np.astype(theano.config.floatX), labels_np_new.astype(theano.config.floatX), test_size=0.1, random_state=88)
X_train,y_train,N = upsample((X_train,y_train),(0,1),0)
#X_train,y_train = cps_generator((X_train,y_train),([1.0,0.0],[0.0,1.0]))
print "train dataset size=%d"%(len(X_train))
        ############################# train the modals
#weight = {0:w,1:1-w}
#clf = tree.DecisionTreeClassifier(max_depth=depth, criterion='entropy',min_samples_leaf=msl,class_weight=weight)
        #clf = tree.DecisionTreeClassifier(max_depth=8, criterion='entropy')
        #weight = {0:w,1:1-w}
        #clf = ensemble.RandomForestClassifier(n_estimators=tree_num,class_weight=weight,max_depth=depth,criterion='entropy')
clf = ensemble.RandomForestClassifier(n_estimators=tree_num,oob_score=True,n_jobs=-1,max_features="auto",min_samples_leaf=leaf_size,criterion='entropy')
clf.fit(X_train, y_train)
 

Y_true = np.asarray(y_test)
Y_pre = clf.predict(X_test)

np.set_printoptions(threshold='nan')
accuracy,TPR,FPR,TNR,FNR= cal_accuracy(Y_true,Y_pre)
#print("%d: ACC=%f, TPR=%f, FPR=%f,TNR=%f,FNR=%f"%(depth,accuracy,TPR, FPR,TNR,FNR))
print("%d,%d: ACC=%f, TPR=%f, FPR=%f,TNR=%f,FNR=%f"%(tree_num,leaf_size,accuracy,TPR, FPR,TNR,FNR))

print(classification_report(Y_true,Y_pre))


#print('N:'+str(N))





