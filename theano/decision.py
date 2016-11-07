__author__ = 'peichanghua(me@zhuzhusir.com) @ Tsinghua 4rd July '
import os
import urllib
import traceback
import sys
import commands
import socket
import time
import glob
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn import ensemble
import pydot
from IPython.display import Image
from numpy import *
import numpy as np
from scipy.stats import itemfreq
import random
########## reference
# http://scikit-learn.org/dev/modules/tree.html#tree

FAST = 0
SLOW = 1
VERY_SLOW = 2
TH_1 = 12
TH_2 = 50
#>>> iris.feature_names
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
F = ['T_rx','T_tx','retry-ratio','rssi','rate_tx','rate_rx','au','au_inf','compete','q']
#F = ['T_rx','T_tx','retry-ratio','rssi','rate_tx','rate_rx','au']
#F = ['retry-ratio','rssi','rate_tx','rate_rx','au']

C =array(['FAST','SLOW','VERY SLOW'],dtype='S10')
C1 =['FAST','SLOW','VERY_SLOW']
N=10
def shuffle(X,Y):
	X_s = []
	Y_s = []
	if len(X) != len(Y):
		print "length mismatch"
		return
	indexes = range(len(X))
	random.shuffle(indexes)
	for index in indexes:
		X_s.append(X[index])
		Y_s.append(Y[index])
	return X_s,Y_s

def classify(y):
	if float(y) < TH_1:
		return FAST
	else:
		return SLOW
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
	return float(tp+tn)/float(num),float(tp)/float(tp + fn),float(fp)/float(fp+tn)
def balanced_subsample(x,y,subsample_size=1.0):

	class_xs = []
	min_elems = None

	for yi in np.unique(y):
		elems = x[(y == yi)]
		class_xs.append((yi, elems))
		if min_elems == None or elems.shape[0] < min_elems:
			min_elems = elems.shape[0]

	use_elems = min_elems
	if subsample_size < 1:
		use_elems = int(min_elems*subsample_size)

	xs = []
	ys = []

	for ci,this_xs in class_xs:
		if len(this_xs) > use_elems:
			np.random.shuffle(this_xs)

		x_ = this_xs[:use_elems]
		y_ = np.empty(use_elems)
		y_.fill(ci)

		xs.append(x_)
		ys.append(y_)

	xs = np.concatenate(xs)
	ys = np.concatenate(ys)

	return xs,ys
if __name__ == '__main__':
	# input is in dataTmp/result/down.txt and up.txt
	depth = int(sys.argv[1])
	msl = int(sys.argv[2])
	w= float(sys.argv[3])
	HOME_DIR= "/home/viki/capstoneProject/theano/"
	#for subdir in ["up/6CB0CE0FB9CF-303a6496d264"]:
	for subdir in ["up.txt"]:
		file_name_origin = HOME_DIR+"/"+subdir

		X=[]
		Y=[]
		for line in open(file_name_origin,"r"):
			values = line.strip('\n').split(',')
			if values[0] != '': # para
				per_x_list=[]
				#throughput = float(values[2])
				#if throughput < 20000: # 20KBps
				X.append(values[4:9])
			else:
				delays = values[1:]
				delays.sort(cmp=lambda x,y : cmp(float(x),float(y)))
				#delay_avg = np.average(np.asarray(delays,dtype='float'))
				delay_75per = delays[int(0.75*len(delays))]
				delay_binned=classify(delay_75per)
				Y.append(delay_binned)


		############################ pick the train and test part
		all_points = len(Y)
		X_s,Y_s = shuffle(X,Y)
		#X_s = X
		#Y_s = Y
		sample_len = int(len(Y)/N)
		X_test = X_s[:sample_len]
		Y_test = Y_s[:sample_len]
		X_train = X_s[sample_len:]
		Y_train = Y_s[sample_len:]

		############################ balance the class
		X_balance,Y_balance = balanced_subsample(np.asarray(X_train),np.asarray(Y_train),1)
		# X_balance= X_train
		# Y_balance = Y_train
		#print "X=%s"%(str(X_balance))
		# print "Y=%s"%(str(Y_balance))
		print "train dataset size=%d"%(len(X_balance))
		############################# train the modals
		weight = {0:w,1:1-w}
		clf = tree.DecisionTreeClassifier(max_depth=depth, criterion='entropy',min_samples_leaf=msl,class_weight=weight)
		#clf = tree.DecisionTreeClassifier(max_depth=8, criterion='entropy')
		#weight = {0:w,1:1-w}
		#clf = ensemble.RandomForestClassifier(n_estimators=tree_num,class_weight=weight,max_depth=depth,criterion='entropy')
		clf.fit(X_balance, Y_balance)
		dot_data = StringIO()
		tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=F,
                         class_names=C,
                         filled=True, rounded=True,
                         special_characters=True)
		graph = pydot.graph_from_dot_data(dot_data.getvalue())
		out_file = "%d_down_decision.png"%(depth)
		graph.write_png(out_file)


		########### cross validation of the predication

		Y_true = np.asarray(Y_test)
		Y_pre = clf.predict(X_test)

		np.set_printoptions(threshold='nan')
		#print "Y_true=%s"%(str(Y_true))
		#print "Y_pre=%s"%(str(Y_pre))
		#print itemfreq(Y_true)
		#print itemfreq(Y_pre)
		accuracy,TPR,FPR= cal_accuracy(Y_true,Y_pre)
		print "%d: ACC=%f, TPR=%f, FPR=%f"%(depth,accuracy,TPR, FPR)
		#dot_data = StringIO()
		#tree.export_graphviz(clf, out_file=dot_data)
		#graph = pydot.graph_from_dot_data(dot_data.getvalue())
		#out_file = "%s.pdf"%(subdir.split('.')[0])
		#graph.write_pdf(out_file)
