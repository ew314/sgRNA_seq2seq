##### basic #######
import sys,os,re
import time
import warnings
from tqdm import tqdm
from PIL import Image
import gzip
import math
import numpy as np
from numpy import array
import random
from random import randint
#######  scipy #########
import scipy
from scipy import stats
from scipy.stats import chisquare,kstest,pearsonr,spearmanr
from scipy.stats import ttest_ind as ttest
from sklearn import  linear_model,metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc as sklearn_auc
from sklearn.preprocessing import StandardScaler
import pickle
stdsc=StandardScaler()
warnings.filterwarnings('ignore')
#######  sklearn #########
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
#######  tensorflow #########
import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper,LuongAttention,AttentionWrapper,BahdanauAttention
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.util import nest
#######  keras #########
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
####### matplotlib ######## 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig



feedback_data_pre_result='./data/feedback_data.txt'
validation_data_pre_result='./data/vaildation_data.txt'
model_save_loc='./model_save_loc'
figure_save_loc='./figure_save_loc'

os.system('mkdir %s %s'%(model_save_loc,figure_save_loc))

def RGB(n):
	if n>9:
		n=n-10+65
		m=chr(n)
	else:
		m=str(n)
	return m

#(255,0,0) -> 'r'
#int->color
def color(R,G,B):
	RGB101=int(R)
	RGB102=int(G)
	RGB103=int(B)
	n11=RGB101/16
	n11=int(n11)
	n12=RGB101-16*n11
	n21=RGB102/16
	n21=int(n21)
	n22=RGB102-16*n21
	n31=RGB103/16
	n31=int(n31)
	n32=RGB103-16*n31
	p1=RGB(n11)
	p2=RGB(n12)
	p3=RGB(n21)
	p4=RGB(n22)
	p5=RGB(n31)
	p6=RGB(n32)
	p=p1+p2+p3+p4+p5+p6
	return p

def Confusion_Matrix_softmax(label,result):
	label_af=[]
	for tp in label:
		label_af.append(list(tp).index(1))
	label=np.array(label_af)
	result=np.array(result)
	TP=float(list(label+result).count(2))
	FN=float(list(label-result).count(1))
	FP=float(list(label-result).count(-1))
	TN=float(list(label+result).count(0))
	try:
		acc=(TP+TN)/(TP+FN+FP+TN)
	except:
		acc=0
	try:
		pre=(TP)/(TP+FP)
	except:
		pre=0
	try:
		tpr=(TP)/(TP+FN)
	except:
		tpr=0
	try:
		tnr=(TN)/(TN+FP)
	except:
		tnr=0
	try:
		F1=(2*pre*tpr)/(pre+tpr)
	except:
		F1=0
	return acc,pre,tpr,tnr,F1
	
def AUC(pre_score,label):
	y_true=[]
	y_score=[]
	for i in range(0,len(pre_score)):
		y_true.append(label[i][0])
		y_score.append(pre_score[i][0])
	fpr, tpr, _ = metrics.roc_curve(y_true,y_score,pos_label=1)	
	auc=metrics.auc(fpr, tpr)	
	return auc	


def binarize_sequence_single(sequence1):
	arr = np.zeros((4,22,1))
	for i in range(22):
		if sequence1[i] == 'A':
			arr[0][i] = 1
		if sequence1[i] == 'C':
			arr[1][i]= 1
		if sequence1[i] == 'G':
			arr[2][i]= 1
		if sequence1[i] == 'T':
			arr[3][i]= 1
	return arr


	
	
	

loc=os.getcwd()

X_data=[[],[]]
y_data=[[],[]]
f1=open(feedback_data_pre_result,'r')
m1=f1.readlines()
f1.close()
for i in range(1,len(m1)):
	p1=m1[i].strip().split('\t')
	if p1[0] != p1[1]:
		X_data[0].append(binarize_sequence_single(p1[1]))
		y_data[0].append(np.array([1,0]))
	if p1[0] == p1[1] :
		X_data[0].append(binarize_sequence_single(p1[1]))
		y_data[0].append(np.array([0,1]))		


f1=open(validation_data_pre_result,'r')
m1=f1.readlines()
f1.close()
for i in range(1,len(m1)):
	p1=m1[i].strip().split('\t')
	if p1[0] != p1[1]:
		X_data[1].append(binarize_sequence_single(p1[1]))
		y_data[1].append(np.array([1,0]))
	if p1[0] == p1[1] :
		X_data[1].append(binarize_sequence_single(p1[1]))
		y_data[1].append(np.array([0,1]))		


X_train=np.array(X_data[0])
y_train=np.array(y_data[0])
X_val=np.array(X_data[1])
y_val=np.array(y_data[1])

#############  classfied weights ##################


sample_weights_c=[]
class_weights_c={}
class_number_c=[0,0]
for i in range(0,len(y_train)):
	if y_train[i][0] == 1:
		class_number_c[1]=class_number_c[1]+1
	else:
		class_number_c[0]=class_number_c[0]+1

print(class_number_c)

class_bin_c=[1/float(class_number_c[0]),1/float(class_number_c[1])]

class_weights_c['0']=class_bin_c[0]/sum(class_bin_c)
class_weights_c['1']=class_bin_c[1]/sum(class_bin_c)


print(class_weights_c)
	
for i in range(0,len(y_train)):
	if y_train[i][0] ==1:
		sample_weights_c.append(class_weights_c['1'])
	else:
		sample_weights_c.append(class_weights_c['0'])

sample_weights_c=np.array(sample_weights_c)


model_class = Sequential()
model_class.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', input_shape=(4,22,1), data_format='channels_last', name = 'dense_1'))
model_class.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', data_format='channels_last', name = 'dense_2'))
model_class.add(MaxPool2D(pool_size=(1,2), padding='same', data_format='channels_last', name = 'dense_3'))
model_class.add(Dropout(0.25))
model_class.add(Flatten())
model_class.add(Dense(units=128, activation='sigmoid', name = 'dense_4'))
model_class.add(Dropout(0.25))
model_class.add(Dense(units=256, activation='sigmoid', name = 'class_dense_5'))
model_class.add(Dropout(0.25))
model_class.add(Dense(2, activation='softmax', name='class_dense_6'))
model_class.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam')



epoch=2
batch_size=1024
model_class.fit(X_train, y_train, batch_size=batch_size,sample_weight=np.array(sample_weights_c), epochs=epoch, validation_data=(X_val, y_val))
pre_score=model_class.predict(X_val)
pre_label=model_class.predict_classes(X_val)

acc,pre,tpr,tnr,F1=Confusion_Matrix_softmax(y_val,pre_label)
print('class','acc:',acc,'  pre:',pre,'  tpr:',tpr,'  tnr',tnr)
print('F1:',F1)

mp = "%s/train_by_feedback_model.h5"%model_save_loc
model_class.save(mp)
	

y_true=[]
y_score=[]
for i in range(0,len(pre_score)):
	y_true.append(y_val[i][0])
	y_score.append(pre_score[i][0])
fpr, tpr, thresholds_keras = metrics.roc_curve(y_true,y_score,pos_label=1)	
auc=metrics.auc(fpr, tpr)	
print("AUC : ", auc)
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='S3< val (AUC = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('%s/train_by_feedback.ROC.png'%figure_save_loc, dpi=300)
plt.close('all')
plt.clf()	



######## PR AUC  #################
lr_precision, lr_recall, _ = precision_recall_curve(y_true,y_score,pos_label=1)	
lr_auc=sklearn_auc(lr_recall, lr_precision)
print('PR-AUC=%.3f' %(lr_auc))
plt.figure(figsize=(5,5))
plt.plot(lr_recall, lr_precision,label='S3< val (AUC = {:.3f})'.format(lr_auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.savefig('%s/train_by_feedback.PR_AUC.png'%figure_save_loc, dpi=300)
plt.close('all')
plt.clf()			

lr_precision_feed=lr_precision
lr_recall_feed=lr_recall






X_train=np.array(X_data[1])
y_train=np.array(y_data[1])
X_val=np.array(X_data[0])
y_val=np.array(y_data[0])

#############  classfied weights ##################

sample_weights_c=[]
class_weights_c={}
class_number_c=[0,0]
for i in range(0,len(y_train)):
	if y_train[i][0] == 1:
		class_number_c[1]=class_number_c[1]+1
	else:
		class_number_c[0]=class_number_c[0]+1

print(class_number_c)

class_bin_c=[1/float(class_number_c[0]),1/float(class_number_c[1])]

class_weights_c['0']=class_bin_c[0]/sum(class_bin_c)
class_weights_c['1']=class_bin_c[1]/sum(class_bin_c)


print(class_weights_c)
	
for i in range(0,len(y_train)):
	if y_train[i][0] ==1:
		sample_weights_c.append(class_weights_c['1'])
	else:
		sample_weights_c.append(class_weights_c['0'])

sample_weights_c=np.array(sample_weights_c)


model_class = Sequential()
model_class.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', input_shape=(4,22,1), data_format='channels_last', name = 'dense_1'))
model_class.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', data_format='channels_last', name = 'dense_2'))
model_class.add(MaxPool2D(pool_size=(1,2), padding='same', data_format='channels_last', name = 'dense_3'))
model_class.add(Dropout(0.25))
model_class.add(Flatten())
model_class.add(Dense(units=128, activation='sigmoid', name = 'dense_4'))
model_class.add(Dropout(0.25))
model_class.add(Dense(units=256, activation='sigmoid', name = 'class_dense_5'))
model_class.add(Dropout(0.25))
model_class.add(Dense(2, activation='softmax', name='class_dense_6'))
model_class.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adam')


epoch=2
batch_size=1024
model_class.fit(X_train, y_train, batch_size=batch_size,sample_weight=np.array(sample_weights_c), epochs=epoch, validation_data=(X_val, y_val))
pre_score=model_class.predict(X_val)
pre_label=model_class.predict_classes(X_val)

acc,pre,tpr,tnr,F1=Confusion_Matrix_softmax(y_val,pre_label)
print('class','acc:',acc,'  pre:',pre,'  tpr:',tpr,'  tnr',tnr)
print('F1:',F1)

mp = "%s/train_by_validation_model.h5"%model_save_loc
model_class.save(mp)
	

y_true=[]
y_score=[]
for i in range(0,len(pre_score)):
	y_true.append(y_val[i][0])
	y_score.append(pre_score[i][0])
fpr, tpr, thresholds_keras = metrics.roc_curve(y_true,y_score,pos_label=1)	
auc=metrics.auc(fpr, tpr)	
print("AUC : ", auc)
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='S3< val (AUC = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('%s/train_by_validation.ROC.png'%figure_save_loc, dpi=300)
plt.close('all')
plt.clf()	


lr_precision, lr_recall, _ = precision_recall_curve(y_true,y_score,pos_label=1)	
lr_auc=sklearn_auc(lr_recall, lr_precision)
print('PR-AUC=%.3f' %(lr_auc))
plt.figure(figsize=(5,5))

plt.plot(lr_recall_feed, lr_precision_feed,'#63C960',label='feed (AUC = {:.3f})'.format(lr_auc))
plt.plot(lr_recall, lr_precision,'#1F77B4',label='val (AUC = {:.3f})'.format(lr_auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.savefig('%s/train_by_validation.PR_AUC.png'%figure_save_loc, dpi=300)
plt.close('all')
plt.clf()		