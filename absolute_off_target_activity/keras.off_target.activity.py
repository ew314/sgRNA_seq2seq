##### basic #######
import sys,os,re
import math
import time
import numpy as np
from numpy import array
import random
from random import randint
import scipy
from scipy import stats
from scipy.stats import chisquare,kstest,pearsonr,spearmanr
from sklearn import  linear_model,metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
stdsc=StandardScaler()
from sklearn import metrics
#######  tensorflow #########
import tensorflow as tf
#######  keras #########
import keras
from keras.layers import Input, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.constraints import max_norm
from keras.optimizers import RMSprop
####### matplotlib ######## 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig


on_filename='/data/Wt_data'
rel_filename='/data/Table_S8_machine_learning_input.txt'
model_save_loc='./model_save_loc'
figure_save_loc='./figure_save_loc'

os.system('mkdir %s %s'%(model_save_loc,figure_save_loc))





def binarize_sequence_single(sequence):
	arr = np.zeros((4, 23, 1))
	for i in range(23):
		if sequence[i] == 'A':
			arr[0][i][0] = 1
		if sequence[i] == 'C':
			arr[1][i][0]= 1
		if sequence[i] == 'G':
			arr[2][i][0]= 1
		if sequence[i] == 'T':
			arr[3][i][0]= 1
	return arr

def sequence_binarize_double(arr):
	a=''
	for i in range(0,23):
		if arr[0][i][0]==1:
			a=a+'A'
		if arr[1][i][0]==1:
			a=a+'C'
		if arr[2][i][0]==1:
			a=a+'G'
		if arr[3][i][0]==1:
			a=a+'T'
	return a

			
def random_fold(X,y,rate):
	indices=np.arange(len(X))
	np.random.shuffle(indices)
	X_train = X[int(len(indices)/rate):] 
	X_val = X[:int(len(indices)/rate)] 	
	y_train = y[int(len(indices)/rate):] 
	y_val = y[:int(len(indices)/rate)]
	return X_train,X_val,y_train,y_val


loc=os.getcwd()

X=[]
y=[]

f1=open(filename,'r')
m1=f1.readlines()
f1.close()
for i in range(0,len(m1)):
	p1=m1[i].strip().split('\t')
	if len(p1[0])==23:
		X.append(binarize_sequence_single(p1[0]))
		y.append(float(p1[1]))

X=np.array(X)
y=np.array(y)
X_train,X_val,y_train,y_val=random_fold(X,y,5)




#############  float weights ##################

sample_weights_f=[]
class_weights_f={}
class_number_f=[0,0,0,0,0]
for i in range(0,len(y_train)):
	if y_train[i] <= 0.2:
		class_number_f[0]=class_number_f[0]+1
	elif y_train[i] > 0.2 and y_train[i] <= 0.4 :
		class_number_f[1]=class_number_f[1]+1
	elif y_train[i] > 0.4 and y_train[i] <= 0.6 :
		class_number_f[2]=class_number_f[2]+1
	elif y_train[i] > 0.6 and y_train[i] <= 0.8 :
		class_number_f[3]=class_number_f[3]+1
	elif y_train[i] > 0.8 :
		class_number_f[4]=class_number_f[4]+1
		
class_bin_f=[1/float(class_number_f[0]),1/float(class_number_f[1]),1/float(class_number_f[2]),1/float(class_number_f[3]),1/float(class_number_f[4])]

class_weights_f['0']=class_bin_f[0]/sum(class_bin_f)
class_weights_f['1']=class_bin_f[1]/sum(class_bin_f)
class_weights_f['2']=class_bin_f[2]/sum(class_bin_f)
class_weights_f['3']=class_bin_f[3]/sum(class_bin_f)
class_weights_f['4']=class_bin_f[4]/sum(class_bin_f)	

for i in range(0,len(y_train)):
	if y_train[i] <= 0.2:
		sample_weights_f.append(class_weights_f['0'])
	elif y_train[i] > 0.2 and y_train[i] <= 0.4 :
		sample_weights_f.append(class_weights_f['1'])
	elif y_train[i] > 0.4 and y_train[i] <= 0.6 :
		sample_weights_f.append(class_weights_f['2'])
	elif y_train[i] > 0.6 and y_train[i] <= 0.8 :
		sample_weights_f.append(class_weights_f['3'])
	elif y_train[i] > 0.8  :
		sample_weights_f.append(class_weights_f['4'])
sample_weights_f=np.array(sample_weights_f)


## DNA_input
model_float = Sequential()
model_float.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', input_shape=(4,23,1), data_format='channels_last', name = 'dense_1'))
model_float.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', data_format='channels_last', name = 'dense_2'))
model_float.add(MaxPool2D(pool_size=(1,2), padding='same', data_format='channels_last', name = 'dense_3'))
model_float.add(Dropout(0.25))
model_float.add(Flatten())
model_float.add(Dense(units=256, activation='sigmoid', name = 'dense_4'))
model_float.add(Dropout(0.25))
model_float.add(Dense(units=128, activation='sigmoid', name = 'float_dense_5'))
model_float.add(Dropout(0.25))
model_float.add(Dense(1, activation='linear',name = 'float_dense_7'))
model_float.compile(loss='logcosh', metrics=['mse'], optimizer='adam')

epoch=64
model_history = model_float.fit(X_train,y_train.ravel(),sample_weight=np.array(sample_weights_f),batch_size=256,epochs=epoch,validation_data=(X_val, y_val.ravel()))


fig,ax = plt.subplots(figsize=(5,5))
plt.xlim(-0.5,2.5)
plt.ylim(-0.5,2.5)
ax.scatter(model_float.predict(X_val), y_val, marker='.', alpha=.2)
ax.set_xlabel('predicted activity')
ax.set_ylabel('measured activity')
ax.set_title('NC.WT.performance on validation set');
xx=1.5
yy=0.8
a,b,r_value,p_value,std_err=stats.linregress(model_float.predict(X_val).ravel(),y_val.ravel())   
t=plt.text(1.4,0.8,'Y=%sX+%s'%(str(a)[:6],str(b)[:6]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)
t=plt.text(1.4,0.6,'P=%s'%(str(p_value)[:8]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)
t=plt.text(1.4,0.4,'R2=%s'%(str(r_value*r_value)[:6]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)							
fig.savefig('%s/on_target.validation.png'%figure_save_loc, dpi=300)
plt.close('all')
plt.clf()	

mp = "%s/NC_WT_float_model.h5"%model_save_loc
model_float.save(mp)







def binarize_sequence_double(sequence1,sequence2):
	arr = np.zeros((4, 22, 2))
	for i in range(22):
		if sequence1[i] == 'A':
			arr[0][i][0] = 1
		if sequence1[i] == 'C':
			arr[1][i][0]= 1
		if sequence1[i] == 'G':
			arr[2][i][0]= 1
		if sequence1[i] == 'T':
			arr[3][i][0]= 1
			
		if sequence2[i] == 'A':
			arr[0][i][1] = 1
		if sequence2[i] == 'C':
			arr[1][i][1]= 1
		if sequence2[i] == 'G':
			arr[2][i][1]= 1
		if sequence2[i] == 'T':
			arr[3][i][1]= 1	
	return arr

def sequence_binarize_double(arr):
	a=''
	b=''
	for i in range(0,22):
		if arr[0][i][0]==1:
			a=a+'A'
		if arr[1][i][0]==1:
			a=a+'C'
		if arr[2][i][0]==1:
			a=a+'G'
		if arr[3][i][0]==1:
			a=a+'T'
		if arr[0][i][1]==1:
			b=b+'A'
		if arr[1][i][1]==1:
			b=b+'C'
		if arr[2][i][1]==1:
			b=b+'G'
		if arr[3][i][1]==1:
			b=b+'T'		
	return [a,b]
		
	
def random_fold(X,y,rate):
	indices=np.arange(len(X))
	np.random.shuffle(indices)
	X_train = X[int(len(indices)/rate):] 
	X_val = X[:int(len(indices)/rate)] 	
	y_train = y[int(len(indices)/rate):] 
	y_val = y[:int(len(indices)/rate)]
	return X_train,X_val,y_train,y_val




############  WT ###############
WT_float_predict = load_model("%s/NC_WT_float_model.h5"%model_save_loc)


DNA_set=[]
f1=open(rel_filename,'r')
m1=f1.readlines()
f1.close()
for i in range(1,len(m1)):
	p1=m1[i].strip().split('\t')
	DNA=p1[9][2:25]
	DNA_set.append(DNA)


DNA_set=list(set(DNA_set))
X=[]
for DNA in DNA_set:
	X.append(binarize_sequence_single(DNA))
X=np.array(X)
WT_score=list(WT_float_predict.predict(X).ravel())

WT_DNA_score={}
for i in range(0,len(DNA_set)):
	WT_DNA_score[DNA_set[i]]=WT_score[i]


f1=open('%s/NBT.DNA.pre.ab.act.txt'%model_save_loc,'w')
f1.write('DNA')
f1.write('\t')
f1.write('sgRNA')
f1.write('\t')
f1.write('rel act')
f1.write('\t')
f1.write('WT_DNA_act')
f1.write('\t')
f1.write('WT_sgRNA_act')
f1.write('\n')
for i in range(1,len(m1)):
	p1=m1[i].strip().split('\t')
	DNA=p1[9][2:25]
	sgRNA=p1[10][2:25]
	rel=float(p1[8])
	WT_dna=WT_DNA_score[DNA]
	WT_sgrna=WT_dna*rel
	f1.write(DNA)
	f1.write('\t')
	f1.write(sgRNA)
	f1.write('\t')
	f1.write(str(rel))
	f1.write('\t')
	f1.write(str(WT_dna))	
	f1.write('\t')
	f1.write(str(WT_sgrna))
	f1.write('\n')

for seq in DNA_set:
	DNA=seq
	sgRNA=seq
	rel=float(1)
	WT_dna=WT_DNA_score[DNA]
	WT_sgrna=WT_dna*rel
	f1.write(DNA)
	f1.write('\t')
	f1.write(sgRNA)
	f1.write('\t')
	f1.write(str(rel))
	f1.write('\t')
	f1.write(str(WT_dna))	
	f1.write('\t')
	f1.write(str(WT_sgrna))
	f1.write('\n')
f1.close()






filename='%s/NBT.DNA.pre.ab.act.txt'%model_save_loc
f1=open(filename,'r')
m1=f1.readlines()
f1.close()
X=[]
y=[]
DNA_act={}
DNA_set=[]
DNA_sgRNA_rel={}
for i in range(1,len(m1)):
	p1=m1[i].strip().split('\t')
	DNA=p1[0][1:]
	sgRNA=p1[1][1:]
	DNA_set.append(DNA)
	DNA_act[DNA]=float(p1[3])
	X.append(binarize_sequence_double(DNA,sgRNA))
	y.append(float(p1[4]))
	DNA_sgRNA_rel[DNA+';'+sgRNA]=float(p1[2])
		
X=np.array(X)
y=np.array(y)
X_train,X_val,y_train,y_val=random_fold(X,y,5)



sample_weights=[]
class_weights={}
class_number=[0,0,0,0,0,0]
for i in range(0,len(y_train)):
	if y_train[i] < 0.2:
		class_number[0]=class_number[0]+1
	elif y_train[i] >= 0.2 and y_train[i] < 0.4 :
		class_number[1]=class_number[1]+1
	elif y_train[i] >= 0.4 and y_train[i] < 0.6 :
		class_number[2]=class_number[2]+1
	elif y_train[i] >= 0.6 and y_train[i] < 0.8 :
		class_number[3]=class_number[3]+1
	elif y_train[i] >= 0.8 and y_train[i] < 1.0 :
		class_number[4]=class_number[4]+1
	elif y_train[i] >= 1.0:
		class_number[5]=class_number[5]+1


class_bin=[1/float(class_number[0]),1/float(class_number[1]),1/float(class_number[2]),1/float(class_number[3]),1/float(class_number[4]),1/float(class_number[5])]


class_weights['0']=class_bin[0]/sum(class_bin)
class_weights['1']=class_bin[1]/sum(class_bin)
class_weights['2']=class_bin[2]/sum(class_bin)
class_weights['3']=class_bin[3]/sum(class_bin)
class_weights['4']=class_bin[4]/sum(class_bin)	
class_weights['5']=class_bin[5]/sum(class_bin)

print(class_weights)
	
for i in range(0,len(y_train)):
	if y_train[i] < 0.2:
		sample_weights.append(class_weights['0'])
	elif y_train[i] >= 0.2 and y_train[i] < 0.4 :
		sample_weights.append(class_weights['1'])
	elif y_train[i] >= 0.4 and y_train[i] < 0.6 :
		sample_weights.append(class_weights['2'])
	elif y_train[i] >= 0.6 and y_train[i] < 0.8 :
		sample_weights.append(class_weights['3'])
	elif y_train[i] >= 0.8 and y_train[i] < 1.0 :
		sample_weights.append(class_weights['4'])
	elif y_train[i] >= 1.0:
		sample_weights.append(class_weights['5'])	

sample_weights=np.array(sample_weights)

model_float = Sequential()
model_float.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', input_shape=(4,22,2), data_format='channels_last', name = 'dense_1'))
model_float.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', data_format='channels_last', name = 'dense_2'))
model_float.add(MaxPool2D(pool_size=(1,2), padding='same', data_format='channels_last', name = 'dense_3'))
model_float.add(Dropout(0.25))
model_float.add(Flatten())
model_float.add(Dense(units=128, activation='sigmoid', name = 'dense_4'))
model_float.add(Dropout(0.25))
model_float.add(Dense(units=256, activation='sigmoid', name = 'float_dense_5'))
model_float.add(Dropout(0.25))
model_float.add(Dense(1, activation='linear',name = 'float_dense_7'))
model_float.compile(loss='logcosh', metrics=['logcosh'], optimizer='adam')


batch_size=256
epoch=64
model_history = model_float.fit(X_train,y_train.ravel(), batch_size=batch_size, epochs=epoch, validation_data=(X_val, y_val.ravel()))



fig,ax = plt.subplots(figsize=(5,5))
plt.xlim(-0.5,2.5)
plt.ylim(-0.5,2.5)

ax.scatter(model_float.predict(np.array(X_val)).ravel(), y_val.ravel(), marker='.', alpha=.2)
ax.set_xlabel('predicted activity / match score')
ax.set_ylabel('measured activity')
ax.set_title('rel act off_target performance on validation set');
xx=1.5
yy=0.8
a,b,r_value,p_value,std_err=stats.linregress(model_float.predict(np.array(X_val)).ravel(), y_val.ravel())   
t=plt.text(1.4,0.8,'Y=%sX+%s'%(str(a)[:6],str(b)[:6]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)
t=plt.text(1.4,0.6,'P=%s'%(str(p_value)[:8]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)
t=plt.text(1.4,0.4,'R2=%s'%(str(r_value*r_value)[:6]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)							
fig.savefig('%s/off_target_ab.validation.png'%figure_save_loc, dpi=300)
plt.close('all')
plt.clf()	






rel_score_set=[]	
rel_class_set=[]
rel_act=[]

y_val=[]
for i in range(0,len(X_val)):
	arr=X_val[i]
	[DNAseq,sgRNA]=sequence_binarize_double(arr)
	Xaft=[binarize_sequence_double(DNAseq,sgRNA)]
	Xaft=np.array(Xaft)
	Xord=[binarize_sequence_double(DNAseq,DNAseq)]
	Xord=np.array(Xord)		
	aft_score=float(model_float.predict(Xaft).ravel()[0])
	ord_score=float(model_float.predict(Xord).ravel()[0])
	rel_score_set.append(aft_score/ord_score)	
	tp_val=DNA_sgRNA_rel[DNAseq+';'+sgRNA]
	rel_act.append(tp_val)




fig,ax = plt.subplots(figsize=(5,5))
plt.xlim(-0.5,2.5)
plt.ylim(-0.5,2.5)

ax.scatter(rel_score_set, rel_act, marker='.', alpha=.2)
ax.set_xlabel('predicted mismatch activity / predicted matched activity')
ax.set_ylabel('measured relative activity')
ax.set_title('rel act off_target performance on validation set');
xx=1.5
yy=0.8
a,b,r_value,p_value,std_err=stats.linregress(rel_score_set, rel_act)   
t=plt.text(1.4,0.8,'Y=%sX+%s'%(str(a)[:6],str(b)[:6]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)
t=plt.text(1.4,0.6,'P=%s'%(str(p_value)[:8]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)
t=plt.text(1.4,0.4,'R2=%s'%(str(r_value*r_value)[:6]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)							
fig.savefig('%s/off_target_rel.validation.png'%figure_save_loc, dpi=300)
plt.close('all')
plt.clf()	

                    
mp = "%s/NBT_AB_float_model.h5"%model_save_loc
model_float.save(mp)


