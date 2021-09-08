##### basic #######
import numpy as np
import random
from random import randint
import scipy
from scipy import stats
from scipy.stats import chisquare,kstest,pearsonr,spearmanr
from sklearn import  linear_model,metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc as sklearn_auc
from sklearn.preprocessing import StandardScaler
import pickle
stdsc=StandardScaler()
from sklearn import metrics
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
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

############  one-hot code  ##################

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
	
############# train data ################			
	


def random_seq(X,y,rate):
	indices=np.arange(len(X))
	np.random.shuffle(indices)
	X_train = X[indices[int(len(indices)/rate):]] 
	X_val = X[indices[:int(len(indices)/rate)]] 	
	y_train = y[indices[int(len(indices)/rate):]] 
	y_val = y[indices[:int(len(indices)/rate)]]

	return X_train,X_val,y_train,y_val
	



############# model check #####################


X=[]
y=[]
f1=open('./data/Table_S8_machine_learning_input.txt','r')
m1=f1.readlines()
f1.close()
DNAset=[]
for i in range(1,len(m1)):
	p1=m1[i].strip().split('\t')
	a=p1[9][3:25]
	b=p1[10][3:25]
	DNAset.append(a)
	X.append(binarize_sequence_double(a,b))
	y.append(float(p1[8]))

X=np.array(X)
y=np.array(y)
X_train,X_val,y_train,y_val=random_seq(X,y,5)


################# val save #################

f1=open('./val_seq','w')
for i in range(0,len(X_val)):
	arr=X_val[i]
	[DNAseq,sgRNA]=sequence_binarize_double(arr)
	f1.write(DNAseq)
	f1.write('\t')
	f1.write(sgRNA)
	f1.write('\t')
	f1.write(str(y_val[i]))
	f1.write('\n')
f1.close()

#############  float weights ##################

sample_weights_f=[]
class_weights_f={}
class_number_f=[0,0,0,0,0,0]
for i in range(0,len(y_train)):
	if y_train[i] <= 0.2:
		class_number_f[0]=class_number_f[0]+1
	elif y_train[i] > 0.2 and y_train[i] <= 0.4 :
		class_number_f[1]=class_number_f[1]+1
	elif y_train[i] > 0.4 and y_train[i] <= 0.6 :
		class_number_f[2]=class_number_f[2]+1
	elif y_train[i] > 0.6 and y_train[i] <= 0.8 :
		class_number_f[3]=class_number_f[3]+1
	elif y_train[i] > 0.8 and y_train[i] <= 1.0 :
		class_number_f[4]=class_number_f[4]+1
	elif y_train[i] > 1.0:
		class_number_f[5]=class_number_f[5]+1

print(class_number_f)

class_bin_f=[1/float(class_number_f[0]),1/float(class_number_f[1]),1/float(class_number_f[2]),1/float(class_number_f[3]),1/float(class_number_f[4]),1/float(class_number_f[5])]


class_weights_f['0']=class_bin_f[0]/sum(class_bin_f)
class_weights_f['1']=class_bin_f[1]/sum(class_bin_f)
class_weights_f['2']=class_bin_f[2]/sum(class_bin_f)
class_weights_f['3']=class_bin_f[3]/sum(class_bin_f)
class_weights_f['4']=class_bin_f[4]/sum(class_bin_f)	
class_weights_f['5']=class_bin_f[5]/sum(class_bin_f)

print(class_weights_f)
	
for i in range(0,len(y_train)):
	if y_train[i] <= 0.2:
		sample_weights_f.append(class_weights_f['0'])
	elif y_train[i] > 0.2 and y_train[i] <= 0.4 :
		sample_weights_f.append(class_weights_f['1'])
	elif y_train[i] > 0.4 and y_train[i] <= 0.6 :
		sample_weights_f.append(class_weights_f['2'])
	elif y_train[i] > 0.6 and y_train[i] <= 0.8 :
		sample_weights_f.append(class_weights_f['3'])
	elif y_train[i] > 0.8 and y_train[i] <= 1.0 :
		sample_weights_f.append(class_weights_f['4'])
	elif y_train[i] > 1.0:
		sample_weights_f.append(class_weights_f['5'])	

sample_weights_f=np.array(sample_weights_f)


###############   mutiple size  ################

model_float = Sequential()
model_float.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', input_shape=(4,22,2), data_format='channels_last', name = 'dense_1'))
model_float.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', data_format='channels_last', name = 'dense_2'))
model_float.add(MaxPool2D(pool_size=(1,2), padding='same', data_format='channels_last', name = 'dense_3'))
model_float.add(Dropout(0.25))
model_float.add(Flatten())
model_float.add(Dense(units=256, activation='sigmoid', name = 'dense_4'))
model_float.add(Dropout(0.25))
model_float.add(Dense(units=128, activation='sigmoid', name = 'float_dense_5'))
model_float.add(Dropout(0.25))
model_float.add(Dense(1, activation='linear',name = 'float_dense_7'))
model_float.compile(loss='logcosh', metrics=['mean_absolute_error'], optimizer='Adam')

for step in range(0,128):
	epoch=8
	batch_size=256
	model_float.fit(X_train, y_train.ravel(),	sample_weight=np.array(sample_weights_f), batch_size=batch_size, epochs=epoch, validation_data=(X_val, y_val.ravel()))
	
fig,ax = plt.subplots(figsize=(5,5))
plt.xlim(-0.5,2.5)
plt.ylim(-0.5,2.5)
true_act=y_val.ravel()
pre_act=model_float.predict(X_val).ravel()
ax.scatter(pre_act, true_act, marker='.', alpha=.2)
a,b,r_value,p_value,std_err=stats.linregress(pre_act,true_act)
ax.set_xlabel('predicted activity')
ax.set_ylabel('measured activity')
ax.set_title('off_target performance on validation set');
xx=1.5
yy=0.8
t=plt.text(1.4,0.8,'Y=%sX+%s'%(str(a)[:6],str(b)[:6]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)
t=plt.text(1.4,0.6,'P=%s'%(str(p_value)[:8]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)
t=plt.text(1.4,0.4,'R2=%s'%(str(r_value*r_value)[:6]),va='center',ha='left',color='k',alpha=0.8,fontsize=10)							
fig.savefig('validation.result.png', dpi=300)
plt.close('all')
plt.clf()	
mp = "./NBT_react_model.h5"
model_float.save(mp)	
	
