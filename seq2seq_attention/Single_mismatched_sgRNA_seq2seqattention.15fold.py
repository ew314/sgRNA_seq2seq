##### basic #######
import sys,os,re
import math
import numpy as np
from numpy import array
import random
from random import randint
import scipy
from scipy import stats
#######  tensorflow #########
import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper,LuongAttention,AttentionWrapper,BahdanauAttention
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.util import nest
#######  keras #########
from keras.models import load_model
####### matplotlib ######## 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig


sourcefile='./data/train_data/letters_source_uniqe_train_1.txt'
targetfile='./data/train_data/letters_target_uniqe_train_1.txt'
feedback_filename='./data/feedback_data'
vaildation_filename='./data/vaildation_data'
rel_predictor='./NBT_react_model.h5'
model_save_loc='./model_save'
final_model_loc='./final_model'
vocab_dict_path='./vocab_int'

############ keras encode ###########

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
	
	
####### load val data ###########
		

def load_val_data(source,size):
	tpind = np.arange(len(source))
	np.random.shuffle(tpind)
	source_int=np.array(source)[tpind[:size]]
	source_lengths = []
	target_lengths = []
	for ssource in source_int:
		source_lengths.append(len(ssource))
		target_lengths.append(len(ssource)-3)
	return list(source_int),source_lengths,target_lengths

def load_val_data_all(source):
	source_int=np.array(source)[:]
	source_lengths = []
	target_lengths = []
	for ssource in source_int:
		source_lengths.append(len(ssource))
		target_lengths.append(len(ssource)-3)
	return list(source_int),source_lengths,target_lengths
	
####### mismatch test ###########	
	
def zhixin95 (data):
	confidence=0.95
	sample_mean = np.mean(data)
	sample_std = np.std(data)    
	sample_size = len(data)
	alpha = 1 - 0.95
	t_score = scipy.stats.t.isf(alpha / 2, df = (sample_size-1) )	
	ME = t_score * sample_std / np.sqrt(sample_size)
	lower_limit = sample_mean - ME
	upper_limit = sample_mean + ME
	return sample_mean,ME,lower_limit, upper_limit

def mismatchpair(seq1,seq2):
	num=0
	for i in range(0,len(seq1)):
		try:
			if seq1[i]!=seq2[i]:
				num+=1
		except:
			num=len(seq1)
	return num

def mismatchpair_type(seq1,seq2):
	mis_loc=''
	mis_type=''
	for i in range(0,len(seq1)):
		if seq1[i]!=seq2[i]:
			mis_loc=str(i)+';'+mis_loc
			mis_type=(seq1[i]+'_'+seq2[i])+';'+mis_type
	mis_loc=mis_loc.strip(';')
	mis_type=mis_type.strip(';')
	return mis_loc,mis_type
	
def mismatch(seq_set):
	num=[]
	for p in seq_set:
		tpn=0
		try:
			for i in range(0,len(p[0])):
				if p[0][i] != p[1][i]:
					tpn+=1
			if tpn != 0:
				num.append(float(tpn))
		except:
			num.append(float(len(p[0][0])))
	sample_mean,ME,lower_limit, upper_limit=zhixin95(num)
	return sample_mean,ME	


####### seq2seq encode ###########		
	
def int_sourceseq(intset):
	seq=''
	for p in intset:
		tp=source_int_to_letter[p]
		seq=seq+tp
	return seq

def int_targetseq(intset):
	seq=''
	for p in intset:
		tp=target_int_to_letter[p]
		if tp != '<EOS>' and tp != '<GO>':
			seq=seq+tp
	return seq	
	

def extract_character_vocab(typename,vocab_dict_path):
	int_to_vocab = np.load('%s/%s.int_to_vocab.npy'%(vocab_dict_path,typename),allow_pickle=True).item()
	vocab_to_int = np.load('%s/%s.vocab_to_int.npy'%(vocab_dict_path,typename),allow_pickle=True).item()
	return int_to_vocab, vocab_to_int



def binary_train_data(source_seq,target_seq,source_letter_to_int,target_letter_to_int):
	source_int = [[source_letter_to_int.get(letter) for letter in line] for line in source_seq]
	target_int = [[target_letter_to_int.get(letter) for letter in line] + [target_letter_to_int['<EOS>']] for line in target_seq] 
	return source_int,target_int

def binary_val_seq(val_seq,source_letter_to_int):
	val_int = [[source_letter_to_int.get(letter) for letter in line] for line in val_seq]
	return val_int	
	



#################    load & add data ########################

def load_train_data(sourcefile,targetfile):
	source_seq=[]
	target_seq=[]
	source_target={}
	DNA_score={}
	f1=open(sourcefile, 'r')
	m1=f1.readlines()
	f1.close()
	f2=open(targetfile, 'r')
	m2=f2.readlines()
	f2.close()
	for i in range(0,len(m1)):
		p1=m1[i].strip().split('\t')
		p2=m2[i].strip().split('\t')
		source_seq.append(p1[0])
		target_seq.append(p2[0])
		source_target[p1[0]]=p2[0]
		DNA_score[p1[0]+';'+p2[0]]=float(p2[1])
	return source_seq,target_seq,DNA_score,source_target



source_seq,target_seq,DNA_score,source_target=load_train_data(sourcefile,targetfile)	
source_int_to_letter, source_letter_to_int = extract_character_vocab('source',vocab_dict_path)
target_int_to_letter, target_letter_to_int = extract_character_vocab('target',vocab_dict_path)   
source_int,target_int=binary_train_data(source_seq,target_seq,source_letter_to_int,target_letter_to_int)


def load_train_val_seq(filename):
	f1=open(filename,'r')
	m1=f1.readlines()
	f1.close()
	all_seq=[]
	for p1 in m1:
		p1=p1.strip().split('\t')
		all_seq.append(p1[0])
	all_seq=np.array(all_seq)	
	return all_seq


train_seq=load_train_val_seq(feedback_filename)
val_seq=load_train_val_seq(vaildation_filename)
train_int=binary_val_seq(train_seq,source_letter_to_int)
val_int=binary_val_seq(val_seq,source_letter_to_int)



	

#################    train & feedbak ########################
def bp_pool(source_int_22,int_target_19):
	valseqset=[]
	equalnum=0
	for iii in range(0,len(source_int_22)):
		source_seq=int_sourceseq(source_int_22[iii])
		target_seq=int_targetseq(int_target_19[iii])+source_seq[-3:]
		valseqset.append([source_seq,target_seq])		
		if source_seq==target_seq:
			equalnum+=1
	return valseqset,equalnum


def filter_length(DNA_sgRNA_seq_row):
	Xaft=[]	
	Xord=[]
	DNA_sgRNA_seq=[]
	for i in range(0,len(DNA_sgRNA_seq_row)):
		if len(DNA_sgRNA_seq_row[i][0])==22 and len(DNA_sgRNA_seq_row[i][1])==22:
			Xaft.append(binarize_sequence_double(DNA_sgRNA_seq_row[i][0],DNA_sgRNA_seq_row[i][1]))
			DNA_sgRNA_seq.append(DNA_sgRNA_seq_row[i])
			Xord.append(binarize_sequence_double(DNA_sgRNA_seq_row[i][0],DNA_sgRNA_seq_row[i][0]))
	Xaft = np.array(Xaft)
	Xord = np.array(Xord)	
	return Xaft,Xord,DNA_sgRNA_seq


def result_test_add(pre_set,pre_seq,DNA_score):
	prescore=float_predict.predict(pre_set).ravel()
	presample_mean,preME=mismatch(pre_seq)
	add_seq_seq=[]
	for i in range(0,len(pre_seq)):
		if prescore[i] > 1 and pre_seq[i][0] != pre_seq[i][1]:
			if mismatchpair(pre_seq[i][0],pre_seq[i][1]) == 1:	
				add_seq_seq.append(pre_seq[i])		
				DNA_score[pre_seq[i][0]+';'+pre_seq[i][1]]=prescore[i]						
		if pre_seq[i][0]==pre_seq[i][1]:	
			add_seq_seq.append(pre_seq[i])		
			DNA_score[pre_seq[i][0]+';'+pre_seq[i][1]]=1.			
			if prescore[i] > 1:
				prescore[i]=.9999		
	return prescore,presample_mean,preME,add_seq_seq,DNA_score


def load_int_data_equal_group(source,target,size):
	tpind = np.arange(len(source))
	np.random.shuffle(tpind)	
	mis_source_int_group=[]
	mis_target_int_group=[]
	mis_source_lengths_group = []	
	mis_target_lengths_group = []
	mis_num=0
	eql_source_int_group=[]
	eql_target_int_group=[]
	eql_source_lengths_group = []	
	eql_target_lengths_group = []
	eql_num=0
	for i in tpind:
		if int_targetseq(source[i]) != int_targetseq(target[i]) :
			mis_source_int_group.append(source[i])
			tp=target[i][:]
			del tp[19:22] 
			mis_target_int_group.append(tp)
			mis_source_lengths_group.append(len(source[i]))
			mis_target_lengths_group.append(len(target[i])-3)
			mis_num+=1
		if int_targetseq(source[i]) == int_targetseq(target[i]) :
			eql_source_int_group.append(source[i])
			tp=target[i][:]
			del tp[19:22] 
			eql_target_int_group.append(tp)
			eql_source_lengths_group.append(len(source[i]))
			eql_target_lengths_group.append(len(target[i])-3)
			eql_num+=1		
	group_num=int(min([mis_num,eql_num])/size)-1	
	source_int_group=[]
	source_lengths_group=[]
	target_int_group=[]
	target_lengths_group=[]
	for i in range(0,group_num):
		tp_source_int=[]
		tp_source_lengths=[]
		tp_target_int=[]
		tp_target_lengths=[]		
		tp_source_int.extend(mis_source_int_group[i*size:i*size+size])
		tp_source_lengths.extend(mis_source_lengths_group[i*size:i*size+size])
		tp_target_int.extend(mis_target_int_group[i*size:i*size+size])
		tp_target_lengths.extend(mis_target_lengths_group[i*size:i*size+size])
		tp_source_int.extend(eql_source_int_group[i*size:i*size+size])
		tp_source_lengths.extend(eql_source_lengths_group[i*size:i*size+size])
		tp_target_int.extend(eql_target_int_group[i*size:i*size+size])
		tp_target_lengths.extend(eql_target_lengths_group[i*size:i*size+size])
		source_int_group.append(list(tp_source_int))
		source_lengths_group.append(tp_source_lengths)	
		target_int_group.append(list(tp_target_int))
		target_lengths_group.append(tp_target_lengths)			
	return source_int_group,source_lengths_group,target_int_group,target_lengths_group


def add_train_data(source_seq,add_source_seq,target_seq,add_target_seq,source_target,DNA_score):
	for i in range(0,len(add_source_seq)):
		try:
			old_sgRNA=source_target[add_source_seq[i]]
			if DNA_score[add_source_seq[i]+';'+add_target_seq[i]] > DNA_score[add_source_seq[i]+';'+old_sgRNA]:
				source_target[add_source_seq[i]]=add_target_seq[i]		
		except:
			source_target[add_source_seq[i]]=add_target_seq[i]		
			source_seq.append(add_source_seq[i])			
	new_source=[]
	new_target=[]
	inf=[0,0]
	for DNAseq in source_seq:	
		sgRNA=source_target[DNAseq]
		new_source.append(DNAseq)
		new_target.append(sgRNA)		
		if DNAseq != sgRNA:
			inf[0]=inf[0]+1
		else:
			inf[1]=inf[1]+1
	return new_source,new_target,source_target,inf



############# seq2seq model ########################
def get_rnn_layer(layer_size,num_units,keep_prob):
	def get_rnn_cell():
		single_cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
		cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=keep_prob)
		return cell
	return tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell() for _ in range(layer_size)])

source_vocab_size=6
encoding_embedding_size=6
target_vocab_size=7
decoding_embedding_size=7
rnn_size=2
num_units=256
beam_width=4

tf.reset_default_graph()
train_graph = tf.Graph()
infer_graph = tf.Graph()
with train_graph.as_default():
	mode='train'
	train_inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
	train_targets = tf.placeholder(tf.int32, [None, None], name='targets')
	train_source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
	train_target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
	train_max_target_sequence_length = tf.reduce_max(train_target_sequence_length, name='max_target_len')
	train_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	train_batch_size= tf.placeholder(tf.int32, [], name='batch_size')
	train_learning_rate = tf.placeholder(tf.float32, name='learning_rate')
	decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
	encoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, train_inputs)	
	ending = tf.strided_slice(train_targets, [0, 0], [train_batch_size, -1], [1, 1])
	decoder_input = tf.concat([tf.fill([train_batch_size, 1], target_letter_to_int['<GO>']), ending], 1)
	decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)	
	with tf.name_scope("Encoder"):
		encoder_cell=get_rnn_layer(rnn_size,num_units,train_keep_prob)
		encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embed_input, sequence_length=train_source_sequence_length, dtype=tf.float32)
	with tf.variable_scope("Decoder") as decoder_scope:
		out_layer = tf.layers.Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		memory = encoder_output
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=num_units, memory=memory, memory_sequence_length=train_source_sequence_length)
		cell = get_rnn_layer(rnn_size,num_units,train_keep_prob)
		cell = tf.contrib.seq2seq.AttentionWrapper(cell,attention_mechanism,attention_layer_size=num_units,name="attention")
		initial_state = cell.zero_state(train_batch_size, tf.float32).clone( cell_state=encoder_state )
		helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,sequence_length=train_target_sequence_length )
		decoder = tf.contrib.seq2seq.BasicDecoder( cell = cell, helper = helper, initial_state = initial_state,output_layer=out_layer) 
		outputs,_,_= tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=train_max_target_sequence_length,scope=decoder_scope)
		logits = outputs.rnn_output
	masks = tf.sequence_mask(train_target_sequence_length, train_max_target_sequence_length, dtype=tf.float32, name='masks')	
	cost = tf.contrib.seq2seq.sequence_loss(logits,train_targets,masks)
	optimizer = tf.train.AdamOptimizer(train_learning_rate)
	gradients = optimizer.compute_gradients(cost)
	capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
	train_op = optimizer.apply_gradients(capped_gradients)		
	initializer = tf.global_variables_initializer()
	train_saver = tf.train.Saver()

with infer_graph.as_default():
	mode='infer'
	pre_inputs = tf.placeholder(tf.int32, [None, None], name='pre_inputs')
	pre_source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
	pre_target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
	pre_max_target_sequence_length = tf.reduce_max(pre_target_sequence_length, name='max_target_len')
	pre_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	pre_batch_size= tf.placeholder(tf.int32, [], name='batch_size')
	decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
	encoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, pre_inputs)	
	with tf.name_scope("Encoder"):
		encoder_cell=get_rnn_layer(rnn_size,num_units,pre_keep_prob)
		encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embed_input, sequence_length=pre_source_sequence_length, dtype=tf.float32)	
	with tf.variable_scope("Decoder") as decoder_scope:
		out_layer = tf.layers.Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))	
		start_tokens = tf.ones([pre_batch_size, ], tf.int32) *target_letter_to_int['<GO>']
		end_token = target_letter_to_int['<EOS>']		
		memory = tf.contrib.seq2seq.tile_batch(encoder_output,multiplier=beam_width )
		pre_source_sequence_lengths = tf.contrib.seq2seq.tile_batch(pre_source_sequence_length, multiplier=beam_width)
		encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width )
		batch_size = pre_batch_size * beam_width
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=num_units, memory=memory, memory_sequence_length=pre_source_sequence_lengths)
		cell = get_rnn_layer(rnn_size,num_units,pre_keep_prob)
		cell = tf.contrib.seq2seq.AttentionWrapper(cell,attention_mechanism,attention_layer_size=num_units,name="attention")
		initial_state = cell.zero_state(batch_size, tf.float32).clone( cell_state=encoder_state )
		my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell,embedding = decoder_embeddings,start_tokens = start_tokens,end_token = end_token,initial_state = initial_state,beam_width = beam_width,output_layer = out_layer )
		outputs,dec_hidden,attention_output= tf.contrib.seq2seq.dynamic_decode(my_decoder,maximum_iterations=pre_max_target_sequence_length,scope=decoder_scope)
		logits = outputs.predicted_ids[:, :, 0]	
		attention_outputs=attention_output[:]
	infer_saver = tf.train.Saver()	    	
				
				


os.system('mkdir %s %s'%(model_save_loc,final_model_loc))


fr=open('./train.record.txt','w')
fr.write('epoh')
fr.write('\t')
fr.write('seq2seq_traindata_loss')
fr.write('\t')
fr.write('seq2seq_traindata_95CI')
fr.write('\t')
fr.write('seq2seq_traindata_higher_rate')
fr.write('\t')
fr.write('seq2seq_traindata_mismatch')
fr.write('\t')

fr.write('train_af_95CI')
fr.write('\t')
fr.write('train_af_higher_than_ord_rate')
fr.write('\t')
fr.write('train_af_higher_than_1_rate')
fr.write('\t')
fr.write('train_af_equal_rate')
fr.write('\t')
fr.write('train_af_mismatch')
fr.write('\t')

fr.write('val_af_95CI')
fr.write('\t')
fr.write('val_af_higher_than_ord_rate')
fr.write('\t')
fr.write('val_af_higher_than_1_rate')
fr.write('\t')
fr.write('val_af_equal_rate')
fr.write('\t')
fr.write('val_af_mismatch')
fr.write('\n')	



epoch=1200
lr_set={}
for i in range(1,400):
	lr_set[i]=0.0005
for i in range(400,600):
	lr_set[i]=0.0003
for i in range(600,800):
	lr_set[i]=0.0002
for i in range(800,900):
	lr_set[i]=0.0001	
for i in range(900,1000):
	lr_set[i]=0.00001		
for i in range(1000,epoch+1):
	lr_set[i]=0.000001
	

train_sess = tf.Session(graph=train_graph)
infer_sess = tf.Session(graph=infer_graph)
train_sess.run(initializer)	
float_predict = load_model(rel_predictor)
for num in range(1,epoch+1):
	##### train part ####
	batch_size_num=256
	half_batch=128
	ttloss=0
	higherrate=0
	all_prescore=[]
	train_valseqset=[]
	epoh_learning_rate=lr_set[num]
	train_num=0

	source_int_group,source_lengths_group,target_int_group,target_lengths_group=load_int_data_equal_group(source_int,target_int,half_batch)		
	for ii in range(0,len(source_int_group)):
		source_int_train=source_int_group[ii]
		source_lengths_train=source_lengths_group[ii]
		target_int_train=target_int_group[ii]
		target_lengths_train=target_lengths_group[ii]
				
		valseqset,equalnum=bp_pool(source_int_train,target_int_train)			
		train_valseqset.extend(valseqset)		
		Xaft_s,Xord_s,valseq=filter_length(valseqset)
		prescore=float_predict.predict(Xaft_s).ravel()
		higherthan1=(prescore > 1).sum()
		higherrate=higherrate+higherthan1/float(batch_size_num)
		all_prescore.extend(list(prescore))				
		tloss,_=train_sess.run([cost,train_op],feed_dict={train_inputs:source_int_train,train_source_sequence_length:source_lengths_train,train_targets:target_int_train,train_target_sequence_length:target_lengths_train,train_batch_size:batch_size_num,train_keep_prob:.5,train_learning_rate:epoh_learning_rate})
		ttloss+=tloss
		train_num+=1
		
	train_saver.save(train_sess,"%s/%s.train.highest_unique_train.add_highest_and_unique.add_all.model.ckpt"%(model_save_loc,str(num)), global_step = epoch)
	infer_saver.restore(infer_sess, "%s/%s.train.highest_unique_train.add_highest_and_unique.add_all.model.ckpt-%s"%(model_save_loc,str(num),str(epoch)))
	
	ttloss=ttloss/float(train_num)	
	higherrate=higherrate/float(train_num)
	mis_sample_mean,mis_ME=mismatch(train_valseqset)
	sample_mean,ME,lower_limit, upper_limit=zhixin95(all_prescore)	
	print('')						
	print('###########      ','epoch: ',num,' train_loss: ',ttloss,'95_CI: ',lower_limit,'~',upper_limit,'        ###########')	
	print('###########      ','epoch: ',num,' train_loss: ',ttloss,'higherrate: ',higherrate,'        ###########')
	fr.write(str(num)) 
	fr.write('\t')
	fr.write(str(ttloss)) 
	fr.write('\t')
	fr.write(str(sample_mean)+';'+str(ME))
	fr.write('\t')		
	fr.write(str(higherrate))
	fr.write('\t')	
	fr.write(str(mis_sample_mean)+';'+str(mis_ME))
	fr.write('\t')							
		

	####  feedback ######
	print('')
	print('## feedback part ##')
	print('')

	train_af_score=[]
	train_af_higherrate=0
	train_af_equalrate=0
	train_af_higher_ord=0		
		
	source_int_val,source_lengths_val,target_lengths_val=load_val_data_all(train_int)				
	batch_size_num=len(source_lengths_val)
	pre=infer_sess.run(logits,feed_dict={pre_inputs:source_int_val, pre_source_sequence_length:source_lengths_val, pre_target_sequence_length:target_lengths_val,pre_batch_size:batch_size_num,pre_keep_prob:0.5})
	target_int_19=[]
	equalnum=0
	for i in range(0,len(source_int_val)):
		target_int_19.append(pre[i].ravel())
	DNA_sgRNA_seq_row,equalnum=bp_pool(source_int_val,target_int_19)
	train_af_equalrate=float(equalnum)/len(source_int_val)
	Xaft_s,Xord_s,DNA_sgRNA_seq=filter_length(DNA_sgRNA_seq_row)
			
	if len(DNA_sgRNA_seq) > 0:
		prescore,presample_mean,preME,add_seq_seq,DNA_score=result_test_add(Xaft_s,DNA_sgRNA_seq,DNA_score)
		train_af_score=list(prescore[:])
		higherthan_1=(1 < prescore).sum()
		train_af_higherrate=higherthan_1/float(len(source_int_val))			
		ord_score=float_predict.predict(Xord_s).ravel()
		aft_score=float_predict.predict(Xaft_s).ravel()
		train_af_higher_ord=(ord_score<aft_score).sum()/float(len(source_int_val))				
						
		if len(add_seq_seq) > 0:
			add_source_seq=[]
			add_target_seq=[]
			for seq in add_seq_seq:
				add_source_seq.append(seq[0])
				add_target_seq.append(seq[1])
				
			source_seq,target_seq,source_target,data_inf=add_train_data(source_seq,add_source_seq,target_seq,add_target_seq,source_target,DNA_score)
			source_int,target_int=binary_train_data(source_seq,target_seq,source_letter_to_int,target_letter_to_int)					
			print('###########      ','data inf','mis_pair: ',data_inf[0],' equal_pair: ',data_inf[1],'  ###########')		
		
	mis_sample_mean,mis_ME=mismatch(DNA_sgRNA_seq)							
	sample_mean,ME,lower_limit, upper_limit=zhixin95(train_af_score)		
				
	fr.write(str(sample_mean)+';'+str(ME))
	fr.write('\t')				
	fr.write(str(train_af_higher_ord))
	fr.write('\t')		
	fr.write(str(train_af_higherrate))
	fr.write('\t')			
	fr.write(str(train_af_equalrate))
	fr.write('\t')		
	fr.write(str(mis_sample_mean)+';'+str(mis_ME))
	fr.write('\t')			

	print('###########      ','train','af_equalrate: ',train_af_equalrate,' af_higherrate: ',train_af_higherrate,' af_higher_ord:',train_af_higher_ord,'  ###########')					
			
	####  val part #####
	
	print('## val part ##')
	val_af_score=[]
	val_af_higherrate=0
	val_af_equalrate=0
	val_af_higher_ord=0		
		
	source_int_val,source_lengths_val,target_lengths_val=load_val_data_all(val_int)	
	batch_size_num=len(source_lengths_val)			
	pre=infer_sess.run(logits,feed_dict={pre_inputs:source_int_val, pre_source_sequence_length:source_lengths_val, pre_target_sequence_length:target_lengths_val,pre_batch_size:batch_size_num,pre_keep_prob:1.0})
	target_int_19=[]
	equalnum=0
	for i in range(0,len(source_int_val)):
		target_int_19.append(pre[i].ravel())
	for i in range(0,5):
		print(int_targetseq(pre[i].ravel()),pre[i].ravel())
	DNA_sgRNA_seq_row,equalnum=bp_pool(source_int_val,target_int_19)
	print(DNA_sgRNA_seq_row[:5])
	
	val_af_equalrate=float(equalnum)/len(source_int_val)
									
	Xaft_s,Xord_s,DNA_sgRNA_seq=filter_length(DNA_sgRNA_seq_row)
		
			
	if len(DNA_sgRNA_seq) > 0:
		prescore,presample_mean,preME,add_seq_seq,DNA_score=result_test_add(Xaft_s,DNA_sgRNA_seq,DNA_score)
		val_af_score=list(prescore[:])
		higherthan_1=(1 < prescore).sum()
		val_af_higherrate=higherthan_1/float(len(source_int_val))			
		ord_score=float_predict.predict(Xord_s).ravel()
		aft_score=float_predict.predict(Xaft_s).ravel()
		val_af_higher_ord=(ord_score<aft_score).sum()/float(len(source_int_val))				
			
			
	mis_sample_mean,mis_ME=mismatch(DNA_sgRNA_seq)							
	sample_mean,ME,lower_limit, upper_limit=zhixin95(val_af_score)
				
	fr.write(str(sample_mean)+';'+str(ME))
	fr.write('\t')				
	fr.write(str(val_af_higher_ord))
	fr.write('\t')	
	fr.write(str(val_af_higherrate))
	fr.write('\t')			
	fr.write(str(val_af_equalrate))
	fr.write('\t')		
	fr.write(str(mis_sample_mean)+';'+str(mis_ME))
	fr.write('\n')			

	print('###########      ','val','val_af_equalrate: ',val_af_equalrate,' val_af_higherrate: ',val_af_higherrate,' val_af_higher_ord:',val_af_higher_ord,'mis:',str(mis_sample_mean)+';'+str(mis_ME),'  ###########')		
	
train_saver.save(train_sess,"%s/Single_mismatched_sgRNA_seq2seqattention.model.ckpt"%(final_model_loc), global_step = epoch)		
fr.close()	 


#####  training progress ploting  #####

def RGB(n):
	if n>9:
		n=n-10+65
		m=chr(n)
	else:
		m=str(n)
	return m

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


def progress_plot(af_highrate,af_equalrate,af_mismatch,filename,x_bar):

	fig = plt.figure(figsize=(10,5),dpi=300)
	fig.subplots_adjust(left=-0.01, right=1.01, top=1.01, bottom=-0.01)				
	plt.xlim(-200,x_bar[-1]+250)
	plt.ylim(-0.2,1.3)
	
	C1="#%s"%color(237,89,67)	
	C2="#%s"%color(123,201,107)		
	C3="#%s"%color(228,107,211)		
	
	C_d="#%s"%color(241,241,241)	
	xpit=[0,0,1200,1200]
	ypit=[1,0,0,1]
	plt.fill(xpit,ypit,color=C_d,alpha=1,edgecolor='none',linewidth=0)
	
	for x in [200,400,600,800,1000]:
		plt.plot([x,x],[0,1],color='gray',alpha=.6,linewidth=1,linestyle='dashed')		

	for y in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
		plt.plot([0,1200],[y,y],color='gray',alpha=.6,linewidth=1,linestyle='dashed')		
	
	#############  highrate  ################ 	
	plt.plot(af_highrate[0],af_highrate[1],color=C1,alpha=.8,linewidth=2,linestyle='solid')
	plt.plot([300,500],[1.05,1.05],color=C1,alpha=.8,linewidth=4,linestyle='solid')	
	plt.text(400,1.075,'highr rate',ha='center',va='center',color='k',alpha=0.8,fontsize=8)
	
	
	#############  equalrate  ################ 	
	plt.plot(af_equalrate[0],af_equalrate[1],color=C2,alpha=.8,linewidth=2,linestyle='dashed')
	plt.plot([600,800],[1.05,1.05],color=C2,alpha=0.8,linewidth=4,linestyle='dashed')	
	plt.text(700,1.075,'equal rate',ha='center',va='center',color='k',alpha=0.8,fontsize=8)
	
	
	
	#############  mismatch  ################ 	
	plt.plot(af_mismatch[0],af_mismatch[1],color=C3,alpha=.8,linewidth=2,linestyle='dotted')
	plt.plot([900,1100],[1.05,1.05],color=C3,alpha=0.8,linewidth=4,linestyle='dotted')
	plt.text(1000,1.075,'mismatch',ha='center',va='center',color='k',alpha=0.8,fontsize=8)
	
		
	ine=[0,x_bar[-1]]
	out=[0,0]
	plt.plot(ine,out,color='k',alpha=1,linewidth=1)
	ine=[0,0]
	out=[0,1]
	plt.plot(ine,out,color='k',alpha=1,linewidth=1)
	ine=[x_bar[-1],x_bar[-1]]
	out=[0,1]
	plt.plot(ine,out,color='k',alpha=1,linewidth=1)
	
	
	for i in range(0,len(x_bar)):
		x=[x_bar[i],x_bar[i]]
		y=[0,-0.02]
		plt.plot(x,y,alpha=1,color='k',linewidth=1)
		plt.text(x[0],y[1]-0.05,'%s'%str(x_bar[i]),ha='center',va='center',color='k',alpha=0.8,fontsize=20)
	
	y_bar=[0.2,0.4,0.6,0.8,1.0]
	y_bar_1=['0.2','0.4','0.6','0.8','1.0']
	y_bar_2=['4','8','12','16','20']
	
	for i in range(0,len(y_bar)):
		x=[0,-10]
		y=[y_bar[i],y_bar[i]]
		plt.plot(x,y,alpha=1,color='k',linewidth=1)
		plt.text(x[1]-30,y[0], '%s'%str(y_bar_2[i]),ha='right',va='center',color='k',alpha=0.8,fontsize=20)
	
		x=[x_bar[-1],x_bar[-1]+10]
		y=[y_bar[i],y_bar[i]]
		plt.plot(x,y,alpha=1,color='k',linewidth=1)
		plt.text(x[1]+30,y[0], '%s'%str(y_bar_1[i]),ha='left',va='center',color='k',alpha=0.8,fontsize=20)
	

	fig.savefig(filename, dpi=300)
	plt.close('all')
	plt.clf()		


def record_load_withassist(filename_set):
	f1=open(filename,'r')
	m1=f1.readlines()
	f1.close()
	af_highrate=[[],[]]
	af_equalrate=[[],[]]
	af_mismatch=[[],[]]
	
	for i in range(1,len(m1)):
		p1=m1[i].strip().split('\t')
		try:
			tp=p1[12]
			af_highrate[1].append(float(tp))
			af_highrate[0].append(float(p1[0]))
		except:
			no=1	
		try:
			tp=p1[13]
			af_equalrate[1].append(float(tp))
			af_equalrate[0].append(float(p1[0]))
		except:
			no=1	
		try:
			tp=p1[14].split(';')
			if(str(tp[0]))!='nan':
				af_mismatch[1].append(float(tp[0])/20)
				af_mismatch[0].append(float(p1[0]))
			else:
				af_mismatch[1].append(float(1))
				af_mismatch[0].append(float(p1[0]))			
		except:
			no=1		

	return af_highrate,af_equalrate,af_mismatch
	

#########  train.high_than_1.equal.record #######
filename='./train.record.txt'
af_highrate,af_equalrate,af_mismatch=record_load_withassist(filename)
print(af_highrate,af_equalrate,af_mismatch)
figurename='./Single_mismatched_sgRNA_seq2seqattention.png'
x_bar=[0,200,400,600,800,1000,1200]
progress_plot(af_highrate,af_equalrate,af_mismatch,figurename,x_bar)
