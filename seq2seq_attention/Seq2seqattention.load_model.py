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
from tqdm import tqdm



final_model="./final_model/Multi_mismatched_sgRNA_seq2seqattention.model.ckpt-1200"
vocab_dict_path='./vocab_int'
pre_filename='./data/prediction_data'
output_file='./predict.sgRNAs.txt'
rel_predictor='./NBT_react_model.h5'

############ keras encode ###########

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
	for i in range(0,23):
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
	mis_loc=[]
	mis_type=[]
	for i in range(0,len(seq1)):
		if seq1[i]!=seq2[i]:
			mis_loc.append(i+1)
			mis_type.append(seq1[i]+'_'+seq2[i])
	return mis_loc,mis_type

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

source_int_to_letter, source_letter_to_int = extract_character_vocab('source',model_path)
target_int_to_letter, target_letter_to_int = extract_character_vocab('target',model_path)  	
	
def binary_val_seq(val_seq,source_letter_to_int):
	val_int = [[source_letter_to_int.get(letter) for letter in line] for line in val_seq]
	return val_int	
	



#################    load & add data ########################

def pre_seq_int(input_seq_set,source_letter_to_int):
	val_int=binary_val_seq(input_seq_set,source_letter_to_int)
	batch_size_num=len(input_seq_set)
	return val_int,batch_size_num
	

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
		#attention_mechanism = tf.contrib.seq2seq.BahdanauAttention( num_units = num_units, memory=memory, normalize=True,memory_sequence_length=source_sequence_length)
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
		outputs,_,_= tf.contrib.seq2seq.dynamic_decode(my_decoder,maximum_iterations=pre_max_target_sequence_length,scope=decoder_scope)
		logits = outputs.predicted_ids[:, :, 0]	
	infer_saver = tf.train.Saver()	    	
				
	


off_float_predict = load_model(rel_predictor)


train_sess = tf.Session(graph=train_graph)
infer_sess = tf.Session(graph=infer_graph)
train_sess.run(initializer)	
infer_saver.restore(infer_sess, final_model)


def load_pre_seq(prefile_set):
	all_seq=[]
	all_seq_len=[]
	for prefile in prefile_set:
		f1=open(prefile, 'r')
		m1=f1.readlines()
		f1.close()
		for i in range(0,len(m1)):
			p1=m1[i].strip().split('\t')
			all_seq.append(p1[0].upper())	
			all_seq_len.append(len(p1[0].upper()))
	all_seq_len=np.array(all_seq_len)
	return all_seq,all_seq_len



zf_seq,zf_seq_len=load_pre_seq([pre_filename])
zf_int,zf_size_num=pre_seq_int(zf_seq,source_letter_to_int)
source_int_val,source_lengths_val,target_lengths_val=load_val_data_all(zf_int)				
batch_size_num=len(source_lengths_val)
pre=infer_sess.run(logits,feed_dict={pre_inputs:source_int_val, pre_source_sequence_length:source_lengths_val, pre_target_sequence_length:target_lengths_val,pre_batch_size:batch_size_num,pre_keep_prob:1.0})
zf_DNA_sgRNA=[]
for i in range(0,len(source_int_val)):
	zf_DNA_sgRNA.append(int_sourceseq(source_int_val[i])+';'+int_targetseq(pre[i].ravel()))



fr=open(output_file,'w')
fr.write('on_target')
fr.write('\t')
fr.write('sgRNA_output')
fr.write('\t')
fr.write('CBT_relative_score')
fr.write('\t')
fr.write('mismatch_loc')
fr.write('\t')
fr.write('mismatch_type')	
fr.write('\n')	

for tp in zf_DNA_sgRNA:
	p1=tp.split(';')
	DNA_seq=p1[0]
	sgRNA=p1[1]+DNA_seq[-3:]
	fr.write(DNA_seq)
	fr.write('\t')			
	mis_loc,mis_type=mismatchpair(DNA_seq,sgRNA)
	Xaft=[binarize_sequence_double(DNA_seq,sgRNA)]
	Xaft=np.array(Xaft)
	prescore=off_float_predict.predict(Xaft).ravel()[0]	
	fr.write(sgRNA)
	fr.write('\t')
	fr.write(str(prescore))
	fr.write('\t')
	if len(mis_loc) > 0:
		for tp in mis_loc:
			fr.write(str(tp))
			fr.write(';')
	else:
		fr.write('nan')
	fr.write('\t')
	if len(mis_type) > 0:
		for tp in mis_type:
			fr.write(str(tp))
			fr.write(';')
	else:
		fr.write('nan')		
	fr.write('\n')	
fr.close()
