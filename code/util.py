from __future__ import division
import json
import sys, random
import re
import nltk
import torch
import numpy as np
from nltk import Tree
from os.path import expanduser
from nltk.corpus import stopwords
# from pycorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP('http://localhost:9000')

stop_words = list(stopwords.words('english'))
# stop_words.extend(['.', ',', 'must'])
# from pytorch_pretrained_bert import BertTokenizer, BertModel
seed=123456
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
Wuwei_server=torch.cuda.is_available()
if Wuwei_server:
	print('CUDA is available!')
	my_device = torch.device('cuda:0')
	torch.cuda.manual_seed_all(seed)
	base_path = '../' # working space
	embedding_path = '../data/glove'  # glove path
else:
	my_device = torch.device('cpu')
	if False:
		base_path = expanduser("~") + '/Documents/neural-word-aligner/'
		embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
	else:
		base_path = expanduser("~") + '/Documents/research/twitterPPDB/neural-word-aligner/'
		embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'


max_span_size=3
max_sent_length=150
ENCODER_TYPE='bert' # ['lstm', 'transformer', 'bert']
num_dep_tags=43
num_pos_tags=40
multi_task_parsing=False # It is kind of useless...
multi_task_pos_tagging=False
multi_task_chunking=False
GOLD_CHUNK=True
bidirectional_training=True
bidirectional_decoding=True

def ortho_weight(ndim):
	"""
	Random orthogonal weights

	Used by norm_weights(below), in which case, we
	are ensuring that the rows are orthogonal
	(i.e W = U \Sigma V, U has the same
	# of rows, V has the same # of cols)
	"""
	W = np.random.randn(ndim, ndim)
	u, s, v = np.linalg.svd(W)
	return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
	"""
	Random weights drawn from a Gaussian
	"""
	if nout is None:
		nout = nin
	if nout == nin and ortho:
		W = ortho_weight(nin)
	else:
		W = scale * np.random.randn(nin, nout)
	return W.astype('float32')

def readiSTS16Data(folder,flag='train', HEADLINE=True, GOLD_CHUNK=GOLD_CHUNK):
	label_set = ['REL', 'SIMI', 'SPE2', 'SPE1', 'OPPO', 'EQUI', 'NOALI']
	if flag=='train':
		folder=folder+'train_2015_10_22.utf-8/'
		if HEADLINE:
			sent1_tag_file = folder + 'STSint.input.headlines.sent1.tag.txt'
		else:
			sent1_tag_file = folder + 'STSint.input.images.sent1.tag.txt'
		sent1_tag_labels = []
		tmp_tag = []
		for line in open(sent1_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent1_tag_labels.append(tmp_tag)
				tmp_tag = []
		if HEADLINE:
			sent2_tag_file = folder + 'STSint.input.headlines.sent2.tag.txt'
		else:
			sent2_tag_file = folder + 'STSint.input.images.sent2.tag.txt'
		sent2_tag_labels = []
		tmp_tag = []
		for line in open(sent2_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent2_tag_labels.append(tmp_tag)
				tmp_tag = []
		sent1_list=[]
		counter=0
		if HEADLINE:
			if GOLD_CHUNK:
				chunk_file = 'STSint.input.headlines.sent1.chunk.txt'
			else:
				chunk_file = 'STSint.input.headlines.sent1.chunk.system.txt'
		else:
			if GOLD_CHUNK:
				chunk_file='STSint.input.images.sent1.chunk.txt'
			else:
				chunk_file = 'STSint.input.images.sent1.chunk.system.txt'
		for line in open(folder + chunk_file):
			# line = line.encode('utf-8')
			# print(line)
			sent1_list.append((line.strip().lower(), sent1_tag_labels[counter], None))
			counter+=1
		sent2_list = []
		counter=0
		if HEADLINE:
			if GOLD_CHUNK:
				chunk_file='STSint.input.headlines.sent2.chunk.txt'
			else:
				chunk_file = 'STSint.input.headlines.sent2.chunk.system.txt'
		else:
			if GOLD_CHUNK:
				chunk_file='STSint.input.images.sent2.chunk.txt'
			else:
				chunk_file='STSint.input.images.sent2.chunk.system.txt'
		for line in open(folder + chunk_file):
			# line = line.encode('utf-8')
			sent2_list.append((line.strip().lower(), sent2_tag_labels[counter], None))
			counter+=1
		alignment_list = []
		start_flag=False
		tmp_label=[]
		if HEADLINE:
			if GOLD_CHUNK:
				label_file='STSint.input.headlines.wa'
			else:
				label_file = 'STSint.input.headlines.system.wa'
		else:
			if GOLD_CHUNK:
				label_file='STSint.input.images.wa'
			else:
				label_file = 'STSint.input.images.system.wa'
		for line in open(folder + label_file):
			# line = line.encode('utf-8')
			# print(line)
			# sys.exit()
			if line.strip()=='<alignment>':
				start_flag=True
				tmp_label=[]
				continue
			elif line.strip()=='</alignment>':
				start_flag=False
				alignment_list.append(tmp_label)
				continue
			if start_flag:
				my_label, my_tag, my_score, _ = line.strip().split('//')
				tmp_label.append((my_label, my_tag, my_score))
	else:
		folder=folder+'test_goldStandard/'
		sent1_list = []
		if HEADLINE:
			if GOLD_CHUNK:
				file_path=folder + 'STSint.testinput.headlines.sent1.chunk.txt'
			else:
				file_path = folder + 'STSint.testinput.headlines.sent1.chunk.system.txt'
			sent1_tag_file = folder + 'STSint.testinput.headlines.sent1.tag.txt'
		else:
			if GOLD_CHUNK:
				file_path = folder + 'STSint.testinput.images.sent1.chunk.txt'
			else:
				file_path=folder + 'STSint.testinput.images.sent1.chunk.system.txt'
			sent1_tag_file = folder + 'STSint.testinput.images.sent1.tag.txt'
		sent1_tag_labels = []
		tmp_tag = []
		for line in open(sent1_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent1_tag_labels.append(tmp_tag)
				tmp_tag = []
		counter=0
		for line in open(file_path):
			# line = line.encode('utf-8')
			sent1_list.append((line.strip().lower(), sent1_tag_labels[counter], None))
			counter+=1
		sent2_list = []
		if HEADLINE:
			if GOLD_CHUNK:
				file_path=folder + 'STSint.testinput.headlines.sent2.chunk.txt'
			else:
				file_path = folder + 'STSint.testinput.headlines.sent2.chunk.system.txt'
			sent2_tag_file = folder + 'STSint.testinput.headlines.sent2.tag.txt'
		else:
			if GOLD_CHUNK:
				file_path=folder + 'STSint.testinput.images.sent2.chunk.txt'
			else:
				file_path = folder + 'STSint.testinput.images.sent2.chunk.system.txt'
			sent2_tag_file = folder + 'STSint.testinput.images.sent2.tag.txt'
		sent2_tag_labels = []
		tmp_tag = []
		for line in open(sent2_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent2_tag_labels.append(tmp_tag)
				tmp_tag = []
		counter=0
		for line in open(file_path):
			# line = line.encode('utf-8')
			sent2_list.append((line.strip().lower(), sent2_tag_labels[counter], None))
			counter+=1
		alignment_list = []
		start_flag = False
		tmp_label = []
		if HEADLINE:
			if GOLD_CHUNK:
				file_path=folder + 'STSint.testinput.headlines.wa'
			else:
				file_path = folder + 'STSint.testinput.headlines.system.wa'
		else:
			if GOLD_CHUNK:
				file_path=folder + 'STSint.testinput.images.wa'
			else:
				file_path = folder + 'STSint.testinput.images.system.wa'
		for line in open(file_path):
			# line=line.encode('utf-8')
			if line.strip() == '<alignment>':
				start_flag = True
				tmp_label = []
				continue
			elif line.strip() == '</alignment>':
				start_flag = False
				alignment_list.append(tmp_label)
				continue
			if start_flag:
				my_label, my_tag, my_score, _ = line.strip().split('//')
				tmp_label.append((my_label, my_tag, my_score))
		#print(label_set)
		#print(len(label_set))
	clean_data=[]
	if len(sent1_list)==len(sent2_list) and len(sent1_list)==len(alignment_list):
		clean_data=(sent1_list, sent2_list, alignment_list)
		#for i in range(len(sent1_list)):
			#clean_data.append((sent1_list[i], sent2_list[i],alignment_list[i]))
	else:
		print(len(sent1_list))
		print(len(sent2_list))
		print(len(alignment_list))
		print('number of pairs don\'t match!')
		sys.exit()
	return clean_data

def convert_spanID_to_stateID(span, sent_length):
	span_length=len(span)
	return int(sent_length*(span_length-1)-(span_length-1)*(span_length-2)/2 + span[0] + 1)
	# if span_length==1:
	# 	return span[0]
	# elif span_length==2:
	# 	return sent_length+span[0]
	# elif span_length==3:
	# 	return sent_length+sent_length-1+span[0]

def convert_stateID_to_spanID(stateID, sent_length): # 0 is NULL state
	# print(stateID, sent_length)
	stateID=stateID-1
	if stateID<0:
		return (-1, -1)
	else:
		for span_length in range(1, max_span_size+1):
			lower_bound=(span_length-1)*sent_length-int((span_length-1)*(span_length-2)/2)
			upper_bound=span_length*sent_length-int(span_length*(span_length-1)/2)
			# print(span_length, lower_bound, upper_bound)
			if stateID>=lower_bound and stateID<upper_bound:
				return (stateID-lower_bound, span_length) # return (spanID, span_Length)
	# if stateID<sent_length:
	# 	return (stateID, 1) # start_index, span_size
	# elif stateID>=sent_length and stateID<2*sent_length-1:
	# 	return (stateID-sent_length, 2)
	# else:
	# 	return (stateID-2*sent_length+1, 3)

def readiSTS16DataWithoutGoldenChunk(folder,flag='train', HEADLINE=True):
	# label_set = ['REL', 'SIMI', 'SPE2', 'SPE1', 'OPPO', 'EQUI', 'NOALI']
	if flag=='train':
		folder=folder+'train_2015_10_22.utf-8/'
		sent1_tag_file = folder + 'STSint.input.headlines.sent1.tag.txt'
		sent1_tag_labels = []
		tmp_tag = []
		for line in open(sent1_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent1_tag_labels.append(tmp_tag)
				tmp_tag = []
		sent2_tag_file = folder + 'STSint.input.headlines.sent2.tag.txt'
		sent2_tag_labels = []
		tmp_tag = []
		for line in open(sent2_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent2_tag_labels.append(tmp_tag)
				tmp_tag = []
		sent1_list=[]
		counter=0
		for line in open(folder + 'STSint.input.headlines.sent1.txt'):
			# line = line.encode('utf-8')
			# print(sent1_tag_labels[counter]) #[('B-NP', 'NN'), ('I-NP', 'NN'), ('I-NP', 'NN'), ('B-PP', 'IN'), ('B-NP', 'JJ'), ('I-NP', 'NNP'), ('I-NP', 'NN'), ('I-NP', 'NN')]
			sent1_list.append((line.strip(), ' '.join(item[1] for item in sent1_tag_labels[counter]), None))
			counter+=1
		sent2_list = []
		counter=0
		for line in open(folder + 'STSint.input.headlines.sent2.txt'):
			# line = line.encode('utf-8')
			sent2_list.append((line.strip(), ' '.join(item[1] for item in sent2_tag_labels[counter]), None))
			counter+=1
		alignment_list = []
		start_flag=False
		tmp_label=[]
		for line in open(folder + 'STSint.input.headlines.wa'):
			# line = line.encode('utf-8')
			# print(line)
			# sys.exit()
			if line.strip()=='<alignment>':
				start_flag=True
				tmp_label=[]
				continue
			elif line.strip()=='</alignment>':
				start_flag=False
				alignment_list.append(tmp_label)
				continue
			if start_flag:
				# my_tag=None
				# for item in label_set:
				# 	if item in line.strip().split('//')[1]:
				# 		my_tag=item
				# 		break
				my_label, my_tag, my_score, _ = line.strip().split('//')
				tmp_label.append((my_label, my_tag, my_score))
		# print(sent1_list[-1])
		# print(sent2_list[-1])
		# print(alignment_list[-1])
		# sys.exit()
		''''''
		sent1_tag_file = folder + 'STSint.input.images.sent1.tag.txt'
		sent1_tag_labels = []
		tmp_tag = []
		for line in open(sent1_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent1_tag_labels.append(tmp_tag)
				tmp_tag = []
		sent2_tag_file = folder + 'STSint.input.images.sent2.tag.txt'
		sent2_tag_labels = []
		tmp_tag = []
		for line in open(sent2_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent2_tag_labels.append(tmp_tag)
				tmp_tag = []
		counter=0
		for line in open(folder + 'STSint.input.images.sent1.txt'):
			# line = line.encode('utf-8')
			sent1_list.append((line.strip(), ' '.join(item[1] for item in sent1_tag_labels[counter]), None))
			counter+=1
		counter=0
		for line in open(folder + 'STSint.input.images.sent2.txt'):
			# line = line.encode('utf-8')
			sent2_list.append((line.strip(), ' '.join(item[1] for item in sent2_tag_labels[counter]), None))
			counter+=1
		start_flag = False
		tmp_label = []
		for line in open(folder + 'STSint.input.images.wa'):
			# line = line.encode('utf-8')
			if line.strip() == '<alignment>':
				start_flag = True
				tmp_label = []
				continue
			elif line.strip() == '</alignment>':
				start_flag = False
				alignment_list.append(tmp_label)
				continue
			if start_flag:
				# my_tag = None
				# for item in label_set:
				# 	if item in line.strip().split('//')[1]:
				# 		my_tag = item
				# 		break
				# if my_tag:
				# 	tmp_label.append((line.strip().split('//')[0], my_tag, line.strip().split('//')[2]))
				my_label, my_tag, my_score, _ = line.strip().split('//')
				tmp_label.append((my_label, my_tag, my_score))
		#print(label_set)
		#print(len(label_set))
		#sys.exit()
		# print(sent1_list[0])
		# print(sent2_list[0])
		# print(alignment_list[0])
		# sys.exit()
		''''''
	else:
		folder=folder+'test_goldStandard/'
		sent1_list = []
		if HEADLINE:
			file_path=folder + 'STSint.testinput.headlines.sent1.txt'
			sent1_tag_file = folder + 'STSint.testinput.headlines.sent1.tag.txt'
		else:
			file_path=folder + 'STSint.testinput.images.sent1.txt'
			sent1_tag_file = folder + 'STSint.testinput.images.sent1.tag.txt'
		sent1_tag_labels = []
		tmp_tag = []
		for line in open(sent1_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent1_tag_labels.append(tmp_tag)
				tmp_tag = []
		counter=0
		for line in open(file_path):
			# line = line.encode('utf-8')
			sent1_list.append((line.strip(), ' '.join(item[1] for item in sent1_tag_labels[counter]), None))
			counter+=1
		sent2_list = []
		if HEADLINE:
			file_path=folder + 'STSint.testinput.headlines.sent2.txt'
			sent2_tag_file = folder + 'STSint.testinput.headlines.sent2.tag.txt'
		else:
			file_path=folder + 'STSint.testinput.images.sent2.txt'
			sent2_tag_file = folder + 'STSint.testinput.images.sent2.tag.txt'
		sent2_tag_labels = []
		tmp_tag = []
		for line in open(sent2_tag_file):
			if len(line) > 1:
				tmp_tag.append((line.strip().split()[-1], line.strip().split()[-2]))
			else:
				sent2_tag_labels.append(tmp_tag)
				tmp_tag = []
		counter=0
		for line in open(file_path):
			# line = line.encode('utf-8')
			sent2_list.append((line.strip(), ' '.join(item[1] for item in sent2_tag_labels[counter]), None))
			counter+=1
		alignment_list = []
		start_flag = False
		tmp_label = []
		if HEADLINE:
			file_path=folder + 'STSint.testinput.headlines.wa'
		else:
			file_path=folder + 'STSint.testinput.images.wa'
		for line in open(file_path):
			# line=line.encode('utf-8')
			if line.strip() == '<alignment>':
				start_flag = True
				tmp_label = []
				continue
			elif line.strip() == '</alignment>':
				start_flag = False
				alignment_list.append(tmp_label)
				continue
			if start_flag:
				my_label, my_tag, my_score, _ = line.strip().split('//')
				tmp_label.append((my_label, my_tag, my_score))
				# my_tag = None
				# for item in label_set:
				# 	if item in line.strip().split('//')[1]:
				# 		my_tag = item
				# 		break
				# if my_tag:
				# 	tmp_label.append((line.strip().split('//')[0], my_tag, line.strip().split('//')[2]))
		#print(label_set)
		#print(len(label_set))
	clean_data=[]
	if len(sent1_list)==len(sent2_list) and len(sent1_list)==len(alignment_list):
		clean_data=(sent1_list, sent2_list, alignment_list)
		#for i in range(len(sent1_list)):
			#clean_data.append((sent1_list[i], sent2_list[i],alignment_list[i]))
	else:
		print(len(sent1_list))
		print(len(sent2_list))
		print(len(alignment_list))
		print('number of pairs don\'t match!')
		sys.exit()
	return clean_data

def convert_iSTS_chunk(sent):
	chunks = re.findall(r"\[[^\[]+\]", sent)
	sent_list=[]
	for chunk in chunks:
		chunk=chunk[1:-1]
		sent_list.append(chunk.split())
	return sent_list

def readSPMDataset(filename):
	sent1_list=[]
	sent2_list=[]
	label_list=[]
	for line in open(filename):
		label, id1, id2, sent1, sent2=line.strip().split('\t')
		if sent1=='#1 String' and sent2=='#2 String':
			continue
		sent1_list.append((sent1, None, None))
		sent2_list.append((sent2, None, None))
		label_list.append(int(label))
	return (sent1_list, sent2_list, label_list)

def readEdinburghDataset(filename):
	data = json.load(open(filename))
	#print(len(data))
	sent1_list = []
	sent2_list = []
	alignment_list = []
	for i in range(len(data)):
		source=data[i]['source']
		# source_pos_tag = nltk.pos_tag(source.split())
		# pos_source=data[i]['pos_source']
		target=data[i]['target']
		# target_pos_tag = nltk.pos_tag(target.split())
		# pos_target=data[i]['pos_target']
		alignment=data[i]['sureAlign']
		source_pos_tag = ' '.join([item[1] for item in nltk.pos_tag(source.split())])  # data[i]['target.pos']
		target_pos_tag = ' '.join([item[1] for item in nltk.pos_tag(target.split())])  # data[i]['source.pos']
		# if sure_and_possible:
		# 	alignment+=' '+data[i]['possibleAlign']
		# if len(pos_source.split())!=len(source.split()):
		# 	print(data[i])
		# if len(pos_target.split())!=len(target.split()):
		# 	print(data[i])
		sent1_list.append((source, source_pos_tag, None))
		sent2_list.append((target, target_pos_tag, None))
		alignment_list.append(alignment)
	return (sent1_list,sent2_list, alignment_list)

def readEdinburghDataset_reverse(filename, sure_and_possible=False, transpose=False):
	if 'human_errors' in filename:
		if '.tsv' in filename:
			data=[]
			for line in open(filename):
				# print(line)
				ID, sent1, _, sent2, _, _, _, sure_align, poss_align, *useless = line.strip('\n').split('\t')
				my_dict = {}
				my_dict['source'] = sent1
				my_dict['target'] = sent2
				my_dict['sureAlign'] = sure_align
				my_dict['possibleAlign'] = poss_align
				data.append(my_dict)
			if 'clean-dev' in filename:
				data2 = json.load(open(base_path + 'data/mt-reference/clean-version/clean-dev.json')) # clean-test-merged.json
			else:
				data2 = json.load(open(base_path + 'data/mt-reference/clean-version/clean-test-merged.json'))
			for i in range(len(data)):
				if data[i]['source']==data2[i]['source'] and data[i]['target']==data2[i]['target']:
					continue
				else:
					print(i)
					print(data[i]['source'])
					print(data2[i]['source'])
					print(data[i]['target'])
					print(data2[i]['target'])
					sys.exit()
		else:
			data=json.load(open(filename))
			if sure_and_possible:
				for i in range(len(data)):
					data[i]['sureAlign'] = data[i]['sure_poss_LUC']
			else:
				for i in range(len(data)):
					data[i]['sureAlign'] = data[i]['sure_only_LUC']
	elif ('newsela' in filename) or ('arxiv' in filename):
		data = []
		for line in open(filename):
			# print(line)
			ID, sent1, _, sent2, _, _, _, sure_align, poss_align, *useless = line.strip('\n').split('\t')
			my_dict = {}
			my_dict['source'] = sent1
			my_dict['target'] = sent2
			my_dict['sureAlign'] = sure_align + ' '+ poss_align
			my_dict['possibleAlign'] = ''
			data.append(my_dict)
	elif 'wikipedia' in filename:
		data = []
		for line in open(filename):
			# print(line)
			sent1, sent2, sure_align, _, _ = line.strip('\n').split('\t')
			my_dict = {}
			my_dict['source'] = sent1
			my_dict['target'] = sent2
			my_dict['sureAlign'] = sure_align
			my_dict['possibleAlign'] = ''
			data.append(my_dict)
	else:
		data = json.load(open(filename))
	#print(len(data))
	sent1_list = []
	sent2_list = []
	alignment_list = []
	for i in range(len(data)):
		if transpose:
			source=data[i]['target']
			target = data[i]['source']
		else:
			source = data[i]['source']
			target = data[i]['target']
		if 'target.parents' in data[i]:
			if transpose:
				source_dep_tree = (data[i]['target.parents'],data[i]['target.rels'])
				target_dep_tree = (data[i]['source.parents'], data[i]['source.rels'])
			else:
				target_dep_tree = (data[i]['target.parents'], data[i]['target.rels'])
				source_dep_tree = (data[i]['source.parents'], data[i]['source.rels'])
			# source_pos_tag = ' '.join([item[1] for item in nltk.pos_tag(source.split())]) # data[i]['target.pos']
			# target_pos_tag = ' '.join([item[1] for item in nltk.pos_tag(target.split())]) # data[i]['source.pos']
			# print(source_pos_tag)
		else:
			source_dep_tree = None
			target_dep_tree = None
			# source_pos_tag = None
			# target_pos_tag = None
		source_pos_tag = ' '.join([item[1] for item in nltk.pos_tag(source.split())])  # data[i]['target.pos']
		target_pos_tag = ' '.join([item[1] for item in nltk.pos_tag(target.split())])  # data[i]['source.pos']
		# print(source_dep_tree)
		# sys.exit()
		# pos_source=data[i]['pos_source']
		# pos_target=data[i]['pos_target']
		alignment=data[i]['sureAlign']
		if sure_and_possible:
			alignment+=' '+data[i]['possibleAlign']
		my_label=[]
		for item in alignment.split(): # reverse the alignment
			i, j = item.split('-')
			if transpose:
				my_label.append(str(j)+'-'+str(i))
			else:
				my_label.append(str(i) + '-' + str(j))
		alignment=' '.join(my_label)
		# if len(pos_source.split())!=len(source.split()):
		# 	print(data[i])
		# if len(pos_target.split())!=len(target.split()):
		# 	print(data[i])
		# print(source)
		# print(source.lower())
		# sys.exit()
		sent1_list.append((source.lower(), source_pos_tag, source_dep_tree))
		sent2_list.append((target.lower(), target_pos_tag, target_dep_tree))
		# sent1_list.append((source, None, None))
		# sent2_list.append((target, None, None))
		alignment_list.append(alignment)
	return (sent1_list,sent2_list, alignment_list)

def readWikismall(folder, data_type):
	sent1_list=[]
	sent2_list=[]
	alignment_list=[]
	for line in open(folder+'PWKP_108016.tag.80.aner.ori.'+data_type+'.src'):
		sent1_list.append((line.strip().lower(), None, None))
		alignment_list.append(None)
	for line in open(folder+'PWKP_108016.tag.80.aner.ori.'+data_type+'.dst'):
		sent2_list.append((line.strip().lower(), None, None))
	return (sent1_list, sent2_list, alignment_list)

def readNewselaold(folder, data_type):
	sent1_list=[]
	sent2_list=[]
	alignment_list=[]
	for line in open(folder+'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.'+data_type+'.src'):
		sent1_list.append((line.strip().lower(), None, None))
		alignment_list.append(None)
	for line in open(folder+'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.'+data_type+'.dst'):
		sent2_list.append((line.strip().lower(), None, None))
	return (sent1_list, sent2_list, alignment_list)

def readWikiQA(folder, data_type):
	sent1_list=[]
	sent2_list=[]
	alignment_list=[]
	for line in open(folder+data_type+'/a.toks'):
		sent1_list.append((line.strip().lower(), None, None))
		alignment_list.append(None)
	for line in open(folder+data_type+'/b.toks'):
		sent2_list.append((line.strip().lower(), None, None))
	return (sent1_list, sent2_list, alignment_list)

def generate_vocab(data_set_list):
	vocab={}
	word_freq={}
	count=0
	dep_tag_dict={}
	dep_tag_count=0
	vocab['__PAD__']=count
	count+=1
	for dataset in data_set_list:
		for item, dep_tag, _ in dataset[0]:
			for word in item.split():
				word_freq[word]=word_freq.get(word, 0)+1
				try:
					vocab[word]=vocab[word]+0
				except:
					vocab[word]=count
					count+=1
			# if pos:
			# 	for tag in pos.split():
			# 		try:
			# 			pos_dict[tag]+=0
			# 		except:
			# 			pos_dict[tag]=pos_count
			# 			pos_count+=1
		for item, dep_tag, _ in dataset[1]:
			for word in item.split():
				word_freq[word] = word_freq.get(word, 0) + 1
				try:
					vocab[word]=vocab[word]+0
				except:
					vocab[word]=count
					count+=1
			# if pos:
			# 	for tag in pos.split():
			# 		try:
			# 			pos_dict[tag]+=0
			# 		except:
			# 			pos_dict[tag]=pos_count
			# 			pos_count+=1
	return vocab, word_freq, dep_tag_dict

def eval_score(pred_seq, gold_seq):
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	acc=0
	for i in range(len(pred_seq)):
		if pred_seq[i]==1 and gold_seq[i]==1:
			tp+=1
		elif pred_seq[i]==1 and gold_seq[i]==0:
			fp+=1
		elif pred_seq[i]==0 and gold_seq[i]==1:
			fn+=1
		elif pred_seq[i]==0 and gold_seq[i]==0:
			tn+=1
	precision=0
	try:
		precision=tp/(tp+fp)
	except:
		pass
	recall=0
	try:
		recall=tp/(tp+fn)
	except:
		pass
	if precision+recall>0:
		f_score=2*precision*recall/(precision+recall)
	else:
		f_score=0
	acc=(tp+tn)/(len(pred_seq))
	return (precision, recall, f_score, acc)

def extract_phrase_alignment_from_ists_wa_file():
	chunk_statistics={}
	start_flag = False
	tmp_phrase=None
	for my_file in ['my_test_headlines_without_gold_chunks.wa', 'my_test_images_without_gold_chunks.wa']:
		for line in open(base_path + my_file):
			# line = line.encode('utf-8')
			# print(line)
			# sys.exit()
			if line.strip() == '<alignment>':
				start_flag = True
				tmp_phrase=None
				continue
			elif line.strip() == '</alignment>':
				if tmp_phrase:
					my_key = str(len(tmp_phrase[0])) + '_' + str(len(tmp_phrase[1]))
					chunk_statistics[my_key] = chunk_statistics.get(my_key, 0) + 1
				# print(chunk_statistics)
				# sys.exit()
				start_flag = False
				continue
			if start_flag:
				# my_tag=None
				# for item in label_set:
				# 	if item in line.strip().split('//')[1]:
				# 		my_tag=item
				# 		break
				my_label, my_tag, my_score, _ = line.strip().split('//')
				if 'NOALI' in my_tag:
					continue
				pair_left, pair_right = my_label.split('<==>')
				pair_left=int(pair_left)
				pair_right=int(pair_right)
				# print(tmp_phrase)
				if tmp_phrase:
					if tmp_phrase[0][-1]+1==pair_left and tmp_phrase[1][-1]==pair_right:
						tmp_phrase[0].append(pair_left)
					else:
						my_key=str(len(tmp_phrase[0]))+'_'+str(len(tmp_phrase[1]))
						chunk_statistics[my_key]=chunk_statistics.get(my_key,0)+1
						tmp_phrase=([pair_left], [pair_right])
				else:
					tmp_phrase=([pair_left], [pair_right])
	for key in sorted(chunk_statistics.keys()):
		print(key, chunk_statistics[key])

def neighboring_points(pair, e_len, f_len):
	"""
	A function that returns list of neighboring points in
	an alignment matrix for a given alignment (pair of indexes)
	"""
	result = []
	e_index, f_index = pair.split('-')
	e_index, f_index = int(e_index), int(f_index)
	if e_index > 0:
		result.append((e_index - 1, f_index))
	if f_index > 0:
		result.append((e_index, f_index - 1))
	if e_index < e_len - 1:
		result.append((e_index + 1, f_index))
	if f_index < f_len - 1:
		result.append((e_index, f_index + 1))
	if e_index > 0 and f_index > 0:
		result.append((e_index - 1, f_index - 1))
	if e_index > 0 and f_index < f_len - 1:
		result.append((e_index - 1, f_index + 1))
	if e_index < e_len - 1 and f_index > 0:
		result.append((e_index + 1, f_index - 1))
	if e_index < e_len - 1 and f_index < f_len - 1:
		result.append((e_index + 1, f_index + 1))

	return result

def merge_alignment(predA, predB, len_A, len_B):
	final_alignment=predA & predB
	union=predA | predB
	left2right={}
	right2left={}
	for pair in final_alignment:
		left, right = pair.split('-')
		left, right = int(left), int(right)
		if left in left2right:
			left2right[left].append(right)
		else:
			left2right[left]=[right]
		if right in right2left:
			right2left[right].append(left)
		else:
			right2left[right]=[left]
	new_points_added=True
	while new_points_added:
		new_points_added=False
		for pair in (predA & predB):
			for new_pair in neighboring_points(pair, len_A, len_B):
				left, right = new_pair
				if not (left in left2right and right in right2left) and str(left)+'-'+str(right) in union:
					if left in left2right:
						left2right[left].append(right)
					else:
						left2right[left] = [right]
					if right in right2left:
						right2left[right].append(left)
					else:
						right2left[right] = [left]
					final_alignment.add(str(left)+'-'+str(right))
					new_points_added=True
	for i in range(len_A):
		for j in range(len_B):
			if (i not in left2right or j not in right2left) and str(i)+'-'+str(j) in union:
				final_alignment.add(str(i)+'-'+str(j))
	return final_alignment
