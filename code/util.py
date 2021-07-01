from __future__ import division
import random
import torch
import numpy as np

seed=123456
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

base_path = '../' # working directory

if torch.cuda.is_available():
	print('CUDA is available!')
	my_device = torch.device('cuda:0')
	torch.cuda.manual_seed_all(seed)
else:
	my_device = torch.device('cpu')

max_span_size=3
max_sent_length=150
bidirectional_training=True
bidirectional_decoding=True


def convert_spanID_to_stateID(span, sent_length):
	span_length=len(span)
	return int(sent_length*(span_length-1)-(span_length-1)*(span_length-2)/2 + span[0] + 1)

def convert_stateID_to_spanID(stateID, sent_length): # 0 is NULL state
	stateID=stateID-1
	if stateID<0:
		return (-1, -1)
	else:
		for span_length in range(1, max_span_size+1):
			lower_bound=(span_length-1)*sent_length-int((span_length-1)*(span_length-2)/2)
			upper_bound=span_length*sent_length-int(span_length*(span_length-1)/2)
			if stateID>=lower_bound and stateID<upper_bound:
				return (stateID-lower_bound, span_length) # return (spanID, span_Length)

def read_Word_Alignment_Dataset(filename, sure_and_possible=False, transpose=False):
	data = []
	for line in open(filename):
		ID, sent1, _, sent2, _, _, _, sure_align, poss_align, *useless = line.strip('\n').split('\t')
		my_dict = {}
		my_dict['source'] = sent1
		my_dict['target'] = sent2
		my_dict['sureAlign'] = sure_align
		my_dict['possibleAlign'] = poss_align
		data.append(my_dict)
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
		alignment=data[i]['sureAlign']
		if sure_and_possible:
			alignment+=' '+data[i]['possibleAlign']
		my_label=[]
		for item in alignment.split(): # reverse the alignment
			i, j = item.split('-')
			if transpose:
				my_label.append(str(j) + '-' + str(i))
			else:
				my_label.append(str(i) + '-' + str(j))
		alignment=' '.join(my_label)
		sent1_list.append((source, None, None))
		sent2_list.append((target, None, None))
		alignment_list.append(alignment)
	return (sent1_list,sent2_list, alignment_list)

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
