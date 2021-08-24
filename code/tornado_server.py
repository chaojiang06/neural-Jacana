import asyncio
import secrets

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler, Application
import os,yaml

import argparse
import numpy as np
import torch,random
import torch.nn as nn
from transformers import *
from torch import logsumexp
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

def convert_examples_to_features(data_examples, set_type, max_seq_length, tokenizer, pad_token=0, pad_token_segment_id=0):
	features = []
	len_examples=len(data_examples)
	for ex_index, example in enumerate(data_examples):
		if ex_index % 200 == 0:
			print("Processed "+set_type+" examples %d/%d" % (ex_index, len_examples))

		if len(example.text_a.split())>args.max_sent_length or len(example.text_b.split())>args.max_sent_length:
			if set_type=='train':
				continue
			else:
				example.text_a=' '.join(example.text_a.split()[:args.max_sent_length])
				example.text_b=' '.join(example.text_b.split()[:args.max_sent_length])
		inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_seq_length)
		input_ids_a_and_b, token_type_ids_a_and_b = inputs["input_ids"], inputs["token_type_ids"]
		# print(inputs)
		inputs = tokenizer.encode_plus(example.text_b, example.text_a, add_special_tokens=True, max_length=max_seq_length)
		input_ids_b_and_a, token_type_ids_b_and_a = inputs["input_ids"], inputs["token_type_ids"]
		# print(example)
		# print(input_ids_a_and_b)
		# print(input_ids_b_and_a)
		# sys.exit()
		attention_mask = inputs["attention_mask"] #[1] * len(input_ids_a_and_b)
		padding_length = max_seq_length - len(input_ids_a_and_b)
		input_ids_a_and_b = input_ids_a_and_b + ([pad_token] * padding_length)
		input_ids_b_and_a = input_ids_b_and_a + ([pad_token] * padding_length)
		attention_mask = attention_mask + ([0] * padding_length)
		token_type_ids_a_and_b = token_type_ids_a_and_b + ([pad_token_segment_id] * padding_length)
		token_type_ids_b_and_a = token_type_ids_b_and_a + ([pad_token_segment_id] * padding_length)
		sent1_valid_ids = []  #
		sent2_valid_ids = []
		sent1_wordpiece_length = 0
		sent2_wordpiece_length = 0
		total_length = 1  # [CLS]
		for idx, word in enumerate(example.text_a.split()):
			token = tokenizer.tokenize(word)
			sent1_valid_ids.append(total_length)
			total_length += len(token)
			sent1_wordpiece_length += len(token)
		total_length += 1  # [SEP]
		for idx, word in enumerate(example.text_b.split()):
			token = tokenizer.tokenize(word)
			sent2_valid_ids.append(total_length)
			total_length += len(token)
			sent2_wordpiece_length += len(token)
		# print(total_length, sum(attention_mask))
		# sys.exit()
		sent1_valid_ids = sent1_valid_ids + ([0] * (max_seq_length - len(sent1_valid_ids)))
		sent2_valid_ids = sent2_valid_ids + ([0] * (max_seq_length - len(sent2_valid_ids)))
		assert len(input_ids_a_and_b) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids_a_and_b), max_seq_length)
		assert len(attention_mask) == max_seq_length, "Error with input length {} vs {}".format(len(attention_mask), max_seq_length)
		assert len(token_type_ids_a_and_b) == max_seq_length, "Error with input length {} vs {}".format(len(token_type_ids_a_and_b), max_seq_length)
		assert len(sent1_valid_ids) == max_seq_length, "Error with input length {} vs {}".format(len(sent1_valid_ids), max_seq_length)
		assert len(sent2_valid_ids) == max_seq_length, "Error with input length {} vs {}".format(len(sent2_valid_ids), max_seq_length)
		label_s2t=np.zeros((args.max_sent_length, args.max_span_size), dtype=int) -1
		label_t2s=np.zeros((args.max_sent_length, args.max_span_size), dtype=int) -1
		left2right = {}
		for pair in example.label.split():
			left, right = pair.split('-')
			left, right = int(left), int(right)
			if left in left2right:
				left2right[left].append(right)
			else:
				left2right[left] = [right]
		for key in left2right:
			left2right[key]=sorted(left2right[key])
			# print(target_span_list, target_span_list[:target_i+1])
		lenA=len(example.text_a.split())
		lenB=len(example.text_b.split())
		alignment_pair=[]
		new_key = []
		for i in range(lenA):
			if i in left2right:
				if len(new_key) == 0:
					new_key = [i]
				elif left2right[new_key[-1]] == left2right[i] and i == (new_key[-1] + 1):
					new_key.append(i)
				else:
					alignment_pair.append((new_key, left2right[new_key[-1]]))
					new_key = [i]
		alignment_pair.append((new_key, left2right[new_key[-1]]))
		source_id_tagged=set()
		target_id_tagged=set()
		for pair in alignment_pair:
			source_id_list, target_id_list = pair
			# source_id_list=source_id_list[:args.max_span_size]
			# target_id_list=target_id_list[:args.max_span_size]
			source_id_tagged = source_id_tagged | set(source_id_list)
			target_id_tagged = target_id_tagged | set(target_id_list)
			tag_idx=0
			for d in range(min(len(target_id_list), args.max_span_size)):
				if d==len(target_id_list)-1 or d==args.max_span_size-1:
					tag_idx+=target_id_list[0]+1
				else:
					tag_idx+=lenB-d
			if len(source_id_list)>args.max_span_size:
				label_s2t[source_id_list[args.max_span_size-1]][args.max_span_size-1]=tag_idx
				for my_idx in range(args.max_span_size, len(source_id_list)):
					label_s2t[source_id_list[my_idx]][0]=tag_idx
			else:
				label_s2t[source_id_list[-1]][len(source_id_list)-1]=tag_idx
			tag_idx=0
			for d in range(min(len(source_id_list), args.max_span_size)):
				if d==len(source_id_list)-1 or d==args.max_span_size-1:
					tag_idx+=source_id_list[0]+1
				else:
					tag_idx+=lenA-d
			if len(target_id_list)>args.max_span_size:
				label_t2s[target_id_list[args.max_span_size-1]][args.max_span_size - 1] = tag_idx
				for my_idx in range(args.max_span_size, len(target_id_list)):
					label_t2s[target_id_list[my_idx]][0]=tag_idx
			else:
				label_t2s[target_id_list[-1]][len(target_id_list)-1]=tag_idx
		for i in range(lenA):
			if i not in source_id_tagged:
				label_s2t[i][0]=0
		for i in range(lenB):
			if i not in target_id_tagged:
				label_t2s[i][0]=0
		# print(example)
		# print(label_s2t)
		# print(label_t2s)
		# sys.exit()
		features.append(Features(input_ids_a_and_b=input_ids_a_and_b, input_ids_b_and_a=input_ids_b_and_a, attention_mask=attention_mask, segment_ids_a_and_b=token_type_ids_a_and_b,
		                         segment_ids_b_and_a=token_type_ids_b_and_a,sent1_valid_ids=sent1_valid_ids, sent2_valid_ids=sent2_valid_ids, sent1_wordpiece_length=sent1_wordpiece_length,
		                         sent2_wordpiece_length=sent2_wordpiece_length,label_s2t=label_s2t, label_t2s=label_t2s))
	return features

def create_Data_Loader(data_examples, set_type='train', batchsize=16, max_seq_length=128, tokenizer=None):
	data_features = convert_examples_to_features(data_examples, set_type, max_seq_length, tokenizer)
	all_input_ids_a_and_b = torch.tensor([f.input_ids_a_and_b for f in data_features], dtype=torch.long)
	all_input_ids_b_and_a = torch.tensor([f.input_ids_b_and_a for f in data_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.attention_mask for f in data_features], dtype=torch.long, requires_grad=False)
	all_segment_ids_a_and_b = torch.tensor([f.segment_ids_a_and_b for f in data_features], dtype=torch.long)
	all_segment_ids_b_and_a = torch.tensor([f.segment_ids_b_and_a for f in data_features], dtype=torch.long)
	all_sent1_valid_ids = torch.tensor([f.sent1_valid_ids for f in data_features], dtype=torch.long)
	all_sent2_valid_ids = torch.tensor([f.sent2_valid_ids for f in data_features], dtype=torch.long)
	all_sent1_wordpiece_length = torch.tensor([f.sent1_wordpiece_length for f in data_features], dtype=torch.long)
	all_sent2_wordpiece_length = torch.tensor([f.sent2_wordpiece_length for f in data_features], dtype=torch.long)
	if set_type == 'train':
		all_label_s2t = torch.tensor([f.label_s2t for f in data_features])
		all_label_t2s = torch.tensor([f.label_t2s for f in data_features])
		dataset = TensorDataset(all_input_ids_a_and_b, all_input_ids_b_and_a, all_input_mask, all_segment_ids_a_and_b, all_segment_ids_b_and_a, all_sent1_valid_ids, all_sent2_valid_ids, all_sent1_wordpiece_length, all_sent2_wordpiece_length, all_label_s2t, all_label_t2s)
		data_sampler = RandomSampler(dataset)
	else:
		dataset = TensorDataset(all_input_ids_a_and_b, all_input_ids_b_and_a, all_input_mask, all_segment_ids_a_and_b, all_segment_ids_b_and_a, all_sent1_valid_ids, all_sent2_valid_ids, all_sent1_wordpiece_length, all_sent2_wordpiece_length)
		data_sampler = SequentialSampler(dataset)
	dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batchsize)
	return dataloader

def put_a_number_in_bin_and_binary(a, return_fired_idx = False):
	bins = np.array([-11, -6, -4, -3, -2, -1, 0, 1, 2, 3, 5, 10])
	fired_idx = np.digitize(a, bins, right = True)

	if return_fired_idx == True:
		return fired_idx

	d = np.zeros(13)
	d[fired_idx] = 1

	return d

def convert_stateID_to_spanID(stateID, sent_length): # 0 is NULL state
	stateID=stateID-1
	if stateID<0:
		return (-1, -1)
	else:
		for span_length in range(1, args.max_span_size+1):
			lower_bound=(span_length-1)*sent_length-int((span_length-1)*(span_length-2)/2)
			upper_bound=span_length*sent_length-int(span_length*(span_length-1)/2)
			if stateID>=lower_bound and stateID<upper_bound:
				return (stateID-lower_bound, span_length) # return (spanID, span_Length)

class Features(object):
	def __init__(self, input_ids_a_and_b, input_ids_b_and_a, attention_mask, segment_ids_a_and_b, segment_ids_b_and_a, sent1_valid_ids, sent2_valid_ids, sent1_wordpiece_length, sent2_wordpiece_length, label_s2t, label_t2s):
		self.input_ids_a_and_b = input_ids_a_and_b
		self.input_ids_b_and_a = input_ids_b_and_a
		self.attention_mask = attention_mask
		self.segment_ids_a_and_b = segment_ids_a_and_b
		self.segment_ids_b_and_a = segment_ids_b_and_a
		self.sent1_valid_ids = sent1_valid_ids
		self.sent2_valid_ids = sent2_valid_ids
		self.sent1_wordpiece_length=sent1_wordpiece_length
		self.sent2_wordpiece_length=sent2_wordpiece_length
		self.label_s2t = label_s2t
		self.label_t2s = label_t2s

class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super().__init__()
		self.gamma = nn.Parameter(torch.ones(features))
		self.beta = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta

class NeuralWordAligner(nn.Module):
	def __init__(self, args):
		super(NeuralWordAligner, self).__init__()
		self.args = args
		# self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
		self.bert_model = BertModel.from_pretrained('/data/lan/neural-semi-crf/src/spanbert_hf_base/', output_hidden_states=True, output_attentions=True)
		self.attn_proj = nn.Linear(768, 100)
		self.attn_embd = nn.Embedding(1, 100)
		self.default_tag = -1
		self.layer_norm = LayerNorm(768 * 3)
		self.sim_feature_num=12
		span_representation_dim = 768
		self.span_representation_dim=span_representation_dim
		self.mlp1 = nn.Sequential(nn.Linear(self.sim_feature_num, self.sim_feature_num), nn.PReLU(),nn.Linear(self.sim_feature_num, 1))
		self.FF_kernel_spanbert = nn.Sequential(*[nn.Linear((span_representation_dim*3)*4, (span_representation_dim*3)), nn.PReLU(),nn.Linear((span_representation_dim*3), (span_representation_dim*3)), nn.PReLU(), nn.Linear((span_representation_dim*3), self.sim_feature_num)])
		self.mlp2 = nn.Sequential(nn.Linear(args.distance_embedding_size, 1))
		# self.mlp2 = nn.Sequential(nn.Linear(args.distance_embedding_size, args.distance_embedding_size), nn.PReLU(), nn.Linear(args.distance_embedding_size,1))
		self.distance_embedding = nn.Embedding(13, args.distance_embedding_size)
		self.distance_embedding.weight.requires_grad = False
		self.transition_matrix_dict = {}
		for len_B in range(args.max_span_size, args.max_sent_length + 1, 1):
			extended_length_B = args.max_span_size * len_B - int(args.max_span_size * (args.max_span_size - 1) / 2)

			tmp_dist_idx_list = np.zeros((extended_length_B + 1, extended_length_B + 1))
			if args.use_transition_layer=='True':
				for j in range(extended_length_B + 1):  # 0 is NULL state
					for k in range(extended_length_B + 1):  # k is previous state, j is current state
						tmp_dist_idx_list[j][k]=put_a_number_in_bin_and_binary(self.distortionDistance(k, j, len_B), return_fired_idx=True)
			self.transition_matrix_dict[extended_length_B] = tmp_dist_idx_list

	def _input_likelihood(self,
	                      logits: torch.Tensor,
	                      T: torch.Tensor,
	                      text_mask: torch.Tensor,
	                      hamming_cost: torch.Tensor,
	                      tag_mask: torch.Tensor) -> torch.Tensor:
		"""
		Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible segmentations.
		:param logits: shape (batch_size, sequence_length, max_span_width, num_tags)
		:param T: shape (batch_size, num_tags, num_tags)
		:param text_mask: shape (batch_size, sequence_length)
		:param hamming_cost: shape (batch_size, sequence_length, max_span_width, num_tags)
		:param tag_mask: shape (batch_size, num_tags, num_tags)
		:return: (batch_size,) denominator score
		"""
		batch_size, sequence_length, max_span_width, num_tags = logits.size()
		logits = logits.permute(1,2,0,3).contiguous() # shape: (sequence_length, max_span_width, batch_size, num_tags)
		hamming_cost = hamming_cost.permute(1,2,0,3).contiguous() # shape: (sequence_length, max_span_width, batch_size, num_tags)

		if self.args.use_transition_layer == "False":
			alpha = Variable(torch.FloatTensor([[0.0 for _ in range(batch_size)]]).to(my_device), requires_grad=True) # 1 * batch_size
		else:
			alpha = Variable(torch.FloatTensor(np.zeros((1, batch_size, num_tags))).to(my_device), requires_grad=True) # 1 * batch_size * num_tags

		# For each j we compute logits for all the segmentations of length j.
		for j in range(sequence_length):
			width = max_span_width
			if j < max_span_width - 1:
				width = j + 1

			# Reverse the alpha so it gets added to the correct logits.
			idx = Variable(torch.LongTensor([i for i in range(j, j - width, -1)]).to(my_device))
			reversed_alpha = alpha.index_select(dim=0, index=idx)
			# Tensorize and broadcast along the max_span_width dimension.
			broadcast_alpha = reversed_alpha.view(width, batch_size, -1) # shape: (max_span_width, batch_size, num_tags)
			logits_at_j = logits[j]
			start_indices = Variable(torch.LongTensor(range(width)).to(my_device))
			span_factors = logits_at_j.index_select(dim=0, index=start_indices)
			span_costs = hamming_cost[j].index_select(dim=0, index=start_indices)


			# if j==5 and False:
			# 	print(broadcast_alpha.size()) # span_width * bsz * 1 or span_width * bsz * num_tags
			# 	print(span_factors.size()) # span_width * bsz * num_tags
			# 	print(span_costs.size()) # span_width * bsz * num_tags
			# 	sys.exit()
			if self.args.use_transition_layer == "False":
				# Logsumexp the scores over the num_tags axis.
				alpha_along_arglabels = logsumexp(broadcast_alpha + (span_factors + span_costs) * tag_mask[:,0, :], dim=-1)
				# Logsumexp the scores over the width axis.
				alpha_at_j = logsumexp(alpha_along_arglabels, dim=0).view(1, batch_size)
				alpha = torch.cat([alpha, alpha_at_j], dim=0)
			else:
				alpha_along_arglabels = []
				for k in range(num_tags):
					emit_score = ((span_factors[:,:,k] + span_costs[:,:,k]) * tag_mask[:, 0, k]).unsqueeze(-1).repeat(1, 1, num_tags)
					tran_score = T[:,k,:]*tag_mask[:,k,:] # bsz * num_tags
					if j<width:
						tran_score = tran_score.unsqueeze(0).repeat(width, 1, 1)
						tran_score[j:,:,:]=0
					# Logsumexp the scores over the num_tags axis.
					alpha_along_arglabels.append(logsumexp(broadcast_alpha+tran_score+emit_score, dim=-1, keepdim=True))
				alpha_along_arglabels = torch.cat(alpha_along_arglabels, dim=-1)
				# Logsumexp the scores over the width axis.
				alpha_at_j = logsumexp(alpha_along_arglabels, dim=0, keepdim=True) # 1*bsz *num_tags
				alpha = torch.cat([alpha, alpha_at_j], dim=0)

		if self.args.use_transition_layer == "True":
			alpha = logsumexp(alpha, dim=-1)
		# Get the last positions for all alphas in the batch.
		actual_lengths = torch.sum(text_mask, dim=1).view(1, batch_size)
		# Finally we return the alphas along the last "valid" positions.
		# shape: (batch_size)
		partition = alpha.gather(dim=0, index=actual_lengths)
		return partition

	def _joint_likelihood(self,
	                      logits: torch.Tensor,
	                      T: torch.Tensor,
	                      tags: torch.Tensor,
	                      text_mask: torch.LongTensor) -> torch.Tensor:
		"""
		Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
		:param logits: shape (batch_size, sequence_length, max_span_width, num_tags)
		:param T: shape (batch_size, num_tags, num_tags)
		:param tags: golden label shape (batch_size, sequence_length, max_span_width)
		:param text_mask: shape (batch_size, sequence_length)
		:return: (batch_size,) numerator score
		"""
		batch_size, sequence_length, max_span_width, num_tags = logits.shape
		logits = logits.permute(1,2,0,3).contiguous() # shape: (sequence_length, max_span_width, batch_size, num_tags)
		# Transpose to shape: (sequence_length, batch_size)
		text_mask = text_mask.float().transpose(0, 1).contiguous()
		# Transpose to shape: (sequence_length, max_span_width, batch_size)
		tags = tags.permute(1, 2, 0).contiguous()

		default_tags = Variable(self.default_tag * torch.ones(batch_size).long().to(my_device))

		numerator = 0.0
		for j in range(sequence_length):
			for d in range(min(max_span_width, j+1)):
				current_tag = tags[j][d]
				valid_tag_mask = (current_tag != default_tags).long()
				current_tag = current_tag*valid_tag_mask
				# Reshape for gather operation to follow.
				current_tag = current_tag.view(batch_size, 1)
				# The score for using current_tag
				emit_score = logits[j][d].gather(dim=1, index=current_tag).squeeze(1) * valid_tag_mask * text_mask[j]
				numerator += emit_score
				if j>=1 and self.args.use_transition_layer=='True' and j-(d+1) >=0 and valid_tag_mask.sum()>0:
					for last_d in range(min(max_span_width, j)):
						last_tag = tags[j-(d+1)][last_d]
						last_valid_tag_mask = (last_tag != default_tags).long()
						# print(last_tag, last_valid_tag_mask)
						# print(last_tag, last_valid_tag_mask, current_tag.view(-1), valid_tag_mask)
						# print(last_valid_tag_mask)
						last_tag = last_tag*last_valid_tag_mask
						last_tag = last_tag.view(batch_size, 1, 1)
						last_tag = last_tag.repeat(1, num_tags, 1)
						trans_score = T.gather(dim=2, index=last_tag).squeeze(-1)
						trans_score = trans_score.gather(dim=1, index=current_tag).squeeze(1)
						trans_score = trans_score * last_valid_tag_mask * valid_tag_mask * text_mask[j]
						# print(trans_score)
						numerator += trans_score
		# print(tags)
		# sys.exit()
		return numerator

	def viterbi_tags(self,
	                 logits: Variable,
	                 T: Variable,
	                 text_mask: Variable,
	                 tag_masks: Variable,
	                 reverse = False):
		"""
		:param logits: shape (batch_size, sequence_length, max_span_width, num_tags)
		:param T: shape (batch_size, num_tags, num_tags)
		:param text_mask: shape (batch_size, sequence_length)
		:param tag_mask: shape (batch_size, num_tags, num_tags)
		:return:
		"""
		batch_size, max_seq_length, max_span_width, num_classes = logits.size()

		# Get the tensors out of the variables
		tag_masks = tag_masks[:,0,:] # shape (batch_size, num_tags)
		logits, text_mask, tag_masks = logits.data, text_mask.data, tag_masks.data
		sequence_lengths = torch.sum(text_mask, dim=-1)

		all_tags = []
		for logits_ex, tag_mask, sequence_length, transition_matrix in zip(logits, tag_masks, sequence_lengths, T):

			# logits_ex: max_seq_length * max_span_width * num_classes
			emission_matrix=[]
			for d in range(max_span_width):
				extracted_logit=logits_ex[d:sequence_length,d:d+1,:int(sum(tag_mask))] # num_spans * 1 * num_tags
				extracted_logit=extracted_logit.squeeze(1)
				extracted_logit=torch.cat([extracted_logit[:,1:], extracted_logit[:,0:1]], 1)
				emission_matrix.append(extracted_logit)
			emission_matrix=torch.cat(emission_matrix, dim=0)
			# We pass the logits to ``viterbi_decoder``.
			optimal_sequence = self.viterbi_decoder(emission_matrix, transition_matrix, sequence_length, sum(tag_mask)-1)
			predA = set()
			target_sent_length = int((args.max_span_size - 1) / 2 + (sum(tag_mask) - 1) / self.args.max_span_size)
			for i, stateID in enumerate(optimal_sequence):
				if stateID==0:
					continue
				stateID-=1
				for d in range(1, self.args.max_span_size+1):
					lower_bound=(d-1)*target_sent_length - int((d-1)*(d-2)/2)
					upper_bound = d * target_sent_length - int(d * (d - 1) / 2)
					if stateID >= lower_bound and stateID < upper_bound:
						spanID, spanLength = stateID - lower_bound, d
						for kk in range(spanLength):
							if reverse:
								predA.add(str(spanID + kk) + '-' + str(i))
							else:
								predA.add(str(i) + '-' + str(spanID + kk))
						break
			all_tags.append(predA)

		return all_tags

	def viterbi_decoder(self, emission_matrix, transition_matrix, len_A, extended_length_B):
		"""
		:param emission_matrix:  extended_length_A * (extended_length_B + 1), word/phrase pair interaction matrix
		:param transition_matrix: (extended_length_B + 1) * (extended_length_B + 1), state transition matrix
		:param len_A: source sentence length
		:param len_B: target sentence length
		:return: optimal sequence
		"""
		emission_matrix=emission_matrix.data.cpu().numpy()
		transition_matrix=transition_matrix.data.cpu().numpy()
		len_A=len_A.data.cpu().numpy()
		extended_length_B=int(extended_length_B.data.cpu().numpy())
		T1=np.zeros((len_A, extended_length_B + 1), dtype=float)
		T2=np.zeros((len_A, extended_length_B + 1), dtype=int)
		T3=np.zeros((len_A, extended_length_B + 1), dtype=int)
		for j in range(extended_length_B+1):
			# T1[0][j]=start_transition[j] + emission_matrix[0][j-1]
			T1[0][j] = emission_matrix[0][j - 1]
			T2[0][j]=-1
			T3[0][j]=1 # span size

		for i in range(1, len_A):
			global_max_val = float("-inf")
			for j in range(extended_length_B+1):
				max_val = float("-inf")
				for span_size in range(1, min(i + 1, args.max_span_size) + 1):
					for k in range(extended_length_B+1):
						if i-span_size>=0:
							cur_val = T1[i-span_size][k] + transition_matrix[j][k] + emission_matrix[i - (span_size-1) +(span_size-1)*len_A-int((span_size-1)*(span_size-2)/2)][j-1]
							# if i==len_A-1:
							# 	cur_val+=end_transition[j]
						else:
							cur_val = emission_matrix[i - (span_size-1) + (span_size-1)*len_A - int((span_size-1)*(span_size-2)/2)][j-1]
						if cur_val>max_val:
							T1[i][j]=cur_val
							T2[i][j]=k
							T3[i][j]=span_size
							max_val=cur_val
				if max_val>global_max_val:
					global_max_val=max_val
		optimal_sequence=[]
		max_val=float("-inf")
		max_idx=-1
		for j in range(extended_length_B+1):
			if T1[len_A-1][j]>max_val:
				max_idx=j
				max_val=T1[len_A-1][j]
		# optimal_sequence = [max_idx] + optimal_sequence
		# for i in range(len_A - 1, 0, -1):
		# 	optimal_sequence = [T2[i][max_idx]] + optimal_sequence
		# 	max_idx = T2[i][max_idx]
		i=len_A-1
		while i>=0:
			optimal_element=[max_idx]*T3[i][max_idx]
			optimal_sequence=optimal_element+optimal_sequence
			new_i = i - T3[i][max_idx]
			new_max_idx = T2[i][max_idx]
			i=new_i
			max_idx=new_max_idx

		return optimal_sequence

	def _compose_span_representations(self, seq1, seq2):
		"""

		:param seq1: bsz*max_sent1_length*768, unigram representations arranged first, then followed by CLS representations
		:param seq2: bsz*max_sent2_length*768, unigram representations arranged first, then followed by CLS representations
		:return: (seq1, seq2): bsz* num_all_spans *768
		"""
		seq1_spans=[]
		seq2_spans=[]
		bsz, max_sent1_length, _ = seq1.size()
		bsz, max_sent2_length, _ = seq2.size()
		idx = Variable(torch.LongTensor([0])).to(my_device)
		for d in range(self.args.max_span_size):
			seq1_spans_d=[]
			for i in range(max_sent1_length-d):
				current_ngrams=seq1[:,i:i+d+1,:] # bsz * current_span_size * 768
				alpha = torch.matmul(self.attn_proj(current_ngrams), self.attn_embd(idx).view(-1,1)) # bsz * span_size * 1
				alpha = F.softmax(alpha, dim=1)
				seq1_spans_d.append(torch.cat([current_ngrams[:,:1,:], current_ngrams[:,-1:,:], torch.matmul(alpha.permute(0, 2, 1), current_ngrams)], dim=-1))
			seq1_spans_d = [torch.zeros((bsz, 1, self.span_representation_dim*3)).to(my_device)]* (max_sent1_length - len(seq1_spans_d)) + seq1_spans_d
			seq1_spans_d=torch.cat(seq1_spans_d, dim=1) # bsz * max_sent1_length * dim
			seq1_spans.append(seq1_spans_d)
			seq2_spans_d=[]
			for i in range(max_sent2_length-d):
				current_ngrams=seq2[:,i:i+d+1,:]
				alpha = torch.matmul(self.attn_proj(current_ngrams), self.attn_embd(idx).view(-1, 1))  # bsz * span_size * 1
				alpha = F.softmax(alpha, dim=1)
				seq2_spans_d.append(torch.cat([current_ngrams[:, :1, :], current_ngrams[:, -1:, :], torch.matmul(alpha.permute(0, 2, 1), current_ngrams)], dim=-1))
			seq2_spans_d = [torch.zeros((bsz, 1, self.span_representation_dim*3)).to(my_device)]* (max_sent2_length-len(seq2_spans_d)) + seq2_spans_d
			seq2_spans_d=torch.cat(seq2_spans_d, dim=1) # bsz * max_sent2_length * dim
			seq2_spans.append(seq2_spans_d)
		return torch.cat(seq1_spans, dim=1), torch.cat(seq2_spans, dim=1)

	def compute_sim_cube_FF_spanbert(self, seq1, seq2):
		'''
		:param seq1: bsz * len_seq1 * dim
		:param seq2: bsz * len_seq2 * dim
		:return:
		'''
		def compute_sim_FF(prism1, prism2):
			features = torch.cat([prism1, prism2, torch.abs(prism1 - prism2), prism1 * prism2], dim=-1)
			FF_out = self.FF_kernel_spanbert(features)
			return FF_out.permute(0,3,1,2)

		def compute_prism(seq1, seq2):
			prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
			prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
			prism1 = prism1.permute(1, 2, 0, 3).contiguous()
			prism2 = prism2.permute(1, 0, 2, 3).contiguous()
			return compute_sim_FF(prism1, prism2)

		sim_cube = torch.Tensor(seq1.size(0), self.sim_feature_num, seq1.size(1), seq2.size(1)) # bsz * feature_num * len_seq1 * len_seq2
		sim_cube = sim_cube.to(my_device)
		sim_cube[:, 0:self.sim_feature_num] = compute_prism(seq1, seq2)
		return sim_cube

	def create_pad_cube(self, sent1_lengths, sent2_lengths):
		pad_cube = []
		max_len1 = max(sent1_lengths)
		max_len2 = max(sent2_lengths)

		for s1_length, s2_length in zip(sent1_lengths, sent2_lengths):
			pad_mask = np.ones((max_len1, max_len2))
			pad_mask[:s1_length, :s2_length] = 0
			pad_cube.append(pad_mask)

		pad_cube = np.array(pad_cube)
		return torch.from_numpy(pad_cube).float().to(my_device).unsqueeze(0)

	def distortionDistance(self, state_i, state_j, sent_length):
		start_i, size_i=convert_stateID_to_spanID(state_i, sent_length)
		start_j, size_j=convert_stateID_to_spanID(state_j, sent_length)
		return start_j - (start_i + size_i - 1) - 1

	def forward(
			self,
			input_ids_a_and_b=None,
			input_ids_b_and_a=None,
			attention_mask=None,
			token_type_ids_a_and_b=None,
			token_type_ids_b_and_a=None,
			sent1_valid_ids=None,
			sent2_valid_ids=None,
			sent1_wordpiece_length=None,
			sent2_wordpiece_length=None,
			labels_s2t=None,
			labels_t2s=None,
	):
		sent1_lengths = torch.sum(sent1_valid_ids > 0, dim=1).cpu().numpy()
		sent2_lengths = torch.sum(sent2_valid_ids > 0, dim=1).cpu().numpy()
		max_len1 = max(sent1_lengths)
		max_len2 = max(sent2_lengths)
		# print(sent1_lengths)
		# print(sent2_lengths)
		outputs = self.bert_model(
			input_ids=input_ids_a_and_b,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids_a_and_b,
		)
		all_layer_hidden_states, all_layer_attention_weights = outputs[-2:]
		# print(outputs[0].size()) # bsz*128*768
		# print(all_layer_hidden_states[-1].size()) # bsz*128*768
		# print(all_layer_attention_weights[-1].size()) # bsz*12*128*128
		all_hidden_states = all_layer_hidden_states[-1]  # the last layer, bsz*128*768
		batch_size, max_sequence_length, hidden_dim = outputs[0].size()
		# print(all_hidden_states)
		# print(outputs[0]) # the same thing as all_hidden_states
		outputs_b_and_a = self.bert_model(
			input_ids=input_ids_b_and_a,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids_b_and_a,
		)
		all_layer_hidden_states_b_and_a, all_layer_attention_weights_b_and_a = outputs_b_and_a[-2:]
		all_hidden_states_b_and_a = all_layer_hidden_states_b_and_a[-1]
		seq1 = []
		seq2 = []
		for i in range(batch_size):
			for j in range(sent1_lengths[i]-1):
				diff=sent1_valid_ids[i][j+1]-sent1_valid_ids[i][j]
				if diff>1:
					position=sent1_valid_ids[i][j]
					# print(j,position)
					all_hidden_states[i][position:position+1,:]=torch.mean(all_hidden_states[i][position:position+diff, :], dim=0, keepdim=True)
					position=sent1_valid_ids[i][j]+sent2_wordpiece_length[i]+1
					# print(j,position)
					# print(sent2_wordpiece_length)
					# sys.exit()
					all_hidden_states_b_and_a[i][position:position+1,:]=torch.mean(all_hidden_states_b_and_a[i][position:position+diff, :], dim=0, keepdim=True)
			# print(sent1_valid_ids[i])
			# print(sent1_valid_ids[i]+sent2_wordpiece_length[i]+1)
			sent1_valid_ids_part2 = (sent1_valid_ids[i]+sent2_wordpiece_length[i]+1)*(sent1_valid_ids[i]>0).int()
			# print(sent1_valid_ids_part2)
			# sys.exit()
			seq1.append(torch.index_select(all_hidden_states[i], 0, sent1_valid_ids[i]).unsqueeze(0) + torch.index_select(all_hidden_states_b_and_a[i], 0, sent1_valid_ids_part2).unsqueeze(0))
			for j in range(sent2_lengths[i]-1):
				diff=sent2_valid_ids[i][j+1]-sent2_valid_ids[i][j]
				if diff>1:
					position=sent2_valid_ids[i][j]
					all_hidden_states[i][position:position+1, :]=torch.mean(all_hidden_states[i][position:position+diff,:], dim=0, keepdim=True)
					position=sent2_valid_ids[i][j]-sent1_wordpiece_length[i]-1
					all_hidden_states_b_and_a[i][position:position+1,:]=torch.mean(all_hidden_states_b_and_a[i][position:position+diff, :], dim=0, keepdim=True)
			sent2_valid_ids_part2 = (sent2_valid_ids[i]-sent1_wordpiece_length[i]-1)*(sent2_valid_ids[i]>0).int()
			seq2.append(torch.index_select(all_hidden_states[i], 0, sent2_valid_ids[i]).unsqueeze(0) + torch.index_select(all_hidden_states_b_and_a[i], 0, sent2_valid_ids_part2).unsqueeze(0))
		seq1 = torch.cat(seq1, 0) # bsz*128*768
		seq2 = torch.cat(seq2, 0) # bsz*128*768
		seq1 = seq1[:, :max_len1, :]
		seq2 = seq2[:, :max_len2, :]
		seq1_spans, seq2_spans = self._compose_span_representations(seq1, seq2) # bsz * num_all_spans * dim
		# print(seq1_spans.size())
		# print(seq2_spans.size())

		# bsz * feature_num * num_all_source_spans * num_all_target_spans
		simCube_context_spanbert = self.compute_sim_cube_FF_spanbert(self.layer_norm(seq1_spans), self.layer_norm(seq2_spans))

		simCube_context_spanbert_A2B = F.pad(simCube_context_spanbert, (1, 0), 'constant', 0)
		output_both_A2B = self.mlp1(simCube_context_spanbert_A2B.permute(0, 2, 3, 1)).squeeze(-1) # bsz * num_all_source_spans * (num_all_target_spans + 1), 0 is inserted at the head position

		simCube_context_spanbert_B2A = F.pad(simCube_context_spanbert.permute(0, 1, 3, 2), (1, 0), 'constant', 0)
		output_both_B2A = self.mlp1(simCube_context_spanbert_B2A.permute(0, 2, 3, 1)).squeeze(-1) # bsz * num_all_target_spans * (num_all_source_spans + 1)

		num_tags_B = output_both_A2B.size(-1)
		num_tags_A = output_both_B2A.size(-1)

		tag_B_valid_ids=[[0] for i in range(batch_size)]
		tag_A_valid_ids=[[0] for i in range(batch_size)]
		for i in range(batch_size):
			for d in range(1, args.max_span_size+1, 1):
				tag_B_valid_ids[i]+=[k+(d-1)*max_len2 for k in range(d, sent2_lengths[i]+1, 1)]
				tag_A_valid_ids[i]+=[k+(d-1)*max_len1 for k in range(d, sent1_lengths[i]+1, 1)]
			tag_B_valid_ids[i]+=(num_tags_B-len(tag_B_valid_ids[i]))*[num_tags_B-1]
			tag_A_valid_ids[i]+=(num_tags_A-len(tag_A_valid_ids[i]))*[num_tags_A-1]
		# print(sent1_lengths)
		# print(sent2_lengths)
		# print(tag_A_valid_ids)
		# print(tag_B_valid_ids)
		# sys.exit()
		tag_B_valid_ids = torch.LongTensor(tag_B_valid_ids).to(my_device).view(batch_size, 1, num_tags_B)
		tag_A_valid_ids = torch.LongTensor(tag_A_valid_ids).to(my_device).view(batch_size, 1, num_tags_A)
		tag_B_valid_ids = tag_B_valid_ids.repeat(1, output_both_A2B.size(1), 1)
		tag_A_valid_ids = tag_A_valid_ids.repeat(1, output_both_B2A.size(1), 1)
		output_both_A2B = output_both_A2B.gather(dim=-1, index=tag_B_valid_ids)
		output_both_B2A = output_both_B2A.gather(dim=-1, index=tag_A_valid_ids)

		logits_A2B = output_both_A2B.view(batch_size, args.max_span_size, -1, num_tags_B)
		logits_A2B = logits_A2B.permute(0,2,1,3) # shape (batch_size, sequence_length, max_span_width, num_tags)
		logits_B2A = output_both_B2A.view(batch_size, args.max_span_size, -1, num_tags_A)
		logits_B2A = logits_B2A.permute(0,2,1,3)

		# print(logits_A2B.size())
		# print(logits_B2A.size())

		batch_dist_idx_list_B = np.zeros((batch_size, num_tags_B, num_tags_B), dtype=int) # the last embedding dim is for padding
		tag_masks_sentB = np.zeros((batch_size, num_tags_B, num_tags_B), dtype=int)
		batch_dist_idx_list_A = np.zeros((batch_size, num_tags_A, num_tags_A), dtype=int)
		tag_masks_sentA = np.zeros((batch_size, num_tags_A, num_tags_A), dtype=int)
		for d in range(batch_size):
			extended_length_B = args.max_span_size*sent2_lengths[d]-int(args.max_span_size*(args.max_span_size-1)/2)
			batch_dist_idx_list_B[d,:(extended_length_B+1),:(extended_length_B+1)] = self.transition_matrix_dict[extended_length_B]
			tag_masks_sentB[d,:(extended_length_B+1),:(extended_length_B+1)]=np.ones((extended_length_B+1, extended_length_B+1))

			extended_length_A = args.max_span_size*sent1_lengths[d]-int(args.max_span_size*(args.max_span_size-1)/2)
			batch_dist_idx_list_A[d,:(extended_length_A+1),:(extended_length_A+1)] = self.transition_matrix_dict[extended_length_A]
			tag_masks_sentA[d,:(extended_length_A+1),:(extended_length_A+1)]=np.ones((extended_length_A+1, extended_length_A+1))

		extracted_tensor_idx_B = Variable(torch.from_numpy(batch_dist_idx_list_B).type(torch.LongTensor)).to(my_device)
		transition_matrix_sentB = self.mlp2(self.distance_embedding(extracted_tensor_idx_B)).squeeze(-1)
		extracted_tensor_idx_A = Variable(torch.from_numpy(batch_dist_idx_list_A).type(torch.LongTensor)).to(my_device)
		transition_matrix_sentA = self.mlp2(self.distance_embedding(extracted_tensor_idx_A)).squeeze(-1)
		if self.args.use_transition_layer == "False":
			transition_matrix_sentB = transition_matrix_sentB*0
			transition_matrix_sentA = transition_matrix_sentA*0
		tag_masks_sentB = Variable(torch.from_numpy(tag_masks_sentB).type(torch.FloatTensor)).to(my_device) # bsz * num_tags_B * num_tags_B
		tag_masks_sentA = Variable(torch.from_numpy(tag_masks_sentA).type(torch.FloatTensor)).to(my_device) # bsz * num_tags_A * num_tags_A
		text_masks_sentA=(sent1_valid_ids>0).int()[:,:max_len1] # bsz * max_sent1
		text_masks_sentB=(sent2_valid_ids>0).int()[:,:max_len2] # bsz * max_sent2

		# print(transition_matrix_sentB.size())
		# print(transition_matrix_sentA.size())
		# print(tag_masks_sentB.size())
		# print(tag_masks_sentA.size())
		# print(text_masks_sentB.size())
		# print(text_masks_sentA.size())
		# sys.exit()

		if self.training:
			gold_tags_sentB = labels_s2t[:,:max_len1,:]
			# print(gold_tags_sentB.type()) # torch.LongTensor
			# sys.exit()
			numerator_A2B = self._joint_likelihood(logits_A2B, transition_matrix_sentB, gold_tags_sentB, text_masks_sentA)
			# print(numerator_A2B)
			zeros = Variable(torch.zeros(batch_size, max_len1, args.max_span_size, num_tags_B).float().to(my_device))
			gold_tags_sentB_without_negative_one = gold_tags_sentB * (gold_tags_sentB > -1)
			scattered_tags = zeros.scatter_(-1, gold_tags_sentB_without_negative_one.unsqueeze(dim=-1), 1)
			hamming_cost = (1-scattered_tags)
			# negative_ones_only = (gold_tags_sentB.unsqueeze(-1).repeat(1,1,1,num_tags_B)==-1).float()
			# hamming_cost += negative_ones_only
			# hamming_cost = (hamming_cost > 0).float()
			denominator_A2B = self._input_likelihood(logits_A2B, transition_matrix_sentB, text_masks_sentA, hamming_cost, tag_masks_sentB)
			# print(denominator_A2B)
			loss_A2B = (denominator_A2B-numerator_A2B).sum()#/batch_size

			gold_tags_sentA = labels_t2s[:, :max_len2, :]
			numerator_B2A = self._joint_likelihood(logits_B2A, transition_matrix_sentA, gold_tags_sentA, text_masks_sentB)
			zeros = Variable(torch.zeros(batch_size, max_len2, args.max_span_size, num_tags_A).float().to(my_device))
			gold_tags_sentA_without_negative_one = gold_tags_sentA * (gold_tags_sentA > -1)
			scattered_tags = zeros.scatter_(-1, gold_tags_sentA_without_negative_one.unsqueeze(dim=-1), 1)
			hamming_cost = (1 - scattered_tags)
			# negative_ones_only = (gold_tags_sentA.unsqueeze(-1).repeat(1, 1, 1, num_tags_A) == -1).float()
			# hamming_cost += negative_ones_only
			# hamming_cost = (hamming_cost > 0).float()
			denominator_B2A = self._input_likelihood(logits_B2A, transition_matrix_sentA, text_masks_sentB, hamming_cost, tag_masks_sentA)
			loss_B2A = (denominator_B2A-numerator_B2A).sum()#/batch_size

			return loss_A2B+loss_B2A

		else:
			optimal_sequence_A2B = self.viterbi_tags(logits_A2B, transition_matrix_sentB, text_masks_sentA, tag_masks_sentB, reverse=False)
			optimal_sequence_B2A = self.viterbi_tags(logits_B2A, transition_matrix_sentA, text_masks_sentB, tag_masks_sentA, reverse=True)
			return [optimal_sequence_A2B[i] & optimal_sequence_B2A[i] for i in range(batch_size)]

live_connections = []

class MainHandler(RequestHandler):
    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)
        # self.corpus = None
        # self.ent_model = None
        # self.yaml_cfg = None
        # self.rel_classifier = None
        # self.ex_json_data = json.dumps(get_document("/simple_input/", "ex1_brat"))
        # print("loading ctc")
        # self.ctc_classifier_info = ctc_model_load.load_ctc()
        # print("loading ner")
        # self.ner_classifer_info = softner_ner_predict_from_file.load_ner_model()
        # print("loading ctc")
        # self.seg_classifer_info = softner_segmenter_preditct_from_file.load_seg_model()
        

    def initialize(self,  cp, yaml_cfg):
        # self.corpus = corpus
        # self.ent_model = ent_model
        self.yaml_cfg = yaml_cfg
        # self.rel_classifier = rel_classifier
        

        self.cp = cp

    def get(self):
        self.render("test_visualizer.xhtml",
                    cp=self.cp,
                    title="",
                    original_text="",
                    spinner=False,
                    ent_vis_data="",
                    rel_vis_data="",
                    ex_vis_data="")

    async def post(self, *args, **kwargs):
        ent_brat = "ent_brat_" + secrets.token_hex(16)
        rel_brat = "rel_brat_" + secrets.token_hex(16)
        output_txt_file_name=  ent_brat+".txt"

        # await inference(the_text=self.get_argument('editor'), output_ann_file_path=output_ann_file_path, output_txt_file_name=output_txt_file_name, secret_token = ent_brat)
        await inference(input_text=self.get_argument('editor'), output_txt_file_name=output_txt_file_name)

        alignments=''
        for line in open(output_txt_file_name):
            alignments+=line.strip()

        # await rel_classifier.rel_infer(ent_brat=ent_brat, rel_brat=rel_brat)
        self.render("test_visualizer.xhtml",
                    cp=self.cp,
                    title=self.get_argument('title'),
                    original_text=self.get_argument('editor'),
                    spinner=False,
                    ent_vis_data=alignments,#'0-0 1-1 2-2 3-3',#json.dumps(get_document("/simple_input/", ent_brat)),
                    rel_vis_data=self.get_argument('editor').split('\n')[0].strip(),
                    ex_vis_data=self.get_argument('editor').split('\n')[-1].strip())

        os.remove(output_txt_file_name)
        # self.cleanup(ent_brat)

    @staticmethod
    def cleanup(ent_brat):
        for fn in [ent_brat]:
            for ext in ['.txt', '.ann']:
                path = os.path.join(yaml_cfg['BRAT_SAVE_PATH'], fn + ext)
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    print("Error: {0} file not found".format(path))


class DataHandler(RequestHandler):
    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)

    def get(self, *args, **kwargs):
        file_name = 'data/WLP-Dataset.zip'
        buf_size = 4096
        self.set_header('Content-Type', 'application/force-download')
        self.set_header('Content-Disposition', 'attachment; filename=' + file_name)
        with open(file_name, 'rb') as f:
            while True:
                data = f.read(buf_size)
                if not data:
                    break
                self.write(data)
        self.finish()

def parse_yaml(cfg_path):
    with open(cfg_path, 'r') as stream:
        return yaml.load(stream)

async def inference(input_text, output_txt_file_name):
    source=input_text.split('\n')[0].strip()
    target=input_text.split('\n')[-1].strip()
    data = []
    example = namedtuple('example', 'ID, text_a, text_b, label')
    data.append(example(0, source, target, '0-0'))
    test_dataloader = create_Data_Loader(data_examples=data, set_type='test', batchsize=1, max_seq_length=128, tokenizer=tokenizer)
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(my_device) for t in batch)
        input_ids_a_and_b, input_ids_b_and_a, input_mask, segment_ids_a_and_b, segment_ids_b_and_a, sent1_valid_ids, sent2_valid_ids, sent1_wordpiece_length, sent2_wordpiece_length = batch
        with torch.no_grad():
            decoded_results = model(input_ids_a_and_b=input_ids_a_and_b, input_ids_b_and_a=input_ids_b_and_a,
                                    attention_mask=input_mask, token_type_ids_a_and_b=segment_ids_a_and_b,
                                    token_type_ids_b_and_a=segment_ids_b_and_a,
                                    sent1_valid_ids=sent1_valid_ids, sent2_valid_ids=sent2_valid_ids,
                                    sent1_wordpiece_length=sent1_wordpiece_length,
                                    sent2_wordpiece_length=sent2_wordpiece_length)
        with open(output_txt_file_name,'w') as f:
            f.writelines(' '.join(list(decoded_results[0])))


def init_args():
    parser = argparse.ArgumentParser(description='Sequence labeler.')
    parser.add_argument('--port', type=str,
                        help='port')
    parser.add_argument('--cp', type=str, help='color palette css filename. root folder is static.')
    args = parser.parse_args()
    return args


def init_main():
    yaml_cfg = parse_yaml(os.path.join(os.path.dirname(__file__), "configs/basic.yaml"))
    # corpus, the_model = load_model_and_corpus(yaml_cfg)
    return yaml_cfg


class Args:
    """docstring for Args"""
    def __init__(self):
        self.port = "8924"
        self.cp = "cp_2.css"
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--max_epoch", default=6, type=int)
    parser.add_argument("--max_span_size", default=4, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--max_sent_length", default=70, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--dataset", default='mtref', type=str)
    parser.add_argument("--sure_and_possible", default='True', type=str)
    parser.add_argument("--distance_embedding_size", type=int, default=128)
    parser.add_argument('--use_transition_layer', type=str, default='False',
                        help="if False, will set transition score to 0.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = NeuralWordAligner(args)
    my_device = torch.device('cpu')
    model = model.to(my_device)

    checkpoint = torch.load('Checkpoint_sure_and_possible_True_dataset_mtref_batchsize_1_max_span_size_4_use_transition_layer_False_epoch_2_0.9150.pt', map_location=my_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tornado_args = Args()


    yaml_cfg = init_main()
    brat_filepath = "data/simple_input/"
    # rel_classifier = RelationClassifier()

    app = Application(handlers=[
        (r"/", MainHandler, {"cp": tornado_args.cp,
                                      "yaml_cfg": yaml_cfg}),
        
    ], static_path=os.path.join(os.path.dirname(__file__), "static"),
        template_path=os.path.join(os.path.dirname(__file__), "templates"))

    server = HTTPServer(app)
    server.listen(tornado_args.port)
    server.start(num_processes=1)
    IOLoop.current().start()
