from __future__ import division
import sys, copy
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.nn as nn
from util import *
# from main_semi_crf_chao_run_ckpt_on_dataset import max_span_size, max_sent_length
import numpy as np
import torch
from pytorch_transformers import BertModel, BertTokenizer

def put_a_number_in_bin_and_binary(a, return_fired_idx = False):
	bins = np.array([-11, -6, -4, -3, -2, -1, 0, 1, 2, 3, 5, 10])
	fired_idx = np.digitize(a, bins, right = True)

	if return_fired_idx == True:
		return fired_idx

	d = np.zeros(13)
	d[fired_idx] = 1

	return d

def put_a_number_in_bin_and_binary_abs(a, return_fired_idx = False):
	bins = np.array([0, 1, 2, 3, 5, 10])
	fired_idx = np.digitize(a, bins, right=True)

	if return_fired_idx == True:
		return fired_idx

	d = np.zeros(7)
	d[fired_idx] = 1

	return d

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
	def __init__(self, task, embedding_dim, num_layers,wv_dict, wv_arr, args, max_span_size=3):
		super(NeuralWordAligner, self).__init__()
		self.task=task
		self.num_layers = num_layers
		self.embedding_dim = embedding_dim
		self.wv_dict=wv_dict
		self.wv_arr=wv_arr
		self.max_span_size=max_span_size

		self.use_transition_layer = args.use_transition_layer
		self.transition_layer_info = args.transition_layer_info

		if args.projection=='True':
			span_representation_dim=300
			hidden_dim=300//2
		else:
			span_representation_dim = 768
			hidden_dim = 768 // 2
		self.lstm = nn.LSTM(span_representation_dim, hidden_dim//2, num_layers, bidirectional=True)
		if args.layer=='-1':
			self.bert_embd_reduce = nn.Linear(768*2, 300, bias=False)
		else:
			self.bert_embd_reduce = nn.Linear(768, 300, bias=False)
		self.attn = nn.Linear(span_representation_dim, 100)
		self.attn_glove = nn.Linear(300, 100)
		self.attn_embd = nn.Embedding(3, 100)
		self.bn=LayerNorm(span_representation_dim)
		self.bn_glove = LayerNorm(embedding_dim)
		self.bn_zhengbao=LayerNorm(span_representation_dim+span_representation_dim//2)
		self.bn_spanbert = LayerNorm(span_representation_dim*3)
		self.args=args
		self.span_representation_dim=span_representation_dim
		if 'spanbert' in args.span_representation:
			if args.bert_with_coref_span == 'True':
				print("AA, using bert-base-uncased with coref span representation.")
				self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
				self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
															output_attentions=True)
				self.bert_model = self.bert_model.to(my_device)
			else:
				self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
				self.bert_model = BertModel.from_pretrained('spanbert_hf_base/', output_hidden_states=True,
													   output_attentions=True)
				self.bert_model = self.bert_model.to(my_device)
		else:
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
			self.bert_model = self.bert_model.to(my_device)

		self.sim_feature_num=12
		if args.glove_representation=='no_glove':
			total_features = self.sim_feature_num * 1 # + 17 + 8
		else:
			total_features = self.sim_feature_num * 2
		self.mlp1 = nn.Sequential(nn.Linear(total_features, total_features), nn.PReLU(),nn.Linear(total_features, 1))
		self.FF_kernel = nn.Sequential(*[nn.Linear(span_representation_dim*4, span_representation_dim), nn.PReLU(),nn.Linear(span_representation_dim, span_representation_dim), nn.PReLU(), nn.Linear(span_representation_dim, self.sim_feature_num)])
		self.FF_kernel_glove = nn.Sequential(*[nn.Linear(embedding_dim*4, embedding_dim), nn.PReLU(),nn.Linear(embedding_dim, embedding_dim), nn.PReLU(), nn.Linear(embedding_dim, self.sim_feature_num)])
		self.FF_kernel_zhengbao = nn.Sequential(*[nn.Linear((span_representation_dim+span_representation_dim//2)*4, (span_representation_dim+span_representation_dim//2)), nn.PReLU(),nn.Linear((span_representation_dim+span_representation_dim//2), (span_representation_dim+span_representation_dim//2)), nn.PReLU(), nn.Linear((span_representation_dim+span_representation_dim//2), self.sim_feature_num)])
		self.FF_kernel_spanbert = nn.Sequential(*[nn.Linear((span_representation_dim*3)*4, (span_representation_dim*3)), nn.PReLU(),nn.Linear((span_representation_dim*3), (span_representation_dim*3)), nn.PReLU(), nn.Linear((span_representation_dim*3), self.sim_feature_num)])

		# start to deal with mlp2 in transition layer
		if args.transition_layer_info == 'scalar':
			self.mlp2 = nn.Sequential(nn.Linear(1, 1))
		elif args.transition_layer_info == 'bin' and args.use_absolute_dist == 'False':
			self.mlp2 = nn.Sequential(nn.Linear(13, 1))
		elif args.transition_layer_info == 'bin' and args.use_absolute_dist == 'True':
			self.mlp2 = nn.Sequential(nn.Linear(7, 1))
		elif args.transition_layer_info == 'scalar+bin' and args.use_absolute_dist == 'False':
			self.mlp2 = nn.Sequential(nn.Linear(14, 1))
		elif args.transition_layer_info == 'scalar+bin' and args.use_absolute_dist == 'True':
			self.mlp2 = nn.Sequential(nn.Linear(8, 1))
		elif args.transition_layer_info == 'bin_vector':
			self.mlp2 = nn.Sequential(nn.Linear(args.distance_embedding_size, 1))

		# buffer transition_matrix_dict for transition matrix, intialize embedding for bin_vector
		if args.transition_layer_info == 'scalar':
			self.transition_matrix_dict = {}
			for len_B in range(max_span_size, max_sent_length + 1, 1):
				extended_length_B = self.max_span_size * len_B - int(
					self.max_span_size * (self.max_span_size - 1) / 2)
				transition_matrix = np.zeros((extended_length_B + 1, extended_length_B + 1, 1), dtype=float)
				for j in range(extended_length_B + 1):  # 0 is NULL state
					for k in range(extended_length_B + 1):  # k is previous state, j is current state
						if args.use_absolute_dist == 'False':
							transition_matrix[j][k][0] = self.distortionDistance(k, j, len_B)
						else:
							transition_matrix[j][k][0] = np.abs(self.distortionDistance(k, j, len_B))
				self.transition_matrix_dict[extended_length_B] = transition_matrix

		elif args.transition_layer_info == 'bin':
			self.transition_matrix_dict = {}
			for len_B in range(max_span_size, max_sent_length + 1, 1):
				extended_length_B = self.max_span_size * len_B - int(
					self.max_span_size * (self.max_span_size - 1) / 2)
				if args.use_absolute_dist == 'False':
					transition_matrix = np.zeros((extended_length_B + 1, extended_length_B + 1, 13), dtype=float)
				else:
					transition_matrix = np.zeros((extended_length_B + 1, extended_length_B + 1, 7), dtype=float)
				for j in range(extended_length_B + 1):  # 0 is NULL state
					for k in range(extended_length_B + 1):  # k is previous state, j is current state
						if args.use_absolute_dist == 'False':
							transition_matrix[j][k] = put_a_number_in_bin_and_binary(
								self.distortionDistance(k, j, len_B))
						else:
							transition_matrix[j][k] = put_a_number_in_bin_and_binary_abs(
								np.abs(self.distortionDistance(k, j, len_B)))
				self.transition_matrix_dict[extended_length_B] = transition_matrix

		elif args.transition_layer_info == 'scalar+bin':
			self.transition_matrix_dict = {}
			for len_B in range(max_span_size, max_sent_length + 1, 1):
				extended_length_B = self.max_span_size * len_B - int(
					self.max_span_size * (self.max_span_size - 1) / 2)
				if args.use_absolute_dist == 'False':
					transition_matrix = np.zeros((extended_length_B + 1, extended_length_B + 1, 14), dtype=float)
				else:
					transition_matrix = np.zeros((extended_length_B + 1, extended_length_B + 1, 8), dtype=float)
				for j in range(extended_length_B + 1):  # 0 is NULL state
					for k in range(extended_length_B + 1):  # k is previous state, j is current state
						if args.use_absolute_dist == 'False':
							transition_matrix[j][k][:13] = put_a_number_in_bin_and_binary(
								self.distortionDistance(k, j, len_B))
							transition_matrix[j][k][13] = self.distortionDistance(k, j, len_B)
						else:
							transition_matrix[j][k][:7] = put_a_number_in_bin_and_binary_abs(
								np.abs(self.distortionDistance(k, j, len_B)))
							transition_matrix[j][k][7] = np.abs(self.distortionDistance(k, j, len_B))
				self.transition_matrix_dict[extended_length_B] = transition_matrix

		elif args.transition_layer_info == 'bin_vector':
			if args.use_absolute_dist == 'False':
				self.distance_embedding = nn.Embedding(13, args.distance_embedding_size)
			else:
				self.distance_embedding = nn.Embedding(7, args.distance_embedding_size)

			if args.update_distance_embedding == "False":
				self.distance_embedding.weight.requires_grad = False

			self.transition_matrix_dict = {}
			for len_B in range(max_span_size, max_sent_length + 1, 1):
				extended_length_B = self.max_span_size * len_B - int(
					self.max_span_size * (self.max_span_size - 1) / 2)

				tmp_dist_idx_list = []
				for j in range(extended_length_B + 1):  # 0 is NULL state
					for k in range(extended_length_B + 1):  # k is previous state, j is current state
						if args.use_absolute_dist == 'False':
							tmp_dist_idx_list.append(
								put_a_number_in_bin_and_binary(self.distortionDistance(k, j, len_B),
								                               return_fired_idx=True))
						else:
							tmp_dist_idx_list.append(
								put_a_number_in_bin_and_binary_abs(np.abs(self.distortionDistance(k, j, len_B)),
								                                   return_fired_idx=True))

				self.transition_matrix_dict[extended_length_B] = tmp_dist_idx_list

		# max_extended_length = self.max_span_size * max_sent_length - int(self.max_span_size * (self.max_span_size - 1) / 2)
		# self.START = nn.Embedding(max_extended_length + 1, args.distance_embedding_size)
		if self.args.add_START_symbol == 'START':
			self.tokenizer.add_tokens(["S-START", "T-START"])
			self.bert_model.resize_token_embeddings(len(self.tokenizer))
		# if args.update_START_embedding == 'False':
		# 	self.START.weight.requires_grad = False
		# self.END = nn.Embedding(max_extended_length+1, args.distance_embedding_size)

	def word_layer(self,sent, cur_max_span_size):
		sent_embd=[]
		for word in sent:
			try:
				sent_embd.append(torch.from_numpy(self.wv_arr[self.wv_dict[word]].numpy()).to(my_device).view(1,-1))
			except:
				sent_embd.append(torch.normal(torch.zeros(self.embedding_dim), std=1).to(my_device).view(1,-1))
		sent_embd=torch.cat(sent_embd,0)
		chunk_list = []
		for span_size in range(1, 1+cur_max_span_size):
			for i in range(len(sent)):
				if i+span_size<=len(sent):
					chunk_list.append([i+k for k in range(span_size)])
		return sent_embd, chunk_list

	def get_chunk_list(self,sent, cur_max_span_size):
		chunk_list = []
		for span_size in range(1, 1+cur_max_span_size):
			for i in range(len(sent)):
				if i+span_size<=len(sent):
					chunk_list.append([i+k for k in range(span_size)])
		return chunk_list

	def compute_sim_cube_FF(self, seq1, seq2):
		'''
		:param seq1: bs * len_seq1 * dim
		:param seq2: bs * len_seq2 * dim
		:return:
		'''
		def compute_sim_FF(prism1, prism2):
			features = torch.cat([prism1, prism2, torch.abs(prism1 - prism2), prism1 * prism2], dim=-1)
			if prism1.size(-1)==self.embedding_dim:
				FF_out = self.FF_kernel_glove(features)
			else:
				FF_out = self.FF_kernel(features)
			return FF_out.permute(0,3,1,2)

		def compute_prism(seq1, seq2):
			prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
			prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
			prism1 = prism1.permute(1, 2, 0, 3).contiguous()
			prism2 = prism2.permute(1, 0, 2, 3).contiguous()
			return compute_sim_FF(prism1, prism2)

		sim_cube = torch.Tensor(seq1.size(0), self.sim_feature_num, seq1.size(1), seq2.size(1)) # bs * feature_num * len_seq1 * len_seq2
		sim_cube = sim_cube.to(my_device)
		sim_cube[:, 0:self.sim_feature_num] = compute_prism(seq1, seq2)
		return sim_cube

	def compute_sim_cube_FF_zhengbao(self, seq1, seq2):
		'''
		:param seq1: bs * len_seq1 * dim
		:param seq2: bs * len_seq2 * dim
		:return:
		'''
		def compute_sim_FF(prism1, prism2):
			features = torch.cat([prism1, prism2, torch.abs(prism1 - prism2), prism1 * prism2], dim=-1)
			FF_out = self.FF_kernel_zhengbao(features)
			return FF_out.permute(0,3,1,2)

		def compute_prism(seq1, seq2):
			prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
			prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
			prism1 = prism1.permute(1, 2, 0, 3).contiguous()
			prism2 = prism2.permute(1, 0, 2, 3).contiguous()
			return compute_sim_FF(prism1, prism2)

		sim_cube = torch.Tensor(seq1.size(0), self.sim_feature_num, seq1.size(1), seq2.size(1)) # bs * feature_num * len_seq1 * len_seq2
		sim_cube = sim_cube.to(my_device)
		sim_cube[:, 0:self.sim_feature_num] = compute_prism(seq1, seq2)
		return sim_cube

	def compute_sim_cube_FF_spanbert(self, seq1, seq2):
		'''
		:param seq1: bs * len_seq1 * dim
		:param seq2: bs * len_seq2 * dim
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

		sim_cube = torch.Tensor(seq1.size(0), self.sim_feature_num, seq1.size(1), seq2.size(1)) # bs * feature_num * len_seq1 * len_seq2
		sim_cube = sim_cube.to(my_device)
		sim_cube[:, 0:self.sim_feature_num] = compute_prism(seq1, seq2)
		return sim_cube

	def compute_sim_cube(self, seq1, seq2):
		'''
		:param seq1: bs * len_seq1 * dim
		:param seq2: bs * len_seq2 * dim
		:return:
		'''
		def compute_sim(prism1, prism2):
			prism1_len = prism1.norm(dim=3)
			prism2_len = prism2.norm(dim=3)

			dot_prod = torch.matmul(prism1.unsqueeze(3), prism2.unsqueeze(4))
			dot_prod = dot_prod.squeeze(3).squeeze(3)
			cos_dist = dot_prod / (prism1_len * prism2_len + 1E-8)
			# print(prism1.size())
			# print(prism2.size())
			l2_dist = ((prism1 - prism2).norm(dim=3))
			# print(l2_dist.size())
			# sys.exit()
			return torch.stack([dot_prod, cos_dist, l2_dist], 1)

		def compute_prism(seq1, seq2):
			prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
			prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
			prism1 = prism1.permute(1, 2, 0, 3).contiguous()
			prism2 = prism2.permute(1, 0, 2, 3).contiguous()
			return compute_sim(prism1, prism2)

		sim_cube = torch.Tensor(seq1.size(0), 12, seq1.size(1), seq2.size(1)) # bs * 12 * len_seq1 * len_seq2
		sim_cube = sim_cube.to(my_device)
		half_dim=seq1.size(2)//2
		seq1_f = seq1[:, :, :half_dim]
		seq1_b = seq1[:, :, half_dim:]
		seq2_f = seq2[:, :, :half_dim]
		seq2_b = seq2[:, :, half_dim:]
		sim_cube[:, 0:3] = compute_prism(seq1, seq2)
		sim_cube[:, 3:6] = compute_prism(seq1_f, seq2_f)
		sim_cube[:, 6:9] = compute_prism(seq1_b, seq2_b)
		sim_cube[:, 9:12] = compute_prism(seq1_f + seq1_b, seq2_f + seq2_b)
		return sim_cube

	def viterbi_decoder(self, emission_matrix, transition_matrix, len_A, extended_length_B):
		"""
		:param emission_matrix:  extended_length_A * (extended_length_B + 1), word/phrase pair interaction matrix
		:param transition_matrix: (extended_length_B + 1) * (extended_length_B + 1), state transition matrix
		:param len_A: source sentence length
		:param len_B: target sentence length
		:return:
		"""
		emission_matrix=emission_matrix.data.cpu().numpy()
		transition_matrix=transition_matrix.data.cpu().numpy()
		# start_transition=self.start_transition.data.cpu().numpy()
		# end_transition=self.end_transition.data.cpu().numpy()
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
				for span_size in range(1, min(i + 1, self.max_span_size) + 1):  # span_size can be {1,2,3,4}
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

	def _score_sentence(self, output_both, transition_matrix, golden_sequence, len_A):
		# golden_sequence is a list of states: [1, 2, 3, 33, 33, 33, 8, 9, 10, 11, 41, 15]
		# print(golden_sequence)
		score=0
		# print(output_both.size())
		gold_list=[]
		tmp=golden_sequence[0:1]
		for i, item in enumerate(golden_sequence):
			if i==0:
				continue
			if item==tmp[-1]:
				if len(tmp)==max_span_size:
					gold_list.append((i - len(tmp), tmp, tmp[-1]))
					tmp = [item]
				else:
					tmp.append(item)
			else:
				gold_list.append((i-len(tmp), tmp, tmp[-1]))
				tmp=[item]
		gold_list.append((len_A - len(tmp), tmp, tmp[-1]))
		# print(gold_list)
		# score+=self.start_transition[gold_list[0][-1]]
		score = 0
		for start_i, span, item in gold_list:
			span_size=len(span)
			score+=output_both[start_i+(span_size-1)*len_A-int((span_size-1)*(span_size-2)/2)][item-1]
			if start_i-1>=0:
				score+=transition_matrix[item][golden_sequence[start_i-1]]
		# score+=self.end_transition[gold_list[-1][-1]]
		return score

	def argmax(self, vec):
		# return the argmax as a python int
		_, idx = torch.max(vec, 1)
		return idx.item()

	def log_sum_exp(self, vec):
		max_score = vec[0, self.argmax(vec)]
		max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
		return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

	def backward_beta(self, output_both, transition_matrix, len_A):
		target_size = output_both.size(1)
		beta_all = torch.full((output_both.size(0), target_size), 0).to(my_device)
		forward_var1 = torch.full((1, target_size), 0).to(my_device)
		forward_var2 = torch.full((1, target_size), 0).to(my_device)
		forward_var3 = torch.full((1, target_size), 0).to(my_device)
		tmp_forward_var1 = torch.full((1, target_size), 1).to(my_device)
		tmp_forward_var2 = torch.full((1, target_size), 1).to(my_device)
		tmp_forward_var3 = torch.full((1, target_size), 1).to(my_device)
		for i in range(len_A-1, 0, -1):
			for span_size in range(1, min(len_A - i, self.max_span_size) + 1):
				alphas_t = []
				if span_size == 1:
					feat = output_both[i]
					for j in range(target_size):
						emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
						trans_score = transition_matrix[:,j]  # [:-1]
						if i == len_A-1:
							next_tag_var = forward_var1 + emit_score
						else:
							next_tag_var = forward_var1 + trans_score + emit_score
						beta_all[i][j] = self.log_sum_exp(next_tag_var-emit_score).view(1)
						next_tag_var = self.log_sum_exp(next_tag_var).view(1)
						alphas_t.append(next_tag_var)
					tmp_forward_var1 = torch.cat(alphas_t).view(1, -1)
				elif span_size == 2 and i <= len_A- 2:
					feat = output_both[i + len_A]
					for j in range(target_size):
						emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
						trans_score = transition_matrix[:,j]
						if i == len_A-2:
							next_tag_var = forward_var2 + emit_score
						else:
							next_tag_var = forward_var2 + trans_score + emit_score
						beta_all[i+len_A][j] = self.log_sum_exp(next_tag_var-emit_score).view(1)
						next_tag_var = self.log_sum_exp(next_tag_var).view(1)
						alphas_t.append(next_tag_var)
					tmp_forward_var2 = torch.cat(alphas_t).view(1, -1)
				elif span_size == 3 and i <= len_A-3:
					feat = output_both[i + 2 * len_A - 1]
					for j in range(target_size):
						emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
						trans_score = transition_matrix[:,j]
						if i == len_A-3:
							next_tag_var = forward_var3 + emit_score
						else:
							next_tag_var = forward_var3 + trans_score + emit_score
						beta_all[i+2*len_A-1][j] = self.log_sum_exp(next_tag_var-emit_score).view(1)
						next_tag_var = self.log_sum_exp(next_tag_var).view(1)
						alphas_t.append(next_tag_var)
					tmp_forward_var3 = torch.cat(alphas_t).view(1, -1)

			forward_var3 = forward_var2
			forward_var2 = forward_var1
			if i == len_A-1:
				forward_var1 = tmp_forward_var1
			elif i == len_A-2:
				max_score = torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
				forward_var1 = max_score + torch.log(
					torch.exp(tmp_forward_var1 - max_score) + torch.exp(tmp_forward_var2 - max_score))
			elif i <= len_A-3:
				max_score = torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
				max_score = torch.max(max_score, torch.max(tmp_forward_var3))
				forward_var1 = max_score + torch.log(torch.exp(tmp_forward_var1 - max_score) + torch.exp(tmp_forward_var2 - max_score) + torch.exp(
						tmp_forward_var3 - max_score))

		beta = self.log_sum_exp(forward_var1)
		return beta, beta_all

	def forward_alpha(self, output_both, transition_matrix, len_A):
		target_size=output_both.size(1)
		alpha_all = torch.full((output_both.size(0), target_size), 0).to(my_device)
		forward_var1 = torch.full((1, target_size), 0).to(my_device)
		forward_var2 = torch.full((1, target_size), 0).to(my_device)
		forward_var3 = torch.full((1, target_size), 0).to(my_device)
		tmp_forward_var1 = torch.full((1, target_size), 1).to(my_device)
		tmp_forward_var2 = torch.full((1, target_size), 1).to(my_device)
		tmp_forward_var3 = torch.full((1, target_size), 1).to(my_device)
		for i in range(len_A):
			for span_size in range(1, min(i + 1, self.max_span_size) + 1):
				alphas_t = []
				if span_size==1:
					feat = output_both[i]
					for j in range(target_size):
						emit_score = feat[j-1].view(1, -1).expand(1, target_size)
						trans_score = transition_matrix[j]#[:-1]
						if i<=0:
							next_tag_var = forward_var1 + emit_score
						else:
							next_tag_var = forward_var1 + trans_score + emit_score
						next_tag_var=self.log_sum_exp(next_tag_var).view(1)
						alpha_all[i][j]=next_tag_var
						alphas_t.append(next_tag_var)
					tmp_forward_var1 = torch.cat(alphas_t).view(1, -1)
				elif span_size==2 and i>=1:
					feat = output_both[i -1 +len_A]
					for j in range(target_size):
						emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
						trans_score = transition_matrix[j]
						if i<=1:
							next_tag_var = forward_var2 + emit_score
						else:
							next_tag_var = forward_var2 + trans_score + emit_score
						next_tag_var = self.log_sum_exp(next_tag_var).view(1)
						alpha_all[i-1+len_A][j] = next_tag_var
						alphas_t.append(next_tag_var)
					tmp_forward_var2 = torch.cat(alphas_t).view(1, -1)
				elif span_size==3 and i>=2:
					feat = output_both[i -2 +2*len_A-1]
					for j in range(target_size):
						emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
						trans_score = transition_matrix[j]
						if i<=2:
							next_tag_var = forward_var3 + emit_score
						else:
							next_tag_var = forward_var3 + trans_score + emit_score
						next_tag_var = self.log_sum_exp(next_tag_var).view(1)
						alpha_all[i -2+2*len_A-1][j] = next_tag_var
						alphas_t.append(next_tag_var)
					tmp_forward_var3 = torch.cat(alphas_t).view(1, -1)

			forward_var3 = forward_var2
			forward_var2 = forward_var1
			if i==0:
				forward_var1=tmp_forward_var1
			elif i==1:
				max_score=torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
				forward_var1 = max_score+torch.log(torch.exp(tmp_forward_var1-max_score) + torch.exp(tmp_forward_var2-max_score))
			elif i>=2:
				max_score = torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
				max_score = torch.max(max_score, torch.max(tmp_forward_var3))
				forward_var1=max_score+torch.log(torch.exp(tmp_forward_var1-max_score)+torch.exp(tmp_forward_var2-max_score)+torch.exp(tmp_forward_var3-max_score))

		alpha = self.log_sum_exp(forward_var1)
		return alpha, alpha_all

	def _forward_alg(self, output_both, transition_matrix, len_A, gold_sentence):
		target_size=output_both.size(1)
		forward_var1 = torch.full((1, target_size), 0).to(my_device) #+ self.start_transition
		forward_var2 = torch.full((1, target_size), 0).to(my_device) #+ self.start_transition
		forward_var3 = torch.full((1, target_size), 0).to(my_device) #+ self.start_transition
		tmp_forward_var1 = torch.full((1, target_size), 1).to(my_device)
		tmp_forward_var2 = torch.full((1, target_size), 1).to(my_device)
		tmp_forward_var3 = torch.full((1, target_size), 1).to(my_device)
		for i in range(len_A):
			for span_size in range(1, min(i + 1, self.max_span_size) + 1):
				alphas_t = []
				if span_size==1:
					feat = output_both[i]
					for j in range(target_size):
						emit_score = feat[j-1].view(1, -1).expand(1, target_size)
						trans_score = transition_matrix[j]#[:-1]
						if gold_sentence[i] == j:
							hamming=0
						else:
							hamming=1
						if i<=0:
							next_tag_var = forward_var1 + emit_score + hamming * self.args.hamming
						else:
							next_tag_var = forward_var1 + trans_score + emit_score + hamming * self.args.hamming
						alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
					tmp_forward_var1 = torch.cat(alphas_t).view(1, -1)
				elif span_size==2 and i>=1:
					feat = output_both[i -1 +len_A]
					for j in range(target_size):
						emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
						trans_score = transition_matrix[j]
						if gold_sentence[i] == j and gold_sentence[i-1]==j:
							hamming=0
						else:
							hamming=1
						if i<=1:
							next_tag_var = forward_var2 + emit_score + hamming * self.args.hamming
						else:
							next_tag_var = forward_var2 + trans_score + emit_score + hamming * self.args.hamming
						alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
					tmp_forward_var2 = torch.cat(alphas_t).view(1, -1)
				elif span_size==3 and i>=2:
					feat = output_both[i -2 +2*len_A-1]
					for j in range(target_size):
						emit_score = feat[j - 1].view(1, -1).expand(1, target_size)
						trans_score = transition_matrix[j]
						if gold_sentence[i] == j and gold_sentence[i-1]==j and gold_sentence[i-2]==j:
							hamming=0
						else:
							hamming=1
						if i<=2:
							next_tag_var = forward_var3 + emit_score + hamming * self.args.hamming
						else:
							next_tag_var = forward_var3 + trans_score + emit_score + hamming * self.args.hamming
						alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
					tmp_forward_var3 = torch.cat(alphas_t).view(1, -1)

			forward_var3 = forward_var2
			forward_var2 = forward_var1
			if i==0:
				forward_var1=tmp_forward_var1
			elif i==1:
				max_score=torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
				forward_var1 = max_score+torch.log(torch.exp(tmp_forward_var1-max_score) + torch.exp(tmp_forward_var2-max_score))
			elif i>=2:
				max_score = torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
				max_score = torch.max(max_score, torch.max(tmp_forward_var3))
				# if i==len_A-1:
				# 	max_score = torch.max(max_score, self.end_transition)
				# 	forward_var1 = max_score + torch.log(torch.exp(tmp_forward_var1 - max_score) + torch.exp(tmp_forward_var2 - max_score) + torch.exp(
				# 			tmp_forward_var3 - max_score) + torch.exp(self.end_transition - max_score))
				# else:
				forward_var1=max_score+torch.log(torch.exp(tmp_forward_var1-max_score)+torch.exp(tmp_forward_var2-max_score)+torch.exp(tmp_forward_var3-max_score))

		alpha = self.log_sum_exp(forward_var1)
		return alpha

	def distortionDistance(self, state_i, state_j, sent_length):
		start_i, size_i=convert_stateID_to_spanID(state_i, sent_length)
		start_j, size_j=convert_stateID_to_spanID(state_j, sent_length)
		return start_j - (start_i + size_i - 1) - 1

	def forward(self, raw_input_A, raw_input_B, golden_sequence_sentA, golden_sequence_sentB):
		"""
		:param raw_input_A: (source, source_dep_tag, source_dep_tree)
		:param raw_input_B: (target, target_dep_tag, target_dep_tree)
		:return:
		"""
		# embd_A: # of chunks in A * embedding_dim
		input_sentA, sentA_tag_list, A_dep_tree = raw_input_A
		input_sentB, sentB_tag_list, B_dep_tree = raw_input_B
		# input_A='good time'
		# input_B='I am good'
		input_A=input_sentA.split()
		input_B=input_sentB.split()
		len_A = len(input_A)
		len_B = len(input_B)
		extended_length_A = self.max_span_size * len_A - int(self.max_span_size*(self.max_span_size-1)/2) # if max_span_size=3, then 3*n-3; if max_span_size=4, then 4*n-6
		extended_length_B = self.max_span_size * len_B - int(self.max_span_size*(self.max_span_size-1)/2)

		# print(len_A, len(sentA_tag_list), extended_length_A)
		# print(len_B, len(sentB_tag_list), extended_length_B)
		embd_A_word, chunk_A = self.word_layer(input_A, self.max_span_size)
		embd_B_word, chunk_B = self.word_layer(input_B, self.max_span_size)
		embd_A_word = torch.unsqueeze(embd_A_word, 0).view(-1, 1, self.embedding_dim)
		embd_B_word = torch.unsqueeze(embd_B_word, 0).view(-1, 1, self.embedding_dim)

		input_ids_A = torch.tensor([self.tokenizer.encode(input_sentA, add_special_tokens=True)]).to(my_device)
		input_ids_B = torch.tensor([self.tokenizer.encode(input_sentB, add_special_tokens=True)]).to(my_device)
		if self.args.add_SEP_symbol == 'True':
			input_ids_A_and_B = torch.tensor([self.tokenizer.encode(input_sentA+' [SEP] '+input_sentB, add_special_tokens=True)]).to(my_device)
			input_ids_B_and_A = torch.tensor([self.tokenizer.encode(input_sentB+' [SEP] '+input_sentA, add_special_tokens=True)]).to(my_device)
		else:
			input_ids_A_and_B = torch.tensor([self.tokenizer.encode(input_sentA + ' ' + input_sentB, add_special_tokens=True)]).to(my_device)
			input_ids_B_and_A = torch.tensor([self.tokenizer.encode(input_sentB + ' ' + input_sentA, add_special_tokens=True)]).to(my_device)
		input_ids_A = input_ids_A[:, 1:-1]
		input_ids_B = input_ids_B[:, 1:-1]
		if self.args.freeze_bert=='True':
			with torch.no_grad():
				# all_hidden_states_A, _ = bert_model(input_ids_A)[-2:] # tuple with 13 elements
				# all_hidden_states_B, _ = bert_model(input_ids_B)[-2:]
				all_hidden_states_A_and_B, _ = self.bert_model(input_ids_A_and_B)[-2:]
				all_hidden_states_B_and_A, _ = self.bert_model(input_ids_B_and_A)[-2:]
		else:
			all_hidden_states_A_and_B, _ = self.bert_model(input_ids_A_and_B)[-2:]
			all_hidden_states_B_and_A, _ = self.bert_model(input_ids_B_and_A)[-2:]
		if self.args.layer=='-1':
			fetched_hidden_state_A_and_B = torch.cat([all_hidden_states_A_and_B[-7], all_hidden_states_A_and_B[-5]], -1)
			fetched_hidden_state_B_and_A = torch.cat([all_hidden_states_B_and_A[-7], all_hidden_states_B_and_A[-5]], -1)
		else:
			fetched_hidden_state_A_and_B = all_hidden_states_A_and_B[int(self.args.layer)-13]
			fetched_hidden_state_B_and_A = all_hidden_states_B_and_A[int(self.args.layer)-13]
		# default version
		fetched_hidden_state_A_part1 = fetched_hidden_state_A_and_B[:,1:input_ids_A.size(1)+1, :]
		if self.args.add_SEP_symbol=='True':
			fetched_hidden_state_A_part2 = fetched_hidden_state_B_and_A[:,input_ids_B.size(1) + 2:-1, :] # for [SEP] usage
			fetched_hidden_state_B_part1 = fetched_hidden_state_A_and_B[:, input_ids_A.size(1) + 2:-1, :] # for [SEP] usage
		else:
			fetched_hidden_state_A_part2 = fetched_hidden_state_B_and_A[:, input_ids_B.size(1) + 1:-1, :]
			fetched_hidden_state_B_part1 = fetched_hidden_state_A_and_B[:, input_ids_A.size(1) + 1:-1, :]
		fetched_hidden_state_B_part2 = fetched_hidden_state_B_and_A[:, 1:input_ids_B.size(1)+1, :]
		fetched_hidden_state_A = fetched_hidden_state_A_part1 + fetched_hidden_state_A_part2
		fetched_hidden_state_B = fetched_hidden_state_B_part1 + fetched_hidden_state_B_part2
		if self.args.freeze_bert=='True':
			fetched_hidden_state_A=fetched_hidden_state_A.data
			fetched_hidden_state_B=fetched_hidden_state_B.data
		hidden_A0 = []
		start = 0
		for ii, word in enumerate(input_A):
			if word in ['[CLS]','S-START','T-START']:
				split_tokens = [word]
			else:
				split_tokens = self.tokenizer._tokenize(word)
			tmp = [kk for kk in range(start, start + len(split_tokens))]
			start += len(split_tokens)
			# hidden_A0.append(torch.mean(fetched_hidden_state_A[:, tmp, :].data, 1, keepdim=True))
			hidden_A0.append(torch.mean(fetched_hidden_state_A[:, tmp, :], 1, keepdim=True))
		assert start==len(input_ids_A[0])
		hidden_B0 = []
		start = 0
		for ii, word in enumerate(input_B):
			if word in ['[CLS]','S-START','T-START']:
				split_tokens = [word]
			else:
				split_tokens = self.tokenizer._tokenize(word)
			tmp = [kk for kk in range(start, start + len(split_tokens))]
			start += len(split_tokens)
			# hidden_B0.append(torch.mean(fetched_hidden_state_B[:, tmp, :].data, 1, keepdim=True))
			hidden_B0.append(torch.mean(fetched_hidden_state_B[:, tmp, :], 1, keepdim=True))
		assert start == len(input_ids_B[0])
		# print([tokenizer._convert_id_to_token(int(item)) for item in input_ids_A[0]])
		# print([tokenizer._convert_id_to_token(int(item)) for item in input_ids_B[0]])
		# print(' '.join(recovered_A))
		bert_embd_A = torch.cat(hidden_A0, 0)
		bert_embd_B = torch.cat(hidden_B0, 0)
		# print(hidden_A0.size(), len_A)
		# print(hidden_B0.size(), len_B)
		if len_A!=bert_embd_A.size(0) or len_B!=bert_embd_B.size(0):
			print(input_A)
			print(input_B)
		assert bert_embd_A.size(0)==len_A and bert_embd_B.size(0)==len_B

		if self.args.projection=='True':
			bert_embd_A=self.bert_embd_reduce(bert_embd_A) # len_A * 1 * 300
			bert_embd_B=self.bert_embd_reduce(bert_embd_B)
		lstm_hA, _ = self.lstm(bert_embd_A)
		lstm_hB, _ = self.lstm(bert_embd_B)

		out0, out1=bert_embd_A, bert_embd_B
		# print('embd_A_word: ', embd_A_word.size()) # ? len_A * 1 * 300
		# print('out0: ', out0.size()) # len_A * 1 * 300
		span_A=[]
		span_B=[]
		context_A_weight=[]
		context_B_weight=[]
		context_A_start = []
		context_B_start = []
		context_A_end = []
		context_B_end = []
		context_A_recursive = []
		context_B_recursive = []
		idx_0 = Variable(torch.LongTensor([0])).to(my_device)
		idx_1 = Variable(torch.LongTensor([1])).to(my_device)
		for chunk in chunk_A:
			alpha=torch.matmul(self.attn(out0[chunk]),self.attn_embd(idx_0).view(-1,1))
			alpha=F.softmax(alpha, dim=0)
			if self.args.glove_representation=='attn_glove':
				alpha_1 = torch.matmul(self.attn_glove(embd_A_word[chunk]), self.attn_embd(idx_1).view(-1, 1))
				alpha_1 = F.softmax(alpha_1, dim=0)
				span_A.append(torch.matmul(alpha_1.view(1, -1), embd_A_word[chunk].view(len(chunk), -1)))
			else:
				span_A.append(torch.matmul(alpha.view(1,-1),embd_A_word[chunk].view(len(chunk), -1)))
			context_A_weight.append(torch.matmul(alpha.view(1,-1),out0[chunk].view(len(chunk), -1)))
			context_A_start.append(out0[chunk[0]])
			context_A_end.append(out0[chunk[-1]])
			start_i, end_j = chunk[0], chunk[-1]
			phrase_embeding = []
			if start_i-1>=0:
				phrase_embeding.append(lstm_hA[end_j][:,:self.span_representation_dim//4] - lstm_hA[start_i-1][:,:self.span_representation_dim//4])
				phrase_embeding.append(lstm_hA[start_i-1][:,:self.span_representation_dim//4])
			else:
				phrase_embeding.append(lstm_hA[end_j][:, :self.span_representation_dim//4])
				phrase_embeding.append(lstm_hA[start_i][:, :self.span_representation_dim//4] * 0)
			if end_j+1 < len_A:
				phrase_embeding.append(lstm_hA[start_i][:,self.span_representation_dim//4:] - lstm_hA[end_j+1][:, self.span_representation_dim//4:])
				phrase_embeding.append(lstm_hA[end_j+1][:, self.span_representation_dim//4:])
			else:
				phrase_embeding.append(lstm_hA[start_i][:, self.span_representation_dim//4:])
				phrase_embeding.append(lstm_hA[end_j][:, self.span_representation_dim//4:] * 0)
			context_A_recursive.append(torch.cat(phrase_embeding, 1))
		for chunk in chunk_B:
			alpha=torch.matmul(self.attn(out1[chunk]),self.attn_embd(idx_0).view(-1,1))
			alpha=F.softmax(alpha, dim=0)
			if self.args.glove_representation == 'attn_glove':
				alpha_1 = torch.matmul(self.attn_glove(embd_B_word[chunk]), self.attn_embd(idx_1).view(-1, 1))
				alpha_1 = F.softmax(alpha_1, dim=0)
				span_B.append(torch.matmul(alpha_1.view(1, -1), embd_B_word[chunk].view(len(chunk), -1)))
			else:
				span_B.append(torch.matmul(alpha.view(1,-1),embd_B_word[chunk].view(len(chunk), -1)))
			context_B_weight.append(torch.matmul(alpha.view(1,-1),out1[chunk].view(len(chunk), -1)))
			context_B_start.append(out1[chunk[0]])
			context_B_end.append(out1[chunk[-1]])
			start_i, end_j = chunk[0], chunk[-1]
			phrase_embeding = []
			if start_i - 1 >= 0:
				phrase_embeding.append(lstm_hB[end_j][:, :self.span_representation_dim//4] - lstm_hB[start_i - 1][:, :self.span_representation_dim//4])
				phrase_embeding.append(lstm_hB[start_i - 1][:, :self.span_representation_dim//4])
			else:
				phrase_embeding.append(lstm_hB[end_j][:, :self.span_representation_dim//4])
				phrase_embeding.append(lstm_hB[start_i][:, :self.span_representation_dim//4] * 0)
			if end_j + 1 < len_B:
				phrase_embeding.append(lstm_hB[start_i][:, self.span_representation_dim//4:] - lstm_hB[end_j + 1][:, self.span_representation_dim//4:])
				phrase_embeding.append(lstm_hB[end_j + 1][:, self.span_representation_dim//4:])
			else:
				phrase_embeding.append(lstm_hB[start_i][:, self.span_representation_dim//4:])
				phrase_embeding.append(lstm_hB[end_j][:, self.span_representation_dim//4:] * 0)
			context_B_recursive.append(torch.cat(phrase_embeding, 1))
		span_A = torch.unsqueeze(torch.cat(span_A, 0), 0)
		span_B = torch.unsqueeze(torch.cat(span_B, 0), 0)
		context_A_weight = torch.unsqueeze(torch.cat(context_A_weight, 0), 0)
		context_B_weight = torch.unsqueeze(torch.cat(context_B_weight, 0), 0)
		context_A_recursive = torch.unsqueeze(torch.cat(context_A_recursive, 0), 0)
		context_B_recursive = torch.unsqueeze(torch.cat(context_B_recursive, 0), 0)
		context_A_start = torch.unsqueeze(torch.cat(context_A_start, 0), 0)
		context_B_start = torch.unsqueeze(torch.cat(context_B_start, 0), 0)
		context_A_end = torch.unsqueeze(torch.cat(context_A_end, 0), 0)
		context_B_end = torch.unsqueeze(torch.cat(context_B_end, 0), 0)
		# print(context_A_weight.size()) # 1 * len_A * 300
		# print(context_B_weight.size())
		context_A_zhengbao = torch.cat([context_A_weight,context_A_recursive[:,:,self.span_representation_dim//4:self.span_representation_dim//2], context_A_recursive[:,:,-self.span_representation_dim//4:]], 2)
		context_B_zhengbao = torch.cat([context_B_weight,context_B_recursive[:,:,self.span_representation_dim//4:self.span_representation_dim//2], context_B_recursive[:,:,-self.span_representation_dim//4:]], 2)
		context_A_spanbert = torch.cat([context_A_weight, context_A_start, context_A_end],2)
		context_B_spanbert = torch.cat([context_B_weight, context_B_start, context_B_end],2)
		# print(context_A_zhengbao.size()) # 1 * len_A * 450
		# print(context_B_zhengbao.size()) # 1 * len_B * 450
		simCube = self.compute_sim_cube_FF(self.bn_glove(span_A), self.bn_glove(span_B))[0]
		simCube_context_weight = self.compute_sim_cube_FF(self.bn(context_A_weight), self.bn(context_B_weight))[0]
		simCube_context_recursive = self.compute_sim_cube_FF(self.bn(context_A_recursive), self.bn(context_B_recursive))[0]
		simCube_context_zhengbao = self.compute_sim_cube_FF_zhengbao(self.bn_zhengbao(context_A_zhengbao), self.bn_zhengbao(context_B_zhengbao))[0]
		simCube_context_spanbert = self.compute_sim_cube_FF_spanbert(self.bn_spanbert(context_A_spanbert), self.bn_spanbert(context_B_spanbert))[0]

		simCube_A2B = F.pad(simCube, (0, 1), 'constant', 0) # 12 * extended_lenA * extended_lenB --> 12 * extended_lenA * (extended_lenB + 1)
		simCube_context_weight_A2B = F.pad(simCube_context_weight, (0, 1), 'constant', 0)
		simCube_context_recursive_A2B = F.pad(simCube_context_recursive, (0, 1), 'constant', 0)
		simCube_context_zhengbao_A2B = F.pad(simCube_context_zhengbao, (0, 1), 'constant', 0)
		simCube_context_spanbert_A2B = F.pad(simCube_context_spanbert, (0, 1), 'constant', 0)
		if self.args.glove_representation=='no_glove':
			if self.args.span_representation=='avg_bert':
				focusCube_A2B = torch.cat([simCube_context_weight_A2B], 0)
			elif self.args.span_representation=='zhengbao_bert':
				focusCube_A2B = torch.cat([simCube_context_zhengbao_A2B], 0)
			elif self.args.span_representation=='zhengbao_spanbert':
				focusCube_A2B = torch.cat([simCube_context_zhengbao_A2B], 0)
			elif self.args.span_representation=='coref_spanbert':
				focusCube_A2B = torch.cat([simCube_context_spanbert_A2B], 0)
			else:
				focusCube_A2B = torch.cat([simCube_context_recursive_A2B], 0)
		else:
			if self.args.span_representation=='avg_bert':
				focusCube_A2B = torch.cat([simCube_A2B, simCube_context_weight_A2B], 0)
			elif self.args.span_representation=='zhengbao_bert':
				focusCube_A2B = torch.cat([simCube_A2B, simCube_context_zhengbao_A2B], 0)
			elif self.args.span_representation=='zhengbao_spanbert':
				focusCube_A2B = torch.cat([simCube_A2B, simCube_context_zhengbao_A2B], 0)
			elif self.args.span_representation=='coref_spanbert':
				focusCube_A2B = torch.cat([simCube_A2B, simCube_context_spanbert_A2B], 0)
			else:
				focusCube_A2B = torch.cat([simCube_A2B, simCube_context_recursive_A2B], 0)
		focusCube_A2B=focusCube_A2B.permute(1,2,0)

		simCube_B2A = F.pad(simCube.permute(0, 2, 1), (0, 1), 'constant', 0) # 12 * extended_lenB * extended_lenA --> 12 * extended_lenB * (extended_lenA + 1)
		simCube_context_weight_B2A = F.pad(simCube_context_weight.permute(0, 2, 1), (0, 1), 'constant', 0)
		simCube_context_recursive_B2A = F.pad(simCube_context_recursive.permute(0, 2, 1), (0, 1), 'constant', 0)
		simCube_context_zhengbao_B2A = F.pad(simCube_context_zhengbao.permute(0, 2, 1), (0, 1), 'constant', 0)
		simCube_context_spanbert_B2A = F.pad(simCube_context_spanbert.permute(0, 2, 1), (0, 1), 'constant', 0)
		if self.args.glove_representation=='no_glove':
			if self.args.span_representation=='avg_bert':
				focusCube_B2A = torch.cat([simCube_context_weight_B2A], 0)
			elif self.args.span_representation=='zhengbao_bert':
				focusCube_B2A = torch.cat([simCube_context_zhengbao_B2A], 0)
			elif self.args.span_representation=='zhengbao_spanbert':
				focusCube_B2A = torch.cat([simCube_context_zhengbao_B2A], 0)
			elif self.args.span_representation=='coref_spanbert':
				focusCube_B2A = torch.cat([simCube_context_spanbert_B2A], 0)
			else:
				focusCube_B2A = torch.cat([simCube_context_recursive_B2A], 0)
		else:
			if self.args.span_representation=='avg_bert':
				focusCube_B2A = torch.cat([simCube_B2A, simCube_context_weight_B2A], 0)
			elif self.args.span_representation=='zhengbao_bert':
				focusCube_B2A = torch.cat([simCube_B2A, simCube_context_zhengbao_B2A], 0)
			elif self.args.span_representation=='zhengbao_spanbert':
				focusCube_B2A = torch.cat([simCube_B2A, simCube_context_zhengbao_B2A], 0)
			elif self.args.span_representation=='coref_spanbert':
				focusCube_B2A = torch.cat([simCube_B2A, simCube_context_spanbert_B2A], 0)
			else:
				focusCube_B2A = torch.cat([simCube_B2A, simCube_context_recursive_B2A], 0)
		focusCube_B2A = focusCube_B2A.permute(1, 2, 0)

		output_both_A2B=self.mlp1(focusCube_A2B).squeeze() # extended_length_A * (extended_length_B + 1)
		output_both_B2A = self.mlp1(focusCube_B2A).squeeze()  # extended_length_B * (extended_length_A + 1)

		if self.transition_layer_info == 'bin_vector':
			tmp_dist_idx_list = self.transition_matrix_dict[extended_length_B]
			extracted_tensor_idx = Variable(torch.from_numpy(np.array(tmp_dist_idx_list)).type(torch.LongTensor)).to(my_device)
			transition_matrix_sentB = self.distance_embedding(extracted_tensor_idx).view(
				extended_length_B + 1, extended_length_B + 1, -1)

			tmp_dist_idx_list = self.transition_matrix_dict[extended_length_A]
			extracted_tensor_idx = Variable(torch.from_numpy(np.array(tmp_dist_idx_list)).type(torch.LongTensor)).to(my_device)
			transition_matrix_sentA = self.distance_embedding(extracted_tensor_idx).view(
				extended_length_A + 1, extended_length_A + 1, -1)
		else:
			transition_matrix_sentB = Variable(torch.from_numpy(self.transition_matrix_dict[extended_length_B])).type(
				torch.FloatTensor)
			transition_matrix_sentA = Variable(torch.from_numpy(self.transition_matrix_dict[extended_length_A])).type(
				torch.FloatTensor)
			transition_matrix_sentB = transition_matrix_sentB.to(my_device)
			transition_matrix_sentA = transition_matrix_sentA.to(my_device)

		# self.start_transition = Variable(torch.from_numpy(np.array(range(extended_length_B+1))).type(torch.LongTensor)).to(my_device)
		# self.start_transition = self.mlp2(self.START(self.start_transition)).view(-1)
		# self.end_transition = Variable(torch.from_numpy(np.array(range(extended_length_B+1))).type(torch.LongTensor)).to(my_device)
		# self.end_transition = self.mlp2(self.END(self.end_transition)).view(-1)

		transition_matrix_sentB = self.mlp2(transition_matrix_sentB)
		transition_matrix_sentB=transition_matrix_sentB.view(transition_matrix_sentB.size(0), transition_matrix_sentB.size(1))

		transition_matrix_sentA = self.mlp2(transition_matrix_sentA)
		transition_matrix_sentA = transition_matrix_sentA.view(transition_matrix_sentA.size(0), transition_matrix_sentA.size(1))

		if self.use_transition_layer == "False":
			transition_matrix_sentA = transition_matrix_sentA*0
			transition_matrix_sentB = transition_matrix_sentB*0

		if self.training:
			forward_score_A2B = self._forward_alg(output_both_A2B, transition_matrix_sentB, len_A, golden_sequence_sentA)
			gold_score_A2B = self._score_sentence(output_both_A2B, transition_matrix_sentB, golden_sequence_sentA, len_A)
			loss_sentA_to_sentB = forward_score_A2B  - gold_score_A2B

			if bidirectional_training:
				# self.start_transition = Variable(torch.from_numpy(np.array(range(extended_length_A + 1))).type(torch.LongTensor)).to(my_device)
				# self.start_transition = self.mlp2(self.START(self.start_transition)).view(-1)
				# self.end_transition = Variable(torch.from_numpy(np.array(range(extended_length_A + 1))).type(torch.LongTensor)).to(my_device)
				# self.end_transition = self.mlp2(self.END(self.end_transition)).view(-1)
				forward_score_B2A = self._forward_alg(output_both_B2A, transition_matrix_sentA, len_B, golden_sequence_sentB)
				gold_score_B2A = self._score_sentence(output_both_B2A, transition_matrix_sentA, golden_sequence_sentB, len_B)
				loss_sentB_to_sentA = forward_score_B2A  - gold_score_B2A
			else:
				loss_sentB_to_sentA = 0
			return loss_sentA_to_sentB + loss_sentB_to_sentA
		else:
			return output_both_A2B, transition_matrix_sentB, output_both_B2A, transition_matrix_sentA #output_both_B2A, return_sequence_sentA, return_sequence_sentB

