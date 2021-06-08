from __future__ import division
import argparse
from os import listdir
from scipy.spatial.distance import cosine
from scipy.special import softmax
from model_semi_crf import *
from util import *
import time, gzip
import os,pickle
from datetime import timedelta
from datetime import datetime
# from torchtext.vocab import load_word_vectors
def write_pickle_file(file, path, protocol = None):
    if protocol == None:
        with open(path, 'wb') as handle:
            pickle.dump(file, handle)
    else:
        with open(path, 'wb') as handle:
            pickle.dump(file, handle, protocol = protocol)

def convert_0_1_alignment_label(sentA, sentB, label):
	lenA=len(sentA.split()) # sentA is source, sentB is target
	lenB=len(sentB.split())
	token_labels=label.split()
	golden_sequence=[]
	left_dict={}
	while len(token_labels)>0:
		item=token_labels[0]
		i,j=item.split('-')
		i=int(i)
		j=int(j)
		tmp_right=[j]
		if item in token_labels:
			token_labels.remove(item)
		else:
			continue
		pointer=j+1
		while pointer<lenB:
			if str(i)+'-'+str(pointer) in token_labels:
				tmp_right.append(pointer)
				token_labels.remove(str(i) + '-' + str(pointer))
				pointer+=1
			else:
				break
		pointer=j-1
		while pointer>=0:
			if str(i)+'-'+str(pointer) in token_labels:
				tmp_right=[pointer]+tmp_right
				token_labels.remove(str(i) + '-' + str(pointer))
				pointer-=1
			else:
				break
		left_dict[i]=tmp_right
	for i in range(lenA):
		if i in left_dict:
			golden_sequence.append(convert_spanID_to_stateID(left_dict[i][:max_span_size], lenB))
		else:
			golden_sequence.append(0) # 0 is NULL alignment
	return golden_sequence

def threshold_decoding(probA2B, len_A, len_B, threshold):
	result=set()
	num_row, num_col=probA2B.shape
	for i in range(len_A):
		for j in range(num_col):
			if probA2B[i][j]>=threshold:
				source_start_span, source_span_length = convert_stateID_to_spanID(i + 1, len_A)
				target_start_span, target_span_length = convert_stateID_to_spanID(j + 1, len_B)
				for ii in range(source_span_length):
					for jj in range(target_span_length):
						result.add(str(source_start_span+ii)+'-'+str(target_start_span+jj))
	return result

def generate_test_output_crf_alignment(test_set, model, args, save_name_data_to_write=None):
	collect = []
	test_lsents, test_rsents, test_labels = test_set
	precision_all = []
	recall_all = []
	acc = []
	data_to_write = []
	my_thresholds = {}
	for th in [0.02 * k for k in range(1, 50)]:
		my_thresholds[th] = {'precision': [], 'recall': [], 'acc': []}
	crf_prob_all_pairs_dev_sure_only = []
	# for test_i in range(len(test_labels)):
	for test_i in range(len(test_labels)):
		print(test_i)
		if args.add_START_symbol == 'CLS':
			left_item = ('[CLS] ' + test_lsents[test_i][0], test_lsents[test_i][1], test_lsents[test_i][2])
			right_item = ('[CLS] ' + test_rsents[test_i][0], test_rsents[test_i][1], test_rsents[test_i][2])
		elif args.add_START_symbol == 'START':
			left_item = ('S-START ' + test_lsents[test_i][0], test_lsents[test_i][1], test_lsents[test_i][2])
			right_item = ('T-START ' + test_rsents[test_i][0], test_rsents[test_i][1], test_rsents[test_i][2])
		else:
			left_item = (test_lsents[test_i][0], test_lsents[test_i][1], test_lsents[test_i][2])
			right_item = (test_rsents[test_i][0], test_rsents[test_i][1], test_rsents[test_i][2])
		if len(left_item[0].split()) < max_span_size:
			left_item = (' '.join(left_item[0].split() + [max_span_size * 'pad']), left_item[1], left_item[2])
		if len(right_item[0].split()) < max_span_size:
			right_item = (' '.join(right_item[0].split() + [max_span_size * 'unk']), right_item[1], right_item[2])
		left_item = (' '.join(left_item[0].split()[:max_sent_length]), left_item[1], left_item[2])
		right_item = (' '.join(right_item[0].split()[:max_sent_length]), right_item[1], right_item[2])
		len_A = len(left_item[0].split())
		len_B = len(right_item[0].split())
		extended_length_A = max_span_size * len_A - int(
			max_span_size * (max_span_size - 1) / 2)  # if max_span_size=3, then 3*n-3; if max_span_size=4, then 4*n-6
		extended_length_B = max_span_size * len_B - int(max_span_size * (max_span_size - 1) / 2)
		with torch.no_grad():
			try:
				output_both_A2B, transition_matrix_sentB, output_both_B2A, transition_matrix_sentA = model(left_item,
																										   right_item,
																										   None,
																										   None)  # extended_length_A * (extended_length_B + 1)
			except:
				file_pointer = open('wikismall_neural_semicrf_output' + '.num_' + str(len(test_lsents)), "w")
				file_pointer.write(json.dumps(data_to_write, indent=4))
				file_pointer.close()
				return
			if args.decoding_method == 'bi_avg':
				output_both_B2A, transition_matrix_sentA, _, _ = model(right_item, left_item, None,
																	   None)  # extended_length_B * (extended_length_A + 1)
				if args.marginal == 'True':
					_, alpha_all = model.forward_alpha(output_both_A2B, transition_matrix_sentB, len_A)
					_, beta_all = model.backward_beta(output_both_A2B, transition_matrix_sentB, len_A)
					alpha_all = alpha_all.data.cpu().numpy()
					beta_all = beta_all.data.cpu().numpy()
					probA2B = softmax(alpha_all + beta_all, axis=1)
					_, alpha_all = model.forward_alpha(output_both_B2A, transition_matrix_sentA, len_B)
					_, beta_all = model.backward_beta(output_both_B2A, transition_matrix_sentA, len_B)
					alpha_all = alpha_all.data.cpu().numpy()
					beta_all = beta_all.data.cpu().numpy()
					probB2A = softmax(alpha_all + beta_all, axis=0)
					prob_merged = (probA2B[:, 1:] + np.transpose(probB2A[:, 1:])) / 2
				else:
					probA2B = softmax(output_both_A2B.data.cpu().numpy(), axis=1)
					probB2A = softmax(output_both_B2A.data.cpu().numpy(), axis=0)
					prob_merged = (probA2B[:, :-1] + np.transpose(probB2A[:, :-1])) / 2
				# print(prob_merged[0])
				gold = set(test_labels[test_i].split())
				crf_prob_all_pairs_dev_sure_only.append(prob_merged)
				for th in my_thresholds:
					pred = threshold_decoding(prob_merged, len_A, len_B, th)
					if len(pred) > 0:
						precision = len(gold & pred) / len(pred)
					else:
						if len(gold) == 0:
							precision = 1
						else:
							precision = 0
					if len(gold) > 0:
						recall = len(gold & pred) / len(gold)
					else:
						if len(pred) == 0:
							recall = 1
						else:
							recall = 0
					if len(pred & gold) == len(gold) and len(pred & gold) == len(pred):
						my_thresholds[th]['acc'].append(1)
					else:
						my_thresholds[th]['acc'].append(0)
					my_thresholds[th]['precision'].append(precision)
					my_thresholds[th]['recall'].append(recall)
			else:
				optimal_sequence_A = model.viterbi_decoder(output_both_A2B, transition_matrix_sentB, len_A,
														   extended_length_B)
				optimal_sequence_B = model.viterbi_decoder(output_both_B2A, transition_matrix_sentA, len_B,
														   extended_length_A)
				# forward_score_A2B = model._forward_alg(output_both_A2B, transition_matrix_sentB, len_A, optimal_sequence_A)
				# gold_score_A2B = model._score_sentence(output_both_A2B, transition_matrix_sentB, optimal_sequence_A,len_A)
				# loss_sentA_to_sentB = (forward_score_A2B - gold_score_A2B).data.cpu().numpy()
				# forward_score_B2A = model._forward_alg(output_both_B2A, transition_matrix_sentA, len_B,optimal_sequence_B)
				# gold_score_B2A = model._score_sentence(output_both_B2A, transition_matrix_sentA, optimal_sequence_B,len_B)
				# loss_sentB_to_sentA = (forward_score_B2A - gold_score_B2A).data.cpu().numpy()
				predA = set()
				for i, item in enumerate(optimal_sequence_A):
					stateID = item
					spanID, spanLength = convert_stateID_to_spanID(stateID, len(right_item[0].split()))
					if spanID >= 0:
						for kk in range(spanLength):
							if args.add_START_symbol:
								if i >= 1 and spanID + kk >= 1:  # special operation for START token
									predA.add(str(i - 1) + '-' + str(spanID + kk - 1))
							else:
								predA.add(str(i) + '-' + str(spanID + kk))
				predB = set()
				for i, item in enumerate(optimal_sequence_B):
					stateID = item
					spanID, spanLength = convert_stateID_to_spanID(stateID, len(left_item[0].split()))
					if spanID >= 0:
						for kk in range(spanLength):
							if args.add_START_symbol:
								if spanID + kk >= 1 and i >= 1:
									predB.add(str(spanID + kk - 1) + '-' + str(i - 1))
							else:
								predB.add(str(spanID + kk) + '-' + str(i))
				if args.decoding_method == 'AandB':
					pred = predA & predB
				elif args.decoding_method == 'AorB':
					pred = predA | predB
				elif args.decoding_method == 'A2B':
					pred = predA
				elif args.decoding_method == 'B2A':
					pred = predB
				elif args.decoding_method == 'grow_diag':
					pred = merge_alignment(predA, predB, len_A, len_B)
				# print(pred)
				collect.append(pred)
				my_dict = {}
				my_dict['id'] = str(test_i + 1)
				my_dict['source'] = left_item
				my_dict['target'] = right_item
				my_dict['A'] = (
				output_both_A2B, transition_matrix_sentB, len_A, extended_length_B, optimal_sequence_A, predA)
				my_dict['B'] = (
				output_both_B2A, transition_matrix_sentA, len_B, extended_length_A, optimal_sequence_B, predB)
				my_dict['output'] = {'AandB': predA & predB, 'grow_diag': merge_alignment(predA, predB, len_A, len_B)}
				data_to_write.append(my_dict)
				# data_to_write.append((' '.join(left_item[0].split()[1:]), ' '.join(right_item[0].split()[1:]), ' '.join(list(pred)), str(loss_sentA_to_sentB), str(loss_sentB_to_sentA)))
				# continue
				gold = set(test_labels[test_i].split())
				if len(pred) > 0:
					precision = len(gold & pred) / len(pred)
				else:
					if len(gold) == 0:
						precision = 1
					else:
						precision = 0
				if len(gold) > 0:
					recall = len(gold & pred) / len(gold)
				else:
					if len(pred) == 0:
						recall = 1
					else:
						recall = 0
				precision_all.append(precision)
				recall_all.append(recall)
				if len(pred & gold) == len(gold) and len(pred & gold) == len(pred):
					acc.append(1)
				else:
					acc.append(0)

	# print(collect)
	# file_pointer = open('ll', 'w')
	#	file_pointer.write(json.dumps(collect))
	#	file_pointer.close()
	#	with open('ll1.txt', 'w') as f:
	#		for i in collect:
	#			f.writelines('\t'.join(i) + '\n')
	#	print('done writing txt')

	if save_name_data_to_write != None:
		write_pickle_file(data_to_write, save_name_data_to_write)
		print('done writing pkl')
	# with open('ll2', 'w') as f:
	#		json.dump(f, collect)

	# file_pointer = open('wikiqa_old_neural_semicrf_output' + '.num_' + str(len(test_lsents)), "w")
	# file_pointer.write(json.dumps(data_to_write, indent=4))
	# file_pointer.close()
	# with open(args.alignment_save_path+'align.txt','w') as f:
	# 	for line in data_to_write:
	# 		f.writelines('\t'.join(line)+'\n')
	# return
	if args.decoding_method == 'bi_avg':
		best_f1 = 0
		for th in my_thresholds:
			precision = sum(my_thresholds[th]['precision']) / len(my_thresholds[th]['precision'])
			recall = sum(my_thresholds[th]['recall']) / len(my_thresholds[th]['recall'])
			acc = sum(my_thresholds[th]['acc']) / len(my_thresholds[th]['acc'])
			if precision + recall > 0:
				f1 = 2 * precision * recall / (precision + recall)
			else:
				f1 = 0
			my_thresholds[th]['precision'] = precision
			my_thresholds[th]['recall'] = recall
			my_thresholds[th]['f1'] = f1
			my_thresholds[th]['acc'] = acc
			if f1 > best_f1:
				best_f1 = f1
				best_acc = acc
				best_precision = precision
				best_recall = recall
		# print(my_thresholds)
		# pickle.dump(crf_prob_all_pairs_dev_sure_only, open('crf_prob_all_pairs_dev_sure_poss.p', 'wb'))
		print('threshold\tprecision\trecall\tF1\tacc')
		for th in my_thresholds:
			print(th, my_thresholds[th]['precision'], my_thresholds[th]['recall'], my_thresholds[th]['f1'],
				  my_thresholds[th]['acc'])
		return best_precision, best_recall, best_f1, best_acc
	else:
		precision = sum(precision_all) / len(precision_all)
		recall = sum(recall_all) / len(recall_all)
		if precision + recall > 0:
			f1 = 2 * precision * recall / (precision + recall)
		else:
			f1 = 0
		# file_pointer = open('final_model_dev_sure_poss_semiCRF_output' + '.F1_' + str(f1), "w")
		# file_pointer.write(json.dumps(data_to_write, indent=4))
		# file_pointer.close()
		return precision, recall, f1, sum(acc) / len(acc)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--span_representation', type=str, default='coref_spanbert',
	                    help='Currently supported representations: avg_bert, zhengbao_bert, zhengbao_spanbert, coref_spanbert, lstm_minus_bert')
	parser.add_argument('--glove_representation', type=str, default='no_glove',
	                    help='Currently supported representations: no_glove, cur_glove, attn_glove')
	parser.add_argument('--sure_and_possible', type=str, default='True',
	                    help='if False, will use sure only setting.')
	parser.add_argument('--freeze_bert', type=str, default='False',
	                    help='if False, will update weights in BERT or SpanBERT')
	parser.add_argument('--decoding_method', type=str, default='AandB',
	                    help='Currently supported decoding methods: A2B, B2A, AandB, AorB, grow_diag, bi_avg')
	parser.add_argument('--eval_mode', type=str, default='False',
	                    help='if set True, we start evaluation, otherwise we keep training the model')
	parser.add_argument('--layer', type=str, default='12',
	                    help='Extract hidden representations from layer: 0, 1, 2, ... 12. Note: 12 is the last layer.')
	parser.add_argument('--projection', type=str, default='False',
	                    help='If False, will not project BERT/SpanBERT representations from 768 to 300 dimensions')
	parser.add_argument('--learning_rate', type=float, default=1e-5,
	                    help='Learning rate for neural semi-CRF.')
	parser.add_argument('--hamming', type=float, default=1,
	                    help='Add hamming cost during training')
	parser.add_argument('--marginal', type=str, default='True',
	                    help='if False, will not calculate marginal probability in bidi-avg decoding.')
	parser.add_argument('--use_transition_layer', type=str, default='True',
	                    help="if False, will set transition score to 0.")
	parser.add_argument('--transition_layer_info', type=str, default='bin_vector',
	                    help='scalar, bin, scalar+bin, bin_vector')
	parser.add_argument('--use_absolute_dist', type=str, default='False',
	                    help='if True, will use the absolute value of distance.')
	parser.add_argument("--distance_embedding_size", type=int, default=128)
	parser.add_argument("--update_distance_embedding", type=str, default='False',
	                    help='if False, will freeze distance embedding during training.')
	parser.add_argument("--add_START_symbol", type=str, default='CLS',
	                    help='CLS or START')
	parser.add_argument("--add_SEP_symbol", type=str, default='False',
	                    help='True or False')
	parser.add_argument('--bert_with_coref_span', type=str, default='False',
	                    help='For ablation study, if True, use BERT with coref-spanbert span representation')
	parser.add_argument("--note", type=str, default='',
						help='spacial prefixs added to ckpts')
	parser.add_argument("--dataset", type=str, default='MTReference',
						help='MTReference, Wikipedia, newsela, arxiv')
	args = parser.parse_args()
	print(args)
	print("max_span_size is {}".format(max_span_size))
	print("max_sent_length is {}".format(max_sent_length))

	data_set = args.dataset  # MTReference_LUC, MTReference, iSTS_LDC, iSTS, wikismall
	if 'MTReference' in data_set:
		transpose=True
	else:
		transpose=False
	# save_name='saved_checkpoint_'+'freeze_bert_'+args.freeze_bert+'_glove_representation_'+args.glove_representation+'_'+args.span_representation+'_'+'sure_and_possible_'+args.sure_and_possible+'_projection_'+args.projection \
	#           +'_layer_'+args.layer+'_lr_'+str(args.learning_rate) + '_use_transition_layer_'+args.use_transition_layer+'_transition_layer_info_'+args.transition_layer_info + '_use_absolute_dist_' + args.use_absolute_dist
	save_name=args.note+'_sure_and_possible_'+args.sure_and_possible+'_dataset_'+args.dataset+'_decoding_'+args.decoding_method
	print(save_name)
	org_save_name = save_name
	if data_set=='MTReference':
		train_set = readEdinburghDataset_reverse(base_path + 'data/mt-reference/clean-version/clean-train-dep-old-version.json', args.sure_and_possible=='True',transpose)
		dev_set = readEdinburghDataset_reverse(base_path + 'data/mt-reference/clean-version/human_errors/dev_and_test_after_correction/clean-dev.tsv',args.sure_and_possible=='True', transpose)
		test_set = readEdinburghDataset_reverse(base_path + 'data/mt-reference/clean-version/human_errors/dev_and_test_after_correction/clean-test.tsv',args.sure_and_possible=='True', transpose)
		# newsela_set = readEdinburghDataset_reverse(base_path + 'data/newsela/batch_01.tsv', sure_and_possible, transpose)
		# arxiv_set = readEdinburghDataset_reverse(base_path + 'data/arxiv/batch_01.tsv', sure_and_possible, transpose)
		# wiki_set = readEdinburghDataset_reverse(base_path + 'data/wikipedia/qa-wiki-test.txt', sure_and_possible, transpose)
	elif data_set=='Wikipedia':
		train_set = readEdinburghDataset_reverse(base_path + 'data/wikipedia/qa-wiki-train.txt', True, transpose)
		dev_set = readEdinburghDataset_reverse(base_path + 'data/wikipedia/qa-wiki-dev.txt', True, transpose)
		test_set = readEdinburghDataset_reverse(base_path + 'data/wikipedia/qa-wiki-test.txt', True, transpose)
	elif data_set=='iSTS':
		train_set = readEdinburghDataset_reverse(base_path + 'data/iSTS/train/iSTS.headlines+images.train.json', False, False)
		test_set_headlines = readEdinburghDataset_reverse(base_path + 'data/iSTS/test/iSTS.headlines.test.json', False, False)
		test_set_images = readEdinburghDataset_reverse(base_path + 'data/iSTS/test/iSTS.images.test.json', False, False)
		dev_set= (test_set_headlines[0] + test_set_images[0], test_set_headlines[1] + test_set_images[1], test_set_headlines[2] + test_set_images[2])
		test_set=dev_set
	elif data_set=='wikismall':
		train_set = readWikismall(base_path+'data/wikismall/', 'train')
		dev_set = readWikismall(base_path + 'data/wikismall/', 'valid')
		test_set = readWikismall(base_path + 'data/wikismall/', 'test')
	elif data_set=='newsela-old':
		train_set = readNewselaold(base_path+'data/newsela-old/', 'train')
		dev_set = readNewselaold(base_path + 'data/newsela-old/', 'valid')
		test_set = readNewselaold(base_path + 'data/newsela-old/', 'test')
	elif data_set == 'wikiqa':
		train_set = readWikiQA(base_path+'data/wikiqa/','train')
		dev_set = readWikiQA(base_path+'data/wikiqa/','dev')
		test_set = readWikiQA(base_path+'data/wikiqa/','test')
	print('MTReference training size: %d' % len(train_set[0]))
	print('MTReference dev size: %d' % len(dev_set[0]))
	print('MTReference test size: %d' % len(test_set[0]))
	embedding_dim = 300
	num_layers=1
	num_epochs = 5
	# if args.glove_representation!='no_glove':
	# 	wv_dict, wv_arr, wv_size = load_word_vectors(embedding_path, 'glove.840B', embedding_dim) # len(wv_dict.keys()) = 2196016
	# else:
	# 	wv_dict, wv_arr, wv_size = '', '', ''
	wv_dict, wv_arr, wv_size = '', '', ''
	lsents, rsents, labels = train_set
	model = NeuralWordAligner(data_set,embedding_dim, num_layers,wv_dict, wv_arr, args, max_span_size)
	model = model.to(my_device)
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=1e-5)
	try:
		all_model_names = listdir(base_path + 'checkpoints/')
	except:
		print('please make checkpoints folder first.')
		sys.exit()
	selected_names = []
	for current_model_name in all_model_names:
		if save_name in current_model_name:
			split_name = current_model_name.split('_')
			epoch_index = split_name.index('epoch')
			selected_names.append((int(split_name[epoch_index + 1]), current_model_name))
	selected_names = sorted(selected_names, key=lambda x: x[0])
	if args.eval_mode=='True':
		for (epoch_number, current_model_name) in selected_names:
			print(current_model_name)
			checkpoint = torch.load(base_path+'checkpoints/'+current_model_name)
			model.load_state_dict(checkpoint['model_state_dict'])
			model.eval()
			print('start evaluation...')
			# args.alignment_save_path=base_path+'data/wikiqa/test/'
			# generate_test_output_crf_alignment(test_set, model, args)
			args.alignment_save_path = base_path + 'data/wikiqa/dev/'
			generate_test_output_crf_alignment(dev_set, model, args)
			args.alignment_save_path = base_path + 'data/wikiqa/train/'
			generate_test_output_crf_alignment(train_set, model, args)
			sys.exit()
			precision, recall, f1, acc = generate_test_output_crf_alignment(dev_set, model, args)
			print('MTReference dev score: precision: %.6f  recall: %.6f  f1: %.6f acc: %.6f' % (precision, recall, f1, acc))
			precision, recall, f1, acc = generate_test_output_crf_alignment(test_set, model, args)
			print('MTReference test score: precision: %.6f  recall: %.6f  f1: %.6f acc: %.6f' % (precision, recall, f1, acc))
		sys.exit()
	if len(selected_names)==0:
		print('No checkpoint found, training from epoch 0')
		best_f1 = 0
		best_acc = 0
	else:
		epoch_number, current_model_name=selected_names[-1]
		print('%d checkpoint(s) found, training from epoch %d' % (len(selected_names), len(selected_names)))
		checkpoint = torch.load(base_path + 'checkpoints/' + current_model_name)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		best_f1=checkpoint['best_f1']
		best_acc=checkpoint['best_acc']
		for empty_call in range(len(selected_names)):
			indices = torch.randperm(len(lsents))
	chunk_statistics={}
	for epoch in range(len(selected_names), num_epochs, 1):
		accumulated_loss = 0
		model.train()
		indices = torch.randperm(len(lsents))
		start_time = time.time()
		train_num_correct=0
		valid_train_examples=0
		data_loss = 0
		for index, iter_i in enumerate(indices):
			sentA, _, _ = lsents[iter_i]
			sentB, _, _ = rsents[iter_i]
			if args.add_START_symbol == 'CLS':
				sentA = '[CLS] ' + sentA
				sentB = '[CLS] ' + sentB
				gold_label = ['0-0']
				for pair in labels[iter_i].split():
					left, right = pair.split('-')
					left, right = int(left), int(right)
					gold_label.append(str(left+1)+'-'+str(right+1))
				gold_label=' '.join(gold_label)
			elif args.add_START_symbol == 'START':
				sentA = 'S-START ' + sentA
				sentB = 'T-START ' + sentB
				gold_label = ['0-0']
				for pair in labels[iter_i].split():
					left, right = pair.split('-')
					left, right = int(left), int(right)
					gold_label.append(str(left+1)+'-'+str(right+1))
				gold_label=' '.join(gold_label)
			else:
				gold_label=labels[iter_i]
			if len(sentA.split())> 80 or len(sentB.split())> 80:
				continue
			if len(sentA.split())< max_span_size or len(sentA.split())> max_sent_length or len(sentA.split())==1 or len(sentB.split())< max_span_size or len(sentB.split())> max_sent_length or len(sentB.split())==1:
				continue
			golden_sequence_sentA = convert_0_1_alignment_label(sentA, sentB, gold_label) # sentA as source
			if bidirectional_training:
				reverse_label = ' '.join([item.split('-')[1]+'-'+item.split('-')[0] for item in labels[iter_i].split()])
				if args.add_START_symbol:
					gold_label = ['0-0']
					for pair in reverse_label.split():
						left, right = pair.split('-')
						left, right = int(left), int(right)
						gold_label.append(str(left + 1) + '-' + str(right + 1))
					gold_label = ' '.join(gold_label)
				else:
					gold_label = reverse_label
				golden_sequence_sentB = convert_0_1_alignment_label(sentB, sentA, gold_label) # sentB as source
			else:
				golden_sequence_sentB = None
			optimizer.zero_grad()
			loss = model((sentA, None, None), (sentB, None, None), golden_sequence_sentA, golden_sequence_sentB)
			loss.backward()
			data_loss += loss.data
			optimizer.step()
			valid_train_examples+=1
			if valid_train_examples%50==0:
				print('# ' + str(valid_train_examples) + ': Loss: ' + str(data_loss/valid_train_examples))
				# break
		# start testing
		print('--' * 20)
		msg = '%d completed epochs, %d batches' % (epoch, valid_train_examples)
		if valid_train_examples>0:
			msg += '\t training batch loss: %f' % (data_loss / (valid_train_examples))
		print(msg)
		model.eval()
		print('Results:')
		# precision, recall, dev_f1, acc = generate_test_output_crf_alignment(dev_set, model, args)
		precision, recall, dev_f1, acc = generate_test_output_crf_alignment(dev_set, model, args,  base_path + 'checkpoints/'+ save_name +'_epoch_'+str(epoch)+'_dev_data_to_write.pt')

		print('MTReference dev score: precision: %.6f  recall: %.6f  f1: %.6f acc: %.6f' % (precision, recall, dev_f1, acc))
		if (dev_f1 > best_f1 or acc > best_acc) and True:
			best_f1 = dev_f1
			best_acc = acc
			if 'iSTS' in data_set:
				print('Headlines results:')
				save_name=org_save_name+'Headlines_'
				precision, recall, f1, acc = generate_test_output_crf_alignment(test_set_headlines, model, args)
				print('Test score: precision: %.6f  recall: %.6f  f1: %.6f acc: %.6f' % (precision, recall, f1, acc))
				print('Images results:')
				save_name = org_save_name + 'Images_'
				precision, recall, f1, acc = generate_test_output_crf_alignment(test_set_images, model, args)
				print('Test score: precision: %.6f  recall: %.6f  f1: %.6f acc: %.6f' % (precision, recall, f1, acc))
			else:
				precision, recall, f1, acc = generate_test_output_crf_alignment(test_set, model, args)
				print('MTReference test score: precision: %.6f  recall: %.6f  f1: %.6f acc: %.6f' % (precision, recall, f1, acc))
		torch.save({
			'epoch': epoch,
			'best_f1': best_f1,
			'best_acc': best_acc,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
		}, base_path + 'checkpoints/'+ save_name +'_epoch_'+str(epoch)+'_'+"%.4f" % dev_f1 + '.pt')
		elapsed_time = time.time() - start_time
		print('Epoch ' + str(epoch) + ' finished within ' + str(timedelta(seconds=elapsed_time)) + ', and current time:' + str(datetime.now()))
		model.train()
