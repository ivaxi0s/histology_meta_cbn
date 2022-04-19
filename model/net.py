import torch
import torch.nn as nn
import sys, argparse

from resnet import *
# from var_len_lstm import VariableLengthLSTM
from attention import Attention
import json
import pdb

EPS = 1.0e-10

'''
creates the modified ResNet with CBN with the specified version

Arguments:
	n : resnet version [18, 34, 50, 101, 152]
	lstm_size : size of lstm embedding required for the CBN layer
	emb_size : size of hidden layer of MLP used to predict beta and gamma values

Returns:
	model : requried resnet model
'''

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parser = argparse.ArgumentParser('VQA network baseline!')
# args = parser.parse_args()
# def load_config(config_file):
# 	with open(config_file, 'rb') as f_config:
# 		config_str = f_config.read()
# 		config = json.loads(config_str.decode('utf-8'))

# 	return config

# config = load_config(args.config)

def create_resnet(n, mlp_lstm_size, emb_size, use_pretrained): #change lstm size to mlp size
	if n == 18:
		model = resnet18(mlp_lstm_size, emb_size, pretrained=use_pretrained).to(DEVICE)
	if n == 34:
		model = resnet34(mlp_lstm_size, emb_size, pretrained=use_pretrained).to(DEVICE)
	if n == 50:
		model = resnet50(mlp_lstm_size, emb_size, pretrained=use_pretrained).to(DEVICE)
	if n == 101:
		model = resnet101(mlp_lstm_size, emb_size, pretrained=use_pretrained).to(DEVICE)
	if n == 152:
		model = resnet152(mlp_lstm_size, emb_size, pretrained=use_pretrained).to(DEVICE)

	return model

'''
VQA Architecture : 
			Consists of:
				- Embedding layer : to get the word embedding
				- Variable Length LSTM : used to get lstm representation of the question
										 embedding concatenated with the glove vectors
				- Resnet layer with CBN
				- Attention layer
				- MLP for question embedding
				- MLP for image embedding
				- Dropout
				- MLP for fused question and image embedding (element wise product)
				- Softmax Layer
				- Cross Entropy Loss
'''
class Net(nn.Module):

	def __init__(self, no_words, no_answers, resnet_model, mlp_lstm_size, emb_size, use_pretrained=False): #add config
		super(Net, self).__init__()

		self.use_pretrained = use_pretrained # whether to use pretrained ResNet
		# self.word_cnt = no_words # total count of words
		self.ans_cnt = no_answers # total count of valid answers
		self.mlp_lstm_size = mlp_lstm_size # lstm emb size to be passed to CBN layer
		self.emb_size = emb_size # hidden layer size of MLP used to predict delta beta and gamma parameters
		# self.config = config # config file containing the values of parameters
		
		# self.embedding = nn.Embedding(self.word_cnt, self.emb_size) 
		# self.lstm = VariableLengthLSTM().to(DEVICE) #change here cofig #chnage lstm to mlp
		self.net = create_resnet(resnet_model, self.mlp_lstm_size, self.emb_size, self.use_pretrained)
		self.attention = Attention().to(DEVICE)
		 
		self.que_mlp = nn.Sequential(
						nn.Linear(1024, 1024), #no_hidden_LSTM, no_question_mlp
						nn.Tanh(),
						)

		self.img_mlp = nn.Sequential( 
						nn.Linear(2048, 1024), #, no_image_mlp
						nn.Tanh(),
						)

		self.dropout = nn.Dropout(0.5) #config['model']['dropout_keep_prob']

		self.final_mlp = nn.Linear(1024, self.ans_cnt) #config['model']['no_hidden_final_mlp']

		self.softmax = nn.Softmax()

		self.loss = nn.CrossEntropyLoss()
		# pdb.set_trace()

	'''
	Computes a forward pass through the network

	Arguments:
		image : input image
		tokens : question tokens
		glove_emb : glove embedding of the question
		labels : ground truth tokens

	Retuns: 
		loss : hard cross entropy loss
	'''
	def forward(self, image, labels=None):

		####### Question Embedding #######
		# get the lstm representation of the final state at time t
		# que_emb = self.embedding(tokens)
		# emb = torch.cat([que_emb, glove_emb], dim=2)
		# lstm_emb, internal_state = self.lstm(emb)
		# lstm_emb = lstm_emb[:,-1,:]

		####### Image features using CBN ResNet with Attention ########
		feature = self.net(image, 1024) #change attention to image only
		# l2 normalisation
		sq_sum = torch.sqrt(torch.sum(feature**2, dim=1)+EPS)
		sq_sum = torch.stack([sq_sum]*feature.data.shape[1], dim=1)
		feature = feature / sq_sum
		attn_feature = self.attention(feature) #change to image only

		####### MLP for question and image embedding ########
		# mlp_lstm_emb = mlp_lstm_emb.view(feature.data.shape[0], -1)
		# que_embedding = self.que_mlp(lstm_emb)
		image_embedding = self.img_mlp(attn_feature) 

		####### MLP for fused question and image embedding ########
		full_embedding = que_embedding * image_embedding
		full_embedding = self.dropout(full_embedding) #word embedding for 
		out = self.final_mlp(full_embedding)
		
		prob = self.softmax(out)
		val, ind = torch.max(prob, dim=1)
		# hard cross entropy loss
		if labels is not None:
			loss = self.loss(prob, labels)
			return loss, ind
		else:
			return ind


# testing code
if __name__ == '__main__':
	# torch.cuda.set_device(int(sys.argv[1]))
	net = Net( no_words= 1000, no_answers= 2, resnet_model= 18, mlp_lstm_size= 1024, emb_size= 1024)
	ones = torch.ones(8,3,375,375)
	net(ones)
	pdb.set_trace()
	print("done")
