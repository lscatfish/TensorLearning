#分析模型


import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from minist.model import MNIST_ConvAttnNet,_transform,DEVICE,IMG_FOLDER,train_loader,test_loader,model

def load_model():
	kt=torch.load('./minist/md.pth',map_location=DEVICE)
	print(kt)
	model.load_state_dict(kt )
@torch.no_grad
def eval_meta(img):
	model.eval()
	return model(img)

	

if __name__ == "__main__":
	load_model()
	print(torch.argmax(F.softmax(eval_meta(test_loader[0]))))


	print 

