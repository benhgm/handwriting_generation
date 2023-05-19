import os
import torch
import random
import numpy as np

from collections import Counter
from torch.utils.data import Dataset, DataLoader

# Random seeds
torch.manual_seed(5340)
np.random.seed(5340)
random.seed(5340)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

class HandwritingDataset(Dataset):
    def __init__(self, split='train'):
        self.strokes = np.load("data/strokes.npy", allow_pickle=True, encoding='latin1')
        
        with open('data/sentences.txt') as f:
            self.sentences = f.readlines()

        self.train_strokes = []
        self.train_texts = []
        self.validation_strokes = []
        self.validation_texts = []

        # only get train data with length at most 800
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) <= 801:
                self.train_strokes.append(self.strokes[i])
                self.train_texts.append(self.sentences[i])
            else:
                self.validation_strokes.append(self.strokes[i])
                self.validation_texts.append(self.sentences[i])

        # pad data with zeros to make all training and validation data have same shape
        # train data shape -> 800 x 3
        # val data shape -> 1200 x 3
        # masks are used for loss calculation to find the mean
        self.train_masks = np.zeros((len(self.train_strokes),801))
        for i in range(len(self.train_strokes)):
            self.train_masks[i][0:len(self.train_strokes[i])] = 1
            self.train_strokes[i] = np.vstack([self.train_strokes[i], np.zeros((801-len(self.train_strokes[i]), 3))])
    
        self.validation_masks = np.zeros((len(self.validation_strokes),1201))
        for i in range(len(self.validation_strokes)):
            self.validation_masks[i][0:len(self.validation_strokes[i])] = 1
            self.validation_strokes[i] = np.vstack([self.validation_strokes[i], np.zeros((1201-len(self.validation_strokes[i]), 3))])


        # assign an integer value to each character in the character list
        # there are a total of 50 characters
        char_list = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-'

        char_to_code = {}
        code_to_char = {}
        c = 0
        for _ in char_list:
            char_to_code[_] = c
            code_to_char[c] = _
            c += 1

        torch.save(char_to_code, 'char_to_code.pt')

        # get the maximum text length in the dataset
        max_text_len = np.max(np.array([len(a) for a in self.validation_texts]))

        self.train_onehot = []
        self.train_text_masks = []
        for t in self.train_texts:
            # create a vector of zeros for the one-hot vector. Each element represents 
            # whether a character is present or not. An extra element is added for 'unknown' characters
            onehots = np.zeros((max_text_len, len(char_to_code)+1))
            mask = np.ones(max_text_len)

            for j in range(len(t)):
                # goes through each text sequence and assign a value of 1 to characters that appear in the sequence
                try:
                    onehots[j][char_to_code[t[j]]] = 1
                except:
                    onehots[j][-1] = 1
            
            # zero pad the mask values that are beyond the length of the text
            mask[len(t):] = 0

            self.train_onehot.append(onehots)
            self.train_text_masks.append(mask)
        
        self.train_onehot = np.stack(self.train_onehot)
        self.train_text_masks = np.stack(self.train_text_masks)
        self.train_text_lens = np.array([[len(a)] for a in self.train_texts])

        self.validation_onehot = []
        self.validation_text_masks = []
        for t in self.validation_texts:
            # create a vector of zeros for the one-hot vector. Each element represents 
            # whether a character is present or not. An extra element is added for 'unknown' characters
            onehots = np.zeros((max_text_len, len(char_to_code)+1))
            mask = np.ones(max_text_len)
           
            for k in range(len(t)):
                try:
                    onehots[k][char_to_code[t[k]]] = 1
                except:
                    onehots[k][-1] = 1
            # zero pad the mask values that are beyond the length of the text
            mask[len(t):] = 0

            self.validation_onehot.append(onehots)
            self.validation_text_masks.append(mask)

        self.validation_onehot = np.stack(self.validation_onehot)
        self.validation_text_masks = np.stack(self.validation_text_masks)
        self.validation_text_lens = np.array([[len(a)] for a in self.validation_texts])

        if split == 'train':
            self.strokes = self.train_strokes
            self.stroke_masks = self.train_masks

            self.texts = self.train_texts
            self.text_masks = self.train_text_masks

            self.onehots = self.train_onehot
            
            self.text_lens = self.train_text_lens

        if split == 'val':
            self.strokes = self.validation_strokes
            self.stroke_masks = self.validation_masks

            self.texts = self.validation_texts
            self.text_masks = self.validation_text_masks
            
            self.onehots = self.validation_onehot
            
            self.text_lens = self.validation_text_lens

    def __len__(self):
        return len(self.strokes)
    
    def __getitem__(self, idx):
        stroke = torch.from_numpy(self.strokes[idx]).float()
        stroke_mask = torch.from_numpy(self.stroke_masks[idx]).float()
        onehot = torch.from_numpy(self.onehots[idx]).float()
        text_len = torch.from_numpy(self.text_lens[idx]).float()
        text_mask = torch.from_numpy(self.text_masks[idx]).float()
        return stroke, stroke_mask, onehot, text_mask, text_len

if __name__ == "__main__":
    dataset = HandwritingDataset(split='val')
    data = dataset[1]

    for d in data:
        print(d.shape)
    # onehots = data[2]
    # idxs = torch.arange(0, onehots.shape[-1]).float()
    # text = ''
    # print(onehots.shape)
    # char_to_code = torch.load("char_to_code.pt")
    # print(char_to_code)

    # for i in range(onehots.shape[0]):
    #     idx = torch.dot(onehots[i], idxs).item()
    #     # print(idx)
    #     for k in char_to_code:
    #         if int(idx) == char_to_code[k]:
    #             text += k   
        
    # print(text)
    
