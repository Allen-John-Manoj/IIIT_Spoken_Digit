import torch 
from torch.utils.data import DataLoader
from torch import optim 
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np 
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import os
import argparse
import yaml 
from tqdm import tqdm 
from tabulate import tabulate
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
import torchaudio
from torchaudio.transforms import Resample, MFCC
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import os



class Model(nn.Module): 

	def __init__(self, n_mfcc, n_label, h, d, n_lstm): 
		super().__init__()
		self.lstm_layer = nn.LSTM(input_size=n_mfcc, hidden_size=h, num_layers=n_lstm, batch_first=True, bidirectional=True)
		self.lstm_layer_dropout = nn.Dropout()
		self.linear_layer = nn.Linear(in_features=h*2, out_features=d)
		self.linear_layer_relu = nn.ReLU()
		self.linear_layer_dropout = nn.Dropout()
		self.output_layer = nn.Linear(in_features=d, out_features=n_label)
		self.output_layer_logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x, lengths): 
		batch_size = len(x)
		x = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True)
		x, (hn, cn) = self.lstm_layer(x)
		hn = self.lstm_layer_dropout(hn)
		hn = hn.transpose(1, 2).reshape(-1, batch_size).transpose(1, 0)
		hn = self.linear_layer_relu(self.linear_layer(hn))
		hn = self.linear_layer_dropout(hn)
		return self.output_layer_logsoftmax(self.output_layer(hn))

class TrimMFCCs: 

	def __call__(self, batch): 
		return batch[:, 1:, :]

class Standardize:

	def __call__(self, batch): 
		for sequence in batch: 
			sequence -= sequence.mean(axis=0)
			sequence /= sequence.std(axis=0)
		return batch 

class SpokenDigitDataset(Dataset): 
	def __init__(self, path, sr, n_mfcc): 
		assert os.path.exists(path), f'The path for Spoken digit dataset does not exists' 
		self.path = path 
		self.audio_files = os.listdir(path)
		self.sr = sr
		self.transform = torchvision.transforms.Compose([
				MFCC(sample_rate = sr, n_mfcc = n_mfcc+1), 
				TrimMFCCs(),
				Standardize(),
			])

	def __len__(self): 
		return len(os.listdir(self.path))

	def __getitem__(self, index): 
		audio, sr = torchaudio.load(os.path.join(self.path, self.audio_files[index]))
		audio = Resample(sr, self.sr)(audio)
		mfccs = self.transform(audio)
		return mfccs, int(self.audio_files[index][0])

	def split_dataset(self, split_lengths): 
		valid_dataset_len = int((split_lengths[1]/100)*len(self))
		test_dataset_len = int((split_lengths[2]/100)*len(self))
		train_dataset_len = len(self) - (valid_dataset_len+test_dataset_len)
		train_dataset, valid_dataset, test_dataset = random_split(self, [train_dataset_len, valid_dataset_len, test_dataset_len])
		return train_dataset, valid_dataset, test_dataset

def collate(batch): 
	batch.sort(key = (lambda x: x[0].shape[-1]), reverse=True)
	sequences = [mfccs.squeeze(0).permute(1, 0) for mfccs, _ in batch]
	padded_sequences = pad_sequence(sequences, batch_first=True)
	lengths = torch.LongTensor([len(mfccs) for mfccs in sequences])
	labels = torch.LongTensor([label for _, label in batch])
	return padded_sequences, lengths, labels


def train(hp): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(hp['n_mfcc'], hp['n_label'], hp['h'], hp['d'], hp['n_lstm']).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
    dataset = SpokenDigitDataset(hp['dataset_path'], hp['sampling_rate'], hp['n_mfcc'])
    train_dataset, valid_dataset, test_dataset = dataset.split_dataset(hp['train_valid_test_split'])
    
    accuracy_history = {'train': np.zeros(hp['epochs']), 'valid': np.zeros(hp['epochs'])}    
    
    for epoch in tqdm(range(hp['epochs'])):

        model.train()
        no_audio, accuracy = 0, 0
        for batch, lengths, labels in DataLoader(train_dataset, batch_size=hp['batch_size'], collate_fn=collate, shuffle=True): 
            batch = batch.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            y = model(batch, lengths)
            y_pred = torch.argmax(y, dim=1)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()

            no_audio += len(batch)
            accuracy += (y_pred == labels).sum().item()
        accuracy /= no_audio
        accuracy_history['train'][epoch] = accuracy

        model.eval()
        no_audio, valid_accuracy = 0, 0
        with torch.no_grad(): 
            for batch, lengths, labels in DataLoader(valid_dataset, batch_size=hp['batch_size'], collate_fn=collate, shuffle=True):
                batch = batch.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                y = model(batch, lengths)
                y_pred = torch.argmax(y, dim=1)

                no_audio += len(batch)
                valid_accuracy += (y_pred == labels).sum().item()
            valid_accuracy /= no_audio
            accuracy_history['valid'][epoch] = valid_accuracy

    if not os.path.exists(hp['model_path']): 
        os.mkdir(hp['model_path'])
    torch.save(model.state_dict(), os.path.join(hp['model_path'],'model.pt'))

    plt.figure(figsize=(8, 5))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(accuracy_history['train'], c='r', label='training')
    plt.plot(accuracy_history['valid'], c='b', label='validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(hp['model_path'],'accuracy.png'))
    plt.show()  # Add plt.show() here
    plt.clf()

    model.eval()
    batch, lengths, labels = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=collate, shuffle=True)))
    batch = batch.to(device)
    lengths = lengths.to(device)
    labels = labels.to(device)
    y = model(batch, lengths)
    y_pred = torch.argmax(y, dim=1)
    cm = confusion_matrix(labels.to('cpu').numpy(), y_pred.detach().to('cpu').numpy(), labels=hp['labels'])
    acc = np.diag(cm).sum() / cm.sum()
    df = pd.DataFrame(cm, index=hp['labels'], columns=hp['labels'])
    plt.figure(figsize=(10,7))
    sns.heatmap(df, annot=True, cmap='Greens', cbar_kws={'label': 'Scale'})
    plt.title('Confusion matrix for test dataset predictions', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)
    plt.savefig(os.path.join(hp['model_path'],'confusion_matrix.png'))
    plt.show()  # Add plt.show() here
    plt.clf() 
    display_accuracy_table(accuracy_history, acc)

def display_accuracy_table(accuracy_history, acc):
    table_data = [
        ['Train Dataset', f'{accuracy_history["train"][-1]*100:.2f}%'],
        ['Validation Dataset', f'{accuracy_history["valid"][-1]*100:.2f}%'],
        ['Test Dataset', f'{acc*100:.2f}%']
    ]

    print("\n"+tabulate(table_data, headers=['Dataset', 'Accuracy']))

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameters', type=str, default='hyperparameters.yaml', help='The path for hyperparameters.yaml file')
    args = parser.parse_args()

    hyperparameters = yaml.safe_load(open(args.hyperparameters, 'r'))
    train(hyperparameters)
