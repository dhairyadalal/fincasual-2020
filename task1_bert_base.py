from typing import List
from tqdm import tqdm

from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch import FloatTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple

WEIGHT = "bert-base-uncased"

class Classifier(pl.LightningModule): 
    
    def __init__(self, 
                 num_classes: int,
                 train_dataloader_: DataLoader,
                 val_dataloader_: DataLoader,
                 weights: str = WEIGHT):
        
        super(Classifier, self).__init__()
        self.train_dataloader_ = train_dataloader_
        self.val_dataloader_ = val_dataloader_
        
        self.bert = AutoModel.from_pretrained(weights)
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_classes)
    
    def forward(self, input_ids: torch.tensor):
        bert_logits, bert_pooled = self.bert(input_ids = input_ids)
        out = self.classifier(bert_pooled)
        return out
    
    def training_step(self, batch, batch_idx):
        # batch
        input_ids, labels = batch
    
        # predict
        y_hat = self.forward(input_ids=input_ids)
        
        # loss 
        loss = F.cross_entropy(y_hat, labels)

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        
        y_hat = self.forward(input_ids = input_ids)
        
        loss = F.cross_entropy(y_hat, labels)
        
        a, y_hat = torch.max(y_hat, dim=1)
        y_hat = y_hat.cpu()
        labels = labels.cpu()

        val_acc = accuracy_score(labels, y_hat)
        val_acc = torch.tensor(val_acc)
        
        val_f1 = f1_score(labels, y_hat, average='micro')
        val_f1 = torch.tensor(val_f1)

        return {'val_loss': loss, 'val_acc': val_acc, 'val_f1': val_f1}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc, 'avg_val_f1': avg_val_f1}
        
        return {'avg_val_loss': avg_loss, 'avg_val_f1':avg_val_f1 ,'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], 
                                lr=2e-05, eps=1e-08)
    
    @pl.data_loader
    def train_dataloader(self):
        return self.train_dataloader_
    
    @pl.data_loader
    def val_dataloader(self):
        return self.val_dataloader_

train  = pd.read_csv("data/practice/task1.csv", names =["index", "text", "gold"], sep=";", header=0)
test   = pd.read_csv("data/trial/task1.csv", names =["index", "text", "gold"], sep=";", header=0)

WEIGHTS = ["distilroberta-base", "bert-base-uncased", "roberta-base", "t5-base"]
BATCH_SIZE = 24

random_seed = 1988
train, val = train_test_split(train, stratify=train["gold"], random_state=random_seed)

for weight in WEIGHTS:
    try:
        tokenizer = AutoTokenizer.from_pretrained(weight)

        X_train = [torch.tensor(tokenizer.encode(text, max_length=200)) for text in train["text"]]
        X_train = pad_sequence(X_train, batch_first=True, padding_value=0)
        y_train = torch.tensor(train["gold"].tolist())

        X_val = [torch.tensor(tokenizer.encode(text, max_length=200)) for text in val["text"]]
        X_val = pad_sequence(X_val, batch_first=True, padding_value=0)
        y_val = torch.tensor(val["gold"].tolist())

        ros = RandomOverSampler(random_state=random_seed)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

        X_train_resampled = torch.tensor(X_train_resampled)
        y_train_resampled = torch.tensor(y_train_resampled)

        train_dataset = TensorDataset(X_train_resampled, y_train_resampled)
        train_dataloader_ = DataLoader(train_dataset,
                                    sampler=RandomSampler(train_dataset),
                                    batch_size=BATCH_SIZE)

        val_dataset = TensorDataset(X_val, y_val)
        val_dataloader_ = DataLoader(val_dataset,
                                    batch_size=BATCH_SIZE)
        
        model = Classifier(num_classes=2,
                            train_dataloader_=train_dataloader_,
                            val_dataloader_ = val_dataloader_,
                            weights=weight)

        trainer = pl.Trainer(gpus=1, 
                            default_save_path="data_ext/fincasual/logs/",
                            max_epochs=30,
                            early_stop_callback=True)
        trainer.fit(model)
        
        X_test = [torch.tensor(tokenizer.encode(text, max_length=200)) for text in test["text"].tolist()]
        X_test = pad_sequence(X_test, batch_first=True, padding_value=0)
        y_test = torch.tensor(test["gold"].tolist())

        test_dataset = TensorDataset(X_test, y_test)
        test_dataloader_ = DataLoader(test_dataset, batch_size=16)

        device = "cuda:0"
        model.eval()
        model = model.to(device)

        test_preds = []
        for batch in tqdm(test_dataloader_, total=len(list(test_dataloader_))):
            ii, _ = batch
            ii = ii.to(device)
            preds = model(input_ids = ii)
            preds = torch.argmax(preds, axis=1).detach().cpu().tolist()
            test_preds.extend(preds)    

        from sklearn.metrics import classification_report

        report = classification_report(test["gold"].tolist(), test_preds)

        with open("task1_experiments/"+weight+"_baseline.txt", "w") as f:
            f.write(report)
    except:
        continue
