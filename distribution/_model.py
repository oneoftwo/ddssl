import torch
import torch.nn as nn
import os
import sys


#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# langugate model autoencoder
class LanguageModel(nn.Module):
    
    def __init__(self, in_dim=46, hid_dim=512, n_layers=2, dropout=0.2):
        super().__init__()
        self.GRU = nn.GRU(input_size=hid_dim, hidden_size=hid_dim, \
                num_layers=n_layers, dropout=dropout)
        self.embedding = nn.Embedding(in_dim, hid_dim)
        self.start_codon = nn.Parameter(torch.zeros(hid_dim), \
                requires_grad=True)
        self.fc = nn.Linear(hid_dim, in_dim)

    def forward(self, x): # x[b l f]
        self.GRU.flatten_parameters()
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        start_codon = self.start_codon.unsqueeze(0).unsqueeze(0).\
                repeat(1, x.size(1), 1)
        x = torch.cat([start_codon, x], 0)
        retval, _ = self.GRU(x)
        retval = retval.permute(1, 0, 2)
        retval = self.fc(retval)
        return retval

    def update(self,directory):
        pretrain_dict = torch.load(directory)
        load_dict = dict()
        load_dict.update(pretrain_dict)
        self.load_state_dict(load_dict,strict=False)


if __name__ == '__main__':
    pass
