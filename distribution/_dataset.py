from torch.utils.data import Dataset
import numpy as np
import torch
import random
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from tqdm import tqdm
import _util as UTIL


class SmilesDataset(Dataset):

    def __init__(self, smiles_list, c_to_i=None, stereo=True):
        self.smiles = smiles_list
        self.c_to_i = c_to_i
        self.stereo = stereo
        self.n_char = len(c_to_i)
        self.new_smiles = []
        for s in self.smiles:
            try:
                s = UTIL.sanitize_smiles(s)
                self.new_smiles.append(s)
            except:
                pass

    def __len__(self):
        return len(self.new_smiles)

    def __getitem__(self, idx):
        keydata = self.new_smiles[idx]
        keydata += 'Q'
        sample = dict()
        sample['x'] = torch.from_numpy(np.array([self.c_to_i[c] for c in keydata]))
        sample['len'] = len(keydata)-1
        sample['n_char'] = self.n_char
        sample['smiles'] = self.new_smiles[idx] 
        return sample

    def collate_fn(self, batch) :
        sample = dict()
        n_char = batch[0]['n_char']
        X = torch.nn.utils.rnn.pad_sequence([b['x'] \
                for b in batch], batch_first=True, padding_value = n_char-1)
        L = torch.Tensor([b['len'] for b in batch])
        S = [b['smiles'] for b in batch]
        sample['x'] = X
        sample['len'] = L
        sample['smiles'] = S
        return sample


if __name__ == '__main__':
    pass

