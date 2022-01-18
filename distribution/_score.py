import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from tqdm import tqdm


""" 
functios for score (similarity) measuring
"""


'''
class Quantifier() :

    def __init__(self, model, c_to_i, stereo=True) : 
        self.model = model
        self.c_to_i = c_to_i
        self.stereo = stereo
        
    def __call__(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        with torch.no_grad():
            if mol==None :
                print(smiles)
            if self.stereo: 
                isomers = list(EnumerateStereoisomers(mol))
            else: 
                isomers = [mol]
            best_score = 10000     #minimum value is the best. value = -logP
            for mol in isomers :
                s = Chem.MolToSmiles(mol, isomericSmiles=True)
                s = s+'Q' # Q: End of Codon
                x = torch.tensor([self.c_to_i[i] for i in list(s)]).unsqueeze(0).cuda()
                output = self.model(x)
                p_char = -nn.LogSoftmax(dim = -1)(output)
                p_char = p_char.data.cpu().numpy()
                x = x.data.cpu().numpy()
                isomer_score = 0
                for i in range(len(s)):
                    isomer_score += p_char[0, i, x[0, i]]
                best_score = min(isomer_score, best_score)
        return best_score
'''


# get the score
def get_similarity_score(smiles, model, c_to_i):
    '''
    from trained model, get the score of regeneration for smiles_list
    input:
        smiles_list: smiles list to get the similarity score
        model: trained model
        c_to_i: one-hot encoding dictionary
    output:
        score_list: score list corresponding to smiles list
    '''
    # get c_to_i
    model.cuda()
    model.eval()
    stereo = True
    mol = Chem.MolFromSmiles(smiles)
    with torch.no_grad():
        if mol==None:
            print(smiles)
        if stereo: 
            isomers = list(EnumerateStereoisomers(mol))
        else: 
            isomers = [mol]
        best_score = 99999     #minimum value is the best. value = -logP
        for mol in isomer:
            s = Chem.MolToSmiles(mol, isomericSmiles=True)
            s = s + 'Q' # Q: End of Codon
            x = torch.tensor([c_to_i[i] for i in list(s)]).unsqueeze(0).cuda()
            output = model(x)
            p_char = -nn.LogSoftmax(dim = -1)(output)
            p_char = p_char.data.cpu().numpy()
            x = x.data.cpu().numpy()
            isomer_score = 0
            for i in range(len(s)):
                isomer_score += p_char[0, i, x[0, i]]
            best_score = min(isomer_score, best_score)
        score = 100 - best_score
    return score


# get sim score list from smiles list
def get_similarity_score_list(smiles_list, model, c_to_i):
    score_list = []
    for smiles in tqdm(smiles_list):
        score = get_similarity_score(smiles, model, c_to_i)
        score_list.append(score)
    return score_list


if __name__ == '__main__':
    pass

