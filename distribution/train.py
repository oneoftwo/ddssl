import pickle
from tqdm import tqdm 
import torch
import time
import _dataset as DATASET 
import _util as UTIL
import _model as MODEL
from torch.utils.data import DataLoader
from torch import nn
import numpy as np 
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# train model for given epoch with given data (smiles_list)
def train_autoencoder(model, smiles_list, c_to_i, lr, n_epoch=100):
    train_dataset = DATASET.SmilesDataset(smiles_list, c_to_i)
    loss_list = []
    time_list = []
    # train model
    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, \
            num_workers=4, collate_fn=train_dataset.collate_fn)
    model.train()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_char = train_dataset.n_char
    for epoch in range(1, n_epoch + 1):
        start_time = time.time()
        model, train_loss = \
                train_one_epoch(model, trainloader, optimizer, n_char)
        spent_time = time.time() - start_time
        loss_list.append(train_loss)
        time_list.append(spent_time)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (0.99 ** epoch)
    return model, loss_list


# train single epoch
def train_one_epoch(model, dataloader, optimizer, n_char):
    loss_list = []
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    for sample in tqdm(dataloader):
        x = sample['x'].cuda().long()
        l = sample['len'].cuda().long()
        output = model(x)
        mask = len_mask(l + 1, output.size(1) - 1)
        loss = torch.sum(loss_fn(output[:,:-1].reshape(-1, n_char), \
                x.reshape(-1)) * mask) / mask.sum()
        loss_list.append(loss.data.cpu().numpy())
        if optimizer:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()
    loss = np.mean(np.array(loss_list))
    print(loss)
    return model, loss


def len_mask(l, max_length) :
    """
    Mask the padding part of the result
    example data:
    c1ccccc1Q_________
    c1ccccc1Cc2ccccc2Q
    CNQ_______________
    ...
    We set the value of Padding part to 0
    """
    device = l.device
    mask = torch.arange(0, max_length).repeat(l.size(0)).to(device)
    l = l.unsqueeze(1).repeat(1, max_length).reshape(-1)
    mask = mask-l
    mask[mask>=0] = 0
    mask[mask<0] = 1 
    return mask


def main(label=0):

    # get data 
    processed_data = pickle.load(open('../data/public_data/tox21/tox21_preprocessed.pkl', 'rb'))
    c_to_i = pickle.load(open('./c_to_i.pkl', 'rb'))
    UTIL.set_cuda_visible_devices(1)
    save_model_fn = f'./save/model/model_{label}'

    # filter by label 
    smiles_list = [sample['smiles'] for sample in processed_data if sample['label'] == label]
    smiles_list = smiles_list

    # make torch dataset and model
    model = MODEL.LanguageModel(in_dim=len(c_to_i), hid_dim=128, n_layers=2, dropout=0.5)
    model, loss_history = train_autoencoder(model, smiles_list, c_to_i, lr=1e-5, n_epoch=1000)
    print(loss_history)
    pickle.dump(model, open(save_model_fn, 'wb'))
    

if __name__ == '__main__':
    label = 1
    main(label)

