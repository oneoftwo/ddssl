import numpy as np
import random
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers


def txt_to_smiles_list(fn):
    f = open(fn, 'r')
    smiles_list = []
    for line in f:
        smiles_list.append(line.strip())
    return smiles_list


def smiles_list_to_txt(fn, smiles_list):
    f = open(fn, 'w')
    for idx, smiles in enumerate(smiles_list):
        f.write(smiles)
        if not idx == len(smiles_list) - 1:
            f.write('\n')
    return None


# get visible cuda device
def set_cuda_visible_devices(ngpus):
    import subprocess
    import os
    import numpy as np
    empty = []
    if ngpus>0:
        fn = f'/tmp/empty_gpu_check_{np.random.randint(0,10000000,1)[0]}'
        for i in range(4):
            os.system(f'nvidia-smi -i {i} | grep "No running" | wc -l > {fn}')
            with open(fn) as f:
                out = int(f.read())
            if int(out)==1:
                empty.append(i)
            if len(empty)==ngpus: break
        if len(empty)<ngpus:
            print ('avaliable gpus are less than required', len(empty), ngpus)
            exit(-1)
        os.system(f'rm -f {fn}')
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    os.environ['CUDA_VISIBLE_DEVICES'] = cmd
    return None


# sanitize smiles
def sanitize_smiles(smiles, stereo=True):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    keydata = smiles
    if stereo:
        isomers = list(EnumerateStereoisomers(Chem.MolFromSmiles(keydata)))
        keydata = Chem.MolToSmiles(isomers[0], isomericSmiles=True)
    keydata = Chem.MolToSmiles(Chem.MolFromSmiles(keydata))
    smiles = keydata
    assert not '.' in smiles
    return smiles


if __name__ == '__main__':
    pass

