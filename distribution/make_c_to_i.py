import pickle 
import random
from rdkit import Chem 
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
import _util as UTIL


def main():
    # load
    dataset_dir = '../data/public_data/tox21/tox21_preprocessed.pkl'
    dataset = pickle.load(open(dataset_dir, 'rb'))
    save_fn = './c_to_i.pkl'

    # make c_to_i
    c_to_i = {}
    for data in dataset:
        smiles = data['smiles']
        try:
            smiles = UTIL.sanitize_smiles(smiles)
            for s in smiles:
                if s not in c_to_i and s != 'Q':
                    c_to_i[s] = len(c_to_i)
                    print(s)
        except:
            print(smiles)
    c_to_i['Q'] = len(c_to_i)

    # save
    pickle.dump(c_to_i, open(save_fn, 'wb'))
    print(c_to_i)
    print(f'c_to_i saved at {save_fn}')
    
    return None


if __name__ == '__main__':
    main()

