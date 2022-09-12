"""
Computes the fingerprint similarity of molecules in the validation and test set to
molecules in the training set.
"""
import numpy as np
import pandas as pd
from syn_net.utils.data_utils import *
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing as mp
from scripts._mp_search_similar import func


if __name__ == '__main__':

    NCPU = 64

    DATA_PATH = '/pool001/whgao/data/synth_net/st_hb/st_train.json.gz'
    st_set = SyntheticTreeSet()
    st_set.load(DATA_PATH)
    data = st_set.sts
    data_train = [t.root.smiles for t in data]

    DATA_PATH = '/pool001/whgao/data/synth_net/st_hb/st_test.json.gz'
    st_set = SyntheticTreeSet()
    st_set.load(DATA_PATH)
    data = st_set.sts
    data_test = [t.root.smiles for t in data]

    DATA_PATH = '/pool001/whgao/data/synth_net/st_hb/st_valid.json.gz'
    st_set = SyntheticTreeSet()
    st_set.load(DATA_PATH)
    data = st_set.sts
    data_valid = [t.root.smiles for t in data]

    fps_valid = [
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi),
            2,
            n_bits=1024) for smi in data_valid]
    fps_test = [
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi),
            2,
            n_bits=1024) for smi in data_test]

    with mp.Pool(processes=NCPU) as pool:
        results = pool.map(func, fps_valid)
    similaritys = [r[0] for r in results]
    indices = [data_train[r[1]] for r in results]
    df1 = pd.DataFrame({'smiles': data_valid,
                        'split': 'valid',
                        'most similar': indices,
                        'similarity': similaritys})

    with mp.Pool(processes=NCPU) as pool:
        results = pool.map(func, fps_test)
    similaritys = [r[0] for r in results]
    indices = [data_train[r[1]] for r in results]
    df2 = pd.DataFrame({'smiles': data_test,
                        'split': 'test',
                        'most similar': indices,
                        'similarity': similaritys})

    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    df.to_csv('data_similarity.csv', index=False)
    print('Finish!')
