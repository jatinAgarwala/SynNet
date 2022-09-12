"""
Filters out purchasable building blocks which don't match one of the 3 reaction
templates used for unit testing.
"""
import pandas as pd
from tqdm import tqdm
from syn_net.utils.data_utils import *


if __name__ == '__main__':
    R_PATH = './data/ref/rxns_hb.json.gz'
    BB_PATH = '/home/whgao/scGen/synth_net/data/enamine_us.csv.gz'
    R_SET = ReactionSet()
    R_SET.load(R_PATH)
    MATCHED_MOLS = set()
    for r in tqdm(R_SET.rxns):
        for a_list in r.available_reactants:
            MATCHED_MOLS = MATCHED_MOLS | set(a_list)

    ORIGINAL_MOLS = pd.read_csv(BB_PATH, compression='gzip')['SMILES'].tolist()

    print('Total building blocks number:', len(ORIGINAL_MOLS))
    print('Matched building blocks number:', len(MATCHED_MOLS))

    df = pd.DataFrame({'SMILES': list(MATCHED_MOLS)})
    df.to_csv('./data/building_blocks_matched.csv.gz', compression='gzip')
