"""
Filters out purchasable building blocks which don't match a single template.
"""
from pandas import read_csv, DataFrame
from tqdm import tqdm
from syn_net.utils.data_utils import *


if __name__ == '__main__':
    R_PATH = '/pool001/whgao/data/synth_net/st_pis/reactions_pis.json.gz'
    BB_PATH = '/home/whgao/scGen/synth_net/data/enamine_us.csv.gz'
    R_SET = ReactionSet()
    R_SET.load(R_PATH)
    MATCHED_MOLS = set()
    for r in tqdm(R_SET.rxns):
        for a_list in r.available_reactants:
            MATCHED_MOLS = MATCHED_MOLS | set(a_list)

    ORIGINAL_MOLS = read_csv(BB_PATH, compression='gzip')['SMILES'].tolist()

    print('Total building blocks number:', len(ORIGINAL_MOLS))
    print('Matched building blocks number:', len(MATCHED_MOLS))

    df = DataFrame({'SMILES': list(MATCHED_MOLS)})
    df.to_csv(
        '/pool001/whgao/data/synth_net/st_pis/enamine_us_matched.csv.gz',
        compression='gzip')
