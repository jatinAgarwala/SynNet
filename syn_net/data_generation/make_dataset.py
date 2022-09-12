"""
This file generates synthetic tree data in a sequential fashion.
"""
import gzip
import dill as pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from syn_net.utils.data_utils import SyntheticTreeSet
from syn_net.utils.prep_utils import synthetic_tree_generator


if __name__ == '__main__':
    PATH_REACTION_FILE = \
        '/home/whgao/shared/Data/scGen/reactions_pis.pickle.gz'
    PATH_TO_BUILDING_BLOCKS = \
        '/home/whgao/shared/Data/scGen/enamine_building_blocks_nochiral_matched.csv.gz'

    np.random.seed(6)

    building_blocks = pd.read_csv(
        PATH_TO_BUILDING_BLOCKS,
        compression='gzip')['SMILES'].tolist()
    with gzip.open(PATH_REACTION_FILE, 'rb') as f:
        rxns = pickle.load(f)

    TRIAL = 5
    NUM_FINISH = 0
    NUM_ERROR = 0
    NUM_UNFINISH = 0

    trees = []
    for _ in tqdm(range(TRIAL)):
        tree, action = synthetic_tree_generator(
            building_blocks, rxns, max_step=15)
        if action == 3:
            trees.append(tree)
            NUM_FINISH += 1
        elif action == -1:
            NUM_ERROR += 1
        else:
            NUM_UNFINISH += 1

    print('Total trial: ', TRIAL)
    print('num of finished trees: ', NUM_FINISH)
    print('num of unfinished tree: ', NUM_UNFINISH)
    print('num of error processes: ', NUM_ERROR)

    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save('st_data.json.gz')

    # data_file = gzip.open('st_data.pickle.gz', 'wb')
    # pickle.dump(trees, data_file)
    # data_file.close()
