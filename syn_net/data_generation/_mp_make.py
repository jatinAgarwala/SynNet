"""
This file contains a function to generate a single synthetic tree, prepared for
multiprocessing.
"""
import pandas as pd
import numpy as np
# import dill as pickle
# import gzip

from syn_net.data_generation.make_dataset import synthetic_tree_generator
from syn_net.utils.data_utils import ReactionSet


PATH_REACTION_FILE = '/pool001/whgao/data/synth_net/st_pis/reactions_pis.json.gz'
PATH_TO_BUILDING_BLOCKS = '/pool001/whgao/data/synth_net/st_pis/enamine_us_matched.csv.gz'

building_blocks = pd.read_csv(
    PATH_TO_BUILDING_BLOCKS,
    compression='gzip')['SMILES'].tolist()
R_SET = ReactionSet()
R_SET.load(PATH_REACTION_FILE)
rxns = R_SET.rxns
# with gzip.open(PATH_REACTION_FILE, 'rb') as f:
#     rxns = pickle.load(f)

print('Finish reading the templates and building blocks list!')


def func(_):
    """
    This function is used to generate a single synthetic tree, prepared for multiprocessing. 
    Args:
        None
    Returns: 
        tree: The generated synthetic tree
        action: index corresponding to the specific action for tree generation
    """
    np.random.seed(_)
    tree, action = synthetic_tree_generator(building_blocks, rxns, max_step=15)
    return tree, action
