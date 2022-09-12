"""
This file contains a function to predict a single synthetic tree given a molecular SMILES.
"""
import pandas as pd
import numpy as np
from dgllife.model import load_pretrained
from syn_net.utils.data_utils import ReactionSet
from syn_net.utils.predict_utils import synthetic_tree_decoder, tanimoto_similarity, load_modules_from_checkpoint, mol_fp

# define some constants (here, for the Hartenfeller-Button test set)
NBITS = 4096
OUT_DIM = 256
RXN_TEMPLATE = 'hb'
FEATURIZE = 'fp'
PARAM_DIR = 'hb_fp_2_4096_256'
NCPU = 32

# define model to use for molecular embedding
MODEL_TYPE = 'gin_supervised_contextpred'
DEVICE = 'cpu'
mol_embedder = load_pretrained(MODEL_TYPE).to(DEVICE)
mol_embedder.eval()

# load the purchasable building block embeddings
bb_emb = np.load(
    '/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_256.npy')

# define path to the reaction templates and purchasable building blocks
PATH_TO_RXN_FILE = '/pool001/whgao/data/synth_net/st_' + \
    RXN_TEMPLATE + '/reactions_' + RXN_TEMPLATE + '.json.gz'
PATH_TO_BUILDING_BLOCKS = '/pool001/whgao/data/synth_net/st_' + \
    RXN_TEMPLATE + '/enamine_us_matched.csv.gz'

# define paths to pretrained modules
PARAM_PATH = '/home/whgao/scGen/synth_net/synth_net/params/' + PARAM_DIR + '/'
PATH_TO_ACT = PARAM_PATH + 'act.ckpt'
PATH_TO_RT1 = PARAM_PATH + 'rt1.ckpt'
PATH_TO_RXN = PARAM_PATH + 'rxn.ckpt'
PATH_TO_RT2 = PARAM_PATH + 'rt2.ckpt'

# load the purchasable building block SMILES to a dictionary
building_blocks = pd.read_csv(
    PATH_TO_BUILDING_BLOCKS,
    compression='gzip')['SMILES'].tolist()
bb_dict = {building_blocks[i]: i for i in range(len(building_blocks))}

# load the reaction templates as a ReactionSet object
rxn_set = ReactionSet()
rxn_set.load(PATH_TO_RXN_FILE)
rxns = rxn_set.rxns

# load the pre-trained modules
act_net, rt1_net, rxn_net, rt2_net = load_modules_from_checkpoint(
    path_to_act=PATH_TO_ACT,
    path_to_rt1=PATH_TO_RT1,
    path_to_rxn=PATH_TO_RXN,
    path_to_rt2=PATH_TO_RT2,
    featurize=FEATURIZE,
    rxn_template=RXN_TEMPLATE,
    OUT_DIM=OUT_DIM,
    nbits=NBITS,
    ncpu=NCPU,
)


def func(smi):
    """
    Generates the synthetic tree for the input SMILES.

    Args:
        smi (str): Molecular to reconstruct.

    Returns:
        str: Final product SMILES.
        float: Score of the best final product.
        SyntheticTree: The generated synthetic tree.
    """
    emb = mol_fp(smi)
    try:
        tree, action = synthetic_tree_decoder(
            emb, building_blocks, bb_dict, rxns, mol_embedder, act_net,
            rt1_net, rxn_net, rt2_net, bb_emb, rxn_template=RXN_TEMPLATE,
            n_bits=NBITS, max_step=15)
    except Exception as exception:
        print(exception)
        action = -1

    # tree, action = synthetic_tree_decoder(emb, building_blocks, 
            # bb_dict, rxns, mol_embedder, act_net, rt1_net, rxn_net, rt2_net, max_step=15)

    # import ipdb; ipdb.set_trace(context=9)
    # tree._print()
    # print(action)
    # print(np.max(oracle(tree.get_state())))
    # print()

    if action != 3:
        return None, 0, None

    scores = tanimoto_similarity(emb, tree.get_state())
    max_score_idx = np.where(scores == np.max(scores))[0][0]
    return tree.get_state()[max_score_idx], np.max(scores), tree
