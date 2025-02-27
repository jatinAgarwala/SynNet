"""
This file contains the code to decode synthetic trees using beam search at every
sampling step after the action network (i.e. reactant 1, reaction, and reactant 2
sampling).
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs

from syn_net.utils.data_utils import ReactionSet, SyntheticTreeSet

from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from syn_net.utils.predict_utils import mol_fp, get_mol_embedding
from syn_net.utils.predict_beam_utils import synthetic_tree_decoder_fullbeam, load_modules_from_checkpoint

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("-v", "--version", type=int, default=1,
                        help="Version")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=1024,
                        help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--out_dim", type=int, default=300,
                        help="Output dimension.")
    parser.add_argument("--ncpu", type=int, default=16,
                        help="Number of cpus")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--beam_width", type=int, default=5,
                        help="Beam width to use for Reactant1 search")
    parser.add_argument("-n", "--num", type=int, default=-1,
                        help="Number of molecules to decode.")
    parser.add_argument("-d", "--data", type=str, default='test',
                        help="Choose from ['train', 'valid', 'test']")
    args = parser.parse_args()

    # define model to use for molecular embedding
    readout = AvgPooling()
    MODEL_TYPE = 'gin_supervised_contextpred'
    DEVICE = 'cuda:0'
    mol_embedder = load_pretrained(MODEL_TYPE).to(DEVICE)
    mol_embedder.eval()

    # load the purchasable building block embeddings
    bb_emb = np.load(
        f'/pool001/whgao/data/synth_net/st_{args.rxn_template}/enamine_us_emb.npy')

    # define path to the reaction templates and purchasable building blocks
    path_to_reaction_file = f'/pool001/whgao/data/synth_net/st_{args.rxn_template}/reactions_{args.rxn_template}.json.gz'
    PATH_TO_BUILDING_BLOCKS = f'/pool001/whgao/data/synth_net/st_{args.rxn_template}/enamine_us_matched.csv.gz'

    # define paths to pretrained modules
    param_path = f'/home/rociomer/SynthNet/pre-trained-models/{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_v{args.version}/'
    path_to_act = f'{param_path}act.ckpt'
    path_to_rt1 = f'{param_path}rt1.ckpt'
    path_to_rxn = f'{param_path}rxn.ckpt'
    path_to_rt2 = f'{param_path}rt2.ckpt'

    np.random.seed(6)

    # load the purchasable building block SMILES to a dictionary
    building_blocks = pd.read_csv(
        PATH_TO_BUILDING_BLOCKS,
        compression='gzip')['SMILES'].tolist()
    bb_dict = {building_blocks[i]: i for i in range(len(building_blocks))}

    # load the reaction templates as a ReactionSet object
    rxn_set = ReactionSet()
    rxn_set.load(path_to_reaction_file)
    rxns = rxn_set.rxns

    # load the pre-trained modules
    act_net, rt1_net, rxn_net, rt2_net = load_modules_from_checkpoint(
        path_to_act=path_to_act,
        path_to_rt1=path_to_rt1,
        path_to_rxn=path_to_rxn,
        path_to_rt2=path_to_rt2,
        featurize=args.featurize,
        rxn_template=args.rxn_template,
        out_dim=args.out_dim,
        nbits=args.nbits,
        ncpu=args.ncpu,
    )

    def decode_one_molecule(query_smi):
        """
        Generate a synthetic tree from a given query SMILES.

        Args:
            query_smi (str): SMILES for molecule to decode.

        Returns:
            tree (SyntheticTree): The final synthetic tree
            act (int): The final action (to know if the tree was "properly" terminated)
        """
        if args.featurize == 'fp':
            z_target = mol_fp(query_smi, n_bits=args.nbits)
        elif args.featurize == 'gin':
            z_target = get_mol_embedding(query_smi)
        tree, action = synthetic_tree_decoder_fullbeam(
            z_target=z_target, building_blocks=building_blocks,
            bb_dict=bb_dict, reaction_templates=rxns,
            mol_embedder=mol_embedder, action_net=act_net,
            reactant1_net=rt1_net, rxn_net=rxn_net, reactant2_net=rt2_net,
            bb_emb=bb_emb, beam_width=args.beam_width,
            rxn_template=args.rxn_template, n_bits=args.nbits, max_step=15)
        return tree, action

    # load the purchasable building blocks to decode
    path_to_data = f'/pool001/whgao/data/synth_net/st_{args.rxn_template}/st_{args.data}.json.gz'
    print('Reading data from ', path_to_data)
    sts = SyntheticTreeSet()
    sts.load(path_to_data)
    query_smis = [st.root.smiles for st in sts.sts]
    if args.num == -1:
        pass
    else:
        query_smis = query_smis[:args.num]

    output_smis = []
    similaritys = []
    trees = []
    NUM_FINISH = 0
    NUM_UNFINISH = 0

    print('Start to decode!')
    for smi in tqdm(query_smis):

        try:
            TREE, ACTION = decode_one_molecule(smi)
        except Exception as e:
            print(e)
            ACTION = 1
            TREE = None

        if ACTION != 3:
            NUM_UNFINISH += 1
            output_smis.append(None)
            similaritys.append(None)
            trees.append(None)
        else:
            NUM_FINISH += 1
            output_smis.append(TREE.root.smiles)
            ms = [Chem.MolFromSmiles(sm) for sm in [smi, TREE.root.smiles]]
            fps = [Chem.RDKFingerprint(x) for x in ms]
            similaritys.append(
                DataStructs.FingerprintSimilarity(
                    fps[0], fps[1]))
            trees.append(TREE)

    print('Saving ......')
    save_path = '../results/' + args.rxn_template + '_' + args.featurize + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame({'query SMILES': query_smis,
                       'decode SMILES': output_smis,
                       'similarity': similaritys})
    print("mean similarities", df['similarity'].mean(), df['similarity'].std())
    print("NAs", df.isna().sum())
    df.to_csv(
        f'{save_path}decode_result_{args.data}_bw_{args.beam_width}.csv.gz',
        compression='gzip',
        index=False)

    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(
        f'{save_path}decoded_st_bw_{args.beam_width}_{args.data}.json.gz')

    print('Finish!')
