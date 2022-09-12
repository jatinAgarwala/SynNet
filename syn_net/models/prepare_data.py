"""
Prepares the training, testing, and validation data by reading in the states
and steps for the reaction data and re-writing it as separate one-hot encoded
Action, Reactant 1, Reactant 2, and Reaction files.
"""
from syn_net.utils.prep_utils import prep_data


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan fingerprint.")
    parser.add_argument(
        "--outputembedding",
        type=str,
        default='gin',
        help="Choose from ['fp_4096', 'fp_256', 'gin', 'rdkit2d']")
    args = parser.parse_args()
    rxn_template = args.rxn_template
    featurize = args.featurize
    output_emb = args.outputembedding

    main_dir = '/pool001/whgao/data/synth_net/' + rxn_template + '_' + featurize + '_' + \
        str(args.radius) + '_' + str(args.nbits) + '_' + str(args.outputembedding) + '/'
    if rxn_template == 'hb':
        NUM_RXN = 91
    elif rxn_template == 'pis':
        NUM_RXN = 4700

    if output_emb == 'gin':
        OUT_DIM = 300
    elif output_emb == 'rdkit2d':
        OUT_DIM = 200
    elif output_emb == 'fp_4096':
        OUT_DIM = 4096
    elif output_emb == 'fp_256':
        OUT_DIM = 256

    prep_data(main_dir=main_dir, num_rxn=NUM_RXN, out_dim=OUT_DIM)

    print('Finish!')
