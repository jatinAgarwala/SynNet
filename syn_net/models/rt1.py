"""
Reactant1 network (for predicting 1st reactant).
"""
import time
import numpy as np
import torch
from scipy import sparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from syn_net.models.mlp import MLP, load_array


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
    parser.add_argument("--out_dim", type=int, default=256,
                        help="Output dimension.")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="Number of cpus")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--epoch", type=int, default=2000,
                        help="Maximum number of epoches.")
    args = parser.parse_args()

    if args.out_dim == 300:
        VALIDATION_OPTION = 'nn_accuracy_gin'
    elif args.out_dim == 4096:
        VALIDATION_OPTION = 'nn_accuracy_fp_4096'
    elif args.out_dim == 256:
        VALIDATION_OPTION = 'nn_accuracy_fp_256'
    elif args.out_dim == 200:
        VALIDATION_OPTION = 'nn_accuracy_rdkit2d'
    else:
        raise ValueError

    main_dir = f'/pool001/whgao/data/synth_net/\
        {args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_{VALIDATION_OPTION[12:]}/'
    batch_size = args.batch_size
    ncpu = args.ncpu

    X = sparse.load_npz(main_dir + 'X_rt1_train.npz')
    y = sparse.load_npz(main_dir + 'y_rt1_train.npz')
    X = torch.Tensor(X.A)
    y = torch.Tensor(y.A)
    train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

    X = sparse.load_npz(main_dir + 'X_rt1_valid.npz')
    y = sparse.load_npz(main_dir + 'y_rt1_valid.npz')
    X = torch.Tensor(X.A)
    y = torch.Tensor(y.A)
    _idx = np.random.choice(
        list(
            range(
                X.shape[0])), size=int(
            X.shape[0] / 10), replace=False)
    valid_data_iter = load_array(
        (X[_idx], y[_idx]), batch_size, ncpu=ncpu, is_train=False)

    pl.seed_everything(0)
    if args.featurize == 'fp':
        mlp = MLP(input_dim=int(3 * args.nbits),
                  output_dim=args.out_dim,
                  hidden_dim=1200,
                  num_layers=5,
                  dropout=0.5,
                  num_dropout_layers=1,
                  task='regression',
                  loss='mse',
                  valid_loss=VALIDATION_OPTION,
                  optimizer='adam',
                  learning_rate=1e-4,
                  val_freq=10,
                  ncpu=ncpu)
    elif args.featurize == 'gin':
        mlp = MLP(input_dim=int(2 * args.nbits + args.out_dim),
                  output_dim=args.out_dim,
                  hidden_dim=1200,
                  num_layers=5,
                  dropout=0.5,
                  num_dropout_layers=1,
                  task='regression',
                  loss='mse',
                  valid_loss=VALIDATION_OPTION,
                  optimizer='adam',
                  learning_rate=1e-4,
                  val_freq=10,
                  ncpu=ncpu)
    tb_logger = pl_loggers.TensorBoardLogger(
        f'rt1_{args.rxn_template}_{args.featurize}_{args.radius}_\
            {args.nbits}_{VALIDATION_OPTION[12:]}_logs/'
    )

    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=args.epoch,
        progress_bar_refresh_rate=20,
        logger=tb_logger)
    t = time.time()
    trainer.fit(mlp, train_data_iter, valid_data_iter)
    print(time.time() - t, 's')
    print('Finish!')
