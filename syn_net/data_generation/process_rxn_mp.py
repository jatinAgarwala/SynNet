"""
This file processes a set of reaction templates and finds applicable
reactants from a list of purchasable building blocks.

Usage:
    python process_rxn_mp.py
"""
import multiprocessing as mp
from time import time

import shutup
from syn_net.utils.data_utils import Reaction, ReactionSet
import syn_net.data_generation._mp_process as process
shutup.please()


if __name__ == '__main__':
    NAME = 'pis'
    PATH_TO_RXN_TEMPLATES = '/home/whgao/scGen/synth_net/data/rxn_set_' + NAME + '.txt'
    RXN_TEMPLATES = []
    for line in open(PATH_TO_RXN_TEMPLATES, 'rt'):
        rxn = Reaction(line.split('|')[1].strip())
        RXN_TEMPLATES.append(rxn)

    pool = mp.Pool(processes=64)

    t = time()
    rxns = pool.map(process.func, RXN_TEMPLATES)
    print('Time: ', time() - t, 's')

    r = ReactionSet(rxns)
    r.save('/pool001/whgao/data/synth_net/st_pis/reactions_' + NAME + '.json.gz')
