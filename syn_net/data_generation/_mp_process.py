"""
This file contains a function for search available building blocks
for a matching reaction template. Prepared for multiprocessing.
"""
import pandas as pd

PATH_TO_BUILDING_BLOCKS = '/home/whgao/scGen/synth_net/data/enamine_us.csv.gz'
building_blocks = pd.read_csv(
    PATH_TO_BUILDING_BLOCKS,
    compression='gzip')['SMILES'].tolist()
print('Finish reading the building blocks list!')


def func(rxn_):
    """
    This function is used to search available building blocks for a matching reaction template.
    Args:
        rxn_: a reaction template
    Returns:
        rxn_: set with a list of available building blocks
    """
    rxn_.set_available_reactants(building_blocks)
    return rxn_
