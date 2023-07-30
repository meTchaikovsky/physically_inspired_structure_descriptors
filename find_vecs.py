
from sys import stdout
from os import listdir
from multiprocessing import Pool,sharedctypes

import numpy as np

from pymatgen.core.structure import Structure
from pkg.vis.connected_path import structure_network


def worker(item):

    print(item.split('.')[0])
    stdout.flush()

    stru = Structure.from_file('%s/%s' % ('../source_cifs',item))
    ind = int(item.split('.')[0])
    
    graph_inst = structure_network(stru_name=stru)
    _,_,vec =  graph_inst.get_rep_vector()
    lengths = graph_inst.get_all_path_lengths()
    is_nn = graph_inst.check_nn_chain()
    is_ntiotin = graph_inst.check_side_and_up_nn_chain()
    is_dn = graph_inst.check_dh_up_down()
    is_jpcc = graph_inst.check_jpcc()

    np.ctypeslib.as_array(shared_vecs)[ind] = vec
    np.ctypeslib.as_array(shared_lengths)[ind] = lengths
    np.ctypeslib.as_array(shared_nn_chains)[ind] = is_nn
    np.ctypeslib.as_array(shared_ntiotin_chains)[ind] = is_ntiotin
    np.ctypeslib.as_array(shared_dn_chains)[ind] = is_dn
    np.ctypeslib.as_array(shared_jpcc_chains)[ind] = is_jpcc


all_strus = listdir('../source_cifs')
# the vec representations 
pooled_vecs = np.ctypeslib.as_ctypes(np.zeros((len(all_strus),72)))
shared_vecs= sharedctypes.RawArray(pooled_vecs._type_, pooled_vecs)

# the path lengths 
pooled_lengths = np.ctypeslib.as_ctypes(np.zeros((len(all_strus),3)))
shared_lengths= sharedctypes.RawArray(pooled_lengths._type_, pooled_lengths)

# the nn chain 
nn_chains = np.ctypeslib.as_ctypes(np.zeros(len(all_strus)))
shared_nn_chains = sharedctypes.RawArray(nn_chains._type_, nn_chains)
# the ntiotin chain 
ntiotin_chains = np.ctypeslib.as_ctypes(np.zeros(len(all_strus)))
shared_ntiotin_chains = sharedctypes.RawArray(ntiotin_chains._type_, ntiotin_chains)
# the double hole up and down
dn_chains = np.ctypeslib.as_ctypes(np.zeros(len(all_strus)))
shared_dn_chains = sharedctypes.RawArray(dn_chains._type_, dn_chains)
# the jpcc
jpcc_chains = np.ctypeslib.as_ctypes(np.zeros(len(all_strus)))
shared_jpcc_chains = sharedctypes.RawArray(jpcc_chains._type_, jpcc_chains)

pool = Pool(8)
pool.map(worker,all_strus)
pooled_vecs = np.ctypeslib.as_array(shared_vecs)
pooled_lengths = np.ctypeslib.as_array(shared_lengths)
pooled_nn_chains = np.ctypeslib.as_array(shared_nn_chains)
pooled_ntiotin_chains = np.ctypeslib.as_array(shared_ntiotin_chains)
pooled_dn_chains = np.ctypeslib.as_array(shared_dn_chains)
pooled_jpcc_chains = np.ctypeslib.as_array(shared_jpcc_chains)

np.save('all_vecs',pooled_vecs)
np.save('all_lengths',pooled_lengths)
np.save('ntin_chains',pooled_nn_chains)
np.save('ntiotin_chains',pooled_ntiotin_chains)
np.save('dn_chains',pooled_dn_chains)
np.save('jpcc_chains',pooled_jpcc_chains)
