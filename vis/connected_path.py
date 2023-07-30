
from os import remove 
import pkg_resources
import pickle

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.io.cif import CifWriter

from ..utils import get_hash

import networkx as nx
from networkx import all_shortest_paths
from networkx import shortest_path_length


radii_file = pkg_resources.resource_filename('pkg.data','cordero_covalent_radii_modified.pickle')
with open(radii_file,'rb') as source:
    covalent_radii = pickle.load(source)
    
# all possible angles along the path
chains_and_angles = {
        'TiOTi': [102.67, 154.67],
        'OTiN'  : [77.33, 92.76, 102.67, 154.67, 180.0],
        'TiNTi': [102.67, 154.67],
        'NTiO'  : [77.33, 92.76, 102.67, 154.67, 180.0],
        'OTiO'  : [77.33, 92.76, 102.67, 154.67, 180.0],
        'NTiN'  : [77.33, 92.76, 102.67, 154.67, 180.0]
        }
# the translated inds
count,inds = 0,{}
for chain in chains_and_angles:
    inds[chain] = {}
    for angle in chains_and_angles[chain]:
        inds[chain][angle] = count
        count += 1
    
    
class structure_network():
    
    
    def __init__(self,stru_name):
        """
        This module can transform a crystal structure to a network for performing analysis.
        This module can currently only work with a structure with two dopant N atoms and one oxygen vacancy, 
        the oxygen vacancy must be represented by a C atom. 
        
        In __init__ method, we will first move the center of the supercell to its only oxygen vacancy. Then, 
        we will constructure a network for the structure based on Voronoi connectivity, 
        therefore only strong covalent bonds are considered.
        """
        # center the structure in the first place 
        stru = Structure.from_file(stru_name) if type(stru_name) is str else stru_name
        species_str = [str(item) for item in stru.species]
        c_inds =[ind for ind,item in enumerate(stru.species) if str(item)=='C']
        # Move the center of the structure to its oxygen vacancy
        frac_coords = stru.frac_coords
        displacement = np.array([.5,.5,.5]) - np.mean(frac_coords[c_inds],axis=0)
        frac_coords += displacement
        tmp_stru = Structure(species=stru.species,lattice=stru.lattice.matrix,coords=frac_coords,)
        writer_inside = CifWriter(tmp_stru)
        stru_hash = get_hash(tmp_stru)
        writer_inside.write_file('%s.cif' % stru_hash)
        
        # reload and process the new stru
        stru = Structure.from_file('%s.cif' % stru_hash)
        remove('%s.cif' % stru_hash)
        species_str = [str(item) for item in stru.species]
        c_inds = [ind for ind,item in enumerate(species_str) if item == 'C']
        assert len(c_inds) == 1
        # analyze the neighbors
        all_nbrs = stru.get_all_neighbors(5.,include_index=True,include_image=True)
        voronoi_analyzor = VoronoiConnectivity(stru)
        connected_neighbors = voronoi_analyzor.get_connections()
        connected_nbr = matcher(all_nbrs,connected_neighbors,species_str)
        connected_nbr_mayavi = \
        mayavi_matcher(all_nbrs,connected_neighbors,species_str,stru.cart_coords,stru.lattice.matrix)
        # make sure all Voronoi neighbors are found
        if connected_nbr is False: return 
        # find the three Ti surrounding the ONLY Vo
        three_ti = []
        for item in connected_nbr[c_inds[0]]:
            nc_dist,nc_ind,nc_offset = item[1:]
            if species_str[nc_ind] == 'Ti':three_ti.append(nc_ind)
        assert len(three_ti) == 3
        
        self.stru = stru
        self.species_str = species_str
        self.connected_nbr = connected_nbr
        self.connected_nbr_mayavi = connected_nbr_mayavi
        self.three_ti = three_ti
    
    
    def get_all_path_lengths(self):
        """
        Get the shortest path lengths between:
            1. Vo and the two N atoms 
            2. The two N atoms
        Since we need the shortest path between Vo and N, the C atom (Vo) is kept while building network.
        """
        connected_nbr = self.connected_nbr
        n_inds = [ind for ind,item in enumerate(self.species_str) if item == 'N' ]
        c_inds = [ind for ind,item in enumerate(self.species_str) if item == 'C']
        
        connections = []
        for ind,item in enumerate(connected_nbr): 
            one_node = []
            for inner in item:
                # inner ==> site,dist,ind,offset
                one_node.append([inner[2],inner[1],inner[3]])
            connections.append(one_node)
            
        # build crystal graph with networkx 
        crystal_graph = nx.Graph()
        for ind,item in enumerate(connections):
            for inner in item:
                crystal_graph.add_edge(ind,inner[0],length=inner[1])
                
        # get all path lengths
        all_paths = [[c_ind,n_ind] for c_ind in c_inds for n_ind in n_inds]
        all_paths.append(n_inds)
        assert len(all_paths) == 3
        
        all_path_lengths = []
        for ends in all_paths:
            end_left,end_right = ends
            path_length = shortest_path_length(crystal_graph,end_left,end_right)
            all_path_lengths.append(path_length)
        # get path lengths
        all_path_lengths = sorted(all_path_lengths[:len(all_path_lengths)-1],key=lambda x:x) + [all_path_lengths[-1]]
        
        return all_path_lengths
    
    
    def get_rep_vector(self):
        """
        The method to build the 72 dimensional vector.
        """
        stru = self.stru
        species_str = self.species_str
        connected_nbr = self.connected_nbr
        n_inds = [ind for ind,item in enumerate(self.species_str) if item == 'N' ]
        c_inds = [ind for ind,item in enumerate(self.species_str) if item == 'C']
        three_ti = self.three_ti
        
        connections = []
        for ind,item in enumerate(connected_nbr): 
            # remove graph node Vo
            one_node = []
            if ind not in c_inds:
                for inner in item:
                    # inner ==> site,dist,ind,offset
                    if inner[2] not in c_inds:
                        one_node.append([inner[2],inner[1],inner[3]])
            else:
                one_node.append([ind,0.,np.array([0.,0.,0.])])
            connections.append(one_node)
        # build crystal graph with networkx 
        crystal_graph = nx.Graph()
        for ind,item in enumerate(connections):
            for inner in item:
                crystal_graph.add_edge(ind,inner[0],length=inner[1])
        
        # 2 x 3Ti-N and 1 x N-N
        path_tasks = [[[ti_ind,n_ind] for ti_ind in three_ti] for n_ind in n_inds]
        path_tasks.append([n_inds])
        # enter the main loop
        path_stats,path_vec = [],np.zeros(count*len(path_tasks))
        for ends_list in path_tasks:
            # for each ends_list in path_tasks
            # for example, [Ti0-N0,Ti1-N0,Ti2-N0]
            lengths,stats = [],{}
            for ends in ends_list:
                # for example Ti0-N0
                end_left,end_right = ends
                path_all = list(all_shortest_paths(crystal_graph,end_left,end_right))
                path_length = shortest_path_length(crystal_graph,end_left,end_right)
                lengths.append(path_length)
                # processing the paths because of PBC
                for path in path_all:
                    # if it is not a path connects the two N atoms, its length must be even.
                    if end_left not in n_inds or end_right not in n_inds:assert len(path) % 2 == 0
                    
                    # get the species and the coords of the things along the path
                    one_cart = stru.cart_coords[path]
                    offsets = []
                    for ind_node,node in enumerate(path):
                        
                        if ind_node == 0: 
                            offsets.append([0.,0.,0.])
                        else:
                            prev_nbr = connections[path[ind_node-1]]
                            prev_offset = [inner[2] for inner in prev_nbr if inner[0]==node]
                            assert len(prev_offset) == 1
                            offsets.extend(prev_offset)
                    offsets = np.cumsum(offsets,axis=0) @ stru.lattice.matrix
                    one_cart += offsets
                    one_species = np.array([species_str[node] for node in path])
                    # make sure the path is pointing from Ti to N
                    assert one_species[-2] == 'Ti'
                    
                    # from Ti to N 
                    if end_left not in n_inds or end_right not in n_inds:
                        # get the angles along the path 
                        joints = np.arange(len(one_species))[one_species!='Ti'][:-1]
                        path_angles = [get_angle(
                                one_cart[inner-1]-one_cart[inner],
                                one_cart[inner+1]-one_cart[inner]
                                ) for inner in joints]
                        path_types = [''.join(['Ti',one_species[inner],'Ti']) for inner in joints]
                        if len(path) == 2: 
                            assert len(path_angles) == 0
                            assert len(path_types) == 0
                            
                        if len(one_species) >= 4:
                            path_angles.append(get_angle(one_cart[-3]-one_cart[-2],one_cart[-1]-one_cart[-2]))
                            path_types.append('%sTi%s' % (one_species[-3],one_species[-1]))
                            
                    # from N to N 
                    if end_left in n_inds and end_right in n_inds:
                        # get the angles along the path 
                        joints = np.arange(len(one_species))[one_species=='Ti']
                        path_angles = [get_angle(
                                one_cart[inner-1]-one_cart[inner],
                                one_cart[inner+1]-one_cart[inner]
                                ) for inner in joints]
                        path_types = [''.join([one_species[inner-1],'Ti',one_species[inner+1]]) for inner in joints]
                    
                    for inner_t,inner_a in zip(path_types,path_angles):
                        try:
                            stats[inner_t].append([inner_a,path_length])
                        except KeyError:
                            stats[inner_t] = []
                            stats[inner_t].append([inner_a,path_length])
            
            # summarize from one atom to another 
            # on the same level as "for ends in ends_list:"
            # it is per task type, i.e., 3Ti-N and N-N
            path_stats.append([np.mean(lengths),np.min(lengths),stats])
        
        # the shorter path written to the vector in the first place
        assert len(path_stats) == len(path_tasks)
        path_stats = sorted(path_stats[:len(path_tasks)-1],key=lambda x:x[0]) + [path_stats[-1]]
        tasks_lengths = [item[1] for item in path_stats]
        
        # build path vectors 
        for ind,item in enumerate(path_stats):
    
            one_stats = item[2]
            for chain in one_stats:
                for info in one_stats[chain]:
                    angle,length = info
                    vec_ind = inds[chain][angle] + ind*count
                    path_vec[vec_ind] += 1
        
        return path_stats,tasks_lengths,path_vec
    
    
    def get_path_details_for_mayavi(self):
        """Get the information of the paths for visualizing with mayavi."""
        stru = self.stru
        species_str = self.species_str
        connected_nbr = self.connected_nbr
        n_inds = [ind for ind,item in enumerate(self.species_str) if item == 'N' ]
        c_inds = [ind for ind,item in enumerate(self.species_str) if item == 'C']
        three_ti = self.three_ti

        connections = []
        for ind,item in enumerate(connected_nbr): 
            # remove graph node Vo
            one_node = []
            if ind not in c_inds:
                for inner in item:
                    # inner ==> site,dist,ind,offset
                    if inner[2] not in c_inds:
                        one_node.append([inner[2],inner[1],inner[3]])
            else:
                one_node.append([ind,0.,np.array([0.,0.,0.])])
            connections.append(one_node)
        # build crystal graph with networkx 
        crystal_graph = nx.Graph()
        for ind,item in enumerate(connections):
            for inner in item:
                crystal_graph.add_edge(ind,inner[0],length=inner[1])
 
        # 2 x 3TI-N and 1 x N-N
        ti_pos = stru.cart_coords[three_ti]
        n_pos = stru.cart_coords[n_inds]
        path_tasks = [[[ti_ind,n_ind] for ti_ind in three_ti] for n_ind in n_inds]
        path_tasks.append([n_inds])
        path_details = {0:[],1:[],2:[]}
        
        unique_inds_along_the_path = []
        all_species,all_coords = [],[]
        # the path_tasks 
        for ends_ind,ends_list in enumerate(path_tasks):
            
            one_detail = []
            for ends in ends_list:
                end_left,end_right = ends
                path_all = list(all_shortest_paths(crystal_graph,end_left,end_right))

                # processing the paths because of PBC
                equ_paths = []
                for path in path_all:
                    
                    if end_left not in n_inds or end_right not in n_inds:assert len(path) % 2 == 0
                    
                    # get the species and the coords of the things along the path
                    one_cart = stru.cart_coords[path]
                    offsets = []
                    for ind_node,node in enumerate(path):
                        
                        if ind_node == 0: 
                            offsets.append([0.,0.,0.])
                        else:
                            prev_nbr = connections[path[ind_node-1]]
                            prev_offset = [inner[2] for inner in prev_nbr if inner[0]==node]
                            assert len(prev_offset) == 1
                            offsets.extend(prev_offset)
                    offsets = np.cumsum(offsets,axis=0) @ stru.lattice.matrix
                    one_cart += offsets
                    one_species = np.array([species_str[node] for node in path])
                    # make sure the path is pointing from Ti to N
                    assert one_species[-2] == 'Ti'
                    unique_inds_along_the_path.extend(path)
                    equ_paths.append([path,one_species,one_cart])
                    all_species.extend(one_species)
                    all_coords.extend(one_cart)
                    
                one_detail.append(equ_paths)
                    
            path_details[ends_ind].extend(one_detail)
            
        return crystal_graph,path_details,[ti_pos,n_pos],set(unique_inds_along_the_path)
    
    
    def check_double_hole(self):
        """Find the double_hole strus"""
        stru = self.stru
        species_str = self.species_str
        connected_nbr = self.connected_nbr
        ti_inds = [ind for ind,item in enumerate(self.species_str) if item == 'Ti']
        
        for ti in ti_inds:
            
            ti_nbr = connected_nbr[ti]
            ti_pos = stru.cart_coords[ti]
            nbr_species = np.array([species_str[atom[2]] for atom in ti_nbr])
            if len(np.where(nbr_species == 'N')[0]) == 2:
                ti_nbr_n = np.where(nbr_species == 'N')[0]
                pos_one = stru.cart_coords[ti_nbr[ti_nbr_n[0]][2]] + ti_nbr[ti_nbr_n[0]][3] @ stru.lattice.matrix - ti_pos
                pos_two = stru.cart_coords[ti_nbr[ti_nbr_n[1]][2]] + ti_nbr[ti_nbr_n[1]][3] @ stru.lattice.matrix - ti_pos
                ntin_angle = get_angle(pos_one,pos_two)
                
                if np.abs(ntin_angle - 92.75) < .2: return True
        
        return False
    

    def check_jpcc(self):
        """Find the dimerized N-N spatial orderings"""
        stru = self.stru
        cart_coords = stru.cart_coords
        
        connected_nbr = self.connected_nbr
        ti_inds = [ind for ind,item in enumerate(self.species_str) if item == 'Ti']
        o_inds = [ind for ind,item in enumerate(self.species_str) if item == 'O']
        n_inds = [ind for ind,item in enumerate(self.species_str) if item == 'N' ]
        
        # Find only the two relavent Ti for each N atom  
        n_nbr_ti = [[atom[2] for atom in connected_nbr[n_ind] \
                     if atom[2] in ti_inds and np.abs(atom[1] - 1.94384) < 1e-2] for n_ind in n_inds]
        print(n_nbr_ti)
        nbr_o_of_nbr_ti = [[[atom[2] for atom in connected_nbr[ti_ind] if atom[2] in o_inds] \
                            for ti_ind in nbr_ti] for nbr_ti in n_nbr_ti]
        # nbr_o_of_nbr_ti is (2 (the id of N atoms),2 (two neighboring Ti),6 (the six neighboring O atoms))
        intersect_00 = len(np.intersect1d(nbr_o_of_nbr_ti[0][0],nbr_o_of_nbr_ti[1][0])) == 1 
        intersect_01 = len(np.intersect1d(nbr_o_of_nbr_ti[0][0],nbr_o_of_nbr_ti[1][1])) == 1 
        intersect_10 = len(np.intersect1d(nbr_o_of_nbr_ti[0][1],nbr_o_of_nbr_ti[1][0])) == 1 
        intersect_11 = len(np.intersect1d(nbr_o_of_nbr_ti[0][1],nbr_o_of_nbr_ti[1][1])) == 1 
        overlap_o_atoms = np.sum([intersect_00,intersect_01,intersect_10,intersect_11])
        
        if (overlap_o_atoms == 2) and \
            (np.abs(np.linalg.norm(cart_coords[n_inds[0]] - cart_coords[n_inds[1]]) - 3.79725) < 1e-2):
                return True

        return False
    

    def check_dh_up_down(self):
        """Find the double_hole strus"""
        stru = self.stru
        species_str = self.species_str
        connected_nbr = self.connected_nbr
        ti_inds = [ind for ind,item in enumerate(self.species_str) if item == 'Ti']
        
        for ti in ti_inds:
            
            ti_nbr = connected_nbr[ti]
            ti_pos = stru.cart_coords[ti]
            nbr_species = np.array([species_str[atom[2]] for atom in ti_nbr])
            if len(np.where(nbr_species == 'N')[0]) == 2:
                ti_nbr_n = np.where(nbr_species == 'N')[0]
                pos_one = stru.cart_coords[ti_nbr[ti_nbr_n[0]][2]] + ti_nbr[ti_nbr_n[0]][3] @ stru.lattice.matrix - ti_pos
                pos_two = stru.cart_coords[ti_nbr[ti_nbr_n[1]][2]] + ti_nbr[ti_nbr_n[1]][3] @ stru.lattice.matrix - ti_pos
                ntin_angle = get_angle(pos_one,pos_two)
                
                if np.abs(ntin_angle - 77.33) < .2: return True
        
        return False
    
    
    def check_nn_chain(self):
        """Find the N-Ti-N chain"""
        stru = self.stru
        species_str = self.species_str
        connected_nbr = self.connected_nbr
        ti_inds = [ind for ind,item in enumerate(self.species_str) if item == 'Ti']
        
        for ti in ti_inds:
            
            ti_nbr = connected_nbr[ti]
            ti_pos = stru.cart_coords[ti]
            nbr_species = np.array([species_str[atom[2]] for atom in ti_nbr])
            if len(np.where(nbr_species == 'N')[0]) == 2:
                ti_nbr_n = np.where(nbr_species == 'N')[0]
                pos_one = stru.cart_coords[ti_nbr[ti_nbr_n[0]][2]] + ti_nbr[ti_nbr_n[0]][3] @ stru.lattice.matrix - ti_pos
                pos_two = stru.cart_coords[ti_nbr[ti_nbr_n[1]][2]] + ti_nbr[ti_nbr_n[1]][3] @ stru.lattice.matrix - ti_pos
                ntin_angle = get_angle(pos_one,pos_two)
                
                if np.abs(ntin_angle - 154.66) < .2: return True
        
        return False


    def check_side_and_up_nn_chain(self):
        """Find the N-Ti-O-N-O-Ti-N chain."""
        stru = self.stru
        cart_coords = stru.cart_coords
        lattice = stru.lattice.matrix
        
        connected_nbr = self.connected_nbr
        ti_inds = [ind for ind,item in enumerate(self.species_str) if item == 'Ti']
        o_inds = [ind for ind,item in enumerate(self.species_str) if item == 'O']
        n_inds = [ind for ind,item in enumerate(self.species_str) if item == 'N' ]
        
        # find the two sets of Ti atoms that connects to the two N atoms 
        # the two sets of Ti atoms may overlap 
        n_nbr_ti = [[atom[2] for atom in connected_nbr[n_ind] if atom[2] in ti_inds] for n_ind in n_inds]
        # the idea is to search from the o atom that connects the two Ti atoms
        # all the positions should be calculated w.r.t this o atom 
        for o_count,o in enumerate(o_inds):
            
            o_nbr = connected_nbr[o]
            o_pos = stru.cart_coords[o]
            # find the INDICES of the three neighboring ti atoms
            # if the o atom is the one that we want to find, then
            # two of the ti atoms will connect to the same n atom, while the last ti will connect to the other n atom
            aux_inds = [atom[2] for atom in o_nbr if atom[2] in ti_inds]
            assert len(aux_inds) == 3
            nbr_ti_inds = [[atom for atom in aux_inds if atom in part] for part in n_nbr_ti]
            flattened_nbr_ti_inds = [item for sublist in nbr_ti_inds for item in sublist]
            
            if (len(nbr_ti_inds[0]) + len(nbr_ti_inds[1]) == 3) and len(set(flattened_nbr_ti_inds)) == 3:
                
                nbr_ti_inds = sorted(nbr_ti_inds,key=lambda x:len(x))
                if not tuple((len(nbr_ti_inds[0]),len(nbr_ti_inds[1]))) == (1,2): continue
            
                # get the positions of the three ti atoms and the two n atoms w.r.t the center o atom 
                ti00_ind = nbr_ti_inds[0][0]
                ti10_ind,ti11_ind = nbr_ti_inds[1]
                ti00_offset = np.array([atom[3] for atom in o_nbr if atom[2] == ti00_ind][0])
                ti10_offset = np.array([atom[3] for atom in o_nbr if atom[2] == ti10_ind][0])
                ti11_offset = np.array([atom[3] for atom in o_nbr if atom[2] == ti11_ind][0])
                ti00_pos = cart_coords[ti00_ind] + ti00_offset @ lattice
                ti10_pos = cart_coords[ti10_ind] + ti10_offset @ lattice
                ti11_pos = cart_coords[ti11_ind] + ti11_offset @ lattice
                
                n0_ind,n1_ind = n_inds
                # aux Ti indices for finding the offset of N relative to Ti
                aux_ti_inds = [[atom for atom in aux_inds if atom in part][0] for part in n_nbr_ti]
                n0_offset = np.array([atom[3] for atom in connected_nbr[aux_ti_inds[0]] if atom[2] == n0_ind][0])
                n1_offset = np.array([atom[3] for atom in connected_nbr[aux_ti_inds[1]] if atom[2] == n1_ind][0])
                n0_pos = cart_coords[n0_ind] + (ti00_offset+n0_offset) @ lattice
                n1_pos = cart_coords[n1_ind] + (ti10_offset+n1_offset) @ lattice
                
                # 1. check if the three ti atoms and the two n atoms are in the same plane 
                along_x = np.array([ti00_pos[0],ti10_pos[0],ti11_pos[0],n0_pos[0],n1_pos[0]]) 
                along_y = np.array([ti00_pos[1],ti10_pos[1],ti11_pos[1],n0_pos[1],n1_pos[1]]) 
                all_x = np.all(np.abs(along_x - along_x[0]) < 1e-4)
                all_y = np.all(np.abs(along_y - along_y[0]) < 1e-4)
                
                if not (all_x | all_y): continue 
                
                # calculate the angles 
                # 1. ti10-n1-ti11
                ti10_n1_ti11 = np.abs(get_angle(ti10_pos - n1_pos,ti11_pos - n1_pos) - 102.66) < 1e-1
                # 2. n0-ti00-o
                n0_ti00_o = np.abs(get_angle(ti00_pos - n0_pos,ti00_pos - o_pos) - 154.66) < 1e-1
    
                # both conditions 
                if ti10_n1_ti11 & n0_ti00_o:return True
                        
        return False
        


def get_angle(e1,e2):
    """Get the angle between two vectors."""
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    rad = np.arccos(np.round(e1@e2,decimals=6))
    
    return np.round(rad/np.pi*180,decimals=2)


def matcher(all_nbrs,connected_neighbors,species_str):
    """Convert Voronoi Neighbors to the workable format."""
    connected_nbr = [[] for _ in range(len(all_nbrs))]
    for connection in connected_neighbors:
        ind_c,ind_n,dist_n = connection
        
        center_species = species_str[ind_c]
        nbr_slice = all_nbrs[ind_c]
        
        # get which neighbor of ind_c the connection is referring to
        indicator = 0
        for nbr_c in nbr_slice:
            nbr_dist,nbr_ind,nbr_offset = nbr_c[1:]
            neighbor_species = species_str[nbr_ind]
            if np.abs(nbr_dist-dist_n) < 1e-6 and nbr_ind == ind_n:
                indicator = 1
                if nbr_dist <= covalent_radii[center_species]+covalent_radii[neighbor_species]:
                    connected_nbr[ind_c].append(nbr_c)
        try:
            assert indicator == 1
        except AssertionError:
            print('cutoff too small!')
            return False
            
    return connected_nbr


# connected neighbor builder for mayavi visualization
def mayavi_matcher(all_nbrs,connected_neighbors,species_str,cart_coords,lattice):
        
    connected_nbr = [[] for _ in range(len(all_nbrs))]
    for connection in connected_neighbors:
        ind_c,ind_n,dist_n = connection
        
        center_species = species_str[ind_c]
        nbr_slice = all_nbrs[ind_c]

        # get which neighbor of ind_c the connection is referring to
        indicator = 0
        for nbr_c in nbr_slice:
            nbr_dist,nbr_ind,nbr_offset = nbr_c[1:]
            neighbor_species = species_str[nbr_ind]
            if np.abs(nbr_dist-dist_n) < 1e-6 and nbr_ind == ind_n:
                indicator = 1
                if nbr_dist <= covalent_radii[center_species]+covalent_radii[neighbor_species]:
                    if tuple(sorted([center_species,neighbor_species])) != ('Ti','Ti'):  
                        connected_nbr[ind_c].append([cart_coords[nbr_ind]+nbr_offset@lattice,species_str[nbr_ind],nbr_offset])
        try:
            assert indicator == 1
        except AssertionError:
            print('cutoff too small!')
            return False
            
    return connected_nbr

