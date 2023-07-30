
import re
import hashlib
import numpy as np  

from pymatgen.electronic_structure.core import Spin  


def get_hash(stru):

    string = ''
    for number in stru.lattice.matrix.flatten():
        string += '%.6f' % number
    for number in stru.atomic_numbers:
        string += '%3d' % number
    for number in stru.frac_coords.flatten():
        string += '%.6f' % number

    md5 = hashlib.md5(string.encode('utf-8'))
    hash = md5.hexdigest()

    return hash


def task_generator(num_tasks,num_workers,offset=None):
    
    N, n = num_tasks,num_workers
    work_sizes = [N // n if N % n <= _ else N // n + 1 for _ in range(n)]
    size_cumsum = np.cumsum(work_sizes)
    
    parts = []
    for ind,item in enumerate(work_sizes):
        left_bash = size_cumsum[ind-1] if ind != 0 else 0
        right_bash = size_cumsum[ind-1] + item -1 if ind != 0 else item - 1

        if offset is not None:
            parts.append([left_bash+offset,right_bash+offset])
        else:
            parts.append([left_bash,right_bash])

    return parts


def th_sub(num_tasks,num_workers,num_nodes):
    
    parts = task_generator(num_tasks,num_workers)
    for ind in range(num_workers):
        with open('sub-%i.sh' % ind, 'w') as source:
            
            source.write('#!/bin/bash\n')
            source.write('#SBATCH --job-name=calc\n')
            source.write('#SBATCH --partition=TH_HPC2\n')
            source.write('#SBATCH --nodes=%i\n\n' % num_nodes)
                         
            source.write('cd ..\n')
            source.write('for ind in {%i..%i};do\n' % (parts[ind][0],parts[ind][1]))
            source.write('      cd $ind/\n')
            source.write('      yhrun -N %i -n %i -p TH_HPC2 /THL7/home/lcyin/wwqian/vasp/vasp.5.4.4/bin/vasp_std\n\n' % (num_nodes,28*num_nodes))
            
            source.write('      cp CONTCAR scf/POSCAR\n')
            source.write('      cp POTCAR scf\n')
            source.write('      cd  scf\n')
            source.write('      yhrun -N %i -n %i -p TH_HPC2 /THL7/home/lcyin/wwqian/vasp/vasp.5.4.4/bin/vasp_std\n\n' % (num_nodes,28*num_nodes))
            
            source.write('      cp CHGCAR POSCAR POTCAR dos\n')
            source.write('      cd  dos\n')
            source.write('      yhrun -N %i -n %i -p TH_HPC2 /THL7/home/lcyin/wwqian/vasp/vasp.5.4.4/bin/vasp_std\n\n' % (num_nodes,28*num_nodes))
            
            source.write('      cd ../\n')
            source.write('      cp CHGCAR POSCAR POTCAR band\n')
            source.write('      cd  band\n')
            source.write('      yhrun -N %i -n %i -p TH_HPC2 /THL7/home/lcyin/wwqian/vasp/vasp.5.4.4/bin/vasp_std\n\n' % (num_nodes,28*num_nodes))
            
            source.write('      cd ../../../\n')
            
            source.write('done')
            
            
def th_sub_regain(num_tasks,num_workers,num_nodes):
    
    parts = task_generator(num_tasks,num_workers)
    for ind in range(num_workers):
        with open('sub-%i.sh' % ind, 'w') as source:
            
            source.write('#!/bin/bash\n')
            source.write('#SBATCH --job-name=calc\n')
            source.write('#SBATCH --partition=TH_HPC2\n')
            source.write('#SBATCH --nodes=%i\n\n' % num_nodes)
                         
            source.write('cd ..\n')
            source.write('for ind in {%i..%i};do\n\n' % (parts[ind][0],parts[ind][1]))
            source.write('      cp $ind/scf/CHGCAR $ind/scf/band\n')
            source.write('      cd $ind/scf/band\n')
            source.write('      yhrun -N %i -n %i -p TH_HPC2 /THL7/home/lcyin/wwqian/vasp/vasp.5.4.4/bin/vasp_std\n\n' % (num_nodes,28*num_nodes))
            
            source.write('      cd ../../../\n')
            
            source.write('done')
            
            
def local_sub(num_tasks,num_workers,num_nodes,offset=None):
    
    parts = task_generator(num_tasks,num_workers,offset)
    for ind in range(num_workers):
        with open('sub-%i.sh' % ind, 'w') as source:
            
            source.write('#!/bin/bash\n')
            source.write('#SBATCH --job-name=calc\n')
            source.write('#SBATCH --partition=blade11\n')
            source.write('#SBATCH --cpus-per-task=1\n')
            source.write('#SBATCH --ntasks-per-node=40\n')
            source.write('#SBATCH --nodes=%i\n\n' % num_nodes)
                         
            source.write('cd ..\n')
            source.write('for ind in {%i..%i};do\n\n' % (parts[ind][0],parts[ind][1]))
            source.write('      cd $ind/\n')
            source.write('      mpirun -n %i vasp.5.4.4-optcell\n' % 40*num_nodes)
            
            source.write('      cp CONTCAR scf/POSCAR\n')
            source.write('      cp POTCAR scf\n')
            source.write('      cd  scf\n')
            source.write('      mpirun -n %i vasp.5.4.4-optcell\n' % 40*num_nodes)
            
            source.write('      cp CHGCAR POSCAR POTCAR dos\n')
            source.write('      cd  dos\n')
            source.write('      mpirun -n %i vasp.5.4.4-optcell\n' % 40*num_nodes)
            
            source.write('      cd ../\n')
            source.write('      cp CHGCAR POSCAR POTCAR band\n')
            source.write('      cd  band\n')
            source.write('      mpirun -n %i vasp.5.4.4-optcell\n' % 40*num_nodes)
            
            source.write('      cd ../../../\n')
            
            source.write('done')
            
            
def local_sub_bader(num_tasks,num_workers,num_nodes):
    
    parts = task_generator(num_tasks,num_workers)
    for ind in range(num_workers):
        with open('sub-%i.sh' % ind, 'w') as source:
            
            source.write('#!/bin/bash\n')
            source.write('#SBATCH --job-name=calc\n')
            source.write('#SBATCH --partition=blade11\n')
            source.write('#SBATCH --cpus-per-task=1\n')
            source.write('#SBATCH --ntasks-per-node=40\n')
            source.write('#SBATCH --nodes=%i\n\n' % num_nodes)
                         
            source.write('cd ../strus\n')
            source.write('for ind in {%i..%i};do\n\n' % (parts[ind][0],parts[ind][1]))
            source.write('      cd $ind/\n')
            source.write('      mpirun -n %i vasp.5.4.4-optcell-bader\n' % 40*num_nodes)
            source.write('      cd ../\n')
            
            source.write('done')
            
            
def Eigenval_reader(filename):
    """read the EIGENVAL file"""
    
    with open(filename, 'r') as f:

        ispin = int(f.readline().split()[-1])
        for _ in range(4):
            f.readline()

        nelect,nkpt,nbands = list(map(int,f.readline().split()))
        kpoints,kpoints_weights = [],[]
        if ispin == 2:eigenvalues = {Spin.up:np.zeros((nkpt,nbands,2)),Spin.down:np.zeros((nkpt,nbands,2))}
        else:eigenvalues = {Spin.up:np.zeros((nkpt,nbands,2))}

        ikpt = -1
        for ind_l,line in enumerate(f):
            # the lines that matches the format of the coordinates of k points 
            if re.search(r'(\s+[\-+0-9eE.]+){4}', str(line)):
                ikpt += 1
                kpt = list(map(float,line.split()))
                kpoints.append(kpt[:-1])
                kpoints_weights.append(kpt[-1])

                for i in range(nbands):
                    sl = list(map(float,f.readline().split()))
                    if len(sl) == 3:
                        eigenvalues[Spin.up][ikpt, i, 0] = sl[1]
                        eigenvalues[Spin.up][ikpt, i, 1] = sl[2]
                    elif len(sl) == 5:
                        eigenvalues[Spin.up][ikpt, i, 0] = sl[1]
                        eigenvalues[Spin.up][ikpt, i, 1] = sl[3]
                        eigenvalues[Spin.down][ikpt, i, 0] = sl[2]
                        eigenvalues[Spin.down][ikpt, i, 1] = sl[4]

    energies = eigenvalues[Spin.up][...,0]
    occu = eigenvalues[Spin.up][...,1]

    mask_one = np.abs(occu-1)<0.1
    occu[mask_one] = 1
    mask_zero = np.abs(occu)<0.1
    occu[mask_zero] = 0

    two_ms = []
    for ikpt in occu:

        mask = np.arange(len(ikpt))
        cbm = mask[ikpt==0.][0]
        vbm = mask[ikpt==1.][-1]
        two_ms.append([vbm,cbm])
    two_ms = np.array(two_ms)

    two_vals = []
    for ind,item in enumerate(two_ms):
        two_vals.append([energies[ind][item[0]],energies[ind][item[1]]])
        
    two_vals = np.array(two_vals)
    
    return two_vals,kpoints
    
