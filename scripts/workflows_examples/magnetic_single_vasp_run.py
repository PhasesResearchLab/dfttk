import os
import shutil
import sys
sys.path.append('/storage/home/lam7027/work/bin/vasp-job-automation')
import dfttk.magnetism as magnetism
from pymatgen.core.structure import Structure
from dfttk.workflows import run_configurations

path = '/storage/home/lam7027/scratch/Fe'
incar_name = 'INCAR'
poscar_name = 'POSCAR'
potcar_name = 'POTCAR'
magmoms = {
    'Fe+': '5',
    'Fe-': '-5',
}
dummy_species_pairs = [('Fe+', 'Fe-')]
replace_atoms = {'Fe': 'Fe+,Fe-'}
submit_script='run_vasp'
newgenstrYW_args= [
    "newgenstrYW",
    "-sig",
    "16",
    "-l",
    "lat.in"
]

magnetism.generate_magnetic_configs(
    path=path,
    incar_name=incar_name,
    poscar_name=poscar_name,
    potcar_name=potcar_name,
    magmoms=magmoms,
    dummy_species_pairs=dummy_species_pairs,
    replace_atoms=replace_atoms,
    submit_script='run_vasp',
    newgenstrYW_args=newgenstrYW_args
)

magnetism.scale_poscars(12, os.path.join(path, 'configurations'))

# Call the function
run_configurations(['sbatch', 'run_vasp'], configurations_dir='configurations')