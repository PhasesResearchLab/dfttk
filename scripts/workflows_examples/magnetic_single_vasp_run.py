import os
import shutil
import sys
sys.path.append('/storage/home/lam7027/work/bin/vasp-job-automation')
import dfttk.magnetism as magnetism
from pymatgen.core.structure import Structure
from dfttk.workflows import run_configurations

path = '/storage/home/lam7027/scratch/Fe'
incar = '/storage/home/lam7027/scratch/Fe/INCAR'
potcar = '/storage/home/lam7027/scratch/Fe/POTCAR'
yw_output = '/storage/home/lam7027/scratch/Fe/atat_stuff/YWoutput'
magmoms = {
    'Fe+': '5',
    'Fe-': '-5',
}
dummy_species_pairs = [('Fe+', 'Fe-')]
submit_script='run_vasp'

magnetism.generate_magnetic_configs(
    path=path,
    incar=incar,
    potcar=potcar,
    yw_output=yw_output,
    magmoms=magmoms,
    dummy_species_pairs=dummy_species_pairs,
    submit_script='run_vasp'
)

magnetism.scale_poscars(12, os.path.join(path, 'configurations'))

# Call the function
run_configurations(['sbatch', 'run_vasp'], configurations_dir='configurations')