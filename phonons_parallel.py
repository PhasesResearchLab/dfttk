import sys
import os
import numpy as np
from custodian.vasp.handlers import VaspErrorHandler
import dfttk.workflows as workflows

# Specify custodian handlers
subset = list(VaspErrorHandler.error_msgs.keys())
subset.remove("algo_tet")
subset.remove("brmix")
subset.remove("zbrent")
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

# Input parameters
path = os.getcwd()
phonon_volumes = [1000, 1010, 1030, 1040, 1050, 1060]
supercell_size = [1, 1, 1]
kppa = 2000
vasp_cmd = ["srun", "vasp_std"]

workflows.phonons_parallel(path, phonon_volumes, supercell_size, kppa, "run_phonons_RC")
# This function should take care of everything. Just work on this function for now. 
