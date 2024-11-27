import sys
import os
import numpy as np
from custodian.vasp.handlers import VaspErrorHandler
import dfttk.workflows as workflows

# Specify custodian handlers
subset = list(VaspErrorHandler.error_msgs.keys())
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

# Specify VASP command
vasp_cmd = ["mpirun", "/opt/packages/VASP/VASP6/6.4.3/ONEAPI/vasp_std"]

phonon_volumes = [74, 72, 70, 68, 66, 64, 62, 60]
scaling_matrix = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
kppa = 4001

workflows.phonons_parallel(os.getcwd(), phonon_volumes, kppa, "run_bridges_phonons", scaling_matrix = scaling_matrix)
