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

volumes = [74, 72, 70, 68, 66, 64, 62, 60]
kppa = 4001

workflows.elec_dos_parallel(os.getcwd(), volumes, kppa, "run_bridges_elec")
