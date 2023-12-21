import sys
import os
import numpy as np
from custodian.vasp.handlers import VaspErrorHandler
sys.path.append('/storage/home/lam7027/bin/vasp-job-automation')
import cstdn


# Specify custodian handlers
subset = list(VaspErrorHandler.error_msgs.keys())
subset.remove("algo_tet")
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

# Specify VASP command
vasp_cmd = ["srun", "vasp_std"]

kpoints_list = ['4 4 5', '5 5 6']

cstdn.kpoints_conv_test(os.getcwd(), kpoints_list, vasp_cmd, handlers, backup=False)
