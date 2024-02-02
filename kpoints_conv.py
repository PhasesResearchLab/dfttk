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

kppa_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

cstdn.kpoints_conv_test(os.getcwd(), kppa_list, vasp_cmd, handlers, backup=False)
