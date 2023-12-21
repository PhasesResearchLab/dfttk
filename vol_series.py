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

volumes = list(np.linspace(370, 270, 15))

cstdn.vol_series(os.getcwd(), volumes, vasp_cmd, handlers, restarting=False)

