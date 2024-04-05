import sys
import os
import numpy as np
from custodian.vasp.handlers import VaspErrorHandler

# Replace the path below with your own path to workflows.py
sys.path.append('/storage/home/lam7027/work/bin/vasp-job-automation')
import src.workflows as workflows

# Specify custodian handlers
subset = list(VaspErrorHandler.error_msgs.keys())
subset.remove("algo_tet")
subset.remove("brmix")
subset.remove("zbrent")
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

# Specify VASP command
vasp_cmd = ["srun", "vasp_std"]

volumes = list(np.linspace(370, 270, 15))

workflows.ev_curve_series(os.getcwd(), volumes, vasp_cmd, handlers, restarting=False)

