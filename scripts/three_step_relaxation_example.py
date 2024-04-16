import sys
import os
import numpy as np
from custodian.vasp.handlers import VaspErrorHandler

# Replace the path below with your own path to workflows.py
#sys.path.append('/storage/home/lam7027/work/bin/vasp-job-automation')
import workflows as workflows

# Specify custodian handlers
subset = list(VaspErrorHandler.error_msgs.keys())
subset.remove("algo_tet")
subset.remove("brmix")
subset.remove("zbrent")
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

vasp_cmd = ["srun", "vasp_std"]

settings_override_2relax = [
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
settings_override_3static=[
    {
        "dict": "INCAR",
        "action": {
            "_set": {"ALGO": "Normal", "IBRION": -1, "NSW": 0, "ISMEAR": 0, "SIGMA": 0.05}
        },
    },
    {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
]
        
workflows.three_step_relaxation(os.getcwd(), vasp_cmd, handlers,default_settings=True)

