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

workflows.ev_curve_series(os.getcwd(), volumes, vasp_cmd, handlers, restarting=False, default_settings=True)
workflows.custodian_errors_location(os.getcwd())
workflows.NELM_reached(os.getcwd())
