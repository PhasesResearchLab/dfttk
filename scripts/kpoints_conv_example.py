import sys
import os
import numpy as np
from custodian.vasp.handlers import VaspErrorHandler
#sys.path.append('/storage/home/lam7027/bin/vasp-job-automation')
import src.workflows as workflows


# Specify custodian handlers
subset = list(VaspErrorHandler.error_msgs.keys())
subset.remove("algo_tet")
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

# Specify VASP command
vasp_cmd = ["srun", "vasp_std"]

kppa_list = [400, 800, 1200, 1600, 2000]

workflows.kpoints_conv_test(os.getcwd(), kppa_list, vasp_cmd, handlers)
workflows.calculate_kpoint_conv(os.getcwd(), kppa_list)