import sys
import os
import numpy as np
from custodian.vasp.handlers import VaspErrorHandler
#sys.path.append('/storage/home/lam7027/bin/vasp-job-automation')
import workflows


# Specify custodian handlers
subset = list(VaspErrorHandler.error_msgs.keys())
subset.remove("algo_tet")
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

# Specify VASP command
vasp_cmd = ["srun", "vasp_std"]

encut_list = [200,300,400,500,600]

workflows.encut_conv_test(os.getcwd(), encut_list, vasp_cmd, handlers)
workflows.calculate_encut_conv(os.getcwd(), encut_list)