import os
from custodian.vasp.handlers import VaspErrorHandler
import dfttk.workflows as workflows

# Specify custodian handlers
subset = list(VaspErrorHandler.error_msgs.keys())
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

# Specify VASP command
vasp_cmd = ["mpirun", "/opt/packages/VASP/VASP6/6.4.3/ONEAPI/vasp_std"]

workflows.encut_conv_test(os.getcwd(), vasp_cmd, handlers)
workflows.kpoints_conv_test(os.getcwd(), vasp_cmd, handlers)
