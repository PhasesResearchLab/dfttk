import os
import json
import subprocess
import importlib.resources
import dfttk.vasp_input as vasp_input
from dfttk.aggregate_extraction import calculate_encut_conv, calculate_kpoint_conv


class Configuration:
    def __init__(self, config_path):
        self.config_path = config_path
        self.batch_script = {}
        self.template = None
        self.vasp_cmd = None

    def set_vasp_cmd(self, vasp_cmd):
        self.vasp_cmd = vasp_cmd

    def read_batch_script(self, template):
        self.template = template
        if template == "bridges2":
            with importlib.resources.path(
                "dfttk.job_templates", "bridges2.json"
            ) as batch_script_path:
                with open(batch_script_path, "r") as file:
                    self.batch_script = json.load(file)

    def modify_batch_script(self, key, value, position=None, action="add"):
        if key in self.batch_script and key != "commands":
            self.batch_script[key] = value
        elif key == "commands":
            if action == "add":
                if position is None:
                    self.batch_script["commands"].append(value)
                else:
                    self.batch_script["commands"].insert(position, value)
            elif action == "remove" and position is not None:
                if 0 <= position < len(self.batch_script["commands"]):
                    self.batch_script["commands"].pop(position)

    def write_batch_script(self, batch_script_file="job.sh"):
        batch_script_path = os.path.join(self.config_path, batch_script_file)
        if self.template == "bridges2":
            with open(batch_script_path, "w") as file:
                file.write("#!/bin/bash\n")
                file.write(f"#SBATCH --job-name={self.batch_script['job_name']}\n")
                file.write(f"#SBATCH -A {self.batch_script['account']}\n")
                file.write(f"#SBATCH -p {self.batch_script['partition']}\n")
                file.write(f"#SBATCH -N {self.batch_script['nodes']}\n")
                file.write(
                    f"#SBATCH --ntasks-per-node={self.batch_script['ntasks_per_node']}\n"
                )
                file.write(f"#SBATCH -t {self.batch_script['time']}\n")
                file.write(f"#SBATCH -o {self.batch_script['output_file']}\n")
                file.write(f"#SBATCH -e {self.batch_script['error_file']}\n\n")
                for command in self.batch_script["commands"]:
                    file.write(f"{command}\n")

    def run_volume_relax(
        self,
        material_type: str,
        encut: int = 520,
        kppa: int = 4000,
        magmom_fm: bool = False,
        potcar_functional: str = "PBE_54",
        incar_functional: str = "PBE",
        other_settings: dict = {},
    ):

        vasp_input.volume_relax_set(
            self.config_path,
            material_type,
            encut,
            kppa,
            magmom_fm,
            potcar_functional,
            incar_functional,
            other_settings,
        )

        # Run the job
        subprocess.run(["sbatch", "job.sh"], cwd=self.config_path)

    # TODO: add a way to select the custodian handlers
    def run_conv_test(
        self,
        encut: int = 520,
        kppa: int = 4000,
        magmom_fm: bool = False,
        potcar_functional: str = "PBE_54",
        incar_functional: str = "PBE",
        other_settings: dict = {},
        encut_list: list[int] = [
            270,
            320,
            370,
            420,
            470,
            520,
            570,
            620,
            670,
            720,
            770,
            820,
        ],
        kppa_list: list[float] = [
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ],
        force_gamma: bool = True,
        backup: bool = False,
        max_errors: int = 10,
    ):

        vasp_input.conv_set(
            self.config_path,
            encut,
            kppa,
            magmom_fm,
            potcar_functional,
            incar_functional,
            other_settings,
        )

        # Prepare the run_dfttk.py script
        with open(os.path.join(self.config_path, "run_dfttk.py"), "w") as file:
            file.write("import os\n")
            file.write("from custodian.vasp.handlers import VaspErrorHandler\n")
            file.write("import dfttk.workflows as workflows\n")
            file.write("subset = list(VaspErrorHandler.error_msgs.keys())\n")
            file.write("handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]\n")
            file.write(f"vasp_cmd = {self.vasp_cmd}\n")
            file.write(
                f"workflows.encut_conv_test(os.getcwd(), vasp_cmd, handlers, encut_list={encut_list}, backup={backup}, max_errors={max_errors})\n"
            )
            file.write(
                f"workflows.kpoints_conv_test(os.getcwd(), vasp_cmd, handlers, kppa_list={kppa_list}, force_gamma={force_gamma}, backup={backup}, max_errors={max_errors})\n"
            )

        # Run the job
        subprocess.run(["sbatch", "job.sh"], cwd=self.config_path)

    def analyze_encut_conv(self, plot: bool = True):
        encut_conv_path = os.path.join(self.config_path, "encut_conv")
        encut_conv_df, fig = calculate_encut_conv(encut_conv_path, plot)

        return encut_conv_df, fig
    
    def analyze_kpoints_conv(self, plot: bool = True):
        kpoints_conv_path = os.path.join(self.config_path, "kpoints_conv")
        kpoints_conv_df, fig = calculate_kpoint_conv(kpoints_conv_path, plot)

        return kpoints_conv_df, fig
