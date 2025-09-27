"""
Workflows to automate VASP calculations using custodian.
"""

# Standard library imports
import json
import os
import shutil
import sys
import subprocess
import logging
import fnmatch

# Related third party imports
from natsort import natsorted

# Local application/library specific imports
from custodian.custodian import Custodian
from custodian.vasp.jobs import VaspJob
from custodian.vasp.handlers import VaspErrorHandler
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Incar
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.transformations.standard_transformations import SupercellTransformation

# DFTTK imports
from dfttk.magnetism import get_magnetic_structure
from dfttk.data_extraction import extract_tot_mag_data


def move_folders(src_path: str, dest_path: str, folders: list[str]) -> None:
    """Moves folders from the source path to the destination path. It is used to move previous error folders when restarting
    a job to a temporary folder.

    Args:
        src_path (str): path to the source folder containing the error folders.
        dest_path (str): path to the destination folder to move the error folders.
        folders (list[str]): list of error folders to move.
    """
    for folder in folders:
        src = os.path.join(src_path, folder)
        dest = os.path.join(dest_path, folder)
        subprocess.run(["mv", src, dest], check=True)


def rename_error_folders(path: str, error_folders_old: list[str], prev_error_folders_count: int) -> None:
    """Renames the error folders in the path. It is used to rename the error folders when restarting a job according
    to the number of error folders that were moved to a temporary folder from the previous run. For example, if the
    error folders in the temporary folder are error.1.tar and error.2.tar, the error folders in the path will be renamed
    from error.1.tar to error.3.tar and error.2.tar to error.4.tar.

    Args:
        path (str): path to the folder containing the error folders.
        error_folders_old (list[str]): list of error folders to rename.
        prev_error_folders_count (int): count of error folders that was moved using move_folders.
    """
    for error_folder_old in error_folders_old:
        error_folder_new = error_folder_old.copy()
        index = int(error_folder_new[1])
        error_folder_new[1] = str(index + prev_error_folders_count)
        error_folder_new = ".".join(error_folder_new)
        error_folder_old = ".".join(error_folder_old)

        src_path = os.path.join(path, error_folder_old)
        dest_path = os.path.join(path, error_folder_new)
        subprocess.run(["mv", src_path, dest_path], check=True)


def process_error_folders(path: str, prev_error_folders_count: int) -> None:
    """Processes the error folders in the path by renaming and moving them. It is used to rename the error folders when restarting a job according
    to the number of error folders that were moved to a temporary folder from the previous run. For example, if the
    error folders in the temporary folder are error.1.tar and error.2.tar, the error folders in the path will be renamed
    from error.1.tar to error.3.tar and error.2.tar to error.4.tar. Then, the error folders in the temporary folder will be moved to the path
    and the temporary folder will be removed.

    Args:
        path (str): path to the folder containing the error folders.
        prev_error_folders_count (int): count of error folders that was moved using move_folders.
    """
    error_folders_old = [f for f in os.listdir(path) if f.startswith("error.")]
    error_folders_old = [f.split(".") for f in error_folders_old]
    error_folders_old.reverse()
    rename_error_folders(path, error_folders_old, prev_error_folders_count)

    error_temp_path = os.path.join(path, "error_temp")
    error_temp_folders = [f for f in os.listdir(error_temp_path)]
    move_folders(error_temp_path, path, error_temp_folders)

    subprocess.run(["rm", "-r", error_temp_path])


def merge_custodian_json_files(path: str) -> None:
    """Merges the custodian.json and prev_custodian.json files in the path. The prev_custodian.json file is
    the custodian.json file from the previous run, while the custodian.json file is the custodian.json file from
    the current restart run. The merged custodian.json file will contain the combined settings from the two files.

    Args:
        path (str): path to the folder containing the custodian.json and prev_custodian.json files.
    """
    custodian_json_path = os.path.join(path, "custodian.json")
    prev_custodian_json_path = os.path.join(path, "prev_custodian.json")
    if os.path.isfile(prev_custodian_json_path):
        with open(custodian_json_path, "r") as f:
            custodian_json = json.load(f)
        with open(prev_custodian_json_path, "r") as f:
            prev_custodian_json = json.load(f)

        custodian_json[0].update({"restart": True})
        combined_custodian_json = prev_custodian_json + custodian_json
        with open(custodian_json_path, "w") as f:
            json.dump(combined_custodian_json, f, indent=4)

        os.remove(prev_custodian_json_path)


def custodian_errors_location(path: str) -> list[str]:
    """Prints the calculation folders containing the custodian errors in the specified path.

    Args:
        path (str): path to the folder containing all the calculation folders. E.g. vol_1, phonon_1, etc.

    Returns:
        list[str]: volume folders that encountered VASP errors
        list[str]: phonon folders that encountered VASP errors
    """

    vol_folders_errors = []
    vol_folders = [d for d in os.listdir(path) if d.startswith("vol_") and os.path.isdir(os.path.join(path, d))]
    for vol_folder in vol_folders:
        error_folders = [f for f in os.listdir(os.path.join(path, vol_folder)) if f.startswith("error")]
        if len(error_folders) > 0:
            print(f"In {vol_folder} there are error folders: {error_folders}")
            vol_folders_errors.append(vol_folder)

    phonon_folders_errors = []
    phonon_folders = [d for d in os.listdir(path) if d.startswith("phonon") and os.path.isdir(os.path.join(path, d))]
    phonon_folders = natsorted(phonon_folders)
    for phonon_folder in phonon_folders:
        error_folders = [f for f in os.listdir(os.path.join(path, phonon_folder)) if f.startswith("error")]
        if len(error_folders) > 0:
            error_folders = natsorted(error_folders)
            print(f"In {phonon_folder} there are error folders: {error_folders}")
            phonon_folders_errors.append(phonon_folder)

    return vol_folders_errors, phonon_folders_errors


def NELM_reached(path: str, filename: str = "vasp.out.3static") -> list:
    """Prints the path of the calculations that have reached NELM.

    Args:
        path (str): path to the folder containing all the calculation folders. E.g. vol_1, phonon_1, etc.
        filename (str): only check files with this name or pattern (supports wildcards). If None, check all files.

    Returns:
        list: paths of the calculations that have reached NELM
    """

    target_line = "The electronic self-consistency was not achieved in the given"
    NELM_reached_files = []
    for dirpath, dirs, files in os.walk(path):
        for file in files:
            if filename is not None and not fnmatch.fnmatch(file, filename):
                continue
            filepath = os.path.join(dirpath, file)
            try:
                with open(filepath, "r", errors="ignore") as f:
                    content = f.read()
                    if target_line in content:
                        print(f"{filepath} has reached NELM.")
                        NELM_reached_files.append(filepath)
            except Exception:
                continue
    return NELM_reached_files


class SingleJobWorkflow:
    """
    Workflow for running a single VASP job.

    This class encapsulates the setup and execution of a single VASP calculation,
    including python script generation and running the workflow.
    """

    def __init__(
        self,
        path: str,
        vasp_cmd: list[str],
        error_msgs: list[str] = list(VaspErrorHandler.error_msgs.keys()),
        max_errors: int = 10,
        vaspjob_kwargs: dict = None,
        custodian_kwargs: dict = None,
    ):
        """
        Initialize the SingleJobWorkflow.

        Args:
            path (str): Directory where the calculation will be run.
            vasp_cmd (list[str]): Command to run VASP (e.g., ["srun", "vasp_std"]).
            error_msgs (list[str], optional): List of error messages to catch. Defaults to all known VaspErrorHandler messages.
            max_errors (int, optional): Maximum number of errors before stopping. Defaults to 10.
            vaspjob_kwargs (dict, optional): Additional keyword arguments for VaspJob.
            custodian_kwargs (dict, optional): Additional keyword arguments for Custodian.
        """
        self.path = path
        self.vasp_cmd = vasp_cmd
        self.error_msgs = error_msgs
        self.max_errors = max_errors
        self.vaspjob_kwargs = vaspjob_kwargs or {}
        self.custodian_kwargs = custodian_kwargs or {}

    def write_run_dfttk(self) -> None:
        """
        Write a Python script to run this workflow in the specified directory.

        The script will instantiate this workflow and execute the run method.
        """
        # Prepare the script content for running the workflow
        run_dfttk_script = f"""
import os
from dfttk.workflows import SingleJobWorkflow

workflow = SingleJobWorkflow(
    path=".",
    vasp_cmd={self.vasp_cmd},
    error_msgs={self.error_msgs},
    max_errors={self.max_errors},
    vaspjob_kwargs={self.vaspjob_kwargs},
    custodian_kwargs={self.custodian_kwargs}
)
workflow.run()
""".strip()

        # Write the script to the target directory
        with open(os.path.join(self.path, "run_dfttk.py"), "w") as file:
            file.write(run_dfttk_script)

    def run(self) -> None:
        """
        Run the VASP job in the specified directory using Custodian for error handling.

        Changes to the working directory are handled internally and restored after execution.
        """
        original_dir = os.getcwd()
        os.chdir(self.path)
        # Set up the VaspJob with provided arguments
        step1 = VaspJob(vasp_cmd=self.vasp_cmd, final=True, backup=False, **self.vaspjob_kwargs)

        # Set up error handlers and jobs for Custodian
        handlers = [VaspErrorHandler(errors_subset_to_catch=self.error_msgs)]
        jobs = [step1]
        c = Custodian(handlers, jobs, max_errors=self.max_errors, **self.custodian_kwargs)
        c.run()
        os.chdir(original_dir)


def three_step_relaxation(
    path: str,
    vasp_cmd: list[str],
    handlers: list[str],
    copy_magmom: bool = True,
    backup: bool = False,
    override_2relax: list = None,
    override_3static: list = None,
    max_errors: int = 10,
    restart: str = None,
) -> None:
    """Runs a three-step relaxation - two consecutive relaxations followed by one static.

    Args:
        path (str): path to the folder containing the VASP input files
        vasp_cmd (list[str]): VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (list[str]): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        copy_magmom (bool, optional): If True, copies the magmom from an OUTCAR file of one run to the INCAR
        file of the next run. Defaults to False.
        backup (bool, optional): If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
        override_2relax (list, optional): override settings for the second relaxation step. Defaults to None.
        override_3static (list, optional): override settings for the final static step. Defaults to None.
        max_errors (int, optional): maximum number of errors before stopping the calculation. Defaults to 10.
        restart (str, optional): if not None, restarts the calculation from the specified step. Defaults to None.
    """

    if override_2relax is None:
        override_2relax = [{"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}]

    if override_3static is None:
        override_3static = [
            {
                "dict": "INCAR",
                "action": {"_set": {"ALGO": "Normal", "IBRION": -1, "NSW": 0, "ISMEAR": -5}},
            },
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
        ]

    original_dir = os.getcwd()
    os.chdir(path)
    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix=".1relax",
        backup=backup,
    )
    step2 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix=".2relax",
        backup=backup,
        settings_override=override_2relax,
    )
    step3 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=True,
        suffix=".3static",
        backup=backup,
        settings_override=override_3static,
    )

    if restart is None:
        jobs = [step1, step2, step3]
        c = Custodian(handlers, jobs, max_errors=max_errors)
        c.run()

    elif restart == "step2":
        jobs = [step2, step3]
        c = Custodian(handlers, jobs, max_errors=max_errors)
        c.run()

    elif restart == "step3":
        jobs = [step3]
        c = Custodian(handlers, jobs, max_errors=max_errors)
        c.run()

    os.chdir(original_dir)


def ev_curve_series(
    path: str,
    volumes: list[float],
    vasp_cmd: list[str],
    handlers: list[str],
    restarting: bool = False,
    keep_wavecar: bool = False,
    keep_chgcar: bool = False,
    copy_magmom: bool = True,
    override_2relax: list = None,
    override_3static: list = None,
    max_errors: int = 10,
) -> None:
    """Runs a series of three_step_relaxation calculations for a list of volumes.

    Args:
        path (str): path to the folder containing the VASP input files.
        volumes (list[float]): list of volumes
        vasp_cmd (list[str]): VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (list[str]): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        restarting (bool, optional): for restarting failed jobs. Defaults to False.
        keep_wavecar (bool, optional): if True, does not delete WAVECAR.3static. Defaults to False.
        keep_chgcar (bool, optional): if True, does not delete CHGCAR.3static. Defaults to False.
        copy_magmom (bool, optional): If True, copies the magmom from an OUTCAR file of one run to the INCAR
        file of the next run. Defaults to False.
        override_2relax (list, optional): override settings for the second relaxation step. Defaults to None.
        override_3static (list, optional): override settings for the final static step. Defaults to None.
        max_errors (int, optional): maximum number of errors before stopping the calculation. Defaults to 10.
    """

    # Write a params.json file to track the parameters used
    errors_subset = handlers[0].errors_subset_to_catch
    params = {
        "path": path,
        "volumes": volumes,
        "vasp_cmd": vasp_cmd,
        "handlers": errors_subset,
        "restarting": restarting,
    }

    n = 0
    params_json_path = os.path.join(path, "params_" + str(n) + ".json")
    while os.path.isfile(params_json_path):
        n += 1
        params_json_path = os.path.join(path, "params_" + str(n) + ".json")
    with open(params_json_path, "w") as file:
        json.dump(params, file)

    # Currently, restarting supports the following conditions:
    # 1) The volumes list must be greater than or equal to the number of volume folders.
    # 2) The volumes in the volume folders must match the volumes list in order.
    if restarting:
        vol_folders = [folder for folder in os.listdir(path) if os.path.isdir(folder) and folder.startswith("vol_")]
        vol_folders = natsorted(vol_folders)

        volumes_started = []
        for vol_folder in vol_folders:
            try:
                struct = Structure.from_file(os.path.join(path, vol_folder, "POSCAR.1relax"))
                volume_started = round(struct.volume, 6)
            except Exception as e:
                print(f"possible error: {e}, trying POSCAR")
                try:
                    struct = Structure.from_file(os.path.join(path, vol_folder, "POSCAR"))
                    volume_started = round(struct.volume, 6)
                except Exception as e:
                    print(
                        f"Error: {e}. Failed to extract volumes from POSCAR files. Ensure that POSCAR.1relax or POSCAR exists in each volume folder."
                    )
                    sys.exit(1)

            volumes_started.append(volume_started)
            rounded_volumes_supplied = [round(volume, 6) for volume in volumes]

        if not volumes_started == rounded_volumes_supplied[: len(volumes_started)]:
            print(
                f"Error: The completed volumes do not match the input volumes list from the beginning. \n"
                f"Rounded input volumes: {rounded_volumes_supplied} \n"
                f"Started volumes: {volumes_started} \n"
                "Exiting."
            )
            sys.exit(1)
        else:
            print("The completed volumes match the input volumes list from the beginning. Continuing restart.")

        j = len(vol_folders) - 1
        last_vol_folder_name = "vol_" + str(j)
        last_vol_folder_path = os.path.join(path, last_vol_folder_name)

        # If the job failed at the second step of three_step_relaxation, restart from step 3.
        files = os.listdir(last_vol_folder_path)
        files_exist = any(file.endswith(".1relax") for file in files) and any(
            file.endswith(".2relax") for file in files
        )
        files_missing = not any(file.endswith(".3static") for file in files)

        if files_exist and files_missing:

            custodian_json_path = os.path.join(last_vol_folder_path, "custodian.json")
            custodian_old_json_path = os.path.join(last_vol_folder_path, "custodian_old.json")
            if os.path.isfile(custodian_json_path):
                os.rename(custodian_json_path, custodian_old_json_path)

            error_folders = [f for f in os.listdir(last_vol_folder_path) if f.startswith("error.")]
            prev_error_folders_count = len(error_folders)
            error_temp_path = os.path.join(last_vol_folder_path, "error_temp")
            if not os.path.exists(error_temp_path):
                os.makedirs(error_temp_path)

            move_folders(last_vol_folder_path, error_temp_path, error_folders)

            print(f"Running three step relaxation for volume {str(volumes[j])} at step 3")
            three_step_relaxation(
                last_vol_folder_path,
                vasp_cmd,
                handlers,
                copy_magmom=copy_magmom,
                backup=False,
                override_2relax=override_2relax,
                override_3static=override_3static,
                max_errors=max_errors,
                restart="step3",
            )

            process_error_folders(last_vol_folder_path, prev_error_folders_count)
            merge_custodian_json_files(last_vol_folder_path)

            error_folders = [f for f in os.listdir(last_vol_folder_path) if f.startswith("error.")]

            last_vol_index = j + 1

        # If the job failed at the second step of three_step_relaxation, restart from step 2.
        files_exist = any(file.endswith(".1relax") for file in files)
        files_missing = not any(file.endswith(".3static") for file in files) and not any(
            file.endswith(".2relax") for file in files
        )

        if files_exist and files_missing:

            custodian_json_path = os.path.join(last_vol_folder_path, "custodian.json")
            custodian_old_json_path = os.path.join(last_vol_folder_path, "custodian_old.json")
            if os.path.isfile(custodian_json_path):
                os.rename(custodian_json_path, custodian_old_json_path)

            error_folders = [f for f in os.listdir(last_vol_folder_path) if f.startswith("error.")]
            prev_error_folders_count = len(error_folders)
            error_temp_path = os.path.join(last_vol_folder_path, "error_temp")
            if not os.path.exists(error_temp_path):
                os.makedirs(error_temp_path)

            move_folders(last_vol_folder_path, error_temp_path, error_folders)

            print(f"Running three step relaxation for volume {str(volumes[j])} at step 2")
            three_step_relaxation(
                last_vol_folder_path,
                vasp_cmd,
                handlers,
                copy_magmom=copy_magmom,
                backup=False,
                override_2relax=override_2relax,
                override_3static=override_3static,
                max_errors=max_errors,
                restart="step2",
            )

            process_error_folders(last_vol_folder_path, prev_error_folders_count)
            merge_custodian_json_files(last_vol_folder_path)

            error_folders = [f for f in os.listdir(last_vol_folder_path) if f.startswith("error.")]

            last_vol_index = j + 1

        # If the job failed at the first step of three_step_relaxation, delete the folder.
        files_missing = (
            not any(file.endswith(".3static") for file in files)
            and not any(file.endswith(".2relax") for file in files)
            and not any(file.endswith(".1relax") for file in files)
        )
        if files_missing:
            shutil.rmtree(last_vol_folder_path)
            last_vol_index = j

    files_to_delete = [
        "WAVECAR.1relax",
        "WAVECAR.2relax",
        "WAVECAR.3static",
        "CHGCAR.3static",
        "CHGCAR.1relax",
        "CHGCAR.2relax",
        "CHG.1relax",
        "CHG.2relax",
        "CHG.3static",
        "PROCAR.1relax",
        "PROCAR.2relax",
        "PROCAR.3static",
    ]
    if keep_wavecar:
        files_to_delete.remove("WAVECAR.3static")
    if keep_chgcar:
        files_to_delete.remove("CHGCAR.3static")

    # Start the EV curve calculations in series. If restarting, skip the volumes that have already been completed.
    for i, vol in enumerate(volumes):
        if restarting and i < last_vol_index:
            continue

        vol_folder_name = "vol_" + str(i)
        vol_folder_path = os.path.join(path, vol_folder_name)
        os.makedirs(vol_folder_path)

        if i == 0:
            files_to_copy = ["INCAR", "KPOINTS", "POSCAR", "POTCAR"]
            for file_name in files_to_copy:
                if os.path.isfile(os.path.join(path, file_name)):
                    shutil.copy2(
                        os.path.join(path, file_name),
                        os.path.join(vol_folder_path, file_name),
                    )

        else:
            previous_vol_folder_path = os.path.join(path, f"vol_{i - 1}")
            source_name_dest_name = [
                ("CONTCAR.3static", "POSCAR"),
                ("KPOINTS.3static", "KPOINTS"),
                ("POTCAR", "POTCAR"),
                ("WAVECAR.3static", "WAVECAR"),
                ("CHGCAR.3static", "CHGCAR"),
            ]

            for file_name in source_name_dest_name:
                file_source = os.path.join(previous_vol_folder_path, file_name[0])
                file_dest = os.path.join(vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            incar_path = os.path.join(path, "INCAR")
            shutil.copy2(incar_path, os.path.join(vol_folder_path, "INCAR"))

            if copy_magmom:
                outcar_path = os.path.join(previous_vol_folder_path, "OUTCAR.3static")
                contcar_path = os.path.join(previous_vol_folder_path, "CONTCAR.3static")
                tot_data = extract_tot_mag_data(outcar_path, contcar_path)
                magmom = tot_data["tot"].to_list()
                incar = Incar.from_file(os.path.join(vol_folder_path, "INCAR"))
                incar["MAGMOM"] = magmom
                incar.write_file(os.path.join(vol_folder_path, "INCAR"))

            paths_to_delete = []
            for file_name in files_to_delete:
                file_path = os.path.join(previous_vol_folder_path, file_name)
                paths_to_delete.append(file_path)

            for file_path in paths_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                elif restarting and i == last_vol_index:
                    pass

        poscar = os.path.join(vol_folder_path, "POSCAR")
        struct = Structure.from_file(poscar)
        struct.scale_lattice(vol)
        struct.to_file(poscar, "POSCAR")

        print("Running three step relaxation for volume " + str(vol))
        three_step_relaxation(
            vol_folder_path,
            vasp_cmd,
            handlers,
            backup=False,
            copy_magmom=copy_magmom,
            override_2relax=override_2relax,
            override_3static=override_3static,
        )

        error_folders = [f for f in os.listdir(vol_folder_path) if f.startswith("error.")]

    previous_vol_folder_path = os.path.join(path, "vol_" + str(i))
    paths_to_delete = []
    for file_name in files_to_delete:
        file_path = os.path.join(previous_vol_folder_path, file_name)
        paths_to_delete.append(file_path)

    for file_path in paths_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)


# TODO: add a way to restart the job if it has failed
# TODO: add a way to override the default settings
def run_phonons(
    vasp_cmd: list[str],
    handlers: list[str],
    copy_magmom: bool = True,
    backup: bool = False,
    max_errors: int = 10,
    relax: bool = True,
):
    """Runs a relaxation followed by a phonon calculation.

    Args:
        vasp_cmd (list[str]): VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (list[str]): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        copy_magmom (bool, optional): If True, copies the magmom from an OUTCAR file of one run to the INCAR
        file of the next run. Defaults to False.
        backup (bool, optional): If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
        max_errors (int, optional): maximum number of errors before stopping the calculation. Defaults to 10.
        relax (bool, optional): if True, runs a relaxation before the phonon calculation. Defaults to True.
    """

    settings_override = [
        {
            "dict": "INCAR",
            "action": {
                "_set": {
                    "EDIFF": "1E-6",
                    "IBRION": 6,
                    "NSW": 1,
                    "ISIF": 0,
                    "POTIM": 0.015,
                    "ISYM": 2,
                    "NCORE": 1,
                }
            },
        },
        {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
    ]

    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix=".1relax",
        backup=backup,
    )

    if relax is True:
        step2 = VaspJob(
            vasp_cmd=vasp_cmd,
            copy_magmom=copy_magmom,
            final=True,
            suffix=".2phonons",
            backup=backup,
            settings_override=settings_override,
        )
        jobs = [step1, step2]

    else:
        settings_override = [settings_override[0]]
        step2 = VaspJob(
            vasp_cmd=vasp_cmd,
            copy_magmom=copy_magmom,
            final=True,
            suffix=".2phonons",
            backup=backup,
            settings_override=settings_override,
        )
        jobs = [step2]

    c = Custodian(handlers, jobs, max_errors=max_errors)
    c.run()


def phonons_parallel(
    path: str,
    phonon_volumes: list[float],
    kppa: int,
    run_file: str,
    scaling_matrix: tuple[tuple[int]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
) -> None:
    """Executes the run_phonons function in parallel for a list of phonon volumes.

    Args:
        path: path to the folder containing the vol_* folders.
        phonon_volumes: a list of volumes to run the phonon calculations for. Must match the volumes in the vol_* folders.
        kppa: kppa of the phonon calculations.
        run_file: job script to run the phonon calculations.
        scaling_matrix: scaling matrix for the supercell. Defaults to ((1, 0, 0), (0, 1, 0), (0, 0, 1)).
        relax (bool, optional): whether to run a relaxation before the phonon calculation. Defaults to True.
    """

    # Create a new run_file to run the phonon calculations
    script_name = sys.argv[0]
    with open(script_name, "r") as file:
        script_contents = file.read()
        script_contents = "\n".join(
            [line for line in script_contents.split("\n") if "workflows.phonons_parallel" not in line]
        )

    with open(run_file, "r") as file:
        run_file_contents = file.read()
    new_run_file = run_file_contents + "\n"
    new_run_file += "\n"
    new_run_file += "python << END_OF_PYTHON\n"
    new_run_file += script_contents
    new_run_file += "\nworkflows.run_phonons(vasp_cmd=vasp_cmd, handlers=handlers, copy_magmom=copy_magmom, backup=backup, max_errors=max_errors, relax=relax)\n"
    new_run_file += "workflows.custodian_errors_location(os.getcwd())\n"
    new_run_file += "workflows.NELM_reached(os.getcwd())\n"
    new_run_file += "END_OF_PYTHON\n"

    # Copy files from the vol_* folders to the phonon_* folders
    vol_folders = [
        folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and folder.startswith("vol_")
    ]

    ev_volumes_finished = []
    ev_folder_names = []
    for vol_folder in vol_folders:
        structure_path = os.path.join(path, vol_folder, "CONTCAR.3static")
        structure = Structure.from_file(structure_path)
        ev_volumes_finished.append(round(structure.volume, 6))
        ev_folder_names.append(vol_folder)

    ev_volumes_and_folders_finished = [[a, b] for a, b in zip(ev_volumes_finished, ev_folder_names)]

    for i in range(len(ev_volumes_and_folders_finished)):
        ev_volumes_and_folders_finished[i][1] = ev_volumes_and_folders_finished[i][1].replace("vol_", "")

    phonon_volumes_and_folders = []
    for ev_volume_finished, folder in ev_volumes_and_folders_finished:
        if ev_volume_finished in phonon_volumes:
            phonon_volumes_and_folders.append([ev_volume_finished, folder])

    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        os.makedirs(os.path.join(path, f"phonon_{phonon_folder}"), exist_ok=True)

    source_name_dest_name = [
        ("CONTCAR.3static", "POSCAR"),
        ("INCAR.2relax", "INCAR"),
        ("POTCAR", "POTCAR"),
        ("WAVECAR.3static", "WAVECAR"),
    ]

    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        for source_name, dest_name in source_name_dest_name:
            file_source = os.path.join(path, f"vol_{phonon_folder}", source_name)
            file_dest = os.path.join(path, f"phonon_{phonon_folder}", dest_name)
            if os.path.isfile(file_source):
                shutil.copy2(file_source, file_dest)

    transformation = SupercellTransformation(scaling_matrix)
    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        try:
            # Try to get the magnetic structure
            structure = get_magnetic_structure(
                os.path.join(path, f"vol_{phonon_folder}", "CONTCAR.3static"),
                os.path.join(path, f"vol_{phonon_folder}", "OUTCAR.3static"),
            )
            scaled_structure = transformation.apply_transformation(structure)
            structure_magmoms = scaled_structure.site_properties["magmom"]

            # Read the INCAR file
            incar_path = os.path.join(path, f"phonon_{phonon_folder}", "INCAR")
            incar = Incar.from_file(incar_path)

            # Update the INCAR file with the new MAGMOM values for the scaled structure
            incar["MAGMOM"] = structure_magmoms
            incar.write_file(incar_path)

        except ValueError:
            # Swallow the ValueError and continue without outputting anything
            structure = Structure.from_file(os.path.join(path, f"phonon_{phonon_folder}", "POSCAR"))
            scaled_structure = transformation.apply_transformation(structure)
            pass
        except Exception as e:
            # Handle other exceptions
            print(f"An error occurred: {e}")

        # Write the POSCAR file for the scaled structure
        scaled_structure.to_file(os.path.join(path, f"phonon_{phonon_folder}", "POSCAR"), "POSCAR")

        # Write the KPOINTS file for the scaled structure
        kpoints = Kpoints.automatic_density(scaled_structure, kppa, force_gamma=True)
        kpoints.write_file(os.path.join(path, f"phonon_{phonon_folder}", "KPOINTS"))

    # Run the phonon calculations in parallel
    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        os.chdir(os.path.join(path, f"phonon_{phonon_folder}"))
        with open("run_phonons", "w") as file:
            file.write(new_run_file)
        os.system("sbatch run_phonons")
        os.chdir(path)


def process_phonon_dos_YPHON(path: str):
    """Processes the phonon DOS calculations using YPHON.

    Args:
        path (str): path to the folder containing all the phonon calculation folders. E.g. phonon_1, phonon_2, etc.
    """

    # Reset logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    log_file_path = os.path.join(path, "phonons_parallel.log")

    # If log_file_path already exists, delete it
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    logging.basicConfig(
        filename=log_file_path,
        level=logging.ERROR,  # Set the log level to ERROR
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Process the phonon dos in each phonon_folder
    phonon_folders = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder)) and folder.startswith("phonon")
    ]
    for phonon_folder in phonon_folders:
        try:
            phonon_dos_folder = os.path.join(path, phonon_folder, "phonon_dos")
            phonon_folder = os.path.join(path, phonon_folder)
            if not os.path.exists(phonon_dos_folder):
                os.makedirs(phonon_dos_folder, exist_ok=True)

            # Copy the necessary files to the phonon_dos folder
            shutil.copy(
                os.path.join(phonon_folder, "CONTCAR.2phonons"),
                os.path.join(phonon_dos_folder, "CONTCAR"),
            )
            shutil.copy(
                os.path.join(phonon_folder, "OUTCAR.2phonons"),
                os.path.join(phonon_dos_folder, "OUTCAR"),
            )
            shutil.copy(
                os.path.join(phonon_folder, "vasprun.xml.2phonons"),
                os.path.join(phonon_dos_folder, "vasprun.xml"),
            )

            # Write the volume_per_atom to volph_* files
            index = phonon_folder.split("_")[-1]
            structure = Structure.from_file(os.path.join(phonon_dos_folder, "CONTCAR"))
            number_of_atoms = structure.num_sites
            volume = round(structure.volume, 6)
            volume_per_atom = volume / number_of_atoms
            with open(os.path.join(phonon_dos_folder, "volph_" + index), "w") as f:
                f.write(str(volume_per_atom))

            # Run YPHON commands to generate phonon DOS
            subprocess.run(["vasp_fij"], cwd=phonon_dos_folder)
            subprocess.run(["Yphon <superfij.out"], cwd=phonon_dos_folder, shell=True)

            # Rename the phonon DOS file to vdos_*
            os.rename(
                os.path.join(phonon_dos_folder, "vdos.out"),
                os.path.join(phonon_dos_folder, "vdos_" + index),
            )

        except Exception as e:
            logging.error(f"Error processing folder {phonon_folder}: {e}")

    # Copy all volph_* and vdos_* files to YPHON_results folder
    os.makedirs(os.path.join(path, "YPHON_results"), exist_ok=True)
    for phonon_folder in phonon_folders:
        try:
            phonon_dos_folder = os.path.join(path, phonon_folder, "phonon_dos")
            phonon_folder = os.path.join(path, phonon_folder)
            index = phonon_folder.split("_")[-1]
            shutil.copy(
                os.path.join(phonon_dos_folder, "vdos_" + index),
                os.path.join(path, "YPHON_results", "vdos_" + index),
            )
            shutil.copy(
                os.path.join(phonon_dos_folder, "volph_" + index),
                os.path.join(path, "YPHON_results", "volph_" + index),
            )

        except Exception as e:
            logging.error(f"Error copying files from {phonon_folder}: {e}")

        # Delete log_file_path if it is empty
        if os.path.exists(log_file_path) and os.path.getsize(log_file_path) == 0:
            os.remove(log_file_path)


def run_elec_dos(
    vasp_cmd: list[str],
    handlers: list[str],
    NEDOS: int = 10001,
    backup: bool = False,
    max_errors: int = 10,
):
    """Runs an electronic DOS calculation.

    Args:
        vasp_cmd (list[str]): VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (list[str]): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        copy_magmom (bool, optional): If True, copies the magmom from an OUTCAR file of one run to the INCAR
        file of the next run. Defaults to False.
        backup (bool, optional): If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
        max_errors (int, optional): maximum number of errors before stopping the calculation. Defaults to 10.
    """

    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        final=True,
        suffix=".elec_dos",
        backup=backup,
        settings_override=[
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "EDIFF": "1E-6",
                        "NELM": 200,
                        "IBRION": -1,
                        "NSW": 0,
                        "ISMEAR": -5,
                        "LORBIT": 11,
                        "NEDOS": NEDOS,
                    }
                },
            },
        ],
    )

    jobs = [step1]
    c = Custodian(handlers, jobs, max_errors=max_errors)
    c.run()


def elec_dos_parallel(
    path: str,
    volumes: list[float],
    kppa: float,
    run_file: str,
    scaling_matrix: tuple[tuple[int]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
) -> None:
    """Runs the run_elec_dos function in parallel for a list of volumes.

    Args:
        path: path to the folder containing the VASP input files.
        volumes: a list of volumes to run the electron DOS calculations for.
        kppa: k-point grid density.
        run_file: bash script to run the electron DOS calculations.
        scaling_matrix: scaling matrix for the supercell. The default is the identity matrix.
    """

    # Create a new run_file to run the electron DOS calculations
    script_name = sys.argv[0]
    with open(script_name, "r") as file:
        script_contents = file.read()
        script_contents = "\n".join(
            [line for line in script_contents.split("\n") if "workflows.elec_dos_parallel" not in line]
        )

    with open(run_file, "r") as file:
        run_file_contents = file.read()

    new_run_file = run_file_contents + "\n"
    new_run_file += "\n"
    new_run_file += "python << END_OF_PYTHON\n"
    new_run_file += script_contents
    new_run_file += "\nworkflows.run_elec_dos(vasp_cmd=vasp_cmd, handlers=handlers, NEDOS=NEDOS, backup=backup, max_errors=max_errors)\n"
    new_run_file += "workflows.custodian_errors_location(os.getcwd())\n"
    new_run_file += "workflows.NELM_reached(os.getcwd())\n"
    new_run_file += "END_OF_PYTHON\n"

    # Copy files to elec folders
    vol_folders = [
        folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and folder.startswith("vol_")
    ]
    vol_folders = natsorted(vol_folders)

    ev_volumes_finished = []
    ev_folder_names = []
    for vol_folder in vol_folders:
        structure_path = os.path.join(path, vol_folder, "CONTCAR.3static")
        struct = Structure.from_file(structure_path)
        ev_volumes_finished.append(round(struct.volume, 6))
        ev_folder_names.append(vol_folder)

    ev_volumes_and_folders_finished = [[a, b] for a, b in zip(ev_volumes_finished, ev_folder_names)]

    for i in range(len(ev_volumes_and_folders_finished)):
        ev_volumes_and_folders_finished[i][1] = ev_volumes_and_folders_finished[i][1].replace("vol_", "")

    elec_volumes_and_folders = []
    for ev_volume_finished, folder in ev_volumes_and_folders_finished:
        if ev_volume_finished in volumes:
            elec_volumes_and_folders.append([ev_volume_finished, folder])

    for volume, elec_folder in elec_volumes_and_folders:
        os.makedirs(os.path.join(path, f"elec_{elec_folder}"), exist_ok=True)

    source_name_dest_name = [
        ("CONTCAR.3static", "POSCAR"),
        ("INCAR.2relax", "INCAR"),
        ("POTCAR", "POTCAR"),
        ("WAVECAR.3static", "WAVECAR"),
    ]

    for volume, elec_folder in elec_volumes_and_folders:
        for source_name, dest_name in source_name_dest_name:
            file_source = os.path.join(path, f"vol_{elec_folder}", source_name)
            file_dest = os.path.join(path, f"elec_{elec_folder}", dest_name)
            if os.path.isfile(file_source):
                shutil.copy2(file_source, file_dest)

    # Create a supercell and write the KPOINTS file
    transformation = SupercellTransformation(scaling_matrix)

    for volume, elec_folder in elec_volumes_and_folders:
        try:  # to get a magnetic structure
            structure = get_magnetic_structure(
                os.path.join(path, f"vol_{elec_folder}", "CONTCAR.3static"),
                os.path.join(path, f"vol_{elec_folder}", "OUTCAR.3static"),
            )  # if magnetic data not in OUTCAR, will raise an exception.
            structure = transformation.apply_transformation(structure)
            structure.to_file(os.path.join(path, f"elec_{elec_folder}", "POSCAR"), "POSCAR")
            structure_magmoms = structure.site_properties["magmom"]
            numeric_strings = [str(value) for value in structure_magmoms]
            magmom_string = " ".join(numeric_strings)
            magmom_line = "MAGMOM = " + magmom_string
            # Write the magmom_line to the INCAR file
            incar_file = os.path.join(path, f"elec_{elec_folder}", "INCAR")
            with open(incar_file, "r") as file:
                lines = file.readlines()
            with open(incar_file, "w") as file:
                for line in lines:
                    if line.startswith("MAGMOM ="):
                        file.write(magmom_line + "\n")
                    else:
                        file.write(line)
        except Exception as e:
            structure = Structure.from_file(os.path.join(path, f"elec_{elec_folder}", "POSCAR"))
            structure = transformation.apply_transformation(structure)
            structure.to_file(os.path.join(path, f"elec_{elec_folder}", "POSCAR"), "POSCAR")

        kpoints = Kpoints.automatic_density(structure, kppa, force_gamma=True)
        kpoints.write_file(os.path.join(path, f"elec_{elec_folder}", "KPOINTS"))

    # Run the electronic DOS calculations in parallel
    for volume, elec_folder in elec_volumes_and_folders:
        os.chdir(os.path.join(path, f"elec_{elec_folder}"))
        with open("run_elec_dos", "w") as file:
            file.write(new_run_file)
        os.system("sbatch run_elec_dos")
        os.chdir(path)


def charge_density_difference(
    path: str,
    vasp_cmd: list[str],
    handlers: list[str],
    backup: bool = False,
    max_errors: int = 10,
) -> Chgcar:
    """Runs a charge density difference calculation. The charge_density_difference is calculated as the difference between the
    charge density of the final electronic step and the charge density of a single step.

    Args:
        path (str): path that contains the VASP input files.
        vasp_cmd (list[str]):  VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (list[str]): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        backup (bool, optional): If True, the starting INCAR, KPOINTS, POSCAR and POTCAR files will be copied with a “.orig”
        appended. Defaults to False.
        max_errors (int, optional): maximum number of errors before stopping the calculation. Defaults to 10.

    Returns:
        Chgcar: The charge density difference between the final electronic step and a single step.
    """

    original_dir = os.getcwd()
    os.chdir(path)
    os.mkdir("charge_density_difference")
    shutil.copy2("POSCAR", "charge_density_difference/POSCAR")
    shutil.copy2("POTCAR", "charge_density_difference/POTCAR")
    shutil.copy2("INCAR", "charge_density_difference/INCAR")
    shutil.copy2("KPOINTS", "charge_density_difference/KPOINTS")
    os.chdir("charge_density_difference")

    reference_job = VaspJob(
        vasp_cmd=vasp_cmd,
        final=False,
        suffix=".reference",
        backup=backup,
        settings_override=[
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "EDIFF": "1E-6",
                        "IBRION": -1,
                        "NSW": 1,
                        "ISIF": 2,
                        "NELM": 1,
                        "ISMEAR": -5,
                        "SIGMA": 0.05,
                        "LCHARG": True,
                    }
                },
            },
        ],
    )

    charge_density_job = VaspJob(
        vasp_cmd=vasp_cmd,
        final=True,
        suffix=".charge_density",
        backup=backup,
        settings_override=[
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "EDIFF": "1E-6",
                        "IBRION": -1,
                        "NSW": 1,
                        "ISIF": 2,
                        "NELM": 100,
                        "ISMEAR": -5,
                        "SIGMA": 0.05,
                        "LCHARG": True,
                    }
                },
            },
        ],
    )

    jobs = [reference_job, charge_density_job]
    c = Custodian(handlers, jobs, max_errors=max_errors)
    c.run()

    final = Chgcar.from_file("CHGCAR.charge_density")
    reference = Chgcar.from_file("CHGCAR.reference")
    difference = final - reference
    difference.write_file("CHGCAR.difference")

    os.chdir(original_dir)

    return difference


def kpoints_conv_test(
    path: str,
    vasp_cmd: list[str],
    handlers: list[str],
    kppa_list: list[int] = list(range(1000, 10001, 1000)),
    force_gamma: bool = True,
    backup: bool = False,
    max_errors: int = 10,
):
    """Runs a series of VASP calculations with different k-point densities for convergence testing.

    Args:
        path (str): the path to the folder containing the VASP input files
        vasp_cmd (list[str]): the VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (list[str]): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        kppa_list (list[int], optional): k-point densities. Defaults to [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000].
        force_gamma (bool, optional):If True, forces a gamma-centered mesh. Defaults to True.
        backup (bool, optional): If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
        max_errors (int, optional): maximum number of errors before stopping the calculation. Defaults to 10.
    """

    original_dir = os.getcwd()
    kpoints_conv_dir = os.path.join(path, "kpoints_conv")
    os.makedirs(kpoints_conv_dir)

    shutil.copy2(os.path.join(path, "POSCAR"), os.path.join(kpoints_conv_dir, "POSCAR"))
    shutil.copy2(os.path.join(path, "POTCAR"), os.path.join(kpoints_conv_dir, "POTCAR"))
    shutil.copy2(os.path.join(path, "INCAR"), os.path.join(kpoints_conv_dir, "INCAR"))

    os.chdir(kpoints_conv_dir)
    struct = Structure.from_file("POSCAR")
    for i, kppa in enumerate(kppa_list):
        kpoints = Kpoints.automatic_density(struct, kppa, force_gamma=force_gamma)
        kpoints.write_file("KPOINTS")

        if i == len(kppa_list) - 1:
            final = True
        else:
            final = False

        job = VaspJob(
            vasp_cmd=vasp_cmd,
            final=final,
            backup=backup,
            suffix=f".{kppa}",
            settings_override=[{"dict": "INCAR", "action": {"_set": {"IBRION": -1, "NSW": 0}}}],
        )
        c = Custodian(handlers, [job], max_errors=max_errors)
        c.run()

        if os.path.isfile(f"WAVECAR.{kppa}"):
            os.remove(f"WAVECAR.{kppa}")
        if os.path.isfile(f"CHGCAR.{kppa}"):
            os.remove(f"CHGCAR.{kppa}")
        if os.path.isfile(f"CHG.{kppa}"):
            os.remove(f"CHG.{kppa}")
        if os.path.isfile(f"PROCAR.{kppa}"):
            os.remove(f"PROCAR.{kppa}")
    os.chdir(original_dir)


def encut_conv_test(
    path: str,
    vasp_cmd: list[str],
    handlers: list[str],
    encut_list: list[int] = list(range(270, 821, 50)),
    backup: bool = False,
    max_errors: int = 10,
):
    """Runs a series of VASP calculations with different ENCUT values for convergence testing.

    Args:
        path (str): path to the folder containing the VASP input files
        vasp_cmd (list[str]): VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (list[str]): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        encut_list (list[int], optional): list of ENCUT values to run the calculations for.
        Defaults to [270, 320, 370, 420, 470, 520, 570, 620, 670, 720, 770, 820].
        backup (bool, optional):If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with .orig. Defaults to False.
        max_errors (int, optional): maximum number of errors before stopping the calculation. Defaults to 10.
    """

    original_dir = os.getcwd()
    encut_conv_dir = os.path.join(path, "encut_conv")
    os.makedirs(encut_conv_dir)

    shutil.copy2(os.path.join(path, "POSCAR"), os.path.join(encut_conv_dir, "POSCAR"))
    shutil.copy2(os.path.join(path, "KPOINTS"), os.path.join(encut_conv_dir, "KPOINTS"))
    shutil.copy2(os.path.join(path, "POTCAR"), os.path.join(encut_conv_dir, "POTCAR"))
    shutil.copy2(os.path.join(path, "INCAR"), os.path.join(encut_conv_dir, "INCAR"))

    os.chdir(encut_conv_dir)
    for i, encut in enumerate(encut_list):
        if i == len(encut_list) - 1:
            final = True
        else:
            final = False

        job = VaspJob(
            vasp_cmd=vasp_cmd,
            final=final,
            backup=backup,
            suffix=f".{encut}",
            settings_override=[
                {
                    "dict": "INCAR",
                    "action": {"_set": {"IBRION": -1, "NSW": 0, "ENCUT": encut}},
                }
            ],
        )
        c = Custodian(handlers, [job], max_errors=max_errors)
        c.run()

        if os.path.isfile(f"WAVECAR.{encut}"):
            os.remove(f"WAVECAR.{encut}")
        if os.path.isfile(f"CHGCAR.{encut}"):
            os.remove(f"CHGCAR.{encut}")
        if os.path.isfile(f"CHG.{encut}"):
            os.remove(f"CHG.{encut}")
        if os.path.isfile(f"PROCAR.{encut}"):
            os.remove(f"PROCAR.{encut}")
    os.chdir(original_dir)
