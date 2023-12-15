import os
import shutil
import pandas as pd

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler
from custodian.vasp.jobs import VaspJob
from pymatgen.core import structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

"""
kpoints_list should be a list of strings ex:
    ['1 1 1', '2 2 2', '3 3 3']
incar_tags should be a dictionary ex:
    {'encut' 'ISMEAR': -5, 'IBRION': 2}
only edits the forth line of the KPOINTS file
"""


# Function to extract the last occurrence of volume from OUTCAR files
def extract_volume(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'volume' in line:
                volume = float(line.split()[-1])
                break  # Stop searching after finding the last occurrence
    return volume

# Function to extract the last occurrence of pressure from OUTCAR files
def extract_pressure(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'pressure' in line:
                pressure = float(line.split()[3])
                break  # Stop searching after finding the last occurrence
    return pressure

# Function to extract energy from OSZICAR files
def extract_energy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'F=' in line:
                energy = float(line.split()[4])
                break  # Stop searching after finding the last occurrence
    return energy

def extract_mag_data(outcar_path='OUTCAR'):
    if not os.path.isfile(outcar_path):
        print(f"Warning: File {outcar_path} does not exist. Skipping.")
        return None
    with open(outcar_path, 'r') as file:
        data = []
        step = 0
        found_mag_data = False
        data_start = False
        lines = file.readlines()
        for line in lines:
            if 'magnetization (x)' in line:
                found_mag_data = True
                step += 1
            elif found_mag_data and not data_start and '----' in line:
                data_start = True
            elif data_start and '----' not in line:
                ion = int(line.split()[0])
                s = float(line.split()[1])
                p = float(line.split()[2])
                d = float(line.split()[3])
                tot = float(line.split()[4])
                data.append((step, ion, s, p, d, tot))
            elif data_start and '----' in line:
                data_start = False
                found_mag_data = False
        df = pd.DataFrame(data, columns=['step', '# of ion', 's', 'p', 'd', 'tot'])
        return df

#returns only the 'tot' magnetization of the last step for each specified ion
def extract_simple_mag_data(ion_list, outcar_path='OUTCAR'):
    all_mag_data = get_mag_data(outcar_path)
    last_step_data = all_mag_data[all_mag_data['step'] == all_mag_data['step'].max()]
    simple_data = last_step_data[last_step_data['# of ion'].isin(ion_list)][['# of ion', 'tot']]
    simple_data.reset_index(drop=True, inplace=True)
    return simple_data

def three_step_relaxation(path, vasp_cmd, handlers, backup=True): #path should contain necessary vasp config files
    orginal_dir = os.getcwd()
    os.chdir(path)
    step1 = VaspJob(
    vasp_cmd = vasp_cmd,
    copy_magmom = True,
    final = False,
    suffix = '.1relax',
    backup = backup,
            )
    
    step2 = VaspJob(
    vasp_cmd = vasp_cmd,
    copy_magmom = True,
    final = False,
    suffix = '.2relax',
    backup = backup,
    settings_override = [
        {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
            )
    
    step3 = VaspJob(
    vasp_cmd = vasp_cmd,
    copy_magmom = True,
    final = True,
    suffix = '.3static',
    backup = backup,
    settings_override = [
        {"dict": "INCAR", "action": {"_set": {
            "IBRION": -1,
        "NSW": 0,
            "ISMEAR": -5
            }}},
        {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
            )

    jobs = [step1, step2, step3]
    c = Custodian(handlers, jobs, max_errors = 3)
    c.run()
    os.chdir(orginal_dir)

"""
!!!WARNING!!! You probably want to have volumes in decreasing order eg.
volumes = []
for vol in range(300, 370, 10):
    volumes.append(vol)
    volumes.reverse()
"""
def wavecar_prop_series(path, volumes, vasp_cmd, handlers): #path should contain starting POSCAR, POTCAR, INCAR, KPOINTS
    for i, vol in enumerate(volumes):
        #create vol folder
        vol_folder_name = 'vol_' + str(i)
        vol_folder_path = os.path.join(path, vol_folder_name)
        os.makedirs(vol_folder_path)

        if i == 0: #copy from path
            files_to_copy = ['INCAR', 'KPOINTS', 'POSCAR', 'POTCAR']
            for file_name in files_to_copy:
                if os.path.isfile(os.path.join(path, file_name)):
                    shutil.copy2(os.path.join(path, file_name), os.path.join(vol_folder_path, file_name))
        else: #copy from previous folder and delete WAVECARs, CHGCARs, CHGs, PROCARs from previous volume folder
            previous_vol_folder_path = os.path.join(path, 'vol_' + str(i-1))
            source_name_dest_name = [('CONTCAR.3static', 'POSCAR'),
                                ('INCAR.2relax', 'INCAR'),
                                ('KPOINTS.1relax', 'KPOINTS'),
                                ('POTCAR', 'POTCAR'),
                                ('WAVECAR.3static', 'WAVECAR'),
                                ('CHGCAR.3static', 'CHGCAR')]
            for file_name in source_name_dest_name:
                file_source = os.path.join(previous_vol_folder_path, file_name[0])
                file_dest = os.path.join(vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)
            #after copying, it is safe to delete the WAVECARS, CHGCARS, CHG and PROCARS from the previous volume folder to save space
            #keeps WAVECAR.3static and CHGCAR.3static
            files_to_delete = ['WAVECAR.1relax', 'WAVECAR.2relax',
                            'CHGCAR.1relax', 'CHGCAR.2relax',
                            'CHG.1relax','CHG.2relax', 'CHG.3static',
                            'PROCAR.1relax','PROCAR.2relax', 'PROCAR.3static']
            paths_to_deltete = []
            for file_name in files_to_delete:
                file_path = os.path.join(previous_vol_folder_path, file_name)
                paths_to_deltete.append(file_path)

            for file_path in paths_to_deltete:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        else:
                            print(f"The file {file_path} does not exist.")

        #change the volume of the POSCAR
        poscar = os.path.join(vol_folder_path, 'POSCAR')
        struct = structure.Structure.from_file(poscar)
        struct.scale_lattice(vol)
        struct.to_file(poscar, "POSCAR")
        
        #run vasp
        print('running three step relaxation for volume ' + str(vol))
        three_step_relaxation(vol_folder_path, vasp_cmd, handlers, backup=False)

def kpoints_conv_test(path, kpoints_list, vasp_cmd, handlers, backup=False): #path should contain starting POSCAR, POTCAR, INCAR, KPOINTS
    original_dir = os.getcwd()
    kpoints_conv_dir = os.path.join(path, 'kpoints_conv')
    os.makedirs(kpoints_conv_dir)
    shutil.copy2(os.path.join(path, 'POSCAR'), os.path.join(kpoints_conv_dir, 'POSCAR'))
    shutil.copy2(os.path.join(path, 'POTCAR'), os.path.join(kpoints_conv_dir, 'POTCAR'))
    shutil.copy2(os.path.join(path, 'INCAR'), os.path.join(kpoints_conv_dir, 'INCAR'))
    shutil.copy2(os.path.join(path, 'KPOINTS'), os.path.join(kpoints_conv_dir, 'KPOINTS'))
    os.chdir(kpoints_conv_dir)
    for i, el in enumerate(kpoints_list):
        #change the kpoints file
        with open('KPOINTS', 'r') as file:
            lines = file.readlines()
            lines[3] = el + '\n'

        with open('KPOINTS', 'w') as file:
            file.writelines(lines)
        
        #run the vasp job
        if i == len(kpoints_list) - 1:
            final = True
        else:
            final = False
        job = VaspJob(
        vasp_cmd = vasp_cmd,
        final=False,
        suffix = f'.{i}',
        backup = backup
                )
        c = Custodian(handlers, [job], max_errors = 3)
        c.run()
        if os.path.isfile(f'WAVECAR.[i-1]'):
            os.remove(f'WAVECAR.[i-1]')
        if os.path.isfile(f'CHGCAR.[i-1]'):
            os.remove(f'CHGCAR.[i-1]')
        if os.path.isfile(f'CHG.[i-1]'):
            os.remove(f'CHG.[i-1]')
        if os.path.isfile(f'PROCAR.[i-1]'):
            os.remove(f'PROCAR.[i-1]')
    os.chdir(original_dir)



if __name__ == "__main__":
    subset = list(VaspErrorHandler.error_msgs.keys())
    subset.remove("algo_tet")

    handlers = [VaspErrorHandler(errors_subset_to_catch = subset)]
    vasp_cmd = ["srun", "vasp_std"]

    volumes = []

    # for vol in range(300, 370, 10):
    #     volumes.append(vol)
    # wavecar_prop_series(os.getcwd(), volumes, vasp_cmd, handlers)

    kpoints_list = ['4 4 5', '5 5 6', '6 6 7', '7 7 8', '7 7 9', '8 8 10', '9 9 11']
    kpoints_conv_test(os.getcwd(), kpoints_list, vasp_cmd, handlers, backup=False)